import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from model.model import Scaler


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create a schedule with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a ratio of initial lr (default 0.1 = 10% of initial lr).
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay to min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class Trainer(object):
    def __init__(self, config, policy, qf):
        self.freeze_policy = False
        self.config = config
        self.policy = policy
        self.qf = qf
        assert len(qf) == 4, "Expected two Q-functions and their targets."
        self.optimizers = {}
        self.optimizers["policy"] = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.optimizers['qf1'] = optim.Adam(self.qf['qf1'].parameters(), lr=config.qf_lr)
        self.optimizers['qf2'] = optim.Adam(self.qf['qf2'].parameters(), lr=config.qf_lr)

        # CQL Lagrange multiplier (no entropy regularization)
        self.log_alpha_prime = Scaler(self.config.alpha_prime)
        self.optimizers["log_alpha_prime"] = optim.Adam(self.log_alpha_prime.parameters(), lr=config.alpha_prime_lr)
            
        self._total_steps = 0
        self._modules = [self.policy, self.qf['qf1'], self.qf['qf2'], self.qf['target_qf1'], self.qf['target_qf2']]
        self._modules.append(self.log_alpha_prime)

        # Learning rate schedulers (initialized later with setup_lr_scheduler)
        self.schedulers = {}
        self._use_scheduler = False

    def setup_lr_scheduler(self, num_training_steps, warmup_ratio=0.05, min_lr_ratio=0.1):
        """
        Setup learning rate schedulers with warmup and cosine decay.

        Args:
            num_training_steps: Total number of training steps.
            warmup_ratio: Ratio of warmup steps (default 5%).
            min_lr_ratio: Minimum lr as ratio of initial lr (default 10%).
        """
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        self.schedulers['policy'] = get_cosine_schedule_with_warmup(
            self.optimizers['policy'], num_warmup_steps, num_training_steps, min_lr_ratio
        )
        self.schedulers['qf1'] = get_cosine_schedule_with_warmup(
            self.optimizers['qf1'], num_warmup_steps, num_training_steps, min_lr_ratio
        )
        self.schedulers['qf2'] = get_cosine_schedule_with_warmup(
            self.optimizers['qf2'], num_warmup_steps, num_training_steps, min_lr_ratio
        )
        # alpha_prime uses constant lr (no scheduler)

        self._use_scheduler = True
        print(f"LR scheduler: {num_training_steps} total steps, {num_warmup_steps} warmup steps, min_lr_ratio={min_lr_ratio}")

    def step_scheduler(self):
        """Step all learning rate schedulers."""
        if self._use_scheduler:
            for scheduler in self.schedulers.values():
                scheduler.step()

    def get_lr(self):
        """Get current learning rates."""
        return {
            'policy_lr': self.optimizers['policy'].param_groups[0]['lr'],
            'qf_lr': self.optimizers['qf1'].param_groups[0]['lr'],
        }

    def train(self, batch, cql_min_q_weight=5.0):
        self._total_steps += 1
        metrics = self._train_step(batch, cql_min_q_weight=cql_min_q_weight)

        # Step schedulers after each training step
        self.step_scheduler()

        # Add learning rates to metrics
        if self._use_scheduler:
            lr_info = self.get_lr()
            metrics.update({
                'lr/policy': lr_info['policy_lr'],
                'lr/qf': lr_info['qf_lr'],
            })

        return metrics

    def _train_step(self, batch, cql_min_q_weight=5.0):
        info = {}
        observations = batch["observations"]['proprio'].to(self.device)
        images = batch["observations"]['image'].to(self.device)
        next_observations = batch["next_observations"]['proprio'].to(self.device)
        next_images = batch["next_observations"]['image'].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)
        mc_returns = batch["mc_return"].to(self.device)
        bsize = actions.shape[0]

        new_obs_actions, log_pi= self.policy(observations, images)
        info.update({'actor/log_pi_mean': log_pi.mean().item()})

        # BC monitoring: log_prob of dataset actions
        with torch.no_grad():
            bc_log_prob = self.policy.log_prob(observations, images, actions)
        info.update({
            'actor/bc_log_prob_mean': bc_log_prob.mean().item(),
            'actor/bc_log_prob_std': bc_log_prob.std().item(),
            'actor/in_bc_phase': float(self._total_steps < self.config.bc_start_step),
        })

        # Policy loss with smooth BC -> RL transition
        bc_end_step = self.config.bc_start_step
        transition_steps = getattr(self.config, 'bc_transition_steps', 0)
        transition_end = bc_end_step + transition_steps
        bc_reg_weight = getattr(self.config, 'bc_regularization_weight', 0.0)

        # Compute BC weight for smooth transition
        if self._total_steps < bc_end_step:
            # Pure BC phase
            bc_weight = 1.0
        elif self._total_steps < transition_end and transition_steps > 0:
            # Transition phase: linearly decrease bc_weight from 1.0 to bc_reg_weight
            progress = (self._total_steps - bc_end_step) / transition_steps
            bc_weight = 1.0 - progress * (1.0 - bc_reg_weight)
        else:
            # RL phase with BC regularization
            bc_weight = bc_reg_weight

        # Compute both BC and Q objectives
        bc_log_prob_loss = self.policy.log_prob(observations, images, actions)
        q_value = torch.min(
            self.qf['qf1'](observations, images, new_obs_actions),
            self.qf['qf2'](observations, images, new_obs_actions),
        )

        # Blended policy objective: bc_weight * BC + (1 - bc_weight) * Q
        q_new_actions = bc_weight * bc_log_prob_loss + (1.0 - bc_weight) * q_value

        info.update({
            'actor/q_new_actions_mean': q_new_actions.mean().item(),
            'actor/bc_weight': bc_weight,
            'actor/q_value_mean': q_value.mean().item(),
        })
        policy_loss = -q_new_actions.mean()
        info.update({'actor/policy_loss': policy_loss.item()})

        q1_pred = self.qf['qf1'](observations, images, actions)
        q2_pred = self.qf['qf2'](observations, images, actions)
        info.update({
            'critic/q1_pred_mean': q1_pred.mean().item(),
            'critic/q1_pred_std': q1_pred.std().item(),
            'critic/q2_pred_mean': q2_pred.mean().item(),
            'critic/q2_pred_std': q2_pred.std().item(),
        })

        next_actions_temp, _ = self._get_policy_actions(next_observations, next_images, num_actions=self.config.cql_n_actions, network=self.policy)
        target_qf1_values = self._get_tensor_values(next_observations, next_images, next_actions_temp, network=self.qf['target_qf1']).max(1)[0].view(-1, 1)
        target_qf2_values = self._get_tensor_values(next_observations, next_images, next_actions_temp, network=self.qf['target_qf2']).max(1)[0].view(-1, 1)
        target_q_values = torch.min(target_qf1_values, target_qf2_values)
        q_target = rewards.view(-1, 1) + (1. - dones.view(-1, 1)) * self.config.discount * target_q_values
        q_target = q_target.detach()
        info.update({
            'critic/q_target_mean': q_target.mean().item(),
            'critic/q_target_min': q_target.min().item(),
            'critic/q_target_max': q_target.max().item(),
        })
        qf1_loss = nn.functional.mse_loss(q1_pred, q_target)
        qf2_loss = nn.functional.mse_loss(q2_pred, q_target)
        info.update({
            'critic/qf1_loss_mse': qf1_loss.item(),
            'critic/qf2_loss_mse': qf2_loss.item(),
        })
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.config.cql_n_actions, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(observations, images, num_actions=self.config.cql_n_actions, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_observations, next_images, num_actions=self.config.cql_n_actions, network=self.policy)
        q1_rand = self._get_tensor_values(observations, images, random_actions_tensor, network=self.qf['qf1'])
        q2_rand = self._get_tensor_values(observations, images, random_actions_tensor, network=self.qf['qf2'])
        q1_curr_actions = self._get_tensor_values(observations, images, curr_actions_tensor, network=self.qf['qf1'])
        q2_curr_actions = self._get_tensor_values(observations, images, curr_actions_tensor, network=self.qf['qf2'])
        q1_next_actions = self._get_tensor_values(observations, images, new_curr_actions_tensor, network=self.qf['qf1'])
        q2_next_actions = self._get_tensor_values(observations, images, new_curr_actions_tensor, network=self.qf['qf2'])

        # Cal-QL: bound Q-values with MC return-to-go
        # mc_returns shape: (B,) -> (B, N, 1) to match q values shape
        lower_bounds = mc_returns.view(-1, 1, 1).expand_as(q1_curr_actions)

        # Record bound rates before applying calibration (for debugging)
        num_vals = q1_curr_actions.numel()
        bound_rate_q1_curr = (q1_curr_actions < lower_bounds).sum().item() / num_vals
        bound_rate_q2_curr = (q2_curr_actions < lower_bounds).sum().item() / num_vals
        bound_rate_q1_next = (q1_next_actions < lower_bounds).sum().item() / num_vals
        bound_rate_q2_next = (q2_next_actions < lower_bounds).sum().item() / num_vals

        # Apply Cal-QL calibration: max(Q, V^π_β)
        q1_curr_actions = torch.maximum(q1_curr_actions, lower_bounds)
        q2_curr_actions = torch.maximum(q2_curr_actions, lower_bounds)
        q1_next_actions = torch.maximum(q1_next_actions, lower_bounds)
        q2_next_actions = torch.maximum(q2_next_actions, lower_bounds)

        info.update({
            'critic/cql_q1_rand_mean': q1_rand.mean().item(),
            'critic/cql_q2_rand_mean': q2_rand.mean().item(),
            'critic/cql_q1_curr_actions_mean': q1_curr_actions.mean().item(),
            'critic/cql_q2_curr_actions_mean': q2_curr_actions.mean().item(),
            'critic/cql_q1_next_actions_mean': q1_next_actions.mean().item(),
            'critic/cql_q2_next_actions_mean': q2_next_actions.mean().item(),
            'critic/calql_bound_rate_q1_curr': bound_rate_q1_curr,
            'critic/calql_bound_rate_q2_curr': bound_rate_q2_curr,
            'critic/calql_bound_rate_q1_next': bound_rate_q1_next,
            'critic/calql_bound_rate_q2_next': bound_rate_q2_next,
            'critic/mc_returns_mean': mc_returns.mean().item(),
        })
        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)
        info.update({
            'critic/cql_q1_std_mean': std_q1.mean().item(),
            'critic/cql_q2_std_mean': std_q2.mean().item(),
        })
        random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
        cat_q1 = torch.cat(
            [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
        )
        cat_q2 = torch.cat(
            [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
        )
        min_qf1_loss = torch.logsumexp(cat_q1 / self.config.cql_temp, dim=1,).mean() * cql_min_q_weight * self.config.cql_temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.config.cql_temp, dim=1,).mean() * cql_min_q_weight * self.config.cql_temp
        info.update({
            'critic/cql_logsumexp_loss_q1': min_qf1_loss.item(),
            'critic/cql_logsumexp_loss_q2': min_qf2_loss.item(),
        })
        # Subtract the log likelihood of data (E_s~D,a~D [Q(s,a)])
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * cql_min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * cql_min_q_weight
        info.update({
            'critic/cql_loss_q1_diff': min_qf1_loss.item(),
            'critic/cql_loss_q2_diff': min_qf2_loss.item(),
        })
        # Get alpha_prime: either learnable (Lagrange) or fixed
        if self.config.cql_lagrange:
            # Learnable alpha_prime with clamping
            alpha_prime = torch.clamp(self.log_alpha_prime().exp(), min=0.01, max=1000000.0)
        else:
            # Fixed alpha_prime from config
            alpha_prime = torch.tensor(self.config.alpha_prime, device=self.device)

        min_qf1_loss = alpha_prime * (min_qf1_loss - self.config.cql_target_action_gap)
        min_qf2_loss = alpha_prime * (min_qf2_loss - self.config.cql_target_action_gap)
        info.update({
            'critic/cql_min_qf1_loss_final': min_qf1_loss.item(),
            'critic/cql_min_qf2_loss_final': min_qf2_loss.item(),
            'critic/alpha_prime': alpha_prime.item(),
        })

        # Only update alpha_prime when using Lagrange formulation
        if self.config.cql_lagrange:
            info.update({'critic/log_alpha_prime': self.log_alpha_prime().item()})
            self.optimizers['log_alpha_prime'].zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            info.update({'critic/alpha_prime_loss': alpha_prime_loss.item()})
            alpha_prime_loss.backward(retain_graph=True)
            self.optimizers['log_alpha_prime'].step()

            # Clamp log_alpha_prime to prevent alpha_prime from going too low
            with torch.no_grad():
                self.log_alpha_prime.clamp_(min=-4.6, max=15.0)  # alpha_prime in [0.01, ~3.3M]
        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss
        info.update({
            'critic/total_qf1_loss': qf1_loss.item(),
            'critic/total_qf2_loss': qf2_loss.item(),
        })

        # Update policy network
        self.optimizers['policy'].zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizers['policy'].step()

        # Update Q-functions
        self.optimizers['qf1'].zero_grad()
        self.optimizers['qf2'].zero_grad()
        (qf1_loss + qf2_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.qf['qf1'].parameters(), self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.qf['qf2'].parameters(), self.config.max_grad_norm)
        self.optimizers['qf1'].step()
        self.optimizers['qf2'].step()

        # Soft update target networks
        if self._total_steps % self.config.target_update_interval == 0:
            self._soft_update_target_networks()

        return info

    @torch.no_grad()
    def _soft_update_target_networks(self):
        tau = self.config.soft_target_update_rate
        for target_qf, qf in [('target_qf1', 'qf1'), ('target_qf2', 'qf2')]:
            for target_param, param in zip(self.qf[target_qf].parameters(), self.qf[qf].parameters()):
                target_param.data.lerp_(param.data, tau)

    def _repeat_obs_images(self, obs, images, num_repeat):
        """Repeat observations and images for multiple action samples."""
        batch_size = obs.shape[0]
        obs_expanded = obs.unsqueeze(1).expand(-1, num_repeat, -1).reshape(batch_size * num_repeat, -1)
        images_expanded = images.unsqueeze(1).expand(-1, num_repeat, -1, -1, -1).reshape(
            batch_size * num_repeat, *images.shape[1:]
        )
        return obs_expanded, images_expanded

    def _get_tensor_values(self, obs, images, actions, network=None):
        """Get Q-values for given state-action pairs."""
        if hasattr(network, "evaluate_actions"):
            return network.evaluate_actions(obs, images, actions)

        batch_size = obs.shape[0]
        num_repeat = actions.shape[0] // batch_size
        obs_expanded, images_expanded = self._repeat_obs_images(obs, images, num_repeat)
        preds = network(obs_expanded, images_expanded, actions)
        return preds.view(batch_size, num_repeat, 1)

    def _get_policy_actions(self, obs, images, num_actions, network=None):
        """Sample multiple actions from policy for each state."""
        batch_size = obs.shape[0]

        if hasattr(network, "sample_actions"):
            actions, log_pi = network.sample_actions(obs, images, num_actions=num_actions, deterministic=False)
            actions = actions.reshape(batch_size * num_actions, -1)
            if log_pi is None:
                log_pi = torch.zeros(batch_size, num_actions, device=obs.device)
            return actions.detach(), log_pi.view(batch_size, num_actions, 1).detach()

        obs_expanded, images_expanded = self._repeat_obs_images(obs, images, num_actions)
        actions, log_pi = network(obs_expanded, images_expanded)
        return actions.detach(), log_pi.view(batch_size, num_actions, 1).detach()
    
    def to_device(self, device):
        self.device = device
        self.policy.to(device)
        self.qf['qf1'].to(device)
        self.qf['qf2'].to(device)
        self.qf['target_qf1'].to(device)
        self.qf['target_qf2'].to(device)
        if self.config.cql_lagrange:
            self.log_alpha_prime.to(device)

    def compile(self, mode="default"):
        if mode == "disable":
            print("torch.compile disabled")
            return
        self.policy = torch.compile(self.policy, mode=mode)
        self.qf['qf1'] = torch.compile(self.qf['qf1'], mode=mode)
        self.qf['qf2'] = torch.compile(self.qf['qf2'], mode=mode)
        self.qf['target_qf1'] = torch.compile(self.qf['target_qf1'], mode=mode)
        self.qf['target_qf2'] = torch.compile(self.qf['target_qf2'], mode=mode)
        if self.config.cql_lagrange:
            self.log_alpha_prime = torch.compile(self.log_alpha_prime, mode=mode)
        print(f"Compiled models with mode={mode}")
        
    
    @property
    def modules(self):
        return self._modules

    @property
    def total_steps(self):
        return self._total_steps
    
    
    def setup_multi_gpu(self, local_rank: int):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        self.to_device(device)

        self.policy = DDP(self.policy, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        self.qf['qf1'] = DDP(self.qf['qf1'], device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        self.qf['qf2'] = DDP(self.qf['qf2'], device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        self.optimizers["policy"] = torch.optim.Adam(self.policy.parameters(), lr=self.config.policy_lr)
        self.optimizers["qf"] = torch.optim.Adam(
            list(self.qf['qf1'].parameters()) + list(self.qf['qf2'].parameters()), lr=self.config.qf_lr
        )
        if self.config.cql_lagrange:
            self.optimizers["log_alpha_prime"] = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=self.config.alpha_prime_lr)

        print(f"[Rank {dist.get_rank()}] Trainer multi-GPU setup complete. Device: {device}")
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.qf['qf1'].load_state_dict(checkpoint['qf1_state_dict'])
        self.qf['qf2'].load_state_dict(checkpoint['qf2_state_dict'])
        self.qf['target_qf1'].load_state_dict(checkpoint['target_qf1_state_dict'])
        self.qf['target_qf2'].load_state_dict(checkpoint['target_qf2_state_dict'])
        for k, v in self.optimizers.items():
            if k in checkpoint['optimizers_state_dict']:
                v.load_state_dict(checkpoint['optimizers_state_dict'][k])
        self._total_steps = checkpoint.get('total_steps', 0)
        if self.config.cql_lagrange and 'log_alpha_prime_state_dict' in checkpoint:
            self.log_alpha_prime.load_state_dict(checkpoint['log_alpha_prime_state_dict'])
        # Load scheduler states if available
        if 'schedulers_state_dict' in checkpoint and self._use_scheduler:
            for k, v in self.schedulers.items():
                if k in checkpoint['schedulers_state_dict']:
                    v.load_state_dict(checkpoint['schedulers_state_dict'][k])
        print(f"Loaded checkpoint from {filepath} at total steps {self._total_steps}")
    def load_policy_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded policy checkpoint from {filepath}")
    
    def freeze_policy(self):
        for param_group in self.optimizers["qf"].param_groups:
            param_group['lr'] = self.config.freeze_qf_lr
        self.freeze_policy = True
        self.policy.freeze()
    
    def unfreeze_policy(self):
        for param_group in self.optimizers["qf"].param_groups:
            param_group['lr'] = self.config.qf_lr
        self.freeze_policy = False
        self.policy.unfreeze()

    def save_checkpoint(self, filepath):
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'qf1_state_dict': self.qf['qf1'].state_dict(),
            'qf2_state_dict': self.qf['qf2'].state_dict(),
            'target_qf1_state_dict': self.qf['target_qf1'].state_dict(),
            'target_qf2_state_dict': self.qf['target_qf2'].state_dict(),
            'optimizers_state_dict': {k: v.state_dict() for k, v in self.optimizers.items()},
            'total_steps': self._total_steps,
        }
        if self.config.cql_lagrange:
            checkpoint['log_alpha_prime_state_dict'] = self.log_alpha_prime.state_dict()
        # Save scheduler states
        if self._use_scheduler:
            checkpoint['schedulers_state_dict'] = {k: v.state_dict() for k, v in self.schedulers.items()}
        torch.save(checkpoint, filepath)
