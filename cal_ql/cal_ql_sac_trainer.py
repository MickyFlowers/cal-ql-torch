
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Independent, Normal
from torch.distributions.transformed_distribution import \
    TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from model.model import Scaler


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
        
        self.log_alpha = Scaler(1.0)
        self.optimizers["log_alpha"] = optim.Adam(self.log_alpha.parameters(), lr=config.policy_lr)

        
        self.log_alpha_prime = Scaler(self.config.alpha_prime)
        self.optimizers["log_alpha_prime"] = optim.Adam(self.log_alpha_prime.parameters(), lr=config.qf_lr)
            
        self._total_steps = 0
        self._modules = [self.policy, self.qf['qf1'], self.qf['qf2'], self.qf['target_qf1'], self.qf['target_qf2']]
        self._modules.append(self.log_alpha)
        self._modules.append(self.log_alpha_prime)

    def train(self, batch, cql_min_q_weight=5.0):
        self._total_steps += 1
        metrics = self._train_step(batch, cql_min_q_weight=cql_min_q_weight)
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
        alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
        info.update({'actor/alpha_loss': alpha_loss.item()})
        self.optimizers["log_alpha"].zero_grad()
        alpha_loss.backward()
        self.optimizers["log_alpha"].step()
        alpha = self.log_alpha().exp()
        info.update({
            'actor/alpha': alpha.item(),
            'actor/log_alpha': self.log_alpha().item()
        })

        q_new_actions = torch.min(
            self.qf['qf1'](observations, images, new_obs_actions),
            self.qf['qf2'](observations, images, new_obs_actions),
        ) if self._total_steps >= self.config.bc_start_step else self.policy.log_prob(observations, images, actions)
        info.update({'actor/q_new_actions_mean': q_new_actions.mean().item()})
        policy_loss = (alpha*log_pi - q_new_actions).mean()
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
        info.update({
            'critic/cql_q1_rand_mean': q1_rand.mean().item(),
            'critic/cql_q2_rand_mean': q2_rand.mean().item(),
            'critic/cql_q1_curr_actions_mean': q1_curr_actions.mean().item(),
            'critic/cql_q2_curr_actions_mean': q2_curr_actions.mean().item(),
            'critic/cql_q1_next_actions_mean': q1_next_actions.mean().item(),
            'critic/cql_q2_next_actions_mean': q2_next_actions.mean().item(),
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
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * cql_min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * cql_min_q_weight
        info.update({
            'critic/cql_loss_q1_diff': min_qf1_loss.item(),
            'critic/cql_loss_q2_diff': min_qf2_loss.item(),
        })
        alpha_prime = torch.clamp(self.log_alpha_prime().exp(), min=0.0, max=1000000.0)
        min_qf1_loss = alpha_prime * (min_qf1_loss - self.config.cql_target_action_gap)
        min_qf2_loss = alpha_prime * (min_qf2_loss - self.config.cql_target_action_gap)
        info.update({
            'critic/cql_min_qf1_loss_final': min_qf1_loss.item(),
            'critic/cql_min_qf2_loss_final': min_qf2_loss.item(),
            'critic/alpha_prime': alpha_prime.item(),
            'critic/log_alpha_prime': self.log_alpha_prime().item(),
        })

        self.optimizers['log_alpha_prime'].zero_grad()
        alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
        info.update({'critic/alpha_prime_loss': alpha_prime_loss.item()})
        alpha_prime_loss.backward(retain_graph=True)
        self.optimizers['log_alpha_prime'].step()
        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss
        info.update({
            'critic/total_qf1_loss': qf1_loss.item(),
            'critic/total_qf2_loss': qf2_loss.item(),
        })

        """
        Update networks
        """
        # Update the Q-functions iff '
        self.optimizers['policy'].zero_grad()
        policy_loss.backward(retain_graph=False)
        self.optimizers['policy'].step()

        self.optimizers['qf1'].zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.optimizers['qf1'].step()

        
        self.optimizers['qf2'].zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.optimizers['qf2'].step()

        
        if self._total_steps % self.config.target_update_interval == 0:
            with torch.no_grad():
                for target_param, param in zip(self.qf["target_qf1"].parameters(), self.qf["qf1"].parameters()):
                    new_param = (1 - self.config.soft_target_update_rate) * target_param.data + self.config.soft_target_update_rate * param.data
                    target_param.data.copy_(new_param)
                for target_param, param in zip(self.qf["target_qf2"].parameters(), self.qf["qf2"].parameters()):
                    new_param = (1 - self.config.soft_target_update_rate) * target_param.data + self.config.soft_target_update_rate * param.data
                    target_param.data.copy_(new_param)
        return info

    def _get_tensor_values(self, obs, images, actions, network=None):
        if hasattr(network, "evaluate_actions"):
            return network.evaluate_actions(obs, images, actions)

        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        images_temp = images.unsqueeze(1).repeat(1, num_repeat, 1, 1, 1).view(images.shape[0] * num_repeat, images.shape[1], images.shape[2], images.shape[3])
        preds = network(obs_temp, images_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, images, num_actions, network=None):
        if hasattr(network, "sample_actions"):
            actions, log_pi = network.sample_actions(obs, images, num_actions=num_actions, deterministic=False)
            actions = actions.reshape(obs.shape[0] * num_actions, -1)
            if log_pi is None:
                log_pi = torch.zeros(obs.shape[0], num_actions, device=obs.device)
            return actions.detach(), log_pi.view(obs.shape[0], num_actions, 1).detach()

        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        images_temp = images.unsqueeze(1).repeat(1, num_actions, 1, 1, 1).view(images.shape[0] * num_actions, images.shape[1], images.shape[2], images.shape[3])  
        new_obs_actions, new_obs_log_pi = network(
            obs_temp, images_temp
        )

        return new_obs_actions.detach(), new_obs_log_pi.view(obs.shape[0], num_actions, 1).detach()
    
    def to_device(self, device):
        self.device = device
        self.policy.to(device)
        self.qf['qf1'].to(device)
        self.qf['qf2'].to(device)
        self.qf['target_qf1'].to(device)
        self.qf['target_qf2'].to(device)
        self.log_alpha.to(device)
        if self.config.cql_lagrange:
            self.log_alpha_prime.to(device)

    def compile(self, mode="default"):
        self.policy = torch.compile(self.policy, mode=mode)
        self.qf['qf1'] = torch.compile(self.qf['qf1'], mode=mode)
        self.qf['qf2'] = torch.compile(self.qf['qf2'], mode=mode)
        self.qf['target_qf1'] = torch.compile(self.qf['target_qf1'], mode=mode)
        self.qf['target_qf2'] = torch.compile(self.qf['target_qf2'], mode=mode)
        self.log_alpha = torch.compile(self.log_alpha, mode=mode)
        if self.config.cql_lagrange:
            self.log_alpha_prime = torch.compile(self.log_alpha_prime, mode=mode)
        
    
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
        
        self.optimizers["log_alpha"] = torch.optim.Adam(self.log_alpha.parameters(), lr=self.config.policy_lr)
        if self.config.cql_lagrange:
            self.optimizers["log_alpha_prime"] = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=self.config.qf_lr)

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
        self.log_alpha.load_state_dict(checkpoint['log_alpha_state_dict'])
        if self.config.cql_lagrange and 'log_alpha_prime_state_dict' in checkpoint:
            self.log_alpha_prime.load_state_dict(checkpoint['log_alpha_prime_state_dict'])
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
        checkpoint['log_alpha_state_dict'] = self.log_alpha.state_dict()
        if self.config.cql_lagrange:
            checkpoint['log_alpha_prime_state_dict'] = self.log_alpha_prime.state_dict()
        torch.save(checkpoint, filepath)
