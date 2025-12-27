"""
IQL (Implicit Q-Learning) Trainer

Implementation based on the paper:
"Offline Reinforcement Learning with Implicit Q-Learning"
by Ilya Kostrikov, Ashvin Nair, and Sergey Levine (ICLR 2022)

Reference: https://github.com/ikostrikov/implicit_q_learning

Key ideas:
1. Uses expectile regression to train V function, avoiding OOD action evaluation
2. Q function trained with Bellman update using V instead of max
3. Policy trained with advantage-weighted regression (AWR)
"""

import copy
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from utils.utils import prefix_metrics


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Create a schedule with linear warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def asymmetric_l2_loss(u, tau):
    """
    Asymmetric L2 loss for expectile regression.

    Args:
        u: TD error (target - prediction)
        tau: Expectile value (0 < tau < 1)
            - tau > 0.5 focuses on overestimation
            - tau = 0.5 is symmetric (standard L2)
            - tau < 0.5 focuses on underestimation

    Returns:
        Weighted L2 loss
    """
    # For positive errors (underestimation), weight is tau
    # For negative errors (overestimation), weight is (1 - tau)
    weight = torch.where(u >= 0, tau, 1 - tau)
    return weight * u.pow(2)


class IQLTrainer:
    """
    IQL (Implicit Q-Learning) Trainer.

    IQL learns a value function V(s) using expectile regression on the Q-values,
    which implicitly approximates the maximum Q-value without needing to evaluate
    out-of-distribution actions.

    The key update equations are:
    1. V-function: L_V = E[(τ if Q - V >= 0 else 1-τ) * (Q - V)^2]
    2. Q-function: L_Q = E[(r + γ * V(s') - Q(s,a))^2]
    3. Policy: L_π = E[exp((Q - V) / β) * log π(a|s)]
    """

    def __init__(self, config, policy, qf, vf):
        """
        Initialize IQL Trainer.

        Args:
            config: Configuration object with hyperparameters
            policy: Policy network π(a|s)
            qf: Dictionary with 'qf1', 'qf2', 'target_qf1', 'target_qf2'
            vf: Value function V(s)
        """
        self.config = config
        self.policy = policy
        self.qf = qf
        self.vf = vf

        # IQL hyperparameters (from paper)
        self.discount = config.discount  # γ, typically 0.99
        self.tau = config.soft_target_update_rate  # Soft update rate for target Q
        self.expectile = getattr(config, 'expectile', 0.7)  # τ for expectile regression
        self.beta = getattr(config, 'beta', 3.0)  # Temperature for AWR
        self.clip_score = getattr(config, 'clip_score', 100.0)  # Clip advantage weights
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)

        # Optimizers
        self.optimizers = {}
        self.optimizers['policy'] = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.optimizers['qf1'] = optim.Adam(self.qf['qf1'].parameters(), lr=config.qf_lr)
        self.optimizers['qf2'] = optim.Adam(self.qf['qf2'].parameters(), lr=config.qf_lr)
        self.optimizers['vf'] = optim.Adam(self.vf.parameters(), lr=config.vf_lr)

        self._total_steps = 0
        self._modules = [
            self.policy,
            self.qf['qf1'], self.qf['qf2'],
            self.qf['target_qf1'], self.qf['target_qf2'],
            self.vf
        ]

        # Learning rate schedulers
        self.schedulers = {}
        self._use_scheduler = False

    def setup_lr_scheduler(self, num_training_steps, warmup_ratio=0.05, min_lr_ratio=0.1):
        """Setup learning rate schedulers with warmup and cosine decay."""
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
        self.schedulers['vf'] = get_cosine_schedule_with_warmup(
            self.optimizers['vf'], num_warmup_steps, num_training_steps, min_lr_ratio
        )

        self._use_scheduler = True
        print(f"LR scheduler: {num_training_steps} total steps, {num_warmup_steps} warmup steps")

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
            'vf_lr': self.optimizers['vf'].param_groups[0]['lr'],
        }

    def train(self, batch):
        """
        Single training step.

        Args:
            batch: Dictionary containing:
                - observations: dict with 'proprio' and 'image'
                - next_observations: dict with 'proprio' and 'image'
                - action: actions
                - reward: rewards
                - done: done flags

        Returns:
            metrics: Dictionary of training metrics
        """
        self._total_steps += 1
        metrics = self._train_step(batch)

        # Step schedulers
        self.step_scheduler()

        # Add learning rates to metrics
        if self._use_scheduler:
            lr_info = self.get_lr()
            metrics.update({
                'lr/policy': lr_info['policy_lr'],
                'lr/qf': lr_info['qf_lr'],
                'lr/vf': lr_info['vf_lr'],
            })

        return metrics

    def _train_step(self, batch):
        """Core training step implementing IQL algorithm."""
        info = {}

        # Unpack batch
        observations = batch["observations"]['proprio'].to(self.device)
        images = batch["observations"]['image'].to(self.device)
        next_observations = batch["next_observations"]['proprio'].to(self.device)
        next_images = batch["next_observations"]['image'].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)

        # ==================== Update V function ====================
        # V is trained with expectile regression on Q-values
        # L_V = E[L_τ(Q(s,a) - V(s))] where L_τ is asymmetric L2
        with torch.no_grad():
            # Use minimum of two Q-functions (Double Q trick)
            q1 = self.qf['qf1'](observations, images, actions)
            q2 = self.qf['qf2'](observations, images, actions)
            q_target = torch.min(q1, q2)

        v_pred = self.vf(observations, images)

        # Expectile regression loss
        # When Q > V (underestimation), weight is τ (larger weight for high τ)
        # When Q < V (overestimation), weight is 1-τ (smaller weight for high τ)
        # This makes V approximate a high percentile of Q, close to max
        v_loss = asymmetric_l2_loss(q_target - v_pred, self.expectile).mean()

        self.optimizers['vf'].zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.max_grad_norm)
        self.optimizers['vf'].step()

        info.update({
            'critic/v_loss': v_loss.item(),
            'critic/v_pred_mean': v_pred.mean().item(),
            'critic/v_pred_std': v_pred.std().item(),
        })

        # ==================== Update Q functions ====================
        # Q is trained with standard Bellman backup, but using V instead of max
        # L_Q = E[(r + γ * (1-d) * V(s') - Q(s,a))^2]
        with torch.no_grad():
            next_v = self.vf(next_observations, next_images)
            q_target = rewards.view(-1, 1) + (1.0 - dones.view(-1, 1)) * self.discount * next_v

        q1_pred = self.qf['qf1'](observations, images, actions)
        q2_pred = self.qf['qf2'](observations, images, actions)

        qf1_loss = nn.functional.mse_loss(q1_pred, q_target)
        qf2_loss = nn.functional.mse_loss(q2_pred, q_target)

        self.optimizers['qf1'].zero_grad()
        qf1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qf['qf1'].parameters(), self.max_grad_norm)
        self.optimizers['qf1'].step()

        self.optimizers['qf2'].zero_grad()
        qf2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qf['qf2'].parameters(), self.max_grad_norm)
        self.optimizers['qf2'].step()

        info.update({
            'critic/qf1_loss': qf1_loss.item(),
            'critic/qf2_loss': qf2_loss.item(),
            'critic/q1_pred_mean': q1_pred.mean().item(),
            'critic/q2_pred_mean': q2_pred.mean().item(),
            'critic/q_target_mean': q_target.mean().item(),
        })

        # ==================== Update Policy ====================
        # Policy is trained with advantage-weighted regression (AWR)
        # L_π = E[exp(A / β) * -log π(a|s)] where A = Q(s,a) - V(s)
        with torch.no_grad():
            # Compute advantage A = Q - V
            v = self.vf(observations, images)
            q1 = self.qf['qf1'](observations, images, actions)
            q2 = self.qf['qf2'](observations, images, actions)
            q = torch.min(q1, q2)
            advantage = q - v

            # Compute weights: exp(A / β), clipped for stability
            weights = torch.exp(advantage / self.beta)
            weights = torch.clamp(weights, max=self.clip_score)

        # Log probability of dataset actions under current policy
        log_prob = self.policy.log_prob(observations, images, actions)

        # Advantage-weighted regression loss
        # Maximize weighted log probability: minimize -weight * log_prob
        policy_loss = -(weights * log_prob).mean()

        self.optimizers['policy'].zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizers['policy'].step()

        info.update({
            'actor/policy_loss': policy_loss.item(),
            'actor/log_prob_mean': log_prob.mean().item(),
            'actor/advantage_mean': advantage.mean().item(),
            'actor/advantage_std': advantage.std().item(),
            'actor/weights_mean': weights.mean().item(),
            'actor/weights_max': weights.max().item(),
        })

        # ==================== Soft Update Target Networks ====================
        if self._total_steps % self.config.target_update_interval == 0:
            self._soft_update_target_networks()

        return info

    @torch.no_grad()
    def _soft_update_target_networks(self):
        """Soft update target Q-networks."""
        for target_qf, qf in [('target_qf1', 'qf1'), ('target_qf2', 'qf2')]:
            for target_param, param in zip(self.qf[target_qf].parameters(), self.qf[qf].parameters()):
                target_param.data.lerp_(param.data, self.tau)

    def to_device(self, device):
        """Move all models to device."""
        self.device = device
        self.policy.to(device)
        self.qf['qf1'].to(device)
        self.qf['qf2'].to(device)
        self.qf['target_qf1'].to(device)
        self.qf['target_qf2'].to(device)
        self.vf.to(device)

    def compile(self, mode="default"):
        """Compile models with torch.compile."""
        if mode == "disable":
            print("torch.compile disabled")
            return
        self.policy = torch.compile(self.policy, mode=mode)
        self.qf['qf1'] = torch.compile(self.qf['qf1'], mode=mode)
        self.qf['qf2'] = torch.compile(self.qf['qf2'], mode=mode)
        self.qf['target_qf1'] = torch.compile(self.qf['target_qf1'], mode=mode)
        self.qf['target_qf2'] = torch.compile(self.qf['target_qf2'], mode=mode)
        self.vf = torch.compile(self.vf, mode=mode)
        print(f"Compiled models with mode={mode}")

    @property
    def modules(self):
        return self._modules

    @property
    def total_steps(self):
        return self._total_steps

    def setup_multi_gpu(self, local_rank: int):
        """Setup distributed training with DDP."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        self.to_device(device)

        # Wrap models with DDP
        self.policy = DDP(self.policy, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        self.qf['qf1'] = DDP(self.qf['qf1'], device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        self.qf['qf2'] = DDP(self.qf['qf2'], device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        self.vf = DDP(self.vf, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Recreate optimizers for DDP models
        self.optimizers['policy'] = optim.Adam(self.policy.parameters(), lr=self.config.policy_lr)
        self.optimizers['qf1'] = optim.Adam(self.qf['qf1'].parameters(), lr=self.config.qf_lr)
        self.optimizers['qf2'] = optim.Adam(self.qf['qf2'].parameters(), lr=self.config.qf_lr)
        self.optimizers['vf'] = optim.Adam(self.vf.parameters(), lr=self.config.vf_lr)

        print(f"[Rank {dist.get_rank()}] IQL Trainer multi-GPU setup complete. Device: {device}")

    def save_checkpoint(self, filepath):
        """Save training checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'qf1_state_dict': self.qf['qf1'].state_dict(),
            'qf2_state_dict': self.qf['qf2'].state_dict(),
            'target_qf1_state_dict': self.qf['target_qf1'].state_dict(),
            'target_qf2_state_dict': self.qf['target_qf2'].state_dict(),
            'vf_state_dict': self.vf.state_dict(),
            'optimizers_state_dict': {k: v.state_dict() for k, v in self.optimizers.items()},
            'total_steps': self._total_steps,
        }
        if self._use_scheduler:
            checkpoint['schedulers_state_dict'] = {k: v.state_dict() for k, v in self.schedulers.items()}
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.qf['qf1'].load_state_dict(checkpoint['qf1_state_dict'])
        self.qf['qf2'].load_state_dict(checkpoint['qf2_state_dict'])
        self.qf['target_qf1'].load_state_dict(checkpoint['target_qf1_state_dict'])
        self.qf['target_qf2'].load_state_dict(checkpoint['target_qf2_state_dict'])
        self.vf.load_state_dict(checkpoint['vf_state_dict'])
        for k, v in self.optimizers.items():
            if k in checkpoint['optimizers_state_dict']:
                v.load_state_dict(checkpoint['optimizers_state_dict'][k])
        self._total_steps = checkpoint.get('total_steps', 0)
        if 'schedulers_state_dict' in checkpoint and self._use_scheduler:
            for k, v in self.schedulers.items():
                if k in checkpoint['schedulers_state_dict']:
                    v.load_state_dict(checkpoint['schedulers_state_dict'][k])
        print(f"Loaded checkpoint from {filepath} at total steps {self._total_steps}")

    def load_policy_checkpoint(self, filepath):
        """Load only policy weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded policy checkpoint from {filepath}")
