import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_


class TD3BCTrainer(object):
    """TD3+BC Trainer for offline reinforcement learning.

    TD3+BC combines TD3 with behavior cloning regularization to learn from
    offline datasets while avoiding distribution shift issues.

    The policy loss is: -lambda * Q(s, pi(s)) + BC_loss(pi(s), a)
    where lambda = alpha / |Q(s, pi(s))|.mean()
    """

    def __init__(self, config, policy, qf):
        self.config = config
        self.policy = policy
        self.qf = qf
        assert len(qf) == 4, "Expected two Q-functions and their targets."

        self.optimizers = {}
        self.optimizers["policy"] = optim.Adam(
            self.policy.parameters(),
            lr=config.policy_lr,
            weight_decay=getattr(config, "weight_decay", 0.0),
        )
        self.optimizers["qf1"] = optim.Adam(
            self.qf["qf1"].parameters(),
            lr=config.qf_lr,
            weight_decay=getattr(config, "weight_decay", 0.0),
        )
        self.optimizers["qf2"] = optim.Adam(
            self.qf["qf2"].parameters(),
            lr=config.qf_lr,
            weight_decay=getattr(config, "weight_decay", 0.0),
        )

        # Optional learning rate schedulers
        self.schedulers = {}
        if getattr(config, "use_lr_scheduler", False):
            self.schedulers["policy"] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers["policy"], T_max=config.get("lr_scheduler_steps", 100000)
            )
            self.schedulers["qf1"] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers["qf1"], T_max=config.get("lr_scheduler_steps", 100000)
            )
            self.schedulers["qf2"] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers["qf2"], T_max=config.get("lr_scheduler_steps", 100000)
            )

        self._total_steps = 0
        self._modules = [
            self.policy,
            self.qf["qf1"],
            self.qf["qf2"],
            self.qf["target_qf1"],
            self.qf["target_qf2"],
        ]
        self.device = None

        # Gradient clipping
        self.grad_clip = getattr(config, "grad_clip", None)

    def train(self, batch):
        self._total_steps += 1
        metrics = self._train_step(batch)
        return metrics

    def _train_step(self, batch):
        info = {}
        observations = batch["observations"]["proprio"].to(self.device)
        images = batch["observations"]["image"].to(self.device)
        next_observations = batch["next_observations"]["proprio"].to(self.device)
        next_images = batch["next_observations"]["image"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)
        bsize = actions.shape[0]

        # Update critic
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_actions = self.policy(next_observations, next_images, deterministic=True)
            next_actions = (next_actions + noise).clamp(-1, 1)

            # Compute the target Q value
            target_q1 = self.qf["target_qf1"](next_observations, next_images, next_actions)
            target_q2 = self.qf["target_qf2"](next_observations, next_images, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = (
                rewards.view(-1, 1) + (1.0 - dones.view(-1, 1)) * self.config.discount * target_q
            )

        # Get current Q estimates
        current_q1 = self.qf["qf1"](observations, images, actions)
        current_q2 = self.qf["qf2"](observations, images, actions)

        # Compute critic loss
        qf1_loss = nn.functional.mse_loss(current_q1, target_q)
        qf2_loss = nn.functional.mse_loss(current_q2, target_q)

        info.update(
            {
                "critic/q1_pred_mean": current_q1.mean().item(),
                "critic/q1_pred_std": current_q1.std().item(),
                "critic/q2_pred_mean": current_q2.mean().item(),
                "critic/q2_pred_std": current_q2.std().item(),
                "critic/q_target_mean": target_q.mean().item(),
                "critic/qf1_loss": qf1_loss.item(),
                "critic/qf2_loss": qf2_loss.item(),
            }
        )

        # Optimize the critic
        self.optimizers["qf1"].zero_grad()
        qf1_loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.qf["qf1"].parameters(), self.grad_clip)
        self.optimizers["qf1"].step()

        self.optimizers["qf2"].zero_grad()
        qf2_loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.qf["qf2"].parameters(), self.grad_clip)
        self.optimizers["qf2"].step()

        # Delayed policy updates
        if self._total_steps % self.config.policy_freq == 0:
            # Compute actor loss
            pi_actions = self.policy(observations, images, deterministic=True)
            q_pi = self.qf["qf1"](observations, images, pi_actions)
            lmbda = self.config.alpha / q_pi.abs().mean().detach()

            # TD3+BC policy loss: -Q(s,pi(s)) + lambda * (pi(s) - a)^2
            policy_loss = -lmbda * q_pi.mean() + nn.functional.mse_loss(pi_actions, actions)

            info.update(
                {
                    "actor/policy_loss": policy_loss.item(),
                    "actor/q_pi_mean": q_pi.mean().item(),
                    "actor/bc_mse": nn.functional.mse_loss(pi_actions, actions).item(),
                    "actor/lambda": lmbda.item(),
                }
            )

            # Optimize the actor
            self.optimizers["policy"].zero_grad()
            policy_loss.backward()
            if self.grad_clip is not None:
                clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.optimizers["policy"].step()

            # Update the frozen target models
            with torch.no_grad():
                for target_param, param in zip(
                    self.qf["target_qf1"].parameters(), self.qf["qf1"].parameters()
                ):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )
                for target_param, param in zip(
                    self.qf["target_qf2"].parameters(), self.qf["qf2"].parameters()
                ):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )

            # Update learning rate schedulers
            for scheduler in self.schedulers.values():
                scheduler.step()

        return info

    def to_device(self, device):
        self.device = device
        self.policy.to(device)
        self.qf["qf1"].to(device)
        self.qf["qf2"].to(device)
        self.qf["target_qf1"].to(device)
        self.qf["target_qf2"].to(device)

    def compile(self, mode="default"):
        self.policy = torch.compile(self.policy, mode=mode)
        self.qf["qf1"] = torch.compile(self.qf["qf1"], mode=mode)
        self.qf["qf2"] = torch.compile(self.qf["qf2"], mode=mode)
        self.qf["target_qf1"] = torch.compile(self.qf["target_qf1"], mode=mode)
        self.qf["target_qf2"] = torch.compile(self.qf["target_qf2"], mode=mode)

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

        self.policy = DDP(
            self.policy, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )
        self.qf["qf1"] = DDP(
            self.qf["qf1"],
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        self.qf["qf2"] = DDP(
            self.qf["qf2"],
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

        self.optimizers["policy"] = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.policy_lr
        )
        self.optimizers["qf1"] = torch.optim.Adam(self.qf["qf1"].parameters(), lr=self.config.qf_lr)
        self.optimizers["qf2"] = torch.optim.Adam(self.qf["qf2"].parameters(), lr=self.config.qf_lr)

        print(f"[Rank {dist.get_rank()}] Trainer multi-GPU setup complete. Device: {device}")

    def load_checkpoint(self, filepath, load_optimizer=True):
        """Load model checkpoint.

        Args:
            filepath: Path to the checkpoint file.
            load_optimizer: Whether to load optimizer states. Set to False when
                transitioning from offline to online training with different optimizers.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.qf["qf1"].load_state_dict(checkpoint["qf1_state_dict"])
        self.qf["qf2"].load_state_dict(checkpoint["qf2_state_dict"])
        self.qf["target_qf1"].load_state_dict(checkpoint["target_qf1_state_dict"])
        self.qf["target_qf2"].load_state_dict(checkpoint["target_qf2_state_dict"])
        if load_optimizer:
            for k, v in self.optimizers.items():
                if k in checkpoint.get("optimizers_state_dict", {}):
                    v.load_state_dict(checkpoint["optimizers_state_dict"][k])
            # Load scheduler states if available
            for k, v in self.schedulers.items():
                if k in checkpoint.get("schedulers_state_dict", {}):
                    v.load_state_dict(checkpoint["schedulers_state_dict"][k])
        self._total_steps = checkpoint.get("total_steps", 0)
        print(f"Loaded checkpoint from {filepath} at total steps {self._total_steps}")
        return checkpoint

    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "qf1_state_dict": self.qf["qf1"].state_dict(),
            "qf2_state_dict": self.qf["qf2"].state_dict(),
            "target_qf1_state_dict": self.qf["target_qf1"].state_dict(),
            "target_qf2_state_dict": self.qf["target_qf2"].state_dict(),
            "optimizers_state_dict": {k: v.state_dict() for k, v in self.optimizers.items()},
            "schedulers_state_dict": {k: v.state_dict() for k, v in self.schedulers.items()},
            "total_steps": self._total_steps,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath} at total steps {self._total_steps}")
