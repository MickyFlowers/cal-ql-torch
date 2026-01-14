import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from model.model import Scaler
from utils.utils import prefix_metrics


class BehaviorCloneTrainer(object):
    def __init__(self, lr, policy, input_modality="both"):
        self.policy = policy
        self._policy_module = policy  # Keep reference to original module for method access
        self.lr = lr
        self.input_modality = self._normalize_modality(input_modality)

        self.optimizers = {}
        self.optimizers["policy"] = optim.Adam(self.policy.parameters(), lr=lr)
        self.scaler = GradScaler()

        self._total_steps = 0

    def train(self, batch):
        self._total_steps += 1
        metrics = self._train_step(batch)
        return metrics

    def _train_step(self, batch):
        observations = batch["observations"]["proprio"].to(self.device)
        images = batch["observations"]["image"].to(self.device)
        actions = batch["action"].to(self.device)
        observations, images = self._apply_modality_mask(observations, images)
        # observations = batch["observations"]
        # actions = batch["actions"]
        # rewards = batch["rewards"]
        # next_observations = batch["next_observations"]
        # dones = batch["dones"]

        # Policy forward (use _policy_module for method access when wrapped by DDP)
        with autocast(device_type=observations.device.type, enabled=torch.is_autocast_enabled()):
            log_probs = self._policy_module.log_prob(observations, images, actions)
            policy_loss = -log_probs.mean()

        self.optimizers["policy"].zero_grad()
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizers["policy"])
        self.scaler.update()

        metrics = prefix_metrics(
            dict(
                policy_loss=policy_loss,
                log_prob=log_probs.mean(),
            ),
            "bc",
        )
        return metrics

    def to_device(self, device):
        self.device = device
        self.policy.to(device)

    def compile(self, mode="default"):
        self.policy = torch.compile(self.policy, mode=mode)

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
        # Update _policy_module to point to the underlying module after DDP wrap
        self._policy_module = self.policy.module

        # Recreate optimizer for DDP model
        self.optimizers["policy"] = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        print(f"[Rank {dist.get_rank()}] BC Trainer multi-GPU setup complete. Device: {device}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        for k, v in self.optimizers.items():
            if k in checkpoint["optimizers_state_dict"]:
                v.load_state_dict(checkpoint["optimizers_state_dict"][k])
        self._total_steps = checkpoint.get("total_steps", 0)
        print(f"Loaded checkpoint from {filepath} at total steps {self._total_steps}")

    def save_checkpoint(self, filepath):
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizers_state_dict": {k: v.state_dict() for k, v in self.optimizers.items()},
            "total_steps": self._total_steps,
        }
        torch.save(checkpoint, filepath)

    @torch.no_grad()
    def evaluate(self, batch):
        """Evaluate on a batch without computing gradients or updating weights."""
        observations = batch["observations"]["proprio"].to(self.device)
        images = batch["observations"]["image"].to(self.device)
        actions = batch["action"].to(self.device)
        observations, images = self._apply_modality_mask(observations, images)

        # Policy forward (use _policy_module for method access when wrapped by DDP)
        with autocast(device_type=observations.device.type, enabled=torch.is_autocast_enabled()):
            log_probs = self._policy_module.log_prob(observations, images, actions)
            policy_loss = -log_probs.mean()

        metrics = prefix_metrics(
            dict(
                policy_loss=policy_loss,
                log_prob=log_probs.mean(),
            ),
            "val",
        )
        return metrics

    def _normalize_modality(self, modality):
        if modality is None:
            return "both"
        normalized = str(modality).strip().lower()
        if normalized in ("both", "all"):
            return "both"
        if normalized in ("force", "proprio", "ft"):
            return "force"
        if normalized in ("vision", "image"):
            return "vision"
        raise ValueError(
            f"Unsupported input_modality: {modality}. Use 'both', 'force', or 'vision'."
        )

    def _apply_modality_mask(self, observations, images):
        if self.input_modality == "force":
            images = torch.zeros_like(images)
        elif self.input_modality == "vision":
            observations = torch.zeros_like(observations)
        return observations, images
