"""
Diffusion Policy Trainer

Trainer for diffusion-based action prediction with EMA model averaging.
"""

import copy
import math

import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

from model.ema_model import EMAModel
from utils.utils import prefix_metrics


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create a schedule with linear warmup and cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class DiffusionPolicyTrainer:
    """
    Trainer for Diffusion Policy.

    Features:
    - Mixed precision training with GradScaler
    - EMA model averaging for both policy and vision encoder
    - Gradient clipping
    - Learning rate scheduling
    """

    def __init__(self, policy, vision_encoder, config):
        self.policy = policy
        self.vision_encoder = vision_encoder
        self.config = config

        # Optimizer
        params_to_optimize = list(self.policy.parameters()) + list(self.vision_encoder.parameters())
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config.learning_rate,
            betas=tuple(config.betas),
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon
        )

        # Learning rate scheduler (initialized later with setup_lr_scheduler)
        self.lr_scheduler = None
        self._use_scheduler = False
        self.warmup_ratio = getattr(config, 'warmup_ratio', 0.05)
        self.min_lr_ratio = getattr(config, 'min_lr_ratio', 0.1)

        # EMA models
        self.policy_ema_model = copy.deepcopy(self.policy)
        self.vision_encoder_ema_model = copy.deepcopy(self.vision_encoder)
        self.policy_ema = EMAModel(
            self.policy_ema_model,
            update_after_step=config.policy_ema.update_after_step,
            inv_gamma=config.policy_ema.inv_gamma,
            power=config.policy_ema.power,
            min_value=config.policy_ema.min_value,
            max_value=config.policy_ema.max_value
        )
        self.vision_encoder_ema = EMAModel(
            self.vision_encoder_ema_model,
            update_after_step=config.vision_encoder_ema.update_after_step,
            inv_gamma=config.vision_encoder_ema.inv_gamma,
            power=config.vision_encoder_ema.power,
            min_value=config.vision_encoder_ema.min_value,
            max_value=config.vision_encoder_ema.max_value
        )

        # Mixed precision training
        self.use_amp = getattr(config, 'use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)

        # Training state
        self._total_steps = 0

    def setup_lr_scheduler(self, num_training_steps):
        """
        Setup learning rate scheduler with warmup and cosine decay.

        Args:
            num_training_steps: Total number of training steps.
        """
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps, self.min_lr_ratio
        )
        self._use_scheduler = True
        print(f"LR scheduler setup: {num_training_steps} total steps, {num_warmup_steps} warmup steps, min_lr_ratio={self.min_lr_ratio}")

    def train(self, batch):
        """Single training step."""
        self._total_steps += 1
        metrics = self._train_step(batch)

        # Step scheduler after each training step
        if self._use_scheduler:
            self.lr_scheduler.step()

        # Add learning rate to metrics
        metrics['dp/lr'] = self.optimizer.param_groups[0]['lr']

        return metrics

    def _train_step(self, batch):
        self.policy.train()
        self.vision_encoder.train()

        observations = batch["observations"]['proprio'].to(self.device)
        images = batch["observations"]['image'].to(self.device)
        actions = batch["action"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            image_embeds = self.vision_encoder(images)[1]  # Use patch tokens
            loss = self.policy(
                img_tokens=image_embeds,
                state_tokens=observations,
                action_gt=actions,
            )

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.vision_encoder.parameters()),
            self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update EMA models
        self.policy_ema.step(self.policy)
        self.vision_encoder_ema.step(self.vision_encoder)

        metrics = prefix_metrics(
            dict(
                loss=loss,
                ema_decay=self.policy_ema.decay,
            ),
            "dp"
        )
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader, num_batches=None):
        """
        Evaluate using EMA models.

        Args:
            dataloader: validation dataloader
            num_batches: optional limit on number of batches

        Returns:
            metrics: dict of evaluation metrics
        """
        self.policy_ema_model.eval()
        self.vision_encoder_ema_model.eval()

        total_action_error = 0
        num_samples = 0

        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            observations = batch["observations"]['proprio'].to(self.device)
            images = batch["observations"]['image'].to(self.device)
            actions = batch["action"].to(self.device)

            # Use EMA models for evaluation
            image_embeds = self.vision_encoder_ema_model(images)[1]
            sample_action = self.policy_ema_model.predict_action(image_embeds, observations)
            action_error = torch.nn.functional.mse_loss(sample_action, actions, reduction='sum')

            total_action_error += action_error.item()
            num_samples += observations.shape[0]

        metrics = {
            'eval/action_error': total_action_error / num_samples,
        }
        return metrics

    def to_device(self, device):
        """Move trainer to device."""
        self.device = device
        self.policy.to(device)
        self.vision_encoder.to(device)
        self.policy_ema_model.to(device)
        self.vision_encoder_ema_model.to(device)

    @property
    def total_steps(self):
        return self._total_steps

    def save_checkpoint(self, filepath):
        """Save training checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "vision_encoder_state_dict": self.vision_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "policy_ema_state_dict": self.policy_ema_model.state_dict(),
            "vision_encoder_ema_state_dict": self.vision_encoder_ema_model.state_dict(),
            "total_steps": self._total_steps,
        }
        if self._use_scheduler and self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.policy_ema_model.load_state_dict(checkpoint["policy_ema_state_dict"])
        self.vision_encoder_ema_model.load_state_dict(checkpoint["vision_encoder_ema_state_dict"])
        self._total_steps = checkpoint.get("total_steps", 0)
        if "lr_scheduler_state_dict" in checkpoint and self._use_scheduler and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        print(f"Loaded checkpoint from {filepath} at step {self._total_steps}")


# Backward compatibility alias
Trainer = DiffusionPolicyTrainer
