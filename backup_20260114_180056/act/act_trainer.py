"""
ACT (Action Chunking with Transformers) Trainer

Training loop for ACT policy using CVAE objective:
L = L_reconstruction + beta * L_KL
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import prefix_metrics


class ACTTrainer:
    """
    Trainer for ACT policy.

    Uses CVAE training objective:
    - Reconstruction loss: L1 between predicted and target actions
    - KL divergence: Regularizes latent space to unit Gaussian
    """

    def __init__(self, config, policy):
        self.config = config
        self.policy = policy

        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.lr * 0.01,
        )

        # Mixed precision training (always use scaler like BC trainer)
        self.use_amp = config.use_amp
        self.scaler = GradScaler()

        # Training state
        self._total_steps = 0
        self._modules = [self.policy]

    def train(self, batch):
        """
        Single training step.

        Args:
            batch: dict with keys:
                - observations: dict with 'proprio' and 'image'
                - action_chunk: (B, chunk_size, action_dim) target actions

        Returns:
            metrics: dict of training metrics
        """
        self._total_steps += 1
        return self._train_step(batch)

    def _train_step(self, batch):
        # Unpack batch (same format as BC trainer)
        proprio = batch["observations"]["proprio"].to(self.device)
        images = batch["observations"]["image"].to(self.device)
        target_actions = batch["action_chunk"].to(self.device)

        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(device_type=proprio.device.type, enabled=self.use_amp):
            loss, metrics = self._compute_loss(proprio, images, target_actions)

        # Backward pass
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return metrics

    def _compute_loss(self, proprio, images, target_actions):
        """
        Compute CVAE loss.

        Args:
            proprio: (B, proprio_dim)
            images: (B, C, H, W) or (B, N_cam, C, H, W)
            target_actions: (B, chunk_size, action_dim)

        Returns:
            loss: scalar tensor
            info: dict of loss components
        """
        # Forward pass with actions (training mode)
        pred_actions, mu, logvar = self.policy(proprio, images, actions=target_actions)

        # Reconstruction loss (L1)
        recon_loss = nn.functional.l1_loss(pred_actions, target_actions)

        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.config.kl_weight * kl_loss

        # Use prefix_metrics like BC trainer
        metrics = prefix_metrics(
            dict(
                total_loss=total_loss,
                recon_loss=recon_loss,
                kl_loss=kl_loss,
                mu_mean=mu.mean(),
                mu_std=mu.std(),
                logvar_mean=logvar.mean(),
            ),
            "act",
        )

        return total_loss, metrics

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()

    def to_device(self, device):
        """Move trainer to device."""
        self.device = device
        self.policy.to(device)

    def compile(self, mode="default"):
        """Compile policy with torch.compile."""
        self.policy = torch.compile(self.policy, mode=mode)

    @property
    def modules(self):
        return self._modules

    @property
    def total_steps(self):
        return self._total_steps

    def setup_multi_gpu(self, local_rank: int):
        """Setup distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        self.to_device(device)

        self.policy = DDP(
            self.policy,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

        # Recreate optimizer for DDP model
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.lr * 0.01,
        )

        print(f"[Rank {dist.get_rank()}] ACT Trainer multi-GPU setup complete. Device: {device}")

    def save_checkpoint(self, filepath):
        """Save training checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "total_steps": self._total_steps,
        }
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._total_steps = checkpoint.get("total_steps", 0)
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Loaded checkpoint from {filepath} at step {self._total_steps}")

    def load_policy_checkpoint(self, filepath):
        """Load only policy weights (for fine-tuning or evaluation)."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded policy from {filepath}")

    @torch.no_grad()
    def evaluate(self, dataloader, num_batches=None):
        """
        Evaluate policy on validation data.

        Args:
            dataloader: validation dataloader
            num_batches: optional limit on number of batches

        Returns:
            metrics: dict of evaluation metrics
        """
        self.policy.eval()

        total_recon_loss = 0
        total_kl_loss = 0
        total_sample_error = 0
        num_batches_processed = 0

        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            proprio = batch["observations"]["proprio"].to(self.device)
            images = batch["observations"]["image"].to(self.device)
            target_actions = batch["action_chunk"].to(self.device)

            # Reconstruction loss (with teacher forcing)
            pred_actions, mu, logvar = self.policy(proprio, images, actions=target_actions)
            recon_loss = nn.functional.l1_loss(pred_actions, target_actions)  # mean reduction

            # KL loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Sample error (without teacher forcing, using prior z)
            sample_action, _, _ = self.policy(proprio, images, actions=None)
            sample_error = nn.functional.l1_loss(sample_action, target_actions)  # mean reduction

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_sample_error += sample_error.item()
            num_batches_processed += 1

        self.policy.train()

        # Average over batches (each batch already uses mean reduction)
        avg_recon_loss = total_recon_loss / num_batches_processed
        avg_kl_loss = total_kl_loss / num_batches_processed
        avg_sample_error = total_sample_error / num_batches_processed

        metrics = {
            "eval/recon_loss": avg_recon_loss,
            "eval/kl_loss": avg_kl_loss,
            "eval/total_loss": avg_recon_loss + self.config.kl_weight * avg_kl_loss,
            "eval/sample_error": avg_sample_error,
        }

        return metrics
