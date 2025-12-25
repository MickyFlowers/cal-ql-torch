"""
Diffusion Policy Training Script

Train Diffusion Policy for action prediction with transformer-based denoising.
"""

import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DiffusionPolicyDataset
from diffusion_policy.trainer import DiffusionPolicyTrainer
from model.diffusion_policy import DiffusionPolicy
from model.vision_model import VitFeatureExtractor
from utils.logger import WandBLogger


def _dict_to_device(batch, device):
    """Recursively move batch to device."""
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = _dict_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device, non_blocking=True)
    return batch


@hydra.main(config_path="../config", config_name="train_diffusion_policy", version_base=None)
def main(cfg: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup logging
    variant = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandBLogger(config=cfg.logging, variant=variant)

    # Create dataset
    print(f"Loading dataset from {cfg.dataset.root_path}")
    dataset = DiffusionPolicyDataset(cfg.dataset)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")

    # Create policy
    policy = DiffusionPolicy(
        action_dim=cfg.action_dim,
        pred_horizon=cfg.pred_horizon,
        config=cfg.dp_config,
        img_token_dim=cfg.img_token_dim,
        state_token_dim=cfg.state_token_dim,
        img_cond_len=cfg.img_cond_len,
        img_pos_embed_config=cfg.img_pos_embed_config,
        dtype=torch.bfloat16 if cfg.use_bf16 else torch.float32,
    )

    # Create vision encoder
    vision_encoder = VitFeatureExtractor(
        cfg.model_name,
        True,
        cfg.trainable_layers,
        dtype=torch.bfloat16 if cfg.use_bf16 else torch.float32,
    )

    # Create trainer
    trainer = DiffusionPolicyTrainer(policy, vision_encoder, cfg.trainer)
    trainer.to_device(device)

    # Setup learning rate scheduler
    num_training_steps = cfg.max_epochs * len(dataloader)
    trainer.setup_lr_scheduler(num_training_steps)

    # Load checkpoint if specified
    if cfg.load_ckpt_path and cfg.load_ckpt_path != "":
        trainer.load_checkpoint(cfg.load_ckpt_path)

    # Create checkpoint directory
    ckpt_dir = os.path.join(
        cfg.ckpt_path,
        f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {cfg.max_epochs} epochs")
    global_step = 0

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_metrics = {
            'epoch': epoch,
            'dp/loss': 0,
        }
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.max_epochs}")
        for batch in pbar:
            # Move batch to device
            batch = _dict_to_device(batch, device)

            # Train step
            metrics = trainer.train(batch)
            global_step += 1

            # Accumulate metrics
            for k, v in metrics.items():
                val = v.item() if hasattr(v, 'item') else v
                if k in epoch_metrics:
                    epoch_metrics[k] += val
                else:
                    epoch_metrics[k] = val
            num_batches += 1

            # Update progress bar
            loss = metrics['dp/loss'].item() if hasattr(metrics['dp/loss'], 'item') else metrics['dp/loss']
            pbar.set_postfix({'loss': f"{loss:.4f}"})

            # Log per-step metrics
            if global_step % cfg.log_every_n_step == 0:
                step_metrics = {k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.items()}
                wandb_logger.log(step_metrics, step=global_step)

        # Average epoch metrics
        for k in epoch_metrics:
            if k != 'epoch':
                epoch_metrics[k] /= num_batches

        # Add current learning rate
        epoch_metrics['lr'] = trainer.optimizer.param_groups[0]['lr']

        # Log epoch metrics
        wandb_logger.log(epoch_metrics, step=global_step)
        print(f"Epoch {epoch}: loss={epoch_metrics['dp/loss']:.4f}")

        # Evaluation
        if epoch % cfg.eval_every_n_epochs == 0:
            eval_metrics = trainer.evaluate(dataloader, num_batches=10)
            wandb_logger.log(eval_metrics, step=global_step)
            print(f"  Eval: action_error={eval_metrics['eval/action_error']:.4f}")

        # Save checkpoint
        if epoch % cfg.save_every_n_epoch == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:05d}.pt")
            trainer.save_checkpoint(ckpt_path)

    # Save final checkpoint
    final_ckpt_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
    trainer.save_checkpoint(final_ckpt_path)
    print(f"\nTraining complete! Final checkpoint saved to {final_ckpt_path}")


if __name__ == "__main__":
    main()
