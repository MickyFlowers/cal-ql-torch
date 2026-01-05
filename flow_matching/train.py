"""
Flow Matching Policy Training Script

Train Flow Matching Policy for action prediction with transformer-based flow prediction.
Supports both single-GPU and multi-GPU (distributed) training automatically.

Usage:
    Single GPU:  python -m flow_matching.train [args]
    Multi GPU:   torchrun --nproc_per_node=N -m flow_matching.train [args]
"""

import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.dataset import FlowMatchingDataset
from flow_matching.trainer import FlowMatchingPolicyTrainer
from model.flow_matching_policy import FlowMatchingPolicy
from model.vision_model import VitFeatureExtractor
from utils.distributed import (
    barrier,
    cleanup_distributed,
    is_main_process,
    setup_training,
    sync_metrics,
)
from utils.logger import WandBLogger


def _dict_to_device(batch, device):
    """Recursively move batch to device."""
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = _dict_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device, non_blocking=True)
    return batch


@hydra.main(config_path="../config", config_name="train_flow_matching", version_base=None)
def main(cfg: DictConfig):
    # Setup training environment (auto-detects single vs multi GPU)
    local_rank, device, world_size = setup_training(cfg.device)
    is_distributed = world_size > 1

    # Print config only on main process
    if is_main_process():
        print(OmegaConf.to_yaml(cfg))
        print(f"Training mode: {'Distributed' if is_distributed else 'Single GPU'}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")

    # Set seeds (add local_rank offset for distributed)
    np.random.seed(cfg.seed + local_rank)
    torch.manual_seed(cfg.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + local_rank)

    # Setup logging (only on main process)
    wandb_logger = None
    if is_main_process():
        variant = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger = WandBLogger(config=cfg.logging, variant=variant)

    # Create dataset
    if is_main_process():
        print(f"Loading dataset from {cfg.dataset.root_path}")
    dataset = FlowMatchingDataset(cfg.dataset)

    # Create dataloader with optional distributed sampler
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),  # Only shuffle if not using distributed sampler
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if is_main_process():
        print(f"Dataset: {len(dataset)} samples")
        print(f"Batches per epoch{' (per GPU)' if is_distributed else ''}: {len(dataloader)}")

    # Create policy
    policy = FlowMatchingPolicy(
        action_dim=cfg.action_dim,
        pred_horizon=cfg.pred_horizon,
        config=cfg.fm_config,
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

    # Create trainer and setup device/distributed
    trainer = FlowMatchingPolicyTrainer(policy, vision_encoder, cfg.trainer)
    if is_distributed:
        trainer.setup_multi_gpu(local_rank)
    else:
        trainer.to_device(device)

    # Setup learning rate scheduler
    num_training_steps = cfg.max_epochs * len(dataloader)
    trainer.setup_lr_scheduler(num_training_steps)

    # Load checkpoint if specified
    if cfg.load_ckpt_path and cfg.load_ckpt_path != "":
        trainer.load_checkpoint(cfg.load_ckpt_path)

    # Create checkpoint directory (only on main process)
    ckpt_dir = None
    if is_main_process():
        ckpt_dir = os.path.join(
            cfg.ckpt_path,
            f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}'
        )
        os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    if is_main_process():
        print(f"\nStarting training for {cfg.max_epochs} epochs")
    global_step = 0

    for epoch in range(1, cfg.max_epochs + 1):
        # Set epoch for distributed sampler (important for shuffling)
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_metrics = {
            'epoch': epoch,
            'fm/loss': 0,
        }
        num_batches = 0

        # Progress bar only on main process
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.max_epochs}", disable=not is_main_process())
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
            if is_main_process():
                loss = metrics['fm/loss'].item() if hasattr(metrics['fm/loss'], 'item') else metrics['fm/loss']
                pbar.set_postfix({'loss': f"{loss:.4f}"})

            # Log per-step metrics (only main process)
            if global_step % cfg.log_every_n_step == 0 and is_main_process() and wandb_logger is not None:
                step_metrics = {k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.items()}
                wandb_logger.log(step_metrics, step=global_step)

        # Average epoch metrics
        for k in epoch_metrics:
            if k != 'epoch':
                epoch_metrics[k] /= num_batches

        # Sync metrics across GPUs (if distributed)
        if is_distributed:
            epoch_metrics = sync_metrics(epoch_metrics)

        # Add current learning rate
        epoch_metrics['lr'] = trainer.optimizer.param_groups[0]['lr']

        # Log epoch metrics (only main process)
        if is_main_process():
            if wandb_logger is not None:
                wandb_logger.log(epoch_metrics, step=global_step)
            print(f"Epoch {epoch}: loss={epoch_metrics['fm/loss']:.4f}")

        # Evaluation (only main process)
        if epoch % cfg.eval_every_n_epochs == 0 and is_main_process():
            eval_metrics = trainer.evaluate(dataloader, num_batches=10)
            if wandb_logger is not None:
                wandb_logger.log(eval_metrics, step=global_step)
            print(f"  Eval: sample_error={eval_metrics['eval/sample_error']:.4f}")

        # Save checkpoint (only main process)
        if epoch % cfg.save_every_n_epoch == 0 and is_main_process():
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:05d}.pt")
            trainer.save_checkpoint(ckpt_path)

        # Synchronize before next epoch (if distributed)
        barrier()

    # Save final checkpoint (only main process)
    if is_main_process():
        final_ckpt_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
        trainer.save_checkpoint(final_ckpt_path)
        print(f"\nTraining complete! Final checkpoint saved to {final_ckpt_path}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
