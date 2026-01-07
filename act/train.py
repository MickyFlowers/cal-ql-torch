"""
ACT Training Script

Train ACT (Action Chunking with Transformers) policy using imitation learning.
Supports both single-GPU and multi-GPU (distributed) training automatically.

Usage:
    Single GPU:  python -m act.train [args]
    Multi GPU:   torchrun --nproc_per_node=N -m act.train [args]
"""

import os
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from act.act_model import ACTPolicy
from act.act_trainer import ACTTrainer
from data.dataset import ACTDataset
from utils.distributed import (
    barrier,
    cleanup_distributed,
    is_distributed_available,
    is_main_process,
    setup_training,
    sync_metrics,
)
from utils.logger import WandBLogger


@hydra.main(config_path="../config", config_name="train_act", version_base=None)
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
    dataset = ACTDataset(cfg.dataset)

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
        print(f"Action dim: {dataset.action_dim}")
        print(f"Proprio dim: {dataset.proprio_dim}")
        print(f"Batches per epoch{' (per GPU)' if is_distributed else ''}: {len(dataloader)}")

    # Create model
    policy = ACTPolicy(
        action_dim=dataset.action_dim,
        proprio_dim=dataset.proprio_dim,
        hidden_dim=cfg.act.hidden_dim,
        latent_dim=cfg.act.latent_dim,
        num_encoder_layers=cfg.act.num_encoder_layers,
        num_decoder_layers=cfg.act.num_decoder_layers,
        num_heads=cfg.act.num_heads,
        chunk_size=cfg.act.chunk_size,
        dim_feedforward=cfg.act.dim_feedforward,
        backbone_name=cfg.act.backbone_name,
        pretrained_backbone=cfg.act.pretrained_backbone,
        train_backbone=cfg.act.train_backbone,
    )

    # Count parameters
    if is_main_process():
        num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

    # Create trainer and setup device/distributed
    trainer = ACTTrainer(cfg.act, policy)
    if is_distributed:
        trainer.setup_multi_gpu(local_rank)
    else:
        trainer.to_device(device)

    # Compile if requested
    if cfg.torch_compile_mode is not None and cfg.torch_compile_mode != "disable":
        if is_main_process():
            print(f"Compiling model with mode: {cfg.torch_compile_mode}")
        trainer.compile(mode=cfg.torch_compile_mode)

    # Create checkpoint directory and save config (only on main process)
    ckpt_dir = None
    if cfg.save_every_n_epoch > 0 and is_main_process():
        ckpt_dir = os.path.join(cfg.ckpt_path, f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save config to checkpoint directory
        config_save_path = os.path.join(ckpt_dir, "config.yaml")
        with open(config_save_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        print(f"Saved config to {config_save_path}")

    # Training loop
    if is_main_process():
        print(f"\nStarting training for {cfg.num_epochs} epochs")
    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        # Set epoch for distributed sampler (important for shuffling)
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_metrics = {
            'epoch': epoch,
            'act/total_loss': 0,
            'act/recon_loss': 0,
            'act/kl_loss': 0,
        }
        num_batches = 0

        # Progress bar only on main process
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.num_epochs}", disable=not is_main_process())
        for batch in pbar:
            # Train step
            metrics = trainer.train(batch)
            global_step += 1

            # Accumulate metrics
            for k in ['act/total_loss', 'act/recon_loss', 'act/kl_loss']:
                if k in metrics:
                    epoch_metrics[k] += metrics[k].item() if hasattr(metrics[k], 'item') else metrics[k]
            num_batches += 1

            # Update progress bar
            if is_main_process():
                total_loss = metrics['act/total_loss'].item() if hasattr(metrics['act/total_loss'], 'item') else metrics['act/total_loss']
                recon_loss = metrics['act/recon_loss'].item() if hasattr(metrics['act/recon_loss'], 'item') else metrics['act/recon_loss']
                kl_loss = metrics['act/kl_loss'].item() if hasattr(metrics['act/kl_loss'], 'item') else metrics['act/kl_loss']
                pbar.set_postfix({
                    'loss': f"{total_loss:.4f}",
                    'recon': f"{recon_loss:.4f}",
                    'kl': f"{kl_loss:.4f}",
                })

            # Log per-step metrics (only main process)
            if global_step % cfg.log_every_n_step == 0 and is_main_process() and wandb_logger is not None:
                step_metrics = {k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.items()}
                wandb_logger.log(step_metrics, step=global_step)

        # Average epoch metrics
        for k in ['act/total_loss', 'act/recon_loss', 'act/kl_loss']:
            epoch_metrics[k] /= num_batches

        # Sync metrics across GPUs (if distributed)
        if is_distributed:
            epoch_metrics = sync_metrics(epoch_metrics)

        # Step scheduler
        trainer.step_scheduler()
        epoch_metrics['lr'] = trainer.optimizer.param_groups[0]['lr']

        # Log epoch metrics (only main process)
        if is_main_process():
            if wandb_logger is not None:
                wandb_logger.log(epoch_metrics, step=global_step)
            print(f"Epoch {epoch}: loss={epoch_metrics['act/total_loss']:.4f}, "
                  f"recon={epoch_metrics['act/recon_loss']:.4f}, "
                  f"kl={epoch_metrics['act/kl_loss']:.4f}")

        # Evaluation (only main process)
        if epoch % cfg.eval_every_n_epoch == 0 and is_main_process():
            eval_metrics = trainer.evaluate(dataloader, num_batches=10)
            if wandb_logger is not None:
                wandb_logger.log(eval_metrics, step=global_step)
            print(f"  Eval: loss={eval_metrics['eval/total_loss']:.4f}")

        # Save checkpoint (only main process)
        if cfg.save_every_n_epoch > 0 and epoch % cfg.save_every_n_epoch == 0 and is_main_process():
            ckpt_file_path = os.path.join(ckpt_dir, f'checkpoint_{epoch:05d}.pt')
            trainer.save_checkpoint(ckpt_file_path)

        # Synchronize before next epoch (if distributed)
        barrier()

    # Save final checkpoint (only main process)
    if cfg.save_every_n_epoch > 0 and is_main_process():
        final_ckpt_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
        trainer.save_checkpoint(final_ckpt_path)
        print(f"\nTraining complete! Final checkpoint saved to {final_ckpt_path}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
