"""
Behavior Cloning Training Script

Train BC policy using imitation learning.
Supports both single-GPU and multi-GPU (distributed) training automatically.

Usage:
    Single GPU:  python -m cal_ql.train_bc [args]
    Multi GPU:   torchrun --nproc_per_node=N -m cal_ql.train_bc [args]
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

from cal_ql.bc_trainer import BehaviorCloneTrainer
from data.dataset import BCDatasetLMDB
from model.model import ResNetPolicy
from utils.distributed import (
    barrier,
    cleanup_distributed,
    is_main_process,
    setup_training,
    sync_metrics,
)
from utils.logger import WandBLogger
from utils.utils import Timer
from viskit.logging import logger, setup_logger


def dict_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = dict_to_device(v, device)
        else:
            batch[k] = v.to(device=device, non_blocking=True)
    return batch


@hydra.main(config_path="../config", config_name="train_bc", version_base=None)
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

    # Setup logging (only on main process)
    variant = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = None
    if is_main_process():
        wandb_logger = WandBLogger(config=cfg.logging, variant=variant)
        setup_logger(
            variant=variant,
            exp_id=wandb_logger.experiment_id,
            seed=cfg.seed,
            base_log_dir=cfg.logging.output_dir,
            include_exp_prefix_sub_dir=False,
        )

    # Create dataset (using BC-optimized LMDB for faster data loading)
    # sample_ratio and sample_seed ensure consistent random sampling across all GPUs
    train_dataset = BCDatasetLMDB(
        cfg.dataset,
        sample_ratio=cfg.get('sample_ratio', 1.0),
        sample_seed=cfg.get('sample_seed', 42)
    )

    # Create validation dataset from remaining episodes (if sample_ratio < 1)
    val_dataset = train_dataset.get_validation_dataset()

    # Log dataset info on main process
    if is_main_process():
        print(f"Train dataset size: {len(train_dataset)} samples")
        if cfg.get('sample_ratio', 1.0) < 1.0:
            print(f"Using {cfg.sample_ratio*100:.1f}% of data with seed {cfg.get('sample_seed', 42)}")
        if val_dataset is not None:
            print(f"Validation dataset size: {len(val_dataset)} samples")

    # Create dataloader with optional distributed sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )

    # Create validation dataloader if validation dataset exists
    val_dataloader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True if cfg.num_workers > 0 else False,
        )

    # Set seeds (add local_rank offset for distributed)
    np.random.seed(cfg.seed + local_rank)
    torch.manual_seed(cfg.seed + local_rank)

    observation_dim = cfg.observation_dim
    action_dim = cfg.action_dim

    # Create policy
    policy = ResNetPolicy(
        observation_dim,
        action_dim,
        cfg.policy_obs_proj_arch,
        cfg.policy_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        cfg.policy_log_std_multiplier,
        cfg.policy_log_std_offset,
        train_backbone=cfg.train_policy_backbone,
    )

    # Create trainer and setup device/distributed
    bc = BehaviorCloneTrainer(cfg.learning_rate, policy)
    print("Created BC Trainer")
    if is_distributed:
        bc.setup_multi_gpu(local_rank)
        print(f"[Rank {local_rank}] BC Trainer distributed setup complete.")
    else:
        bc.to_device(device=device)
    print("BC Trainer setup complete.")
    # Load checkpoint if specified
    if cfg.load_ckpt_path != "":
        bc.load_checkpoint(cfg.load_ckpt_path)

    viskit_metrics = {}
    total_grad_steps = 0
    train_timer = None
    epoch = 0
    train_metrics = None

    # Create checkpoint directory and save config (only on main process)
    ckpt_path = None
    if cfg.save_every_n_epoch > 0 and is_main_process():
        ckpt_path = os.path.join(cfg.ckpt_path, f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(ckpt_path, exist_ok=True)
        # Save config to checkpoint directory
        config_save_path = os.path.join(ckpt_path, "config.yaml")
        with open(config_save_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        print(f"Saved config to {config_save_path}")

    while True:
        metrics = {"epoch": epoch}
        metrics["grad_steps"] = total_grad_steps
        metrics["epoch"] = epoch
        metrics["train_time"] = 0 if train_timer is None else train_timer()
        if train_metrics is not None:
            metrics.update(train_metrics)

        # Log only on main process
        if is_main_process():
            if wandb_logger is not None:
                wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # Save checkpoint (only on main process)
        if epoch % cfg.save_every_n_epoch == 0 and epoch != 0 and is_main_process():
            ckpt_file_path = os.path.join(ckpt_path, f'bc_checkpoint_{epoch:05d}.pt')
            bc.save_checkpoint(ckpt_file_path)

        if epoch >= cfg.train_bc_epochs:
            if is_main_process():
                print("Finished Training")
            break

        with Timer() as train_timer:
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Accumulate metrics over the entire epoch
            epoch_metrics = {}
            num_batches = 0
            for batch in tqdm(train_dataloader, desc="Training", disable=not is_main_process()):
                batch = dict_to_device(batch, device=device)
                batch_metrics = bc.train(batch)
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().item()
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0.0
                    epoch_metrics[k] += v
                num_batches += 1

            # Compute epoch average
            train_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

            # Sync metrics across GPUs (if distributed)
            if is_distributed:
                train_metrics = sync_metrics(train_metrics)

            total_grad_steps += len(train_dataloader)

        # Validation loop
        eval_every = cfg.get('eval_every_n_epochs', 10)
        if val_dataloader is not None and epoch % eval_every == 0 and epoch != 0:
            bc.policy.eval()
            val_epoch_metrics = {}
            val_num_batches = 0
            for batch in tqdm(val_dataloader, desc="Validation", disable=not is_main_process()):
                batch = dict_to_device(batch, device=device)
                batch_metrics = bc.evaluate(batch)
                for k, v in batch_metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().item()
                    val_epoch_metrics[k] = val_epoch_metrics.get(k, 0.0) + v
                val_num_batches += 1

            val_metrics = {k: v / val_num_batches for k, v in val_epoch_metrics.items()}
            if is_distributed:
                val_metrics = sync_metrics(val_metrics)
            if is_main_process() and wandb_logger is not None:
                wandb_logger.log(val_metrics)
            bc.policy.train()

        # Synchronize before next epoch (if distributed)
        barrier()
        epoch += 1

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
