"""
IQL (Implicit Q-Learning) Offline Training Script

Train IQL algorithm for offline reinforcement learning.
Supports both single-GPU and multi-GPU (distributed) training automatically.

Reference:
- Paper: "Offline Reinforcement Learning with Implicit Q-Learning" (ICLR 2022)
- Official Implementation: https://github.com/ikostrikov/implicit_q_learning

Usage:
    Single GPU:  python -m iql.train [args]
    Multi GPU:   torchrun --nproc_per_node=N -m iql.train [args]
"""

import copy
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.dataset import CalqlDataset
from iql.iql_trainer import IQLTrainer
from model.model import ResNetPolicy, ResNetQFunction, ResNetVFunction
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


@hydra.main(config_path="../config", config_name="train_iql", version_base=None)
def main(cfg: DictConfig):
    # Setup training environment (auto-detects single vs multi GPU)
    local_rank, device, world_size = setup_training(cfg.device)
    is_distributed = world_size > 1

    torch.autograd.set_detect_anomaly(True)

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

    # Create dataset
    dataset = CalqlDataset(cfg.dataset)

    # Create dataloader with optional distributed sampler
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Set seeds (add local_rank offset for distributed)
    np.random.seed(cfg.seed + local_rank)
    torch.manual_seed(cfg.seed + local_rank)

    observation_dim = cfg.observation_dim
    action_dim = cfg.action_dim

    # Create policy network
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

    # Create Q-function networks
    qf = {}
    qf['qf1'] = ResNetQFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone
    )
    qf['qf2'] = ResNetQFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone
    )
    qf['target_qf1'] = copy.deepcopy(qf['qf1'])
    qf['target_qf2'] = copy.deepcopy(qf['qf2'])

    # Create V-function network (unique to IQL)
    vf = ResNetVFunction(
        observation_dim,
        cfg.v_obs_proj_arch,
        cfg.v_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_v_backbone
    )

    # Create IQL trainer and setup device/distributed
    iql = IQLTrainer(cfg.iql, policy, qf, vf)
    if is_distributed:
        iql.setup_multi_gpu(local_rank)
    else:
        iql.to_device(device=device)

    # Compile if not disabled
    if cfg.torch_compile_mode != "disable":
        iql.compile(mode=cfg.torch_compile_mode)

    # Setup learning rate scheduler
    num_training_steps = cfg.train_iql_epochs * len(dataloader)
    warmup_ratio = getattr(cfg, 'warmup_ratio', 0.05)
    min_lr_ratio = getattr(cfg, 'min_lr_ratio', 0.1)
    if getattr(cfg, 'use_lr_scheduler', True):
        iql.setup_lr_scheduler(num_training_steps, warmup_ratio=warmup_ratio, min_lr_ratio=min_lr_ratio)

    # Load checkpoint if specified
    if cfg.load_ckpt_path != "":
        iql.load_checkpoint(cfg.load_ckpt_path)

    viskit_metrics = {}
    total_grad_steps = 0
    train_timer = None
    epoch = 0
    train_metrics = None

    # Create checkpoint directory (only on main process)
    ckpt_path = None
    if cfg.save_every_n_epoch > 0 and is_main_process():
        ckpt_path = os.path.join(cfg.ckpt_path, f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(ckpt_path, exist_ok=True)

    while True:
        metrics = {"epoch": epoch}
        metrics["grad_steps"] = total_grad_steps
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
            ckpt_file_path = os.path.join(ckpt_path, f'iql_checkpoint_{epoch:05d}.pt')
            iql.save_checkpoint(ckpt_file_path)

        if epoch >= cfg.train_iql_epochs:
            if is_main_process():
                print("Finished Training")
            break

        with Timer() as train_timer:
            # Set epoch for distributed sampler
            if sampler is not None:
                sampler.set_epoch(epoch)

            # Accumulate metrics over the entire epoch
            epoch_metrics = {}
            num_batches = 0
            for batch in tqdm(dataloader, desc="Training", disable=not is_main_process()):
                batch = dict_to_device(batch, device=device)
                batch_metrics = iql.train(batch)
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

            total_grad_steps += len(dataloader)

        # Synchronize before next epoch (if distributed)
        barrier()
        epoch += 1

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
