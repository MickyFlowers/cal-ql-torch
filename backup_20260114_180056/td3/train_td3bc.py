"""
TD3+BC Offline Training Script

Train TD3+BC algorithm for offline reinforcement learning.
Supports both single-GPU and multi-GPU (distributed) training automatically.

TD3+BC combines TD3 with behavior cloning regularization to learn from
static offline datasets.

Usage:
    Single GPU:  python -m td3.train_td3bc [args]
    Multi GPU:   torchrun --nproc_per_node=N -m td3.train_td3bc [args]
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

from data.dataset import RoboMimicDataset
from td3.td3_bc_trainer import TD3BCTrainer
from td3.td3_model import ResNetDeterministicPolicy, ResNetTD3QFunction
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


@hydra.main(config_path="../config", config_name="train_td3bc", version_base=None)
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
    dataset = RoboMimicDataset(cfg.dataset)

    # Create dataloader with optional distributed sampler
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Set seeds (add local_rank offset for distributed)
    np.random.seed(cfg.seed + local_rank)
    torch.manual_seed(cfg.seed + local_rank)

    observation_dim = cfg.observation_dim
    action_dim = cfg.action_dim

    # Create deterministic policy for TD3
    policy = ResNetDeterministicPolicy(
        observation_dim,
        action_dim,
        cfg.policy_obs_proj_arch,
        cfg.policy_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_policy_backbone,
    )

    # Create Q-functions
    qf = {}
    qf["qf1"] = ResNetTD3QFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone,
    )
    qf["qf2"] = ResNetTD3QFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone,
    )
    qf["target_qf1"] = copy.deepcopy(qf["qf1"])
    qf["target_qf2"] = copy.deepcopy(qf["qf2"])

    # Create trainer and setup device/distributed
    td3bc = TD3BCTrainer(cfg.td3bc, policy, qf)
    if is_distributed:
        td3bc.setup_multi_gpu(local_rank)
    else:
        td3bc.to_device(device=device)

    # Load checkpoint if specified
    if cfg.load_ckpt_path != "":
        td3bc.load_checkpoint(cfg.load_ckpt_path)

    viskit_metrics = {}
    total_grad_steps = 0
    train_timer = None
    epoch = 0
    train_metrics = None

    # Create checkpoint directory (only on main process)
    ckpt_path = None
    if cfg.save_every_n_epoch > 0 and is_main_process():
        ckpt_path = os.path.join(
            cfg.ckpt_path, f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}'
        )
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
            ckpt_file_path = os.path.join(ckpt_path, f"checkpoint_{epoch:05d}.pt")
            td3bc.save_checkpoint(ckpt_file_path)

        if epoch >= cfg.train_epochs:
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
                batch_metrics = td3bc.train(batch)

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
