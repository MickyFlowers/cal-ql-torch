"""
Behavior Cloning Finetune Script (Old + New 1:1 Sampling)

Finetune BC policy using a mixed dataset that samples old and new data equally.
"""

import json
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
from data.dataset import BCDatasetLMDBMix
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


@hydra.main(config_path="../config", config_name="train_bc_finetune", version_base=None)
def main(cfg: DictConfig):
    local_rank, device, world_size = setup_training(cfg.device)
    is_distributed = world_size > 1

    if is_main_process():
        print(OmegaConf.to_yaml(cfg))
        print(f"Training mode: {'Distributed' if is_distributed else 'Single GPU'}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")

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

    train_dataset = BCDatasetLMDBMix(
        cfg.dataset,
        sample_seed=cfg.get("sample_seed", 42),
    )

    if is_main_process():
        print(f"Train dataset size: {len(train_dataset)} samples")
        print(f"Old dataset size: {train_dataset.old_len} samples")
        print(f"New dataset size: {train_dataset.new_len} samples")

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

    np.random.seed(cfg.seed + local_rank)
    torch.manual_seed(cfg.seed + local_rank)

    policy = ResNetPolicy(
        cfg.observation_dim,
        cfg.action_dim,
        cfg.policy_obs_proj_arch,
        cfg.policy_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        cfg.policy_log_std_multiplier,
        cfg.policy_log_std_offset,
        train_backbone=cfg.train_policy_backbone,
    )

    bc = BehaviorCloneTrainer(
        cfg.learning_rate,
        policy,
        input_modality=cfg.get("bc_input_modality", "both"),
    )
    if is_distributed:
        bc.setup_multi_gpu(local_rank)
    else:
        bc.to_device(device=device)

    if cfg.load_ckpt_path != "":
        bc.load_checkpoint(cfg.load_ckpt_path)

    viskit_metrics = {}
    total_grad_steps = 0
    train_timer = None
    epoch = 0
    train_metrics = None

    ckpt_path = None
    if cfg.save_every_n_epoch > 0 and is_main_process():
        ckpt_path = os.path.join(cfg.ckpt_path, f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(ckpt_path, exist_ok=True)
        config_save_path = os.path.join(ckpt_path, "config.yaml")
        with open(config_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        print(f"Saved config to {config_save_path}")
        stats_save_path = os.path.join(ckpt_path, "statistics.json")
        with open(stats_save_path, "w") as f:
            json.dump(train_dataset.statistics, f, indent=2, sort_keys=True)
        print(f"Saved statistics to {stats_save_path}")

    while True:
        metrics = {"epoch": epoch}
        metrics["grad_steps"] = total_grad_steps
        metrics["epoch"] = epoch
        metrics["train_time"] = 0 if train_timer is None else train_timer()
        if train_metrics is not None:
            metrics.update(train_metrics)

        if is_main_process():
            if wandb_logger is not None:
                wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        if epoch % cfg.save_every_n_epoch == 0 and epoch != 0 and is_main_process():
            ckpt_file_path = os.path.join(ckpt_path, f'bc_checkpoint_{epoch:05d}.pt')
            bc.save_checkpoint(ckpt_file_path)

        if epoch >= cfg.train_bc_epochs:
            if is_main_process():
                print("Finished Training")
            break

        with Timer() as train_timer:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_metrics = {}
            num_batches = 0
            for batch in tqdm(train_dataloader, desc="Training", disable=not is_main_process()):
                batch = dict_to_device(batch, device=device)
                batch_metrics = bc.train(batch)
                for k, v in batch_metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().item()
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0.0
                    epoch_metrics[k] += v
                num_batches += 1

            train_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

            if is_distributed:
                train_metrics = sync_metrics(train_metrics)

            total_grad_steps += len(train_dataloader)

        barrier()
        epoch += 1

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
