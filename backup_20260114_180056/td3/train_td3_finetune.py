"""TD3 Fine-tuning Script (Data-based Online Training).

This script implements fine-tuning of a TD3 policy from a pretrained TD3+BC
checkpoint using a dataset (without real-time environment interaction).

Use this when:
- You have collected new data and want to fine-tune
- You want to continue training from a checkpoint
- You want to switch from TD3+BC (with BC regularization) to pure TD3

For real-time online training with environment interaction,
use train_td3_online.py instead.
"""

import copy
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CalqlDataset, RoboMimicDataset
from td3.td3_model import ResNetDeterministicPolicy, ResNetTD3QFunction
from td3.td3_trainer import TD3Trainer
from utils.logger import WandBLogger
from utils.utils import Timer
from viskit.logging import logger, setup_logger


def dict_to_device(batch, device):
    """Move batch data to device."""
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = dict_to_device(v, device)
        else:
            batch[k] = v.to(device=device, non_blocking=True)
    return batch


@hydra.main(config_path="../config", config_name="train_td3_finetune", version_base=None)
def main(cfg: DictConfig):
    """Main fine-tuning function."""
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg.device)
    variant = OmegaConf.to_container(cfg, resolve=True)

    # Setup logging
    wandb_logger = WandBLogger(config=cfg.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=cfg.seed,
        base_log_dir=cfg.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )

    # Load dataset
    dataset_type = getattr(cfg, "dataset_type", "calql")
    if dataset_type == "robomimic":
        dataset = RoboMimicDataset(cfg.dataset)
    else:
        dataset = CalqlDataset(cfg.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    observation_dim = cfg.observation_dim
    action_dim = cfg.action_dim

    # Create models
    policy = ResNetDeterministicPolicy(
        observation_dim,
        action_dim,
        cfg.policy_obs_proj_arch,
        cfg.policy_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_policy_backbone,
    )

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

    # Create trainer (pure TD3 without BC)
    td3 = TD3Trainer(cfg.td3bc, policy, qf)
    td3.to_device(device=device)

    # Load pretrained checkpoint
    if cfg.load_ckpt_path != "":
        print(f"Loading pretrained checkpoint from {cfg.load_ckpt_path}")
        # Don't load optimizer states when switching from TD3+BC to TD3
        td3.load_checkpoint(cfg.load_ckpt_path, load_optimizer=cfg.load_optimizer)

    # Setup checkpoint directory
    viskit_metrics = {}
    total_grad_steps = 0
    train_timer = None
    epoch = 0
    train_metrics = None

    if cfg.save_every_n_epoch > 0:
        ckpt_path = os.path.join(
            cfg.ckpt_path, f'{cfg.logging.prefix}_finetune_{time.strftime("%Y%m%d_%H%M%S")}'
        )
        os.makedirs(ckpt_path, exist_ok=True)

    print("Starting fine-tuning...")

    while True:
        metrics = {"epoch": epoch}
        metrics["grad_steps"] = total_grad_steps
        metrics["train_time"] = 0 if train_timer is None else train_timer()

        if train_metrics is not None:
            metrics.update(train_metrics)

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # Save checkpoint
        if cfg.save_every_n_epoch > 0 and epoch % cfg.save_every_n_epoch == 0 and epoch != 0:
            ckpt_file_path = os.path.join(ckpt_path, f"checkpoint_{epoch:05d}.pt")
            td3.save_checkpoint(ckpt_file_path)

        if epoch >= cfg.train_epochs:
            print("Finished Fine-tuning")
            break

        # Reload dataset if it supports dynamic data
        if hasattr(dataset, "reload"):
            dataset.reload()
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

        with Timer() as train_timer:
            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                batch = dict_to_device(batch, device=device)
                train_metrics = td3.train(batch)

                def post_process(m):
                    for k, v in m.items():
                        if isinstance(v, torch.Tensor):
                            m[k] = v.detach().item()
                    return m

                train_metrics = post_process(train_metrics)

            total_grad_steps += len(dataloader)
        epoch += 1

    # Final save
    if cfg.save_every_n_epoch > 0:
        ckpt_file_path = os.path.join(ckpt_path, "checkpoint_final.pt")
        td3.save_checkpoint(ckpt_file_path)


if __name__ == "__main__":
    main()
