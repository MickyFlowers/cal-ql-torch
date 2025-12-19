import copy
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import RoboMimicDataset
from td3.td3_bc_trainer import TD3BCTrainer
from td3.td3_model import ResNetDeterministicPolicy, ResNetTD3QFunction
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
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg.device)
    variant = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandBLogger(config=cfg.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=cfg.seed,
        base_log_dir=cfg.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )

    dataset = RoboMimicDataset(cfg.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

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
    qf['qf1'] = ResNetTD3QFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone
    )
    qf['qf2'] = ResNetTD3QFunction(
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

    td3bc = TD3BCTrainer(cfg.td3bc, policy, qf)
    td3bc.to_device(device=device)

    if cfg.load_ckpt_path != "":
        td3bc.load_checkpoint(cfg.load_ckpt_path)

    viskit_metrics = {}
    total_grad_steps = 0
    train_timer = None
    epoch = 0
    train_metrics = None

    if cfg.save_every_n_epoch > 0:
        ckpt_path = os.path.join(cfg.ckpt_path, f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(ckpt_path, exist_ok=True)

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

        if epoch % cfg.save_every_n_epoch == 0 and epoch != 0:
            ckpt_file_path = os.path.join(ckpt_path, f'checkpoint_{epoch:05d}.pt')
            td3bc.save_checkpoint(ckpt_file_path)

        if epoch >= cfg.train_epochs:
            print("Finished Training")
            break

        with Timer() as train_timer:
            for batch in tqdm(dataloader, desc="Training"):
                batch = dict_to_device(batch, device=device)
                train_metrics = td3bc.train(batch)

                def post_process(metrics):
                    for k, v in metrics.items():
                        if isinstance(v, torch.Tensor):
                            metrics[k] = v.detach().item()
                    return metrics

                train_metrics = post_process(train_metrics)

            total_grad_steps += len(dataloader)
        epoch += 1


if __name__ == "__main__":
    main()
