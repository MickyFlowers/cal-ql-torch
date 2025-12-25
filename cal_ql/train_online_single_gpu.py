import copy
import itertools
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from cal_ql.cal_ql_sac_trainer import Trainer
from data.dataset import CalqlDataset
from model.model import ResNetPolicy, ResNetQFunction
from utils.logger import WandBLogger
from utils.utils import Timer
from viskit.logging import logger, setup_logger

def data_iter_fn(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def dict_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = dict_to_device(v, device)
        else:
            batch[k] = v.to(device=device, non_blocking=True)
    return batch

@hydra.main(config_path="../config", config_name="train_online", version_base=None)
def main(cfg: DictConfig):
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
    dataset = CalqlDataset(cfg.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    data_iter = data_iter_fn(dataloader)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


    observation_dim = cfg.observation_dim
    action_dim = cfg.action_dim

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
    
    cfg.cal_ql.target_entropy = -np.prod((1, action_dim)).item()

    sac = Trainer(cfg.cal_ql, policy, qf)
    # sac.setup_multi_gpu(local_rank)
    sac.to_device(device=device)
    if cfg.load_ckpt_path != "":
        sac.load_checkpoint(cfg.load_ckpt_path)
    # print("compiling sac model...")
    # sac.compile(mode=cfg.torch_compile_mode)

    viskit_metrics = {}
    # n_train_step_per_epoch = cfg.n_train_step_per_epoch_offline
    cql_min_q_weight = cfg.cql_min_q_weight
    enable_calql = cfg.enable_calql
    use_cql = cfg.use_cql
    total_grad_steps = 0
    train_timer = None
    epoch = 0
    train_metrics = None
    expl_metrics = None
    if cfg.save_every_n_epoch > 0:
        ckpt_path = os.path.join(cfg.ckpt_path, f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(ckpt_path, exist_ok=True)
    while True:
        metrics = {"epoch": epoch}
        metrics["grad_steps"] = total_grad_steps
        metrics["epoch"] = epoch
        metrics["train_time"] = 0 if train_timer is None else train_timer()
        if train_metrics is not None:
            metrics.update(train_metrics)
        if expl_metrics is not None:
            metrics.update(expl_metrics)
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        if epoch % cfg.save_every_n_epoch == 0 and epoch != 0:
            ckpt_file_path = os.path.join(ckpt_path, f'checkpoint_{epoch:05d}.pt')
            sac.save_checkpoint(ckpt_file_path)
            
        if epoch >= cfg.train_offline_epochs:
            print("Finished Training")
            break
        dataset.reload()
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        data_iter = data_iter_fn(dataloader)
        with Timer() as train_timer:
            # Accumulate metrics over the entire epoch
            epoch_metrics = {}
            num_batches = 0
            for batch in tqdm(dataloader, desc="Training"):
                batch = next(data_iter)
                batch = dict_to_device(batch, device=device)
                batch_metrics = sac.train(
                    batch, use_cql=use_cql, cql_min_q_weight=cql_min_q_weight, enable_calql=enable_calql
                )
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
            total_grad_steps += len(dataloader)
        epoch += 1


if __name__ == "__main__":
    main()
