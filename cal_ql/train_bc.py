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

from cal_ql.bc_trainer import BehaviorCloneTrainer
from data.dataset import CalqlDataset
from model.model import ResNetPolicy
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

@hydra.main(config_path="../config", config_name="train_bc", version_base=None)
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
    # data_iter = data_iter_fn(dataloader)
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

    bc = BehaviorCloneTrainer(cfg.learning_rate, policy)
    # sac.setup_multi_gpu(local_rank)
    bc.to_device(device=device)
    if cfg.load_ckpt_path != "":
        bc.load_checkpoint(cfg.load_ckpt_path)
    # print("compiling sac model...")
    # sac.compile(mode=cfg.torch_compile_mode)

    viskit_metrics = {}
    # n_train_step_per_epoch = cfg.n_train_step_per_epoch_offline
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
            ckpt_file_path = os.path.join(ckpt_path, f'bc_checkpoint_{epoch:05d}.pt')
            bc.save_checkpoint(ckpt_file_path)
            
        if epoch >= cfg.train_bc_epochs:
            print("Finished Training")
            break

        with Timer() as train_timer:
            for batch in tqdm(dataloader, desc="Training"):
            # for _ in tqdm(range(n_train_step_per_epoch), desc="Training"):
                # batch = next(data_iter)
                batch = dict_to_device(batch, device=device)
                train_metrics = bc.train(batch)
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
