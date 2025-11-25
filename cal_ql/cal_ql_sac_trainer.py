import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from model.model import Scaler
from utils.utils import prefix_metrics


class Trainer(object):
    def __init__(self, config, policy, qf):
        self.config = config
        self.policy = policy
        self.qf = qf
        assert len(qf) == 4, "Expected two Q-functions and their targets."
        self.optimizers = {}
        self.optimizers["policy"] = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.optimizers["qf"] = optim.Adam(
            list(self.qf["qf1"].parameters()) + list(self.qf["qf2"].parameters()), lr=config.qf_lr
        )
        self.scaler = GradScaler()

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scaler(0.0)
            self.optimizers["log_alpha"] = optim.Adam(self.log_alpha.parameters(), lr=config.policy_lr)

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scaler(1.0)
            self.optimizers["log_alpha_prime"] = optim.Adam(self.log_alpha_prime.parameters(), lr=config.qf_lr)
        self._total_steps = 0
        self._modules = [self.policy, self.qf['qf1'], self.qf['qf2'], self.qf['target_qf1'], self.qf['target_qf2']]
        if self.config.use_automatic_entropy_tuning:
            self._modules.append(self.log_alpha)
        if self.config.cql_lagrange:
            self._modules.append(self.log_alpha_prime)

    def train(self, batch,  use_cql=True, cql_min_q_weight=5.0, enable_calql=False):
        self._total_steps += 1
        metrics = self._train_step(batch, use_cql=use_cql, cql_min_q_weight=cql_min_q_weight, enable_calql=enable_calql)
        return metrics
    

    def _train_step(self, batch, use_cql=True, cql_min_q_weight=5.0, enable_calql=False):
        observations = batch["observations"]['proprio'].to(self.device)
        images = batch["observations"]['image'].to(self.device)
        next_observations = batch["next_observations"]['proprio'].to(self.device)
        next_images = batch["next_observations"]['image'].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)
        # observations = batch["observations"]
        # actions = batch["actions"]
        # rewards = batch["rewards"]
        # next_observations = batch["next_observations"]
        # dones = batch["dones"]
        
        # Policy forward
        with autocast(device_type=observations.device.type, enabled=torch.is_autocast_enabled()):
            new_actions, log_pi = self.policy(observations, images)

            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
                alpha = torch.exp(self.log_alpha()) * self.config.alpha_multiplier
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            # Q forward
            
            q_new_actions = torch.min(self.qf["qf1"](observations, images, new_actions), self.qf["qf2"](observations, images, new_actions))
            policy_loss = (alpha.detach() * log_pi - q_new_actions).mean()

            q1_pred = self.qf["qf1"](observations, images, actions)
            q2_pred = self.qf["qf2"](observations, images, actions)
            if self.config.cql_max_target_backup:
                new_next_actions, next_log_pi = self.policy(next_observations, next_images, repeat=self.config.cql_n_actions)
                target_q_values = torch.min(
                    self.qf["target_qf1"](next_observations, next_images, new_next_actions),
                    self.qf["target_qf2"](next_observations, next_images, new_next_actions),
                )
                max_target_indices = torch.argmax(target_q_values, dim=-1, keepdim=True)
                target_q_values = torch.gather(target_q_values, dim=-1, index=max_target_indices).squeeze(-1)
                next_log_pi = torch.gather(next_log_pi, dim=-1, index=max_target_indices).squeeze(-1)
            else:
                new_next_actions, next_log_pi = self.policy(next_observations, next_images)
                target_q_values = torch.min(
                    self.qf["target_qf1"](next_observations, next_images, new_next_actions),
                    self.qf["target_qf2"](next_observations, next_images, new_next_actions),
                )
            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi
            td_target = rewards + (1.0 - dones) * self.config.discount * target_q_values
            qf1_loss = nn.functional.mse_loss(q1_pred, td_target.detach())
            qf2_loss = nn.functional.mse_loss(q2_pred, td_target.detach())
            if use_cql:
                batch_size = actions.shape[0]
                action_dim = actions.shape[-1]
                cql_random_actions = torch.rand(batch_size, self.config.cql_n_actions, action_dim) * 2 - 1
                cql_random_actions = cql_random_actions.to(observations.device)
                cql_current_actions, cql_current_log_pis = self.policy(observations, images, repeat=self.config.cql_n_actions)
                cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
                cql_next_actions, cql_next_log_pis = self.policy(next_observations, next_images, repeat=self.config.cql_n_actions)
                cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()
                cql_q1_rand = self.qf["qf1"](observations, images, cql_random_actions)
                cql_q2_rand = self.qf["qf2"](observations, images, cql_random_actions)
                cql_q1_current_actions = self.qf["qf1"](observations, images, cql_current_actions)
                cql_q2_current_actions = self.qf["qf2"](observations, images, cql_current_actions)
                cql_q1_next_actions = self.qf["qf1"](observations, images, cql_next_actions)
                cql_q2_next_actions = self.qf["qf2"](observations, images, cql_next_actions)
                """ Cal-QL: prepare for Cal-QL, and calculate how much data will be lower bounded for logging """
                mc_returns = batch["mc_return"].to(self.device)
                lower_bounds = mc_returns.view(-1, 1).repeat(1, cql_q1_current_actions.shape[1])
                if enable_calql:
                    cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds)
                    cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds)
                    cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds)
                    cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds)

                cql_cat_q1 = torch.cat(
                    [cql_q1_rand, q1_pred.unsqueeze(1), cql_q1_next_actions, cql_q1_current_actions], dim=1
                )
                cql_cat_q2 = torch.cat(
                    [cql_q2_rand, q2_pred.unsqueeze(1), cql_q2_next_actions, cql_q2_current_actions], dim=1
                )
                cql_std_q1 = torch.std(cql_cat_q1, dim=1)
                cql_std_q2 = torch.std(cql_cat_q2, dim=1)
                if self.config.cql_importance_sample:
                    random_density = np.log(0.5**action_dim)
                    cql_cat_q1 = torch.cat(
                        [
                            cql_q1_rand - random_density,
                            cql_q1_next_actions - cql_next_log_pis,
                            cql_q1_current_actions - cql_current_log_pis,
                        ],
                        dim=1,
                    )
                    cql_cat_q2 = torch.cat(
                        [
                            cql_q2_rand - random_density,
                            cql_q2_next_actions - cql_next_log_pis,
                            cql_q2_current_actions - cql_current_log_pis,
                        ],
                        dim=1,
                    )
                cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.config.cql_temp, dim=1) * self.config.cql_temp
                cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.config.cql_temp, dim=1) * self.config.cql_temp
                cql_qf1_diff = torch.clamp(
                    cql_qf1_ood - q1_pred, self.config.cql_clip_diff_min, self.config.cql_clip_diff_max
                ).mean()
                cql_qf2_diff = torch.clamp(
                    cql_qf2_ood - q2_pred, self.config.cql_clip_diff_min, self.config.cql_clip_diff_max
                ).mean()
                if self.config.cql_lagrange:
                    alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), 0.0, 1000000.0)
                    cql_min_qf1_loss = alpha_prime * cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)
                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
                else:
                    cql_min_qf1_loss = cql_qf1_diff * cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * cql_min_q_weight
                    alpha_prime_loss = torch.tensor(0.0)
                    alpha_prime = torch.tensor(0.0)
                qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
            else:
                qf_loss = qf1_loss + qf2_loss
        if self.config.cql_lagrange and use_cql:
            self.optimizers["log_alpha_prime"].zero_grad()
            self.scaler.scale(alpha_prime_loss).backward(retain_graph=True)
            self.scaler.step(self.optimizers["log_alpha_prime"])

        if self.config.use_automatic_entropy_tuning:
            self.optimizers["log_alpha"].zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.optimizers["log_alpha"])

        self.optimizers["policy"].zero_grad()
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizers["policy"])

        self.optimizers["qf"].zero_grad()
        self.scaler.scale(qf_loss).backward()
        self.scaler.step(self.optimizers["qf"])
        self.scaler.update()

        if self._total_steps % self.config.target_update_interval == 0:
            with torch.no_grad():
                for target_param, param in zip(self.qf["target_qf1"].parameters(), self.qf["qf1"].parameters()):
                    new_param = (1 - self.config.soft_target_update_rate) * target_param.data + self.config.soft_target_update_rate * param.data
                    target_param.data.copy_(new_param)
                for target_param, param in zip(self.qf["target_qf2"].parameters(), self.qf["qf2"].parameters()):
                    new_param = (1 - self.config.soft_target_update_rate) * target_param.data + self.config.soft_target_update_rate * param.data
                    target_param.data.copy_(new_param)

        metrics = dict(
            log_pi=log_pi.mean(),
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
            alpha=alpha,
            average_qf1=q1_pred.mean(),
            average_qf2=q2_pred.mean(),
            average_target_q=target_q_values.mean(),
            total_steps=self._total_steps,
        )
        metrics.update(use_cql=int(use_cql), enable_calql=int(enable_calql), cql_min_q_weight=cql_min_q_weight)
        if use_cql:
            metrics.update(
                prefix_metrics(
                    dict(
                        cql_std_q1=cql_std_q1.mean(),
                        cql_std_q2=cql_std_q2.mean(),
                        cql_q1_rand=cql_q1_rand.mean(),
                        cql_q2_rand=cql_q2_rand.mean(),
                        cql_min_qf1_loss=cql_min_qf1_loss.mean(),
                        cql_min_qf2_loss=cql_min_qf2_loss.mean(),
                        cql_qf1_diff=cql_qf1_diff.mean(),
                        cql_qf2_diff=cql_qf2_diff.mean(),
                        cql_q1_current_actions=cql_q1_current_actions.mean(),
                        cql_q2_current_actions=cql_q2_current_actions.mean(),
                        cql_q1_next_actions=cql_q1_next_actions.mean(),
                        cql_q2_next_actions=cql_q2_next_actions.mean(),
                        alpha_prime_loss=alpha_prime_loss,
                        alpha_prime=alpha_prime,
                    ),
                    "cql",
                )
            )
        return metrics

    def to_device(self, device):
        self.device = device
        self.policy.to(device)
        self.qf['qf1'].to(device)
        self.qf['qf2'].to(device)
        self.qf['target_qf1'].to(device)
        self.qf['target_qf2'].to(device)
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha.to(device)
        if self.config.cql_lagrange:
            self.log_alpha_prime.to(device)

    def compile(self, mode="default"):
        self.policy = torch.compile(self.policy, mode=mode)
        self.qf['qf1'] = torch.compile(self.qf['qf1'], mode=mode)
        self.qf['qf2'] = torch.compile(self.qf['qf2'], mode=mode)
        self.qf['target_qf1'] = torch.compile(self.qf['target_qf1'], mode=mode)
        self.qf['target_qf2'] = torch.compile(self.qf['target_qf2'], mode=mode)
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = torch.compile(self.log_alpha, mode=mode)
        if self.config.cql_lagrange:
            self.log_alpha_prime = torch.compile(self.log_alpha_prime, mode=mode)
        
    
    @property
    def modules(self):
        return self._modules

    @property
    def total_steps(self):
        return self._total_steps
    
    
    def setup_multi_gpu(self, local_rank: int):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        self.to_device(device)
        self.policy = nn.SyncBatchNorm.convert_sync_batchnorm(self.policy)
        self.qf['qf1'] = nn.SyncBatchNorm.convert_sync_batchnorm(self.qf['qf1'])
        self.qf['qf2'] = nn.SyncBatchNorm.convert_sync_batchnorm(self.qf['qf2'])

        self.policy = DDP(self.policy, device_ids=[local_rank], output_device=local_rank)
        self.qf['qf1'] = DDP(self.qf['qf1'], device_ids=[local_rank], output_device=local_rank)
        self.qf['qf2'] = DDP(self.qf['qf2'], device_ids=[local_rank], output_device=local_rank)

        self.optimizers["policy"] = torch.optim.Adam(self.policy.parameters(), lr=self.config.policy_lr)
        self.optimizers["qf"] = torch.optim.Adam(
            list(self.qf['qf1'].parameters()) + list(self.qf['qf2'].parameters()), lr=self.config.qf_lr
        )
        
        if self.config.use_automatic_entropy_tuning:
            self.optimizers["log_alpha"] = torch.optim.Adam(self.log_alpha.parameters(), lr=self.config.policy_lr)
        if self.config.cql_lagrange:
            self.optimizers["log_alpha_prime"] = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=self.config.qf_lr)

        print(f"[Rank {dist.get_rank()}] Trainer multi-GPU setup complete. Device: {device}")
        

        
    def save_checkpoint(self, filepath):
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'qf1_state_dict': self.qf['qf1'].state_dict(),
            'qf2_state_dict': self.qf['qf2'].state_dict(),
            'target_qf1_state_dict': self.qf['target_qf1'].state_dict(),
            'target_qf2_state_dict': self.qf['target_qf2'].state_dict(),
            'optimizers_state_dict': {k: v.state_dict() for k, v in self.optimizers.items()},
            'total_steps': self._total_steps,
        }
        if self.config.use_automatic_entropy_tuning:
            checkpoint['log_alpha_state_dict'] = self.log_alpha.state_dict()
        if self.config.cql_lagrange:
            checkpoint['log_alpha_prime_state_dict'] = self.log_alpha_prime.state_dict()
        torch.save(checkpoint, filepath)
