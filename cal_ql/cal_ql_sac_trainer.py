import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

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
        
        # 1. 将数据移动到设备
        observations = batch["observations"]['proprio'].to(self.device)
        images = batch["observations"]['image'].to(self.device)
        next_observations = batch["next_observations"]['proprio'].to(self.device)
        next_images = batch["next_observations"]['image'].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)
        mc_returns = batch["mc_return"].to(self.device)

        # 2. 准备静态参数
        static_params = {
            "use_cql": use_cql,
            "cql_min_q_weight": cql_min_q_weight,
            "enable_calql": enable_calql,
            "cql_max_target_backup": self.config.cql_max_target_backup,
            "backup_entropy": self.config.backup_entropy,
            "discount": self.config.discount,
            "cql_n_actions": self.config.cql_n_actions,
            "cql_importance_sample": self.config.cql_importance_sample,
            "cql_temp": self.config.cql_temp,
            "cql_clip_diff_min": self.config.cql_clip_diff_min,
            "cql_clip_diff_max": self.config.cql_clip_diff_max,
            "cql_lagrange": self.config.cql_lagrange,
            "cql_target_action_gap": self.config.cql_target_action_gap,
            "use_automatic_entropy_tuning": self.config.use_automatic_entropy_tuning,
            "target_entropy": self.config.target_entropy,
            "alpha_multiplier": self.config.alpha_multiplier,
        }

        # 3. 调用编译后的函数
        losses, computed_metrics = self._compute_loss_and_metrics(
            observations, images, actions, rewards, next_observations, next_images, dones, mc_returns,
            **static_params
        )

        # 4. 执行优化器步骤 (这部分不应被编译)
        if self.config.cql_lagrange and use_cql:
            self.optimizers["log_alpha_prime"].zero_grad(set_to_none=True)
            self.scaler.scale(losses['alpha_prime_loss']).backward(retain_graph=True)
            self.scaler.step(self.optimizers["log_alpha_prime"])

        if self.config.use_automatic_entropy_tuning:
            self.optimizers["log_alpha"].zero_grad(set_to_none=True)
            self.scaler.scale(losses['alpha_loss']).backward(retain_graph=True)
            self.scaler.step(self.optimizers["log_alpha"])

        self.optimizers["policy"].zero_grad(set_to_none=True)
        self.scaler.scale(losses['policy_loss']).backward(retain_graph=True)
        self.scaler.step(self.optimizers["policy"])

        self.optimizers["qf"].zero_grad(set_to_none=True)
        self.scaler.scale(losses['qf_loss']).backward()
        self.scaler.step(self.optimizers["qf"])
        
        self.scaler.update()

        # 5. 软更新目标网络
        if self._total_steps % self.config.target_update_interval == 0:
            self._soft_update_target_networks()

        # 6. 组装最终的 metrics
        final_metrics = {k: v.item() for k, v in computed_metrics.items()}
        final_metrics.update({
            'policy_loss': losses['policy_loss'].item(),
            'qf_loss': losses['qf_loss'].item(),
            'alpha_loss': losses['alpha_loss'].item(),
            'total_steps': self._total_steps,
            'use_cql': int(use_cql),
            'enable_calql': int(enable_calql),
            'cql_min_q_weight': cql_min_q_weight,
        })
        if self.config.cql_lagrange and use_cql:
            final_metrics['alpha_prime_loss'] = losses['alpha_prime_loss'].item()
        
        return final_metrics

    @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def _compute_loss_and_metrics(
        self,
        observations, images, actions, rewards, next_observations, next_images, dones, mc_returns,
        # Static parameters below
        use_cql, cql_min_q_weight, enable_calql, cql_max_target_backup, backup_entropy,
        discount, cql_n_actions, cql_importance_sample, cql_temp, cql_clip_diff_min,
        cql_clip_diff_max, cql_lagrange, cql_target_action_gap, use_automatic_entropy_tuning,
        target_entropy, alpha_multiplier
    ):
        losses = {}
        metrics = {}

        with autocast(device_type=observations.device.type, enabled=torch.is_autocast_enabled()):
            # Policy forward
            new_actions, log_pi = self.policy(observations, images)
            metrics['log_pi'] = log_pi.mean()

            if use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha() * (log_pi + target_entropy).detach()).mean()
                alpha = torch.exp(self.log_alpha()) * alpha_multiplier
            else:
                alpha_loss = torch.tensor(0.0, device=observations.device)
                alpha = torch.tensor(alpha_multiplier, device=observations.device)
            
            losses['alpha_loss'] = alpha_loss
            metrics['alpha'] = alpha

            # Q forward
            q_new_actions = torch.min(
                self.qf['qf1'](observations, images, new_actions),
                self.qf['qf2'](observations, images, new_actions)
            )
            policy_loss = (alpha.detach() * log_pi - q_new_actions).mean()
            losses['policy_loss'] = policy_loss

            q1_pred = self.qf['qf1'](observations, images, actions)
            q2_pred = self.qf['qf2'](observations, images, actions)
            metrics.update({'average_qf1': q1_pred.mean(), 'average_qf2': q2_pred.mean()})

            if cql_max_target_backup:
                new_next_actions, next_log_pi = self.policy(next_observations, next_images, repeat=cql_n_actions)
                target_q_values = torch.min(
                    self.qf['target_qf1'](next_observations, next_images, new_next_actions),
                    self.qf['target_qf2'](next_observations, next_images, new_next_actions),
                )
                max_target_indices = torch.argmax(target_q_values, dim=-1, keepdim=True)
                target_q_values = torch.gather(target_q_values, dim=-1, index=max_target_indices).squeeze(-1)
                next_log_pi = torch.gather(next_log_pi, dim=-1, index=max_target_indices).squeeze(-1)
            else:
                new_next_actions, next_log_pi = self.policy(next_observations, next_images)
                target_q_values = torch.min(
                    self.qf['target_qf1'](next_observations, next_images, new_next_actions),
                    self.qf['target_qf2'](next_observations, next_images, new_next_actions),
                )
            
            metrics['average_target_q'] = target_q_values.mean()

            if backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi
            
            td_target = rewards + (1.0 - dones) * discount * target_q_values
            qf1_loss = nn.functional.mse_loss(q1_pred, td_target.detach())
            qf2_loss = nn.functional.mse_loss(q2_pred, td_target.detach())
            metrics.update({'qf1_loss': qf1_loss, 'qf2_loss': qf2_loss})

            if use_cql:
                # ... (CQL logic remains largely the same, just using static params)
                batch_size = actions.shape[0]
                action_dim = actions.shape[-1]
                cql_random_actions = torch.rand(batch_size, cql_n_actions, action_dim, device=observations.device) * 2 - 1
                
                cql_current_actions, cql_current_log_pis = self.policy(observations, images, repeat=cql_n_actions)
                cql_next_actions, cql_next_log_pis = self.policy(next_observations, next_images, repeat=cql_n_actions)

                cql_q1_rand = self.qf['qf1'](observations, images, cql_random_actions)
                cql_q2_rand = self.qf['qf2'](observations, images, cql_random_actions)
                cql_q1_current_actions = self.qf['qf1'](observations, images, cql_current_actions.detach())
                cql_q2_current_actions = self.qf['qf2'](observations, images, cql_current_actions.detach())
                cql_q1_next_actions = self.qf['qf1'](observations, images, cql_next_actions.detach())
                cql_q2_next_actions = self.qf['qf2'](observations, images, cql_next_actions.detach())

                if enable_calql:
                    lower_bounds = mc_returns.view(-1, 1).repeat(1, cql_q1_current_actions.shape[1])
                    cql_q1_current_actions = torch.max(cql_q1_current_actions, lower_bounds)
                    cql_q2_current_actions = torch.max(cql_q2_current_actions, lower_bounds)
                    cql_q1_next_actions = torch.max(cql_q1_next_actions, lower_bounds)
                    cql_q2_next_actions = torch.max(cql_q2_next_actions, lower_bounds)

                cql_cat_q1 = torch.cat([cql_q1_rand, q1_pred.unsqueeze(1), cql_q1_next_actions, cql_q1_current_actions], dim=1)
                cql_cat_q2 = torch.cat([cql_q2_rand, q2_pred.unsqueeze(1), cql_q2_next_actions, cql_q2_current_actions], dim=1)

                if cql_importance_sample:
                    random_density = np.log(0.5**action_dim)
                    cql_cat_q1 = torch.cat([
                        cql_q1_rand - random_density,
                        cql_q1_next_actions - cql_next_log_pis.detach(),
                        cql_q1_current_actions - cql_current_log_pis.detach(),
                    ], dim=1)
                    cql_cat_q2 = torch.cat([
                        cql_q2_rand - random_density,
                        cql_q2_next_actions - cql_next_log_pis.detach(),
                        cql_q2_current_actions - cql_current_log_pis.detach(),
                    ], dim=1)

                cql_qf1_ood = torch.logsumexp(cql_cat_q1 / cql_temp, dim=1) * cql_temp
                cql_qf2_ood = torch.logsumexp(cql_cat_q2 / cql_temp, dim=1) * cql_temp

                cql_qf1_diff = torch.clip(cql_qf1_ood - q1_pred, cql_clip_diff_min, cql_clip_diff_max).mean()
                cql_qf2_diff = torch.clip(cql_qf2_ood - q2_pred, cql_clip_diff_min, cql_clip_diff_max).mean()

                if cql_lagrange:
                    alpha_prime = torch.clip(torch.exp(self.log_alpha_prime()), 0.0, 1000000.0)
                    cql_min_qf1_loss = alpha_prime * cql_min_q_weight * (cql_qf1_diff - cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * cql_min_q_weight * (cql_qf2_diff - cql_target_action_gap)
                    losses['alpha_prime_loss'] = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
                else:
                    cql_min_qf1_loss = cql_qf1_diff * cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * cql_min_q_weight
                    losses['alpha_prime_loss'] = torch.tensor(0.0, device=observations.device)
                
                qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
            else:
                qf_loss = qf1_loss + qf2_loss
                losses['alpha_prime_loss'] = torch.tensor(0.0, device=observations.device)

            losses['qf_loss'] = qf_loss
        
        return losses, metrics

    def _soft_update_target_networks(self):
        with torch.no_grad():
            for target_param, param in zip(self.qf["target_qf1"].parameters(), self.qf["qf1"].parameters()):
                target_param.data.mul_(1 - self.config.soft_target_update_rate)
                target_param.data.add_(self.config.soft_target_update_rate * param.data)
            for target_param, param in zip(self.qf["target_qf2"].parameters(), self.qf["qf2"].parameters()):
                target_param.data.mul_(1 - self.config.soft_target_update_rate)
                target_param.data.add_(self.config.soft_target_update_rate * param.data)
    
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
