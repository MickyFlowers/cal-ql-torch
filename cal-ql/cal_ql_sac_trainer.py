import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import collect_metrics, grad_norm


class ConservativeSAC:
    def __init__(self, config, policy, qf):
        self.config = config
        self.policy = policy
        self.qf = qf
        assert len(qf) == 4, "Expected two Q-functions and their targets."
        self.optimizers = {}
        self.optimizers["policy"] = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.optimizers["qf1"] = optim.Adam(self.qf["qf1"].parameters(), lr=config.qf_lr)
        self.optimizers["qf2"] = optim.Adam(self.qf["qf2"].parameters(), lr=config.qf_lr)

        model_keys = ["policy", "qf1", "qf2"]

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = nn.Parameter(
                torch.zeros(
                    1,
                ),
                requires_grad=True,
            )
            self.optimizers["log_alpha"] = optim.Adam([self.log_alpha], lr=config.policy_lr)
            model_keys.append("log_alpha")

        if self.config.cql_lagrange:
            self.log_alpha_prime = nn.Parameter(
                torch.zeros(
                    1,
                ),
                requires_grad=True,
            )
            self.optimizers["log_alpha_prime"] = optim.Adam([self.log_alpha_prime], lr=config.qf_lr)
            model_keys.append("log_alpha_prime")
        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(self, batch):
        self._total_steps += 1
        metrics = self._train_step(batch)
        return metrics

    def compute_loss(self, batch, use_cql=True, cql_min_q_weight=5.0, enable_calql=False):
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        loss_collection = {}
        # Policy forward
        new_actions, log_pi = self.policy(observations)

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -self.log_alpha * (log_pi + self.config.target_entropy).mean()
            loss_collection["alpha_loss"] = alpha_loss.item()
        else:
            alpha_loss = 0.0
            alpha = self.config.alpha_multiplier

        # Q forward
        q_new_actions = torch.min(self.qf["qf1"](observations, new_actions), self.qf["qf2"](observations, new_actions))
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        loss_collection["policy"] = policy_loss

        q1_pred = self.qf["qf1"](observations, actions)
        q2_pred = self.qf["qf2"](observations, actions)
        if self.config.cql_max_target_backup:
            new_next_actions, next_log_pi = self.policy(next_observations, repeat=self.config.cql_n_actions)
            target_q_values = torch.min(
                self.qf["target_qf1"](next_observations, new_next_actions),
                self.qf["target_qf2"](next_observations, new_next_actions),
            )
            max_target_indices = torch.argmax(target_q_values, dim=-1, keepdim=True)
            target_q_values = torch.gather(target_q_values, dim=-1, index=max_target_indices).squeeze(-1)
            next_log_pi = torch.gather(next_log_pi, dim=-1, index=max_target_indices).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.policy(next_observations)
            target_q_values = torch.min(
                self.qf["target_qf1"](next_observations, new_next_actions),
                self.qf["target_qf2"](next_observations, new_next_actions),
            )
        if self.config.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi
        td_target = (rewards + (1.0 - dones) * self.config.discount * target_q_values).detach()
        qf1_bellman_loss = nn.functional.mse_loss(q1_pred, td_target)
        qf2_bellman_loss = nn.functional.mse_loss(q2_pred, td_target)
        if use_cql:
            batch_size = actions.shape[0]
            cql_random_actions = torch.rand(batch_size, self.config.cql_n_actions, self.action_dim) * 2 - 1
            cql_current_actions, cql_current_log_pis = self.policy(observations, repeat=self.config.cql_n_actions)
            cql_next_actions, cql_next_log_pis = self.policy(next_observations, repeat=self.config.cql_n_actions)
            cql_q1_rand = self.qf["qf1"](observations, cql_random_actions)
            cql_q2_rand = self.qf["qf2"](observations, cql_random_actions)
            cql_q1_current_actions = self.qf["qf1"](observations, cql_current_actions)
            cql_q2_current_actions = self.qf["qf2"](observations, cql_current_actions)
            cql_q1_next_actions = self.qf["qf1"](observations, cql_next_actions)
            cql_q2_next_actions = self.qf["qf2"](observations, cql_next_actions)
            """ Cal-QL: prepare for Cal-QL, and calculate how much data will be lower bounded for logging """
            lower_bounds = batch["mc_returns"].view(-1, 1).repeat(1, cql_q1_current_actions.shape[1])
            if enable_calql:
                cql_q1_current_actions = torch.max(cql_q1_current_actions, lower_bounds)
                cql_q2_current_actions = torch.max(cql_q2_current_actions, lower_bounds)
                cql_q1_next_actions = torch.max(cql_q1_next_actions, lower_bounds)
                cql_q2_next_actions = torch.max(cql_q2_next_actions, lower_bounds)

            cql_cat_q1 = torch.cat(
                [cql_cat_q1, q1_pred.unsqueeze(1), cql_q1_next_actions, cql_q1_current_actions], dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_cat_q2, q2_pred.unsqueeze(1), cql_q2_next_actions, cql_q2_current_actions], dim=1
            )
            if self.config.cql_importance_sample:
                random_density = np.log(0.5**self.action_dim)
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
                cql_qf1_diff = torch.clip(
                    cql_qf1_ood - q1_pred, self.config.cql_clip_diff_min, self.config.cql_clip_diff_max
                ).mean()
                cql_qf2_diff = torch.clip(
                    cql_qf2_ood - q2_pred, self.config.cql_clip_diff_min, self.config.cql_clip_diff_max
                ).mean()
                if self.config.cql_lagrange:
                    alpha_prime = torch.clip(torch.exp(self.log_alpha_prime), a_min=0.0, a_max=1000000.0)
                    cql_min_qf1_loss = (
                        alpha_prime * cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                    )
                    cql_min_qf2_loss = (
                        alpha_prime * cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)
                    )

                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5

                    loss_collection["log_alpha_prime"] = alpha_prime_loss

                else:
                    cql_min_qf1_loss = cql_qf1_diff * cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * cql_min_q_weight
                    alpha_prime_loss = 0.0
                    alpha_prime = 0.0
                qf1_loss = qf1_bellman_loss + cql_min_qf1_loss
                qf2_loss = qf2_bellman_loss + cql_min_qf2_loss
            else:
                # when use_cql = False
                qf1_loss = qf1_bellman_loss
                qf2_loss = qf2_bellman_loss
            loss_collection["qf1"] = qf1_loss
            loss_collection["qf2"] = qf2_loss

            return tuple(loss_collection[key] for key in self.model_keys), locals()

    def _train_step(self, batch, use_cql=True, cql_min_q_weight=5.0, enable_calql=False):

        loss, aux_values = self.compute_loss(batch, use_cql, cql_min_q_weight, enable_calql)
        grads = {}
        for key, optimizer in zip(self._model_keys, self.optimizers.values()):
            optimizer.zero_grad()
            loss[key].backward(retain_graph=True)  # TODO

        grads["policy"] = grad_norm(self.policy)
        grads["qf1"] = grad_norm(self.qf["qf1"])
        grads["qf2"] = grad_norm(self.qf["qf2"])

        for key, optimizer in self.optimizers.items():
            optimizer.step()

        # update target networks use soft update
        with torch.no_grad():
            for target_param, param in zip(self.qf["target_qf1"].parameters(), self.qf["qf1"].parameters()):
                target_param.data.mul_(1 - self.config.soft_target_update_rate)
                target_param.data.add_(self.config.soft_target_update_rate * param.data)
            for target_param, param in zip(self.qf["target_qf2"].parameters(), self.qf["qf2"].parameters()):
                target_param.data.mul_(1 - self.config.soft_target_update_rate)
                target_param.data.add_(self.config.soft_target_update_rate * param.data)
        metrics = collect_metrics(
            aux_values,
            [
                "log_pi",
                "policy_loss",
                "qf1_loss",
                "qf2_loss",
                "alpha_loss",
                "alpha",
                "q1_pred",
                "q2_pred",
                "target_q_values",
                "policy_loss_gradient",
                "qf1_loss_gradient",
                "qf2_loss_gradient",
            ],
        )

        metrics.update(
            policy_loss_gradient=grads["policy"],
            qf1_loss_gradient=grads["qf1"],
            qf2_loss_gradient=grads["qf2"],
        )
        metrics.update(use_cql=int(use_cql), enable_calql=int(enable_calql), cql_min_q_weight=cql_min_q_weight)
        if use_cql:
            metrics.update(
                collect_metrics(
                    aux_values,
                    [
                        "cql_std_q1",
                        "cql_std_q2",
                        "cql_q1_rand",
                        "cql_q2_rand",
                        "cql_qf1_diff",
                        "cql_qf2_diff",
                        "cql_min_qf1_loss",
                        "cql_min_qf2_loss",
                        "cql_q1_current_actions",
                        "cql_q2_current_actions" "cql_q1_next_actions",
                        "cql_q2_next_actions",
                        "alpha_prime",
                        "alpha_prime_loss",
                        "qf1_bellman_loss",
                        "qf2_bellman_loss",
                        "bound_rate_cql_q1_current_actions",
                        "bound_rate_cql_q2_current_actions",
                        "bound_rate_cql_q1_next_actions",
                        "bound_rate_ql_q2_next_actions",
                        "log_pi_data",
                    ],
                    "cql",
                )
            )
        return metrics

    @property
    def total_steps(self):
        return self._total_steps
