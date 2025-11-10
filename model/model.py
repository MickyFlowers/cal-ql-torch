import torch
import torch.nn as nn
from torch.distributions import Normal

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, arch="256-256", orthogonal_init=False):
        super().__init__()
        hidden_sizes = [int(h) for h in arch.split("-")]
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            linear = nn.Linear(last_dim, h)
            if orthogonal_init:
                nn.init.orthogonal_(linear.weight, gain=2**0.5)
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            last_dim = h
        out_linear = nn.Linear(last_dim, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(out_linear.weight, gain=1e-2)
            nn.init.zeros_(out_linear.bias)
        else:
            nn.init.uniform_(out_linear.weight, a=-1e-2, b=1e-2)
            nn.init.zeros_(out_linear.bias)
        layers.append(out_linear)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FullyConnectedQFunction(nn.Module):
    def __init__(self, observation_dim, action_dim, arch="256-256", orthogonal_init=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.q_net = FullyConnectedNetwork(
            input_dim=observation_dim + action_dim, output_dim=1, arch=arch, orthogonal_init=orthogonal_init
        )

    def forward(self, observations, actions):
        # observations: [batch, obs_dim] or [batch, n, obs_dim]
        # actions: [batch, act_dim] or [batch, n, act_dim]
        if actions.dim() == 3 and observations.dim() == 2:
            batch_size, n, act_dim = actions.shape
            obs_expanded = observations.unsqueeze(1).expand(-1, n, -1)
            obs_flat = obs_expanded.reshape(-1, self.observation_dim)
            actions_flat = actions.reshape(-1, self.action_dim)
            x = torch.cat([obs_flat, actions_flat], dim=-1)
            q_values = self.q_net(x).squeeze(-1)
            return q_values.view(batch_size, n)
        else:
            x = torch.cat([observations, actions], dim=-1)
            q_values = self.q_net(x).squeeze(-1)
            return q_values


class TanhGaussian:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normal = Normal(mean, self.std)

    def sample(self):
        z = self.normal.rsample()
        return torch.tanh(z)

    def log_prob(self, actions):
        # Inverse tanh for actions
        eps = 1e-6
        pre_tanh = torch.atanh(torch.clamp(actions, -1 + eps, 1 - eps))
        log_prob = self.normal.log_prob(pre_tanh)
        # Correction for tanh transformation
        log_prob = log_prob.sum(-1) - torch.log(1 - actions.pow(2) + eps).sum(-1)
        return log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        arch="256-256",
        orthogonal_init=False,
        log_std_multiplier=1.0,
        log_std_offset=-1.0,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.log_std_multiplier = nn.Parameter(torch.tensor(log_std_multiplier), requires_grad=True)
        self.log_std_offset = nn.Parameter(torch.tensor(log_std_offset), requires_grad=True)
        self.base_network = FullyConnectedNetwork(
            input_dim=observation_dim, output_dim=2 * action_dim, arch=arch, orthogonal_init=orthogonal_init
        )

    def log_prob(self, observations, actions):
        # observations: [batch, obs_dim]
        # actions: [batch, act_dim] or [batch, n, act_dim]
        if actions.dim() == 3 and observations.dim() == 2:
            observations = observations.unsqueeze(1).expand(-1, actions.shape[1], -1)

        base_out = self.base_network(observations)
        mean, log_std = torch.chunk(base_out, 2, dim=-1)
        log_std = self.log_std_multiplier * log_std + self.log_std_offset
        log_std = torch.clamp(log_std, -20.0, 2.0)
        dist = TanhGaussian(mean, torch.exp(log_std))
        return dist.log_prob(actions)

    def forward(self, observations, deterministic=False, repeat=None):
        # observations: [batch, obs_dim]
        if repeat is not None:
            observations = observations.unsqueeze(1).expand(-1, repeat, -1)
        base_out = self.base_network(observations)
        mean, log_std = torch.chunk(base_out, 2, dim=-1)
        log_std = self.log_std_multiplier * log_std + self.log_std_offset
        log_std = torch.clamp(log_std, -20.0, 2.0)
        dist = TanhGaussian(mean, torch.exp(log_std))
        if deterministic:
            samples = torch.tanh(mean)
            log_prob = self.log_prob(observations, samples)
        else:
            samples = dist.sample()
            log_prob = self.log_prob(observations, samples)
        return samples, log_prob
