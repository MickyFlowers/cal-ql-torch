import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.transformed_distribution import \
    TransformedDistribution
from torch.distributions.transforms import TanhTransform


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, arch="256-256", orthogonal_init=False):
        super().__init__()
        hidden_sizes = [int(h) for h in arch.split("-")]
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            linear = nn.Linear(last_dim, h)
            if orthogonal_init:
                nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
                nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.ReLU())
            last_dim = h
        out_linear = nn.Linear(last_dim, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(out_linear.weight, gain=1e-2)
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

    @multiple_action_q_function
    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=-1)
        return torch.squeeze(self.q_net(input_tensor), dim=-1)


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
        self.log_std_multiplier = Scaler(log_std_multiplier)
        self.log_std_offset = Scaler(log_std_offset)
        self.base_network = FullyConnectedNetwork(
            input_dim=observation_dim, output_dim=2 * action_dim, arch=arch, orthogonal_init=orthogonal_init
        )

    def log_prob(self, observations, actions):
        # observations: [batch, obs_dim]
        # actions: [batch, act_dim] or [batch, n, act_dim]
        if actions.dim() == 3 and observations.dim() == 2:
            observations = extend_and_repeat(observations, 1, actions.shape[1])

        base_out = self.base_network(observations)
        mean, log_std = torch.split(base_out, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.exp(log_std)
        dist = TransformedDistribution(Normal(mean, std), TanhTransform(cache_size=1))
        return torch.sum(dist.log_prob(actions), dim=-1)

    def forward(self, observations, deterministic=False, repeat=None):
        # observations: [batch, obs_dim]
        if repeat is not None:
            observations = observations.unsqueeze(1).expand(-1, repeat, -1)
        base_out = self.base_network(observations)
        mean, log_std = torch.chunk(base_out, 2, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.exp(log_std)
        dist = TransformedDistribution(Normal(mean, std), TanhTransform(cache_size=1))
        if deterministic:
            samples = torch.tanh(mean)
        else:
            samples = dist.rsample()
        log_prob = torch.sum(dist.log_prob(samples), dim=-1)
        return samples, log_prob


class SamplerPolicy(object):

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions

class Scaler(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.scaler = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    
    def forward(self):
        return self.scaler