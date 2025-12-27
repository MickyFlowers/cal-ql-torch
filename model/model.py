import numpy as np
import timm
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from torch.distributions.transformed_distribution import \
    TransformedDistribution
from torch.distributions.transforms import TanhTransform

from model.distribution import TanhNormal


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)

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

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, orthogonal_init=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.activation = nn.GELU()
        if orthogonal_init:
            nn.init.orthogonal_(self.linear.weight, gain=np.sqrt(2))
            nn.init.constant_(self.linear.bias, 0.0)


    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        out = x
        return out

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, arch="256-256", orthogonal_init=False):
        super().__init__()
        hidden_sizes = [int(h) for h in arch.split("-")]
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            block = MLPBlock(last_dim, h, orthogonal_init)
            layers.append(block)
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
        log_std = torch.clamp(log_std, -5.0, 0.5)  # std range [0.007, 1.65]
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        pre_tanh_actions = torch.log((1 + actions) / (1 - actions)) / 2
        log_prob = dist.log_prob(pre_tanh_actions).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - pre_tanh_actions - nn.functional.softplus(-2 * pre_tanh_actions))).sum(dim=-1)
        return log_prob

    def forward(self, observations, deterministic=False, repeat=None):
        # observations: [batch, obs_dim]
        if repeat is not None:
            observations = observations.unsqueeze(1).expand(-1, repeat, -1)
        base_out = self.base_network(observations)
        mean, log_std = torch.chunk(base_out, 2, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, -5.0, 0.5)  # std range [0.007, 1.65]
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        if deterministic:
            pre_tanh_actions = mean
        else:
            pre_tanh_actions = dist.rsample()
        log_prob = torch.sum(dist.log_prob(pre_tanh_actions), dim=-1)
        log_prob -= (2 * (np.log(2) - pre_tanh_actions - nn.functional.softplus(-2 * pre_tanh_actions))).sum(dim=-1)
        return torch.tanh(pre_tanh_actions), log_prob


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

    def clamp_(self, min=None, max=None):
        """In-place clamp the scaler value."""
        with torch.no_grad():
            self.scaler.clamp_(min=min, max=max)

class ResNetPolicy(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        obs_proj_arch="256-256",
        out_proj_arch="256-256",
        hidden_dim=256,
        orthogonal_init=False,
        log_std_multiplier=1.0,
        log_std_offset=-1.0,
        resnet_model='resnet18',
        image_size=(224, 224),
        train_backbone=False,
        out_indices=3,  
        
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.log_std_multiplier = Scaler(log_std_multiplier)
        self.log_std_offset = Scaler(log_std_offset)
        self.backbone = timm.create_model(resnet_model, pretrained=True, features_only=True, out_indices=(out_indices,))
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        self.train_backbone = train_backbone
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        # Get the number of output channels from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *image_size)
            dummy_output = self.backbone(dummy_input)
            backbone_out_channels = dummy_output[0].shape[1]

        # Feature projection: Global Average Pooling + Linear Layer
        self.image_feature_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out_channels, hidden_dim)
        )
        self.obs_proj = FullyConnectedNetwork(
            input_dim=observation_dim, output_dim=hidden_dim, arch=obs_proj_arch, orthogonal_init=orthogonal_init
        )
        self.out_proj = FullyConnectedNetwork(
            input_dim=2*hidden_dim, output_dim=2 * action_dim, arch=out_proj_arch, orthogonal_init=orthogonal_init
        )

    def _encode_image(self, images):
        # Runs the heavy backbone once and returns pooled image features
        image_ft_map = self.backbone(images)[0]
        return self.image_feature_proj(image_ft_map)

    def _encode_obs(self, observations):
        return self.obs_proj(observations)
        
    def forward(self, observations, images, deterministic=False):
        # observations: [batch, obs_dim]
        # images: [batch, 3, H, W]
        image_ft = self._encode_image(images)
        obs_ft = self._encode_obs(observations)
        ft = torch.cat([obs_ft, image_ft], dim=-1)
        
        base_out = self.out_proj(ft)
        mean, log_std = torch.chunk(base_out, 2, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, -5.0, 0.5)  # std range [0.007, 1.65]
        std = torch.exp(log_std)
        log_prob = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
            log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def sample_actions(self, observations, images, num_actions, deterministic=False):
        # Samples multiple actions per (obs, image) pair with a single backbone pass
        image_ft = self._encode_image(images)
        obs_ft = self._encode_obs(observations)
        base_ft = torch.cat([obs_ft, image_ft], dim=-1)
        base_ft = base_ft.unsqueeze(1).expand(-1, num_actions, -1).reshape(-1, base_ft.shape[-1])

        base_out = self.out_proj(base_ft)
        mean, log_std = torch.chunk(base_out, 2, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, -5.0, 0.5)  # std range [0.007, 1.65]
        std = torch.exp(log_std)
        tanh_normal = TanhNormal(mean, std)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
            log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
            log_prob = log_prob.sum(dim=-1)

        action = action.view(observations.shape[0], num_actions, -1)
        if log_prob is not None:
            log_prob = log_prob.view(observations.shape[0], num_actions)
        return action, log_prob

    def log_prob(self, observations, images, actions):
        raw_actions = atanh(actions)
        image_ft_map = self.backbone(images)[0]
        image_ft = self.image_feature_proj(image_ft_map)
        
        obs_ft = self.obs_proj(observations)
        ft = torch.cat([obs_ft, image_ft], dim=-1)
        
        base_out = self.out_proj(ft)
        mean, log_std = torch.chunk(base_out, 2, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, -5.0, 0.5)  # std range [0.007, 1.65]
        std = torch.exp(log_std)
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        log_prob = torch.clamp(log_prob, min=-100.0, max=10.0)  # clip for numerical stability
        return log_prob.sum(-1)



    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        if not self.train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

class ResNetQFunction(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        obs_proj_arch="256-256",
        out_proj_arch="256-256",
        hidden_dim=256,
        orthogonal_init=False,
        resnet_model='resnet18',
        image_size=(224, 224),
        train_backbone=False,
        out_indices=3,  
        
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.backbone = timm.create_model(resnet_model, pretrained=True, features_only=True, out_indices=(out_indices,))
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        # Get the number of output channels from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *image_size)
            dummy_output = self.backbone(dummy_input)
            backbone_out_channels = dummy_output[0].shape[1]

        # Feature projection: Global Average Pooling + Linear Layer
        self.image_feature_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out_channels, hidden_dim)
        )
        self.obs_proj = FullyConnectedNetwork(
            input_dim=observation_dim + action_dim, output_dim=hidden_dim, arch=obs_proj_arch, orthogonal_init=orthogonal_init
        )
        self.out_proj = FullyConnectedNetwork(
            input_dim=2*hidden_dim, output_dim=1, arch=out_proj_arch, orthogonal_init=orthogonal_init
        )

    def _encode_image(self, images):
        image_ft_map = self.backbone(images)[0]
        return self.image_feature_proj(image_ft_map)
        
    def forward(self, observations, images, actions):
        
        # observations: [batch, obs_dim]
        # images: [batch, 3, H, W]
        
        image_ft_map = self.backbone(images)[0]
        image_ft = self.image_feature_proj(image_ft_map)
        obs_ft = self.obs_proj(torch.cat([observations, actions], dim=-1))
        ft = torch.cat([obs_ft, image_ft], dim=-1)
        
        q_value = self.out_proj(ft)
        return q_value

    def evaluate_actions(self, observations, images, actions):
        # Efficiently evaluates Q for many actions sharing the same obs/image
        action_shape = actions.shape[0]
        obs_shape = observations.shape[0]
        num_repeat = int(action_shape / obs_shape)

        image_ft = self._encode_image(images)
        image_ft = image_ft.repeat_interleave(num_repeat, dim=0)

        obs_rep = observations.repeat_interleave(num_repeat, dim=0)
        obs_actions = torch.cat([obs_rep, actions], dim=-1)
        obs_ft = self.obs_proj(obs_actions)
        ft = torch.cat([obs_ft, image_ft], dim=-1)
        q_value = self.out_proj(ft)
        return q_value.view(obs_shape, num_repeat, 1)


class ResNetVFunction(nn.Module):
    """
    Value Function V(s) using ResNet backbone for image encoding.
    Used in IQL (Implicit Q-Learning) algorithm.
    """

    def __init__(
        self,
        observation_dim,
        obs_proj_arch="256-256",
        out_proj_arch="256-256",
        hidden_dim=256,
        orthogonal_init=False,
        resnet_model='resnet18',
        image_size=(224, 224),
        train_backbone=False,
        out_indices=3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.backbone = timm.create_model(resnet_model, pretrained=True, features_only=True, out_indices=(out_indices,))
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        # Get the number of output channels from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *image_size)
            dummy_output = self.backbone(dummy_input)
            backbone_out_channels = dummy_output[0].shape[1]

        # Feature projection: Global Average Pooling + Linear Layer
        self.image_feature_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out_channels, hidden_dim)
        )
        # Note: V(s) only takes state, no action
        self.obs_proj = FullyConnectedNetwork(
            input_dim=observation_dim, output_dim=hidden_dim, arch=obs_proj_arch, orthogonal_init=orthogonal_init
        )
        self.out_proj = FullyConnectedNetwork(
            input_dim=2 * hidden_dim, output_dim=1, arch=out_proj_arch, orthogonal_init=orthogonal_init
        )

    def forward(self, observations, images):
        """
        Compute V(s) given state observations and images.

        Args:
            observations: [batch, obs_dim] proprioceptive observations
            images: [batch, 3, H, W] images

        Returns:
            V(s): [batch, 1] state values
        """
        image_ft_map = self.backbone(images)[0]
        image_ft = self.image_feature_proj(image_ft_map)
        obs_ft = self.obs_proj(observations)
        ft = torch.cat([obs_ft, image_ft], dim=-1)

        v_value = self.out_proj(ft)
        return v_value

