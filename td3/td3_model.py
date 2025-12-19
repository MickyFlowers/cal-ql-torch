import numpy as np
import timm
import torch
import torch.nn as nn

from model.model import FullyConnectedNetwork


class ResNetDeterministicPolicy(nn.Module):
    """Deterministic policy for TD3 with ResNet vision encoder using fine-grained features"""
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

        # Vision encoder backbone - use pretrained ResNet with fine-grained features
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
            spatial_size = dummy_output[0].shape[2] * dummy_output[0].shape[3]

        # Image feature projection - flatten spatial features and project
        self.image_feature_proj = nn.Sequential(
            nn.Flatten(start_dim=1),  # Flatten spatial dimensions but keep batch
            nn.Linear(backbone_out_channels * spatial_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.obs_proj = FullyConnectedNetwork(
            input_dim=observation_dim, output_dim=hidden_dim, arch=obs_proj_arch, orthogonal_init=orthogonal_init
        )

        self.out_proj = FullyConnectedNetwork(
            input_dim=2*hidden_dim, output_dim=action_dim, arch=out_proj_arch, orthogonal_init=orthogonal_init
        )

    def _encode_image(self, images):
        # Extract fine-grained features from backbone
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

        action = self.out_proj(ft)
        # TD3 uses tanh to bound actions to [-1, 1]
        action = torch.tanh(action)
        return action

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        if not self.train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


class ResNetTD3QFunction(nn.Module):
    """Q-function for TD3 with ResNet vision encoder using fine-grained features"""
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
            spatial_size = dummy_output[0].shape[2] * dummy_output[0].shape[3]

        # Image feature projection - flatten spatial features and project
        self.image_feature_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(backbone_out_channels * spatial_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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
        # actions: [batch, action_dim]

        image_ft = self._encode_image(images)
        obs_ft = self.obs_proj(torch.cat([observations, actions], dim=-1))
        ft = torch.cat([obs_ft, image_ft], dim=-1)

        q_value = self.out_proj(ft)
        return q_value
