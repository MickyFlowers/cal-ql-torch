import torch

from model.model import ResNetPolicy

policy = ResNetPolicy(observation_dim=6, action_dim=6, obs_proj_arch="256-256", out_proj_arch="256-256", hidden_dim=256, orthogonal_init=True, train_backbone=True)


