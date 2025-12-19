"""
Test script to verify TD3+BC implementation
"""
import copy
import torch
from td3.td3_model import ResNetDeterministicPolicy, ResNetTD3QFunction
from td3.td3_bc_trainer import TD3BCTrainer


def test_model_initialization():
    """Test that models can be initialized correctly"""
    print("Testing model initialization...")

    observation_dim = 18
    action_dim = 7
    batch_size = 4

    # Create policy
    policy = ResNetDeterministicPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        obs_proj_arch="256-256",
        out_proj_arch="256-256",
        hidden_dim=256,
        orthogonal_init=False,
        train_backbone=False,
    )

    # Create Q-functions
    qf1 = ResNetTD3QFunction(
        observation_dim=observation_dim,
        action_dim=action_dim,
        obs_proj_arch="256-256",
        out_proj_arch="256-256",
        hidden_dim=256,
        orthogonal_init=False,
        train_backbone=False,
    )

    print("✓ Models initialized successfully")

    # Test forward pass
    obs = torch.randn(batch_size, observation_dim)
    images = torch.randn(batch_size, 3, 224, 224)
    actions = torch.randn(batch_size, action_dim)

    # Policy forward
    policy_actions = policy(obs, images, deterministic=True)
    assert policy_actions.shape == (batch_size, action_dim), f"Policy output shape mismatch: {policy_actions.shape}"
    assert torch.all(policy_actions >= -1) and torch.all(policy_actions <= 1), "Policy actions not in [-1, 1]"

    print(f"✓ Policy forward pass successful, output shape: {policy_actions.shape}")

    # Q-function forward
    q_values = qf1(obs, images, actions)
    assert q_values.shape == (batch_size, 1), f"Q-function output shape mismatch: {q_values.shape}"

    print(f"✓ Q-function forward pass successful, output shape: {q_values.shape}")


def test_trainer_initialization():
    """Test that trainer can be initialized correctly"""
    print("\nTesting trainer initialization...")

    observation_dim = 18
    action_dim = 7

    # Create models
    policy = ResNetDeterministicPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        obs_proj_arch="256-256",
        out_proj_arch="256-256",
        hidden_dim=256,
        orthogonal_init=False,
        train_backbone=False,
    )

    qf = {}
    qf['qf1'] = ResNetTD3QFunction(
        observation_dim=observation_dim,
        action_dim=action_dim,
        obs_proj_arch="256-256",
        out_proj_arch="256-256",
        hidden_dim=256,
        orthogonal_init=False,
        train_backbone=False,
    )
    qf['qf2'] = ResNetTD3QFunction(
        observation_dim=observation_dim,
        action_dim=action_dim,
        obs_proj_arch="256-256",
        out_proj_arch="256-256",
        hidden_dim=256,
        orthogonal_init=False,
        train_backbone=False,
    )
    qf['target_qf1'] = copy.deepcopy(qf['qf1'])
    qf['target_qf2'] = copy.deepcopy(qf['qf2'])

    # Create config
    class Config:
        policy_lr = 3e-4
        qf_lr = 3e-4
        tau = 0.005
        policy_noise = 0.2
        noise_clip = 0.5
        policy_freq = 2
        alpha = 2.5
        discount = 0.99

    config = Config()

    # Initialize trainer
    trainer = TD3BCTrainer(config, policy, qf)
    trainer.to_device('cpu')

    print("✓ Trainer initialized successfully")

    # Test training step
    batch_size = 4
    batch = {
        'observations': {
            'proprio': torch.randn(batch_size, observation_dim),
            'image': torch.randn(batch_size, 3, 224, 224),
        },
        'next_observations': {
            'proprio': torch.randn(batch_size, observation_dim),
            'image': torch.randn(batch_size, 3, 224, 224),
        },
        'action': torch.randn(batch_size, action_dim),
        'reward': torch.randn(batch_size),
        'done': torch.zeros(batch_size),
    }

    metrics = trainer.train(batch)
    print(f"✓ Training step successful, metrics keys: {list(metrics.keys())}")
    print(f"  Sample metrics: critic/q1_pred_mean={metrics['critic/q1_pred_mean']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("TD3+BC Implementation Test")
    print("=" * 60)

    test_model_initialization()
    test_trainer_initialization()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
