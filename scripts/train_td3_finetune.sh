#!/bin/bash
# Training script for TD3 fine-tuning from dataset (no environment interaction)
#
# This script is used for continuing training from a TD3+BC checkpoint
# using collected data, without real-time environment interaction.
#
# Use this when:
#   - You have new data and want to fine-tune
#   - You want to switch from TD3+BC to pure TD3
#   - You want to continue training from a checkpoint
#
# For online training with environment interaction, use train_td3_online.sh
#
# Example usage:
#   bash scripts/train_td3_finetune.sh load_ckpt_path=./checkpoints/td3bc_model.pt
#
# Override config parameters:
#   bash scripts/train_td3_finetune.sh dataset.root_path=/path/to/new/data
#
# Common configurations:
#   - load_ckpt_path: Path to pretrained TD3+BC checkpoint
#   - load_optimizer: Whether to load optimizer state (default false)
#   - td3bc.policy_lr: Policy learning rate (default 1e-4)
#   - train_epochs: Number of training epochs (default 100)
#   - dataset_type: 'calql' or 'robomimic' (default 'calql')

set -e

echo "=========================================="
echo "  TD3 Fine-tuning (Data-based)"
echo "=========================================="

python td3/train_td3_finetune.py "$@"
