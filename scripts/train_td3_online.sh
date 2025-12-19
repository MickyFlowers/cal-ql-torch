#!/bin/bash
# Training script for TD3 online training with environment interaction
#
# This script is used for online fine-tuning after TD3+BC offline pretraining.
# It requires a real robot environment and collects data during training.
#
# Example usage:
#   bash scripts/train_td3_online.sh load_ckpt_path=./checkpoints/td3bc_model.pt
#
# Override config parameters:
#   bash scripts/train_td3_online.sh td3bc.expl_noise=0.2 offline_ratio=0.3
#
# Common configurations:
#   - load_ckpt_path: Path to pretrained TD3+BC checkpoint (required)
#   - td3bc.expl_noise: Exploration noise (default 0.1)
#   - td3bc.policy_lr: Policy learning rate (default 1e-4)
#   - offline_ratio: Ratio of offline data in batches (default 0.5)
#   - n_online_epochs: Number of online epochs (default 100)

set -e

echo "=========================================="
echo "  TD3 Online Training"
echo "=========================================="

if [ -z "$1" ]; then
    echo "Warning: No checkpoint specified. Starting from scratch."
    echo "For fine-tuning, use: bash scripts/train_td3_online.sh load_ckpt_path=<path>"
fi

python td3/train_td3_online.py "$@"
