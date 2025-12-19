#!/bin/bash
# Training script for TD3+BC offline training
#
# TD3+BC is recommended for offline pretraining before online fine-tuning.
# It combines TD3 with behavior cloning regularization to learn from
# static offline datasets.
#
# Example usage:
#   bash scripts/train_td3bc.sh
#
# Override config parameters:
#   bash scripts/train_td3bc.sh td3bc.alpha=3.0 batch_size=128
#
# Common configurations:
#   - td3bc.alpha: BC weight (default 2.5, higher = more conservative)
#   - td3bc.policy_lr: Policy learning rate (default 3e-4)
#   - train_epochs: Number of training epochs (default 100)
#   - batch_size: Batch size (default 256)

set -e

echo "=========================================="
echo "  TD3+BC Offline Training"
echo "=========================================="

python td3/train_td3bc.py "$@"
