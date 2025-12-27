#!/bin/bash
# Training script for TD3+BC offline training
#
# TD3+BC is recommended for offline pretraining before online fine-tuning.
# It combines TD3 with behavior cloning regularization to learn from
# static offline datasets.
#
# Example usage:
#   Single GPU:  bash scripts/train_td3bc.sh
#   Multi-GPU:   NUM_GPUS=2 bash scripts/train_td3bc.sh
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

export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

# Number of GPUs to use (default: 1 for single GPU, set > 1 for multi-GPU)
NUM_GPUS=${NUM_GPUS:-1}
# Master port for distributed training (change if port is occupied)
MASTER_PORT=${MASTER_PORT:-29506}

echo "=========================================="
echo "  TD3+BC Offline Training"
echo "  GPUs: $NUM_GPUS"
echo "=========================================="

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m td3.train_td3bc "$@"
else
    # Single GPU training
    python -m td3.train_td3bc "$@"
fi
