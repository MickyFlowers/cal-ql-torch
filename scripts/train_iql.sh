#!/bin/bash

export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

# Number of GPUs to use (default: 1 for single GPU, set > 1 for multi-GPU)
NUM_GPUS=${NUM_GPUS:-1}
# Master port for distributed training (change if port is occupied)
MASTER_PORT=${MASTER_PORT:-29505}

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with torchrun
    nohup torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m iql.train \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        logging.prefix='iql' \
        iql.policy_lr=3e-4 \
        iql.qf_lr=3e-4 \
        iql.vf_lr=3e-4 \
        iql.expectile=0.7 \
        iql.beta=3.0 \
        batch_size=64 \
        num_workers=8 \
        train_iql_epochs=1000 \
        torch_compile_mode='disable' \
        > logs/train_iql.log 2>&1 &
else
    # Single GPU training
    nohup python -m iql.train \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        logging.prefix='iql' \
        iql.policy_lr=3e-4 \
        iql.qf_lr=3e-4 \
        iql.vf_lr=3e-4 \
        iql.expectile=0.7 \
        iql.beta=3.0 \
        batch_size=64 \
        num_workers=8 \
        train_iql_epochs=1000 \
        torch_compile_mode='disable' \
        > logs/train_iql.log 2>&1 &
fi

echo "IQL training started with $NUM_GPUS GPU(s). Check logs/train_iql.log for progress."
