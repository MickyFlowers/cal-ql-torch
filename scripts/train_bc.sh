#!/bin/bash

export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

# Number of GPUs to use (default: 1 for single GPU, set > 1 for multi-GPU)
NUM_GPUS=${NUM_GPUS:-1}
# Master port for distributed training (change if port is occupied)
MASTER_PORT=${MASTER_PORT:-29503}

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with torchrun
    nohup torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m cal_ql.train_bc \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        torch_compile_mode='disable' \
        logging.prefix='bc' \
        learning_rate=1e-5 \
        batch_size=8 \
        num_workers=8 \
        train_policy_backbone=true \
        train_bc_epochs=100 \
        > logs/train_bc.log 2>&1 &
else
    # Single GPU training
    nohup python -m cal_ql.train_bc \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        torch_compile_mode='disable' \
        logging.prefix='bc' \
        learning_rate=1e-5 \
        batch_size=8 \
        num_workers=8 \
        device=cuda:0 \
        train_policy_backbone=true \
        train_bc_epochs=100 \
        > logs/train_bc.log 2>&1 &
fi

echo "BC training started with $NUM_GPUS GPU(s). Check logs/train_bc.log for progress."
