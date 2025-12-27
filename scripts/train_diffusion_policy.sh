#!/bin/bash

export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

# Number of GPUs to use (default: 1 for single GPU, set > 1 for multi-GPU)
NUM_GPUS=${NUM_GPUS:-1}
# Master port for distributed training (change if port is occupied)
MASTER_PORT=${MASTER_PORT:-29502}

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with torchrun
    nohup torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m diffusion_policy.train \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        logging.prefix='diffusion_policy' \
        batch_size=16 \
        num_workers=16 \
        max_epochs=1000 \
        save_every_n_epoch=200 \
        log_every_n_step=100 \
        eval_every_n_epochs=100 \
        > logs/train_diffusion_policy.log 2>&1 &
else
    # Single GPU training
    nohup python -m diffusion_policy.train \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        logging.prefix='diffusion_policy' \
        batch_size=16 \
        num_workers=16 \
        device=cuda:0 \
        max_epochs=1000 \
        save_every_n_epoch=200 \
        log_every_n_step=100 \
        eval_every_n_epochs=100 \
        > logs/train_diffusion_policy.log 2>&1 &
fi

echo "Diffusion Policy training started with $NUM_GPUS GPU(s). Check logs/train_diffusion_policy.log for progress."
