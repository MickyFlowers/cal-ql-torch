#!/bin/bash

export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

# Number of GPUs to use (default: 1 for single GPU, set > 1 for multi-GPU)
NUM_GPUS=${NUM_GPUS:-1}
# Master port for distributed training (change if port is occupied)
MASTER_PORT=${MASTER_PORT:-29501}

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with torchrun
    nohup torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m act.train \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        logging.prefix='act' \
        act.hidden_dim=512 \
        act.latent_dim=32 \
        act.chunk_size=30 \
        act.num_encoder_layers=4 \
        act.num_decoder_layers=4 \
        act.lr=1e-5 \
        act.kl_weight=10.0 \
        batch_size=8 \
        num_workers=8 \
        num_epochs=2000 \
        > logs/train_act.log 2>&1 &
else
    # Single GPU training
    nohup python -m act.train \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        logging.prefix='act' \
        act.hidden_dim=512 \
        act.latent_dim=32 \
        act.chunk_size=30 \
        act.num_encoder_layers=4 \
        act.num_decoder_layers=4 \
        act.lr=1e-5 \
        act.kl_weight=10.0 \
        batch_size=8 \
        num_workers=8 \
        device=cuda:0 \
        num_epochs=2000 \
        > logs/train_act.log 2>&1 &
fi

echo "ACT training started with $NUM_GPUS GPU(s). Check logs/train_act.log for progress."
