#!/bin/bash

export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

# Number of GPUs to use (default: 1 for single GPU, set > 1 for multi-GPU)
NUM_GPUS=${NUM_GPUS:-1}
# Master port for distributed training (change if port is occupied)
MASTER_PORT=${MASTER_PORT:-29504}

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with torchrun
    nohup torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m cal_ql.train_offline \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        logging.prefix='offline' \
        cal_ql.qf_lr=3e-5 \
        cal_ql.policy_lr=3e-5 \
        cal_ql.alpha_prime_lr=1e-5 \
        batch_size=64 \
        num_workers=64 \
        train_offline_epochs=1000 \
        discount=0.99 \
        bc_start_epochs=50 \
        bc_transition_epochs=10 \
        cql_min_q_weight=5.0 \
        torch_compile_mode='disable' \
        > logs/train_offline.log 2>&1 &
else
    # Single GPU training
    nohup python -m cal_ql.train_offline \
        dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
        logging.output_dir='./experiment_output' \
        logging.online=true \
        logging.prefix='offline' \
        cal_ql.qf_lr=3e-5 \
        cal_ql.policy_lr=3e-5 \
        cal_ql.alpha_prime_lr=1e-5 \
        batch_size=64 \
        num_workers=64 \
        device=cuda:0 \
        train_offline_epochs=1000 \
        discount=0.99 \
        bc_start_epochs=50 \
        bc_transition_epochs=10 \
        cql_min_q_weight=5.0 \
        torch_compile_mode='disable' \
        > logs/train_offline.log 2>&1 &
fi

echo "Cal-QL Offline training started with $NUM_GPUS GPU(s). Check logs/train_offline.log for progress."
