CUDA_VISIBLE_DEVICES=1 nohup python -m act.train     dataset.root_path='/root/autodl-tmp/dataset/processed_insert_2512526/'     logging.output_dir='./experiment_output'     logging.online=true     logging.prefix='act'     act.chunk_size=1     act.lr=5e-6     act.kl_weight=10.0     batch_size=64     num_workers=8     device=cuda:0     num_epochs=200 save_every_n_epoch=50     log_every_n_step=100   eval_every_n_epochs=50  > logs/train_act.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python -m flow_matching.train         dataset.root_path='/root/autodl-tmp/dataset/processed_insert_2512526/'         logging.output_dir='./experiment_output'         logging.online=true         logging.prefix='flow_matching'         batch_size=64         num_workers=8         device=cuda:0         max_epochs=200         save_every_n_epoch=50         log_every_n_step=100         eval_every_n_epochs=50         > logs/train_flow_matching.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python -m cal_ql.train_bc     dataset.root_path='/root/autodl-tmp/dataset/processed_insert_2512526/'     logging.output_dir='./experiment_output'     logging.online=true     torch_compile_mode='disable'     logging.prefix='bc'     learning_rate=1e-4     batch_size=64     num_workers=16     device=cuda:0     train_policy_backbone=true     train_bc_epochs=500  save_every_n_epoch=100  log_every_n_step=100 > logs/train_bc.log 2>&1 &


# Multi-GPU BC Training (using torchrun)
# Adjust --nproc_per_node to match the number of GPUs you want to use
# sample_ratio: controls the fraction of data to use (0.0-1.0), useful for data efficiency studies
# sample_seed: ensures consistent sampling across all GPU processes
CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 -m cal_ql.train_bc \
    dataset.root_path='/mnt/pfs/datasets/rl/processed_insert_2512526/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='disable' \
    logging.project='bc_data' \
    learning_rate=1e-4 \
    batch_size=256 \
    num_workers=16 \
    train_policy_backbone=true \
    train_bc_epochs=500 \
    save_every_n_epoch=100 \
    sample_ratio=0.2 \
    sample_seed=42 \
    > logs/train_bc.log 2>&1 &


# Example: Train with only 50% of data (for data efficiency study)
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m cal_ql.train_bc \
#     dataset.root_path='/path/to/dataset/' \
#     sample_ratio=0.5 \
#     sample_seed=42 \
#     ...

CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 -m iql.train \
    dataset.root_path='/mnt/pfs/datasets/rl/processed_insert_2512526/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    logging.prefix='iql' \
    logging.project='rl' \
    torch_compile_mode='disable' \
    batch_size=256 \
    num_workers=32 \
    train_iql_epochs=500 \
    save_every_n_epoch=100 \
    > logs/train_iql.log 2>&1 &
