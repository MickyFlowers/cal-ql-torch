export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

nohup python -m cal_ql.train_offline_single_gpu \
    dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='default' \
    logging.prefix='offline' \
    cal_ql.qf_lr=3e-5 \
    cal_ql.policy_lr=1e-5 \
    cal_ql.fixed_alpha=0.1 \
    cal_ql.alpha_prime_lr=1e-6 \
    batch_size=64 \
    num_workers=64 \
    device=cuda:1 \
    train_offline_epochs=1000 \
    discount=0.99 \
    bc_start_epochs=50 \
    cql_min_q_weight=1.0 \
    > logs/train_offline_single_gpu.log 2>&1 &