nohup python -m cal_ql.train_offline_single_gpu \
    dataset.root_path='/mnt/pfs/datasets/offline_rl_data_11_17_20/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='default' \
    logging.prefix='offline' \
    cal_ql.qf_lr=1e-6 \
    cal_ql.policy_lr=5e-6 \
    batch_size=128 \
    num_workers=8 \
    > logs/train_offline_single_gpu.log 2>&1 &