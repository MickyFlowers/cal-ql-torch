export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

nohup python -m cal_ql.train_offline_single_gpu \
    dataset.root_path='/mnt/pfs/datasets/offline_rl_data_11_17_20/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='default' \
    logging.prefix='offline' \
    cal_ql.qf_lr=3e-4 \
    cal_ql.policy_lr=1e-4 \
    batch_size=128 \
    num_workers=32 \
    device=cuda:1 \
    train_offline_epochs=500 \
    > logs/train_offline_single_gpu_high_lr.log 2>&1 &