export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000
nohup python -m cal_ql.train_online_single_gpu \
    dataset.root_path='/mnt/pfs/datasets/online_rl_data_11_26_15/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='default' \
    logging.prefix='offline' \
    cal_ql.qf_lr=1e-6 \
    cal_ql.policy_lr=5e-6 \
    batch_size=32 \
    num_workers=8 \
    load_ckpt_path="checkpoints/offline_20251125_191534/checkpoint_00300.pt" \
    device='cuda:1' \
    > logs/train_online_single_gpu.log 2>&1 &