export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

nohup python -m cal_ql.train_offline_from_bc \
    dataset.root_path='/mnt/pfs/datasets/offline_rl_data_11_17_20/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='default' \
    logging.prefix='offline' \
    cal_ql.qf_lr=3e-6 \
    cal_ql.policy_lr=1e-6 \
    cal_ql.freeze_qf_lr=3e-5 \
    batch_size=128 \
    num_workers=32 \
    device=cuda:1 \
    train_offline_epochs=200 \
    freeze_policy_epochs=50 \
    load_policy_ckpt_path=/root/workspace/cal-ql-torch/checkpoints/bc_20251127_162637/bc_checkpoint_00060.pt \
    > logs/train_offline_from_bc.log 2>&1 &