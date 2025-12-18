export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

nohup python -m cal_ql.train_robomimic_offline \
    dataset=robomimic_dataset \
    dataset.root_path='/mnt/pfs/datasets/robotmimic/square/ph/image_v15.hdf5' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='default' \
    logging.prefix='robomimic_offline' \
    cal_ql.qf_lr=3e-5 \
    cal_ql.policy_lr=1e-5 \
    batch_size=64 \
    num_workers=64 \
    device=cuda:1 \
    train_offline_epochs=400 \
    observation_dim=14 \
    action_dim=7 \
    discount=0.95 \
    bc_start_epochs=10 \
    > logs/train_robomimic_offline.log 2>&1 &