export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

nohup python -m cal_ql.train_bc \
    dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='disable' \
    logging.prefix='bc' \
    learning_rate=1e-5 \
    batch_size=8 \
    num_workers=8 \
    device=cuda:0 \
    train_policy_backbone=true \
    train_bc_epochs=100 \
    > logs/train_bc.log 2>&1 &