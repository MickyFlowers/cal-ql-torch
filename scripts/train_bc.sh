export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

nohup python -m cal_ql.train_bc \
    dataset.root_path='/mnt/pfs/datasets/offline_rl_data_11_17_20/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    torch_compile_mode='default' \
    logging.prefix='bc' \
    learning_rate=1e-4 \
    batch_size=128 \
    num_workers=32 \
    device=cuda:0 \
    train_bc_epochs=100 \
    > logs/train_bc.log 2>&1 &