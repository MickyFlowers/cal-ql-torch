export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

nohup python -m diffusion_policy.train \
    dataset.root_path='/mnt/pfs/datasets/rl/processed_data_251218/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    logging.prefix='diffusion_policy' \
    batch_size=16 \
    num_workers=16 \
    device=cuda:0 \
    max_epochs=1000 \
    save_every_n_epoch=200 \
    log_every_n_step=100 \
    eval_every_n_epochs=100 \
    batch_size=16 \
    num_workers=16 \
    > logs/train_diffusion_policy.log 2>&1 &