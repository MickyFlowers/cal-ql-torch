export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000

nohup python -m diffusion_policy.train \
    dataset.root_path='/mnt/pfs/datasets/offline_rl_data_11_17_20/' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    logging.prefix='diffusion_policy' \
    > logs/train_diffusion_policy.log 2>&1 &