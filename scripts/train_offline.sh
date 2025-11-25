nohup python -m cal_ql.train_offline \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    device='cuda:1' \
    torch_compile_mode='default' \
    logging.prefix='offline' \
    cal_ql.qf_lr=1e-6 \
    cal_ql.policy_lr=5e-6 \
    > logs/train_offline.log 2>&1 &