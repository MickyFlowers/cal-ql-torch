export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

python -m cal_ql.train \
    env='antmaze-medium-diverse-v2' \
    logging.output_dir='./experiment_output' \
    logging.online=true \
    device='cuda:0' \