python -m rollout.rollout_online \
    port=603 \
    host_name="120.48.58.215" \
    user_name="root" \
    key_filepath="/home/cyx/.ssh/id_rsa" \
    remote_data_dir="/mnt/pfs/datasets/online_rl_data_11_26_15/" \
    remote_online_ckpt_dir="/root/workspace/cal-ql-torch/checkpoints/online_20251126_171514" \
    remote_offline_ckpt="/root/workspace/cal-ql-torch/checkpoints/offline_20251125_191534/checkpoint_00300.pt" \
    local_ckpt_dir="./checkpoints/online_rollout_11_26_15"

