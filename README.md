# Cal-QL-Torch

PyTorch implementation of offline and online reinforcement learning algorithms for robot learning, with support for single-GPU and multi-GPU distributed training.

## Supported Algorithms

| Algorithm | Description | Training Script |
|-----------|-------------|-----------------|
| **Cal-QL** | Calibrated Q-Learning for offline RL | `cal_ql/train_offline.py` |
| **IQL** | Implicit Q-Learning for offline RL | `iql/train.py` |
| **TD3+BC** | TD3 with Behavior Cloning for offline RL | `td3/train_td3bc.py` |
| **BC** | Behavior Cloning (imitation learning) | `cal_ql/train_bc.py` |
| **ACT** | Action Chunking with Transformers | `act/train.py` |
| **Diffusion Policy** | Diffusion-based policy for action prediction | `diffusion_policy/train.py` |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cal-ql-torch

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

All training scripts support both single-GPU and multi-GPU training automatically. Use the `NUM_GPUS` environment variable to control the number of GPUs.

```bash
# Single GPU training (default)
bash scripts/train_bc.sh

# Multi-GPU training (2 GPUs)
NUM_GPUS=2 bash scripts/train_bc.sh

# Multi-GPU training (4 GPUs)
NUM_GPUS=4 bash scripts/train_offline.sh
```

### Direct Python Execution

```bash
# Single GPU
python -m cal_ql.train_bc dataset.root_path='/path/to/data'

# Multi-GPU with torchrun
torchrun --nproc_per_node=2 -m cal_ql.train_bc dataset.root_path='/path/to/data'
```

## Training Scripts

### Behavior Cloning (BC)

```bash
# Single GPU
bash scripts/train_bc.sh

# Multi-GPU
NUM_GPUS=2 bash scripts/train_bc.sh
```

Key parameters:
- `learning_rate`: Learning rate (default: 1e-5)
- `batch_size`: Batch size (default: 8)
- `train_bc_epochs`: Number of training epochs (default: 100)

### Cal-QL Offline Training

```bash
# Single GPU
bash scripts/train_offline.sh

# Multi-GPU
NUM_GPUS=2 bash scripts/train_offline.sh
```

Key parameters:
- `cal_ql.qf_lr`: Q-function learning rate (default: 3e-5)
- `cal_ql.policy_lr`: Policy learning rate (default: 3e-5)
- `cql_min_q_weight`: CQL regularization weight (default: 5.0)
- `discount`: Discount factor (default: 0.99)

### IQL Training

```bash
# Single GPU
bash scripts/train_iql.sh

# Multi-GPU
NUM_GPUS=2 bash scripts/train_iql.sh
```

Key parameters:
- `iql.expectile`: Expectile for value function (default: 0.7)
- `iql.beta`: Temperature for AWR policy extraction (default: 3.0)
- `iql.policy_lr`: Policy learning rate (default: 3e-4)
- `iql.qf_lr`: Q-function learning rate (default: 3e-4)
- `iql.vf_lr`: Value function learning rate (default: 3e-4)

### ACT Training

```bash
# Single GPU
bash scripts/train_act.sh

# Multi-GPU
NUM_GPUS=2 bash scripts/train_act.sh
```

Key parameters:
- `act.hidden_dim`: Hidden dimension (default: 512)
- `act.chunk_size`: Action chunk size (default: 30)
- `act.kl_weight`: KL divergence weight (default: 10.0)

### Diffusion Policy Training

```bash
# Single GPU
bash scripts/train_diffusion_policy.sh

# Multi-GPU
NUM_GPUS=2 bash scripts/train_diffusion_policy.sh
```

### TD3+BC Training

```bash
# Single GPU
bash scripts/train_td3bc.sh

# Multi-GPU
NUM_GPUS=2 bash scripts/train_td3bc.sh
```

Key parameters:
- `td3bc.alpha`: BC regularization weight (default: 2.5)
- `td3bc.policy_lr`: Policy learning rate (default: 3e-4)

## Project Structure

```
cal-ql-torch/
├── act/                    # ACT (Action Chunking Transformer)
│   ├── act_model.py        # ACT model architecture
│   ├── act_trainer.py      # ACT trainer
│   └── train.py            # Training script (single & multi-GPU)
├── cal_ql/                 # Cal-QL algorithm
│   ├── cal_ql_sac_trainer.py  # Cal-QL SAC trainer
│   ├── bc_trainer.py       # Behavior cloning trainer
│   ├── train_bc.py         # BC training script
│   ├── train_offline.py    # Offline training script
│   └── train_online_single_gpu.py  # Online fine-tuning
├── diffusion_policy/       # Diffusion Policy
│   ├── trainer.py          # Diffusion policy trainer
│   └── train.py            # Training script (single & multi-GPU)
├── iql/                    # IQL algorithm
│   ├── iql_trainer.py      # IQL trainer
│   └── train.py            # Training script (single & multi-GPU)
├── td3/                    # TD3 and TD3+BC
│   ├── td3_trainer.py      # TD3 trainer
│   ├── td3_bc_trainer.py   # TD3+BC trainer
│   └── train_td3bc.py      # TD3+BC training script
├── model/                  # Neural network models
│   ├── model.py            # ResNet-based policy and Q-functions
│   ├── diffusion_policy.py # Diffusion policy model
│   └── vision_model.py     # Vision encoders
├── data/                   # Dataset utilities
│   └── dataset.py          # Dataset classes with LMDB support
├── utils/                  # Utility functions
│   ├── distributed.py      # Distributed training utilities
│   ├── logger.py           # WandB logging
│   └── utils.py            # General utilities
├── config/                 # Hydra configuration files
│   ├── train_bc.yaml
│   ├── train_offline.yaml
│   ├── train_iql.yaml
│   ├── train_act.yaml
│   ├── train_diffusion_policy.yaml
│   └── train_td3bc.yaml
└── scripts/                # Shell scripts
    ├── train_bc.sh
    ├── train_offline.sh
    ├── train_iql.sh
    ├── train_act.sh
    ├── train_diffusion_policy.sh
    └── train_td3bc.sh
```

## Multi-GPU Training

All training scripts automatically detect the training mode:

- **Single GPU**: Run with `python -m <module>` or set `NUM_GPUS=1`
- **Multi-GPU**: Run with `torchrun` or set `NUM_GPUS>1`

The distributed training uses PyTorch DDP (DistributedDataParallel) with NCCL backend.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NUM_GPUS` | Number of GPUs to use | 1 |
| `MASTER_PORT` | Port for distributed communication | 29500-29506 |

### Port Allocation

Each training script uses a different default port to avoid conflicts:

| Script | Default Port |
|--------|--------------|
| train_act.sh | 29501 |
| train_diffusion_policy.sh | 29502 |
| train_bc.sh | 29503 |
| train_offline.sh | 29504 |
| train_iql.sh | 29505 |
| train_td3bc.sh | 29506 |

## Data Collection

### Start camera node:

```bash
roslaunch realsense2_camera rs_camera.launch enable_depth:=false color_height:=480 color_width:=640 color_fps:=30
```

### Start FT sensor node:

```bash
python3 -m env.ft_sensor_node
```

## Configuration

All configurations are managed with [Hydra](https://hydra.cc/). Configuration files are in the `config/` directory.

Override parameters via command line:

```bash
bash scripts/train_bc.sh learning_rate=1e-4 batch_size=16
```

## Logging

Training logs are saved to:
- **WandB**: Set `logging.online=true` to enable
- **Local**: Saved to `logging.output_dir` (default: `./experiment_output`)

## License

MIT License
