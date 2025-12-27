"""
Distributed Training Utilities

Provides common functions for multi-GPU training with PyTorch DDP.
Automatically detects whether to use distributed training based on environment.
"""

import os
from typing import Dict, Any, Optional, Tuple

import torch
import torch.distributed as dist


def is_distributed_available() -> bool:
    """Check if distributed training is available and requested."""
    return "LOCAL_RANK" in os.environ


def setup_training(device: str = "cuda:0") -> Tuple[int, torch.device, int]:
    """
    Setup training environment. Automatically detects distributed vs single GPU.

    Args:
        device: Device string for single-GPU mode (default: "cuda:0")

    Returns:
        tuple: (local_rank, device, world_size)
            - For single GPU: (0, device, 1)
            - For multi GPU: (local_rank, cuda:local_rank, world_size)
    """
    if is_distributed_available():
        # Multi-GPU mode: launched via torchrun
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return local_rank, device, world_size
    else:
        # Single-GPU mode
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        return 0, device, 1


def setup_distributed() -> Tuple[int, torch.device, int]:
    """
    Initialize distributed training environment.

    Note: This function assumes distributed mode. For automatic detection,
    use setup_training() instead.

    Returns:
        tuple: (local_rank, device, world_size)
    """
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, device, world_size


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get world size (number of processes)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def sync_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronize metrics across all processes by averaging.

    Args:
        metrics: Dictionary of metric name to value.

    Returns:
        Dictionary with synchronized (averaged) metrics.
    """
    if not dist.is_initialized():
        return metrics

    world_size = dist.get_world_size()
    synced_metrics = {}

    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.detach()
            v_clone = v.clone()
            dist.all_reduce(v_clone, op=dist.ReduceOp.SUM)
            synced_metrics[k] = (v_clone / world_size).item()
        elif isinstance(v, (int, float)):
            v_tensor = torch.tensor(v, device=torch.cuda.current_device())
            dist.all_reduce(v_tensor, op=dist.ReduceOp.SUM)
            synced_metrics[k] = (v_tensor / world_size).item()
        else:
            synced_metrics[k] = v

    return synced_metrics


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
