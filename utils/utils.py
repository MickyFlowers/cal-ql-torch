import time

import numpy as np
import torch


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    indices = np.random.randint(batch["observations"].shape[0], size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def grad_norm(model: torch.nn.Module):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def collect_metrics(metrics, names, prefix=None):
    collected = {}
    for name in names:
        if name in metrics:
            if (not hasattr(metrics[name], "shape")) or metrics[name].shape == ():
                collected[name] = torch.mean(metrics[name])
            else:
                collected[name + "_mean"] = torch.mean(metrics[name])
                collected[name + "_std"] = torch.std(metrics[name])
                collected[name + "_max"] = torch.max(metrics[name])
                collected[name + "_min"] = torch.min(metrics[name])
    if prefix is not None:
        collected = {"{}/{}".format(prefix, key): value for key, value in collected.items()}
    return collected


class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time
