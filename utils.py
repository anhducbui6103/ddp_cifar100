import csv
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class EpochLog:
    run_name: str
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time_sec: float
    throughput_img_per_sec: float
    lr: float
    local_batch_size: int
    global_batch_size: int
    world_size: int
    peak_memory_mb: float


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def reduce_sum_tensor(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = logits.argmax(dim=1)
    return (preds == targets).sum().item()


def append_csv(path: str, row: EpochLog):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"
