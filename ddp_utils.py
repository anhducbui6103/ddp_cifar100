"""
PyTorch DDP utilities for scaling experiments.
Designed for Kaggle (2x T4) with mp.spawn — no torchrun required.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import timm


@dataclass
class TrainConfig:
    """Training hyperparameters for one experiment run."""

    exp_name: str = "baseline"
    world_size: int = 1
    local_batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 5
    num_workers: int = 2
    model_name: str = "vit_large_patch16_224"
    num_classes: int = 100
    image_size: int = 224
    save_dir: str = "/kaggle/working/results"
    seed: int = 42
    use_amp: bool = True
    warmup_epochs: int = 1

    @property
    def global_batch_size(self) -> int:
        return self.local_batch_size * self.world_size


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp(rank: int, world_size: int, port: str = "12355") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=rank
    )


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def build_transforms(image_size: int, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )


def build_dataloaders(
    config: TrainConfig,
    rank: int,
    world_size: int,
    data_root: str = "/kaggle/working/data",
):
    train_ds = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=build_transforms(config.image_size, train=True),
    )
    val_ds = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=build_transforms(config.image_size, train=False),
    )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.local_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.local_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def build_model(config: TrainConfig, device: torch.device) -> nn.Module:
    model = timm.create_model(
        config.model_name,
        pretrained=False,
        num_classes=config.num_classes,
    )
    return model.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast("cuda", enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / max(total, 1)


def _train_worker(rank: int, world_size: int, config: TrainConfig, shared: dict) -> None:
    set_seed(config.seed + rank)
    is_ddp = world_size > 1

    if is_ddp:
        setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    train_loader, val_loader, train_sampler = build_dataloaders(
        config, rank, world_size
    )
    model = build_model(config, device)

    if is_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.05)
    scaler = GradScaler("cuda", enabled=config.use_amp)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [],
    }

    # Synchronize before timing
    if is_ddp:
        dist.barrier()
    start = time.perf_counter()

    for epoch in range(config.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, config.use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        epoch_time = time.perf_counter() - epoch_start

        if rank == 0:
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epoch_time_sec"].append(epoch_time)
            print(
                f"[{config.exp_name}] epoch {epoch + 1}/{config.epochs} "
                f"loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"time={epoch_time:.1f}s"
            )

        if is_ddp:
            dist.barrier()

    total_time = time.perf_counter() - start

    if rank == 0:
        save_path = Path(config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        state_dict = model.module.state_dict() if is_ddp else model.state_dict()
        ckpt_path = save_path / f"{config.exp_name}_model.pt"
        torch.save(
            {
                "model_state_dict": state_dict,
                "config": asdict(config),
                "history": history,
            },
            ckpt_path,
        )

        result = {
            "exp_name": config.exp_name,
            "world_size": world_size,
            "local_batch_size": config.local_batch_size,
            "global_batch_size": config.global_batch_size,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "total_time_sec": total_time,
            "avg_epoch_time_sec": sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"]),
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
            "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0.0,
            "history": history,
            "checkpoint": str(ckpt_path),
        }
        shared["result"] = result

        metrics_path = save_path / f"{config.exp_name}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in result.items() if k != "history"}, f, indent=2)

        history_path = save_path / f"{config.exp_name}_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    if is_ddp:
        cleanup_ddp()


def run_training(config: TrainConfig) -> dict[str, Any]:
    """Launch single-GPU or DDP training via mp.spawn (Kaggle-safe)."""
    world_size = config.world_size
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {world_size} GPUs but only {torch.cuda.device_count()} available."
        )

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    shared: dict = manager.dict()

    if world_size == 1:
        _train_worker(0, 1, config, shared)
    else:
        mp.spawn(
            _train_worker,
            args=(world_size, config, shared),
            nprocs=world_size,
            join=True,
        )

    return dict(shared.get("result", {}))


def estimate_amdahl_parallel_fraction(speedup: float, num_gpus: int) -> float:
    """
    Given measured speedup S with N GPUs, solve for parallel fraction P:
        S = 1 / ((1 - P) + P/N)  =>  P = (S - 1) / (S - S/N)
    """
    if num_gpus <= 1 or speedup <= 1.0:
        return 0.0
    denom = speedup - (speedup / num_gpus)
    if abs(denom) < 1e-9:
        return 1.0
    p = (speedup - 1.0) / denom
    return max(0.0, min(1.0, p))


def theoretical_amdahl_speedup(parallel_fraction: float, num_gpus: int) -> float:
    serial = 1.0 - parallel_fraction
    return 1.0 / (serial + parallel_fraction / num_gpus)


def load_shared_result(exp_name: str, search_dirs: list[str | Path]) -> dict[str, Any] | None:
    """Load metrics + history exported by another notebook (via Kaggle Dataset / Drive)."""
    for base in search_dirs:
        base = Path(base)
        metrics_path = base / f"{exp_name}_metrics.json"
        history_path = base / f"{exp_name}_history.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, encoding="utf-8") as f:
            result = json.load(f)
        if history_path.exists():
            with open(history_path, encoding="utf-8") as f:
                result["history"] = json.load(f)
        return result
    return None


def backup_result(result: dict[str, Any], exp_name: str, results_dir: Path, drive_path: Path) -> None:
    """Copy metrics, history, and checkpoint to drive_backup for team sharing."""
    import shutil

    for suffix in ("_metrics.json", "_history.json", "_model.pt"):
        src = results_dir / f"{exp_name}{suffix}"
        if src.exists():
            shutil.copy(src, drive_path / src.name)


def get_pipeline_configs(
    world_size: int,
    *,
    base_lr: float = 1e-4,
    local_batch: int = 16,
    epochs: int = 5,
    save_dir: str = "/kaggle/working/results",
) -> list[TrainConfig]:
    """
    Cùng một pipeline 3 bước cho cả 1 GPU và 2 GPU DDP.

    Ý nghĩa tương đương giữa hai notebook:
      - baseline      : cấu hình chuẩn của môi trường (global batch = local × world_size)
      - lr_scaled     : global batch gấp đôi baseline-1GPU + LR tuyến tính (×2)
      - no_lr_scale   : cùng global batch với lr_scaled nhưng giữ LR baseline
    """
    tag = f"{world_size}gpu"
    common = dict(epochs=epochs, save_dir=save_dir)

    if world_size == 1:
        return [
            TrainConfig(
                exp_name=f"pipeline_baseline_{tag}",
                world_size=1,
                local_batch_size=local_batch,
                learning_rate=base_lr,
                **common,
            ),
            TrainConfig(
                exp_name=f"pipeline_lr_scaled_{tag}",
                world_size=1,
                local_batch_size=local_batch * 2,
                learning_rate=base_lr * 2,
                **common,
            ),
            TrainConfig(
                exp_name=f"pipeline_no_lr_scale_{tag}",
                world_size=1,
                local_batch_size=local_batch * 2,
                learning_rate=base_lr,
                **common,
            ),
        ]

    return [
        TrainConfig(
            exp_name=f"pipeline_baseline_{tag}",
            world_size=2,
            local_batch_size=local_batch,
            learning_rate=base_lr,
            **common,
        ),
        TrainConfig(
            exp_name=f"pipeline_lr_scaled_{tag}",
            world_size=2,
            local_batch_size=local_batch,
            learning_rate=base_lr * 2,
            **common,
        ),
        TrainConfig(
            exp_name=f"pipeline_no_lr_scale_{tag}",
            world_size=2,
            local_batch_size=local_batch,
            learning_rate=base_lr,
            **common,
        ),
    ]
