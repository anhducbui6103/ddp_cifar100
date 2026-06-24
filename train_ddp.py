import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import build_dataloaders
from models import build_model
from utils import (
    EpochLog,
    accuracy_top1,
    append_csv,
    format_seconds,
    get_peak_memory_mb,
    is_main_process,
    reduce_sum_tensor,
    set_seed,
    sync_cuda,
)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device


def cleanup_ddp():
    dist.destroy_process_group()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_top1(logits.detach(), targets)
        total_samples += batch_size

    stats = torch.tensor(
        [total_loss, float(total_correct), float(total_samples)],
        dtype=torch.float64,
        device=device,
    )
    reduce_sum_tensor(stats)
    loss = stats[0].item() / stats[2].item()
    acc = stats[1].item() / stats[2].item()
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_top1(logits, targets)
        total_samples += batch_size

    stats = torch.tensor(
        [total_loss, float(total_correct), float(total_samples)],
        dtype=torch.float64,
        device=device,
    )
    reduce_sum_tensor(stats)
    loss = stats[0].item() / stats[2].item()
    acc = stats[1].item() / stats[2].item()
    return loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="ddp")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64, help="Local batch size per GPU")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_ddp()
    set_seed(args.seed + rank)
    use_amp = not args.no_amp

    train_loader, val_loader, train_sampler, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    model = build_model(args.model, num_classes=100, pretrained=not args.no_pretrained)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    global_batch_size = args.batch_size * world_size
    csv_path = os.path.join(args.results_dir, f"{args.run_name}.csv")

    if is_main_process():
        print(f"World size: {world_size}")
        print(f"Model: {args.model}")
        print(f"Local/global batch size: {args.batch_size}/{global_batch_size}")
        print(f"AMP: {use_amp}")

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        sync_cuda()
        start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        sync_cuda()
        epoch_time = time.perf_counter() - start
        throughput = 50000 / epoch_time
        peak_mem = get_peak_memory_mb(device)

        if is_main_process():
            row = EpochLog(
                run_name=args.run_name,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch_time_sec=epoch_time,
                throughput_img_per_sec=throughput,
                lr=optimizer.param_groups[0]["lr"],
                local_batch_size=args.batch_size,
                global_batch_size=global_batch_size,
                world_size=world_size,
                peak_memory_mb=peak_mem,
            )
            append_csv(csv_path, row)

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"time={format_seconds(epoch_time)} | "
                f"throughput={throughput:.1f} img/s | "
                f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
                f"mem_rank0={peak_mem:.0f} MB"
            )

    if is_main_process():
        print(f"Saved log to: {csv_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
