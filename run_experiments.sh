#!/usr/bin/env bash
set -e

MODEL=${1:-resnet50}
EPOCHS=${2:-5}

python train_single.py \
  --run_name single_gpu \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch_size 64 \
  --lr 1e-4

torchrun --nproc_per_node=2 train_ddp.py \
  --run_name ddp_2gpu_fixed_local \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch_size 64 \
  --lr 2e-4

torchrun --nproc_per_node=2 train_ddp.py \
  --run_name ddp_2gpu_fixed_global \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch_size 32 \
  --lr 1e-4

python benchmark.py --results_dir ./results --baseline_csv single_gpu.csv
