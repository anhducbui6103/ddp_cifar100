# PyTorch DDP CIFAR-100 Benchmark

Project cho Đề tài 3: phân tích hiệu năng và khả năng mở rộng của PyTorch Distributed Data Parallel trên CIFAR-100.

## 1. Cài thư viện

```bash
pip install -r requirements.txt
```

## 2. Cấu hình khuyên dùng

Nếu muốn chạy nhanh và ít lỗi:

```text
Model: resnet50
Epochs: 5
Image size: 224
Batch size: 64 nếu đủ VRAM, nếu OOM thì giảm xuống 32
```

Nếu muốn báo cáo đẹp hơn:

```text
Model: convnext_tiny
Epochs: 5 hoặc 10
Image size: 224
Batch size: 32 hoặc 64
```

## 3. Chạy baseline 1 GPU

```bash
python train_single.py \
  --run_name single_gpu \
  --model resnet50 \
  --epochs 5 \
  --batch_size 64 \
  --lr 1e-4
```

## 4. Chạy DDP 2 GPU, fixed local batch size

Mỗi GPU batch 64, global batch = 128. Learning rate scale tuyến tính từ `1e-4` lên `2e-4`.

```bash
torchrun --nproc_per_node=2 train_ddp.py \
  --run_name ddp_2gpu_fixed_local \
  --model resnet50 \
  --epochs 5 \
  --batch_size 64 \
  --lr 2e-4
```

## 5. Chạy DDP 2 GPU, fixed global batch size

Mỗi GPU batch 32, global batch = 64. Learning rate giữ nguyên `1e-4`.

```bash
torchrun --nproc_per_node=2 train_ddp.py \
  --run_name ddp_2gpu_fixed_global \
  --model resnet50 \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-4
```

## 6. Tổng hợp kết quả

Sau khi chạy xong các thí nghiệm:

```bash
python benchmark.py --results_dir ./results --baseline_csv single_gpu.csv
```

Script sẽ tạo:

```text
results/summary.csv
figures/speedup.png
figures/throughput.png
```

## 7. Nếu bị CUDA out of memory

Giảm batch size:

```bash
--batch_size 32
```

Hoặc giảm image size:

```bash
--image_size 160
```

Nhưng nếu dùng image size 160, nhớ ghi rõ trong báo cáo rằng nhóm giảm kích thước ảnh để phù hợp giới hạn VRAM.

## 8. Các metric chính để đưa vào báo cáo

- Time per epoch
- Throughput, images/sec
- Speedup = time 1 GPU / time N GPU
- Efficiency = speedup / số GPU
- Peak GPU memory
- Top-1 accuracy

## 9. Công thức Amdahl

```text
Speedup(N) = 1 / ((1 - P) + P / N)
```

Trong đó:

- N là số GPU.
- P là phần chương trình có thể song song hóa.
- 1 - P là phần không song song hóa được.

Ví dụ với P = 0.9 và N = 2:

```text
Speedup = 1 / (0.1 + 0.9 / 2) = 1.82x
```

Thực tế thường thấp hơn do overhead đồng bộ gradient, truyền dữ liệu giữa GPU, DataLoader, và kích thước dataset nhỏ.
