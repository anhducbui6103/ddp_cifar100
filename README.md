# Phân tích hiệu năng và Scaling của PyTorch DDP

Hai notebook chạy **cùng một pipeline hoàn chỉnh**, chỉ khác `WORLD_SIZE` (1 GPU vs 2 GPU DDP).

Mỗi notebook **nhúng sẵn** toàn bộ code `ddp_utils` — cell Bước 0 ghi ra `ddp_utils.py` rồi import (bắt buộc cho DDP/`mp.spawn` trên Colab).

## Hai notebook — cùng pipeline

| File | GPU | Dòng duy nhất khác biệt |
|------|-----|------------------------|
| `pipeline_1gpu.ipynb` | 1 GPU | `WORLD_SIZE = 1` |
| `pipeline_2gpu_ddp.ipynb` | 2 GPU DDP | `WORLD_SIZE = 2` |

### Pipeline (7 bước, giống nhau 100%)

```
Bước 0  → Ghi + import ddp_utils (cell nhúng) + setup đường dẫn/GPU
Bước 1–3 → Huấn luyện 3 runs: baseline → lr_scaled → no_lr_scale
Bước 4  → Bảng tổng hợp thời gian & accuracy
Bước 5  → Biểu đồ Linear LR Scaling
Bước 6  → Speedup & Amdahl (load kết quả notebook còn lại)
Bước 7  → Kết luận
```

### Ý nghĩa 3 runs (tương đương giữa 2 notebook)

| Run | 1 GPU (`WORLD_SIZE=1`) | 2 GPU DDP (`WORLD_SIZE=2`) |
|-----|------------------------|----------------------------|
| **baseline** | global bs=16, lr=1e-4 | global bs=32, lr=1e-4 |
| **lr_scaled** | global bs=32, lr=2e-4 | global bs=32, lr=2e-4 |
| **no_lr_scale** | global bs=32, lr=1e-4 | = baseline (tái sử dụng, không train lại) |

> Trên 2 GPU, `no_lr_scale` trùng cấu hình `baseline` nên notebook tự bỏ qua lần train thừa.

## Chia việc nhóm 4 người

| Nhóm | Notebook | GPU | Nhiệm vụ |
|------|----------|-----|----------|
| Team A (2 người) | `pipeline_1gpu.ipynb` | ×1 | Chạy pipeline 1 GPU |
| Team B (2 người) | `pipeline_2gpu_ddp.ipynb` | **×2** (bắt buộc) | Chạy pipeline 2 GPU DDP |

Hai team có thể **chạy song song**. Sau đó trao đổi kết quả để hoàn thành Bước 6 (Speedup/Amdahl).

## Cấu trúc dự án

```
pytorch-ddp-scaling-analysis/
├── pipeline_1gpu.ipynb        # Pipeline hoàn chỉnh — 1 GPU (ddp_utils nhúng trong cell)
├── pipeline_2gpu_ddp.ipynb    # Pipeline hoàn chỉnh — 2 GPU DDP (ddp_utils nhúng trong cell)
├── ddp_utils.py               # Bản gốc module (đồng bộ với cell ddp_utils trong notebook)
├── requirements.txt
└── README.md
```

> Khi sửa logic DDP, cập nhật `ddp_utils.py` trong repo rồi đồng bộ lại cell `ddp_utils` trong cả 2 notebook.

## Hướng dẫn chạy trên Google Colab

### Chuẩn bị

1. Upload notebook tương ứng (`pipeline_1gpu.ipynb` hoặc `pipeline_2gpu_ddp.ipynb`)
2. **Runtime → Change runtime type → GPU**
   - Notebook 1 GPU: T4 ×1 là đủ
   - Notebook 2 GPU: cần **2 GPU** (Colab Pro/Pro+ hoặc máy có đủ GPU)

### Chạy

1. **Run All** (hoặc chạy tuần tự từ đầu)
2. Cell `ddp_utils` tự ghi `ddp_utils.py` và import — **không cần upload file thủ công**
3. Kết quả lưu tại `/content/results/` và bản sao tại `/content/drive_backup/`

### Đối chiếu Speedup (Bước 6) trên Colab

Sau khi team kia chạy xong, copy các file `pipeline_baseline_*gpu_metrics.json` (và `_history.json` nếu có) vào:

| Notebook | Đường dẫn chứa kết quả team kia |
|----------|----------------------------------|
| `pipeline_1gpu.ipynb` | `/content/shared/ddp-pipeline-2gpu/` |
| `pipeline_2gpu_ddp.ipynb` | `/content/shared/ddp-pipeline-1gpu/` |

Tạo thư mục và upload file, rồi chạy lại cell Bước 6.

**Gợi ý:** mount Google Drive để chia sẻ `drive_backup/` giữa 2 team:

```python
from google.colab import drive
drive.mount('/content/drive')
# copy drive_backup/ vào /content/shared/...
```

## Hướng dẫn chạy trên Kaggle (tùy chọn)

### Chuẩn bị

1. Upload notebook tương ứng
2. Settings → **Internet: On**

### Team A — `pipeline_1gpu.ipynb`

- Accelerator: GPU (×1 là đủ)
- **Run All**
- Upload `drive_backup/` lên dataset `ddp-pipeline-1gpu`

### Team B — `pipeline_2gpu_ddp.ipynb`

- Accelerator: **GPU T4 ×2** (bắt buộc)
- **Run All**
- Upload `drive_backup/` lên dataset `ddp-pipeline-2gpu`

### Đối chiếu Speedup (Bước 6) trên Kaggle

Trong mỗi notebook, **Add Input** dataset của team kia:

| Notebook | Add Input |
|----------|-----------|
| `pipeline_1gpu.ipynb` | `ddp-pipeline-2gpu` |
| `pipeline_2gpu_ddp.ipynb` | `ddp-pipeline-1gpu` |

Chạy lại cell Bước 6 → có biểu đồ Amdahl và speedup.

## Output mỗi notebook

| File | Mô tả |
|------|-------|
| `pipeline_baseline_{N}gpu_*` | Kết quả run baseline |
| `pipeline_lr_scaled_{N}gpu_*` | Kết quả LR ×2 |
| `pipeline_summary_{N}gpu.csv` | Bảng tổng hợp |
| `lr_scaling_{N}gpu.png` | Biểu đồ accuracy |
| `amdahl_analysis.png` | Speedup & Amdahl (sau khi có cả 2) |
| `amdahl_report.json` | Số liệu speedup |

## Tham số mặc định (đồng bộ giữa 2 team)

| Tham số | Giá trị |
|---------|---------|
| `LOCAL_BATCH` | 16 |
| `BASE_LR` | 1e-4 |
| `EPOCHS` | 5 |
| `model_name` | vit_large_patch16_224 |

## Xử lý lỗi

| Lỗi | Cách xử lý |
|-----|------------|
| `CUDA OOM` | Giảm `LOCAL_BATCH` xuống 8 |
| `Can't get attribute '_train_worker'` | Chạy lại cell `ddp_utils` (Bước 0) — DDP cần worker trong file `.py`, không chạy inline trong notebook |
| `NameError: run_training` | Chạy lại cell `ddp_utils` trước cell huấn luyện |
| 2 GPU notebook báo thiếu GPU | Colab: đổi runtime sang 2 GPU; Kaggle: Settings → T4 ×2 |
| Bước 6 không có speedup | Copy/upload kết quả `drive_backup/` của team kia |

## Tài liệu tham khảo

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Linear LR Scaling (Goyal et al.)](https://arxiv.org/abs/1706.02677)
