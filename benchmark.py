import argparse
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt


def amdahl_speedup(n_gpu: int, p: float) -> float:
    return 1.0 / ((1.0 - p) + p / n_gpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--baseline_csv", type=str, default="single_gpu.csv")
    parser.add_argument("--parallel_fraction", type=float, default=0.9)
    args = parser.parse_args()

    paths = glob.glob(os.path.join(args.results_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {args.results_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        last = df.tail(1).copy()
        last["csv_file"] = os.path.basename(path)
        last["avg_epoch_time_sec"] = df["epoch_time_sec"].mean()
        last["avg_throughput_img_per_sec"] = df["throughput_img_per_sec"].mean()
        frames.append(last)

    summary = pd.concat(frames, ignore_index=True)

    baseline_path = os.path.join(args.results_dir, args.baseline_csv)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_path}")

    baseline_df = pd.read_csv(baseline_path)
    baseline_time = baseline_df["epoch_time_sec"].mean()

    summary["speedup"] = baseline_time / summary["avg_epoch_time_sec"]
    summary["efficiency"] = summary["speedup"] / summary["world_size"]
    summary["amdahl_speedup_p90"] = summary["world_size"].apply(
        lambda n: amdahl_speedup(int(n), args.parallel_fraction)
    )

    out_csv = os.path.join(args.results_dir, "summary.csv")
    summary.to_csv(out_csv, index=False)

    print("\n=== Benchmark Summary ===")
    cols = [
        "run_name",
        "world_size",
        "local_batch_size",
        "global_batch_size",
        "avg_epoch_time_sec",
        "avg_throughput_img_per_sec",
        "speedup",
        "efficiency",
        "amdahl_speedup_p90",
        "val_acc",
        "peak_memory_mb",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nSaved summary to: {out_csv}")

    os.makedirs("figures", exist_ok=True)

    plt.figure()
    plt.bar(summary["run_name"], summary["speedup"])
    plt.ylabel("Speedup vs 1 GPU")
    plt.xlabel("Run")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("figures/speedup.png", dpi=200)
    print("Saved figure: figures/speedup.png")

    plt.figure()
    plt.bar(summary["run_name"], summary["avg_throughput_img_per_sec"])
    plt.ylabel("Throughput, images/sec")
    plt.xlabel("Run")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("figures/throughput.png", dpi=200)
    print("Saved figure: figures/throughput.png")


if __name__ == "__main__":
    main()
