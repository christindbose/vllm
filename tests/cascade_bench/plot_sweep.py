"""Plot Fig-12-style 3-panel TTFT/TPOT/P99 TPOT vs request rate.

Reads JSONs produced by benchmark_serving.py --save-result.
Each file is one (backend, rate) data point. Filename pattern:
  {BACKEND}-{rate}qps.json    (e.g., FA-8.0qps.json)

Usage:
  python plot_sweep.py results/fa results/treewalk -o fig12_local.png
  (each positional arg is a backend's result dir; legend label = dir basename)
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(result_dir):
    """Return list of (rate, metrics_dict) sorted by rate."""
    runs = []
    for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        rate = float(d.get("request_rate", 0))
        runs.append((rate, d))
    runs.sort(key=lambda x: x[0])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+", help="One or more result dirs (one per backend)")
    ap.add_argument("-o", "--output", default="fig12_local.png", help="Output plot filename")
    ap.add_argument("--title", default="vLLM cascade sweep — toolagent (Qwen2.5-7B, H100)",
                    help="Plot title")
    args = ap.parse_args()

    metrics = [
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("mean_itl_ms",  "Mean TBT / ITL (ms)"),
        ("p99_tpot_ms",  "P99 TPOT (ms)"),
    ]
    # Auto-skip metrics that aren't in any of the loaded JSONs (e.g., ITL on
    # older runs). Keep this fail-soft so old result dirs still produce plots.
    have_metrics = []
    for d in args.result_dirs:
        for _, m in load_runs(d):
            for key, label in metrics:
                if key in m and (key, label) not in have_metrics:
                    have_metrics.append((key, label))
    metrics = have_metrics or metrics
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for d in args.result_dirs:
        label = os.path.basename(d.rstrip("/"))
        runs = load_runs(d)
        if not runs:
            print(f"[plot] WARN: no JSONs in {d}", file=sys.stderr)
            continue
        rates = [r for r, _ in runs]
        for ax, (key, _) in zip(axes, metrics):
            ys = [m.get(key, float("nan")) for _, m in runs]
            ax.plot(rates, ys, "o-", label=label)
        print(f"[plot] {label}: {len(runs)} points at rates {rates}")

    for ax, (_, ylabel) in zip(axes, metrics):
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Request rate (req/s)")
    axes[0].set_title(args.title)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {args.output}")


if __name__ == "__main__":
    main()
