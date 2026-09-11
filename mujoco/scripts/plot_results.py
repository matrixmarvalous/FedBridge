#!/usr/bin/env python3
# Codex-added 2026-08-14: dynamic plotting for user-supplied runs; intended
# caller: README post-processing command. No paper seed values are embedded.
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ENV_ORDER = ["HalfCheetah-v5", "Hopper-v4", "Humanoid-v4"]
METHOD_ORDER = [
    "FedBridge",
    "Individual PPO",
    "Push-Avg",
    "PerFedDC",
    "pFedMe",
    "FedAvg",
]
COLORS = {
    "FedBridge": "#D95F5F",
    "Individual PPO": "#E6A553",
    "Push-Avg": "#75A85A",
    "PerFedDC": "#9575A5",
    "pFedMe": "#A5A95B",
    "FedAvg": "#5F8F8F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def ci95(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    if array.size < 2:
        return 0.0
    return float(t.ppf(0.975, array.size - 1) * array.std(ddof=1) / math.sqrt(array.size))


def read_tables(root: Path, filename: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.glob(f"*/{filename}")):
        frame = pd.read_csv(path)
        frame["run_directory"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {filename} files found below {root}")
    return pd.concat(frames, ignore_index=True)


def training_source_data(root: Path) -> pd.DataFrame:
    frame = read_tables(root, "training_log.csv")
    frame["return_mean"] = pd.to_numeric(frame["return_mean"], errors="coerce")
    frame = frame.dropna(subset=["return_mean"])
    per_run = (
        frame.groupby(
            ["environment", "method", "training_seed", "local_step"],
            as_index=False,
        )["return_mean"]
        .mean()
        .rename(columns={"return_mean": "run_mean_return"})
    )
    rows = []
    for keys, group in per_run.groupby(["environment", "method", "local_step"]):
        environment, method, local_step = keys
        rows.append(
            {
                "environment": environment,
                "method": method,
                "local_step": int(local_step),
                "n_runs": int(group["training_seed"].nunique()),
                "mean_return": float(group["run_mean_return"].mean()),
                "ci95": ci95(group["run_mean_return"]),
            }
        )
    return pd.DataFrame(rows)


def zero_shot_source_data(root: Path) -> pd.DataFrame:
    frame = read_tables(root, "heldout_eval.csv")
    frame = frame[frame["primary_eval_policy"].astype(str).str.lower() == "true"]
    per_run = (
        frame.groupby(
            ["environment", "method", "training_seed", "target_cluster"],
            as_index=False,
        )["return"]
        .mean()
        .rename(columns={"return": "run_mean_return"})
    )
    rows = []
    for keys, group in per_run.groupby(
        ["environment", "method", "target_cluster"]
    ):
        environment, method, cluster = keys
        rows.append(
            {
                "environment": environment,
                "method": method,
                "target_cluster": cluster,
                "n_runs": int(group["training_seed"].nunique()),
                "mean_return": float(group["run_mean_return"].mean()),
                "ci95": ci95(group["run_mean_return"]),
            }
        )
    return pd.DataFrame(rows)


def finish_figure(fig, axes: Sequence[plt.Axes]) -> None:
    """Use one ordered legend, including methods absent from the first panel."""
    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(zip(labels, handles))
    labels = [method for method in METHOD_ORDER if method in handles_by_label]
    if labels:
        fig.legend(
            [handles_by_label[label] for label in labels],
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(6, len(labels)),
            frameon=False,
            fontsize=8,
            columnspacing=1.0,
            handlelength=2.0,
        )
        fig.tight_layout(rect=(0, 0.16, 1, 1))
    else:
        fig.tight_layout()


def plot_training(source: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=False)
    for ax, environment in zip(axes, ENV_ORDER):
        env = source[source["environment"] == environment]
        for method in METHOD_ORDER:
            data = env[env["method"] == method].sort_values("local_step")
            if data.empty:
                continue
            x = data["local_step"].to_numpy(dtype=float)
            y = data["mean_return"].to_numpy(dtype=float)
            ci = data["ci95"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                label=method,
                color=COLORS[method],
                linewidth=1.5,
                marker="o",
                markersize=2.5,
            )
            ax.fill_between(x, y - ci, y + ci, color=COLORS[method], alpha=0.18)
        ax.set_title(environment.replace("-v4", "").replace("-v5", ""))
        ax.set_xlabel("Local environment steps")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Training return")
    finish_figure(fig, axes)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_zero_shot(source: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    clusters = ["D", "E", "F"]
    for ax, environment in zip(axes, ENV_ORDER):
        env = source[source["environment"] == environment]
        methods = [method for method in METHOD_ORDER if method in set(env["method"])]
        width = 0.8 / max(1, len(methods))
        x = np.arange(len(clusters), dtype=float)
        for index, method in enumerate(methods):
            data = env[env["method"] == method].set_index("target_cluster")
            means = np.asarray([data.loc[c, "mean_return"] for c in clusters])
            errors = np.asarray([data.loc[c, "ci95"] for c in clusters])
            positions = x - 0.4 + width / 2 + index * width
            ax.bar(
                positions,
                means,
                yerr=errors,
                width=width,
                color=COLORS[method],
                label=method,
                capsize=2,
            )
        ax.set_title(environment.replace("-v4", "").replace("-v5", ""))
        ax.set_xticks(x, clusters)
        ax.set_xlabel("Held-out dynamics cluster")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Zero-shot return")
    finish_figure(fig, axes)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = training_source_data(args.results_root)
    zero_shot = zero_shot_source_data(args.results_root)
    training.to_csv(args.output_dir / "training_source_data.csv", index=False)
    zero_shot.to_csv(args.output_dir / "zero_shot_source_data.csv", index=False)
    plot_training(training, args.output_dir / "training_performance.png")
    plot_zero_shot(zero_shot, args.output_dir / "zero_shot_performance.png")
    print(f"[plot-completed] {args.output_dir}")


if __name__ == "__main__":
    main()
