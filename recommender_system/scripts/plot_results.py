#!/usr/bin/env python3
# Codex-added 2026-08-14
"""Plot EasyRL4Rec learning curves from newly aggregated run data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCENES = ("hetero5", "homo3")
METHODS = ("fedbridge", "individual", "push_avg", "pfedme", "perfeddc", "fedavg")
# Codex-modified 2026-09-10: plots consume generated aggregates only; no
# historical result tables are bundled with the public code repository.
METRICS = ("R_cum", "Len", "Coverage")

METHOD_LABELS = {
    "fedbridge": "FedBridge",
    "individual": "Individual",
    "push_avg": "Push-Avg",
    "pfedme": "pFedMe",
    "perfeddc": "PerFedDC",
    "fedavg": "FedAvg",
}
METHOD_COLORS = {
    "fedbridge": "#D62728",
    "individual": "#7F7F7F",
    "push_avg": "#2CA02C",
    "pfedme": "#1F77B4",
    "perfeddc": "#9467BD",
    "fedavg": "#FF7F0E",
}
METHOD_STYLES = {
    "fedbridge": "-",
    "individual": (0, (2, 1.5)),
    "push_avg": (0, (5, 2)),
    "pfedme": (0, (3, 1, 1, 1)),
    "perfeddc": (0, (6, 2, 1, 2)),
    "fedavg": (0, (1, 1)),
}
SCENE_LABELS = {"hetero5": "Heterogeneous", "homo3": "Homogeneous"}
METRIC_LABELS = {
    "R_cum": "Cumulative reward",
    "Len": "Interaction length",
    "Coverage": "Coverage",
}

EPOCH_REQUIRED_COLUMNS = {
    "scene",
    "method",
    "metric",
    "epoch",
    "n",
    "mean",
    "std",
    "ci95_low",
    "ci95_high",
}
TAIL_REQUIRED_COLUMNS = {
    "scene",
    "method",
    "metric",
    "n",
    "tail10_mean",
    "std",
    "ci95_low",
    "ci95_high",
}


def _default_input_dir() -> Path:
    candidates = (
        PACKAGE_ROOT / "results" / "aggregated",
        PACKAGE_ROOT / "results" / "summary",
        PACKAGE_ROOT / "results" / "aggregate",
    )
    for candidate in candidates:
        if (candidate / "epoch_mean_ci95.csv").is_file() and (
            candidate / "tail10_summary_ci95.csv"
        ).is_file():
            return candidate
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a 2-scene by 3-metric EasyRL4Rec learning-curve figure."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="directory containing epoch_mean_ci95.csv and tail10_summary_ci95.csv",
    )
    parser.add_argument(
        "--epoch-csv",
        type=Path,
        default=None,
        help="explicit epoch aggregate CSV (overrides --input-dir for this file)",
    )
    parser.add_argument(
        "--tail-csv",
        type=Path,
        default=None,
        help="explicit tail-10 aggregate CSV (overrides --input-dir for this file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "figures",
        help="figure output directory (default: recommender_system/figures)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    return parser


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    input_dir = (args.input_dir or _default_input_dir()).expanduser().resolve()
    epoch_path = (
        args.epoch_csv.expanduser().resolve()
        if args.epoch_csv is not None
        else input_dir / "epoch_mean_ci95.csv"
    )
    tail_path = (
        args.tail_csv.expanduser().resolve()
        if args.tail_csv is not None
        else input_dir / "tail10_summary_ci95.csv"
    )
    return epoch_path, tail_path


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _normalize_method_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # The pre-release analysis used "pushavg".  Normalize it when plotting the
    # archived source data; newly generated release outputs use "push_avg".
    result["method"] = result["method"].replace({"pushavg": "push_avg"})
    return result


def _numeric(frame: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        converted = pd.to_numeric(result[column], errors="coerce")
        invalid = converted.isna() & result[column].notna()
        if invalid.any():
            raise ValueError(f"{label} contains non-numeric values in {column}")
        result[column] = converted
    return result


def load_and_validate(epoch_path: Path, tail_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not epoch_path.is_file():
        raise FileNotFoundError("epoch aggregate CSV was not found")
    if not tail_path.is_file():
        raise FileNotFoundError("tail-10 aggregate CSV was not found")

    epoch_frame = _normalize_method_names(pd.read_csv(epoch_path))
    tail_frame = _normalize_method_names(pd.read_csv(tail_path))
    _require_columns(epoch_frame, EPOCH_REQUIRED_COLUMNS, "epoch CSV")
    _require_columns(tail_frame, TAIL_REQUIRED_COLUMNS, "tail CSV")

    epoch_frame = epoch_frame[
        epoch_frame["scene"].isin(SCENES)
        & epoch_frame["method"].isin(METHODS)
        & epoch_frame["metric"].isin(METRICS)
    ].copy()
    tail_frame = tail_frame[
        tail_frame["scene"].isin(SCENES)
        & tail_frame["method"].isin(METHODS)
        & tail_frame["metric"].isin(METRICS)
    ].copy()
    epoch_frame = _numeric(
        epoch_frame,
        ("epoch", "n", "mean", "std", "ci95_low", "ci95_high"),
        "epoch CSV",
    )
    tail_frame = _numeric(
        tail_frame,
        ("n", "tail10_mean", "std", "ci95_low", "ci95_high"),
        "tail CSV",
    )

    if epoch_frame.empty:
        raise ValueError("epoch CSV has no supported scene/method/metric rows")
    if tail_frame.empty:
        raise ValueError("tail CSV has no supported scene/method/metric rows")
    if (epoch_frame["epoch"] <= 0).any() or (epoch_frame["n"] <= 0).any():
        raise ValueError("epoch and n values must be positive")
    if (tail_frame["n"] <= 0).any():
        raise ValueError("tail n values must be positive")

    duplicate_epoch = epoch_frame.duplicated(["scene", "method", "metric", "epoch"])
    if duplicate_epoch.any():
        raise ValueError("epoch CSV has duplicate scene/method/metric/epoch rows")
    duplicate_tail = tail_frame.duplicated(["scene", "method", "metric"])
    if duplicate_tail.any():
        raise ValueError("tail CSV has duplicate scene/method/metric rows")

    expected_conditions = {
        (scene, method, metric)
        for scene in SCENES
        for method in METHODS
        for metric in METRICS
    }
    epoch_conditions = set(
        epoch_frame[["scene", "method", "metric"]].itertuples(index=False, name=None)
    )
    tail_conditions = set(
        tail_frame[["scene", "method", "metric"]].itertuples(index=False, name=None)
    )
    missing_epoch = expected_conditions.difference(epoch_conditions)
    missing_tail = expected_conditions.difference(tail_conditions)
    if missing_epoch:
        raise ValueError(f"epoch CSV lacks {len(missing_epoch)} required condition curves")
    if missing_tail:
        raise ValueError(f"tail CSV lacks {len(missing_tail)} required condition rows")

    finite_columns = ("mean", "std")
    if not np.isfinite(epoch_frame[list(finite_columns)].to_numpy(dtype=float)).all():
        raise ValueError("epoch CSV contains non-finite means or standard deviations")
    if not np.isfinite(tail_frame[["tail10_mean", "std"]].to_numpy(dtype=float)).all():
        raise ValueError("tail CSV contains non-finite means or standard deviations")

    # CI bounds are undefined for n=1.  For n>=2, require finite, ordered bounds.
    for frame, mean_column, label in (
        (epoch_frame, "mean", "epoch CSV"),
        (tail_frame, "tail10_mean", "tail CSV"),
    ):
        with_interval = frame["n"] >= 2
        interval = frame.loc[with_interval, ["ci95_low", mean_column, "ci95_high"]]
        if not np.isfinite(interval.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} contains a non-finite CI for n>=2")
        if not (
            (interval["ci95_low"] <= interval[mean_column])
            & (interval[mean_column] <= interval["ci95_high"])
        ).all():
            raise ValueError(f"{label} contains unordered confidence bounds")

    return epoch_frame, tail_frame


def make_figure(epoch_frame: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.3), sharex="col", squeeze=False)

    for row, scene in enumerate(SCENES):
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            for method in METHODS:
                data = epoch_frame[
                    (epoch_frame["scene"] == scene)
                    & (epoch_frame["method"] == method)
                    & (epoch_frame["metric"] == metric)
                ].sort_values("epoch")
                x = data["epoch"].to_numpy(dtype=float)
                mean = data["mean"].to_numpy(dtype=float)
                low = data["ci95_low"].to_numpy(dtype=float)
                high = data["ci95_high"].to_numpy(dtype=float)
                color = METHOD_COLORS[method]
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=METHOD_STYLES[method],
                    linewidth=1.55 if method == "fedbridge" else 1.25,
                    label=METHOD_LABELS[method],
                    zorder=3 if method == "fedbridge" else 2,
                )
                valid_ci = np.isfinite(low) & np.isfinite(high)
                if valid_ci.any():
                    axis.fill_between(
                        x[valid_ci],
                        low[valid_ci],
                        high[valid_ci],
                        color=color,
                        alpha=0.11 if method == "fedbridge" else 0.065,
                        linewidth=0,
                        zorder=1,
                    )

            axis.set_title(f"{SCENE_LABELS[scene]} · {METRIC_LABELS[metric]}")
            if row == len(SCENES) - 1:
                axis.set_xlabel("Epoch")
            axis.set_ylabel(METRIC_LABELS[metric])
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
            axis.margins(x=0.01)
            if metric == "Coverage":
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.026),
        handlelength=2.7,
        columnspacing=1.5,
    )
    figure.text(
        0.5,
        0.004,
        "Push-Avg uses the original actor-parameter communication update.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.085, 1, 1), h_pad=1.6, w_pad=1.45)
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    epoch_path, tail_path = _resolve_inputs(args)
    try:
        epoch_frame, _tail_frame = load_and_validate(epoch_path, tail_path)
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = make_figure(epoch_frame)
    destinations = [output_dir / f"learning_curves.{extension}" for extension in ("png", "pdf", "svg")]
    try:
        for destination in destinations:
            save_kwargs = {"bbox_inches": "tight"}
            if destination.suffix == ".png":
                save_kwargs["dpi"] = args.dpi
            figure.savefig(destination, **save_kwargs)
    finally:
        plt.close(figure)

    print("Wrote learning_curves.png, learning_curves.pdf, and learning_curves.svg.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
