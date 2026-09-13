#!/usr/bin/env python3
# Codex-added 2026-08-14
"""Aggregate anonymized EasyRL4Rec release runs and Student-t intervals.

The script reads the rank logs in ``results/MovieLensEnv-v0/PPO/logs`` but
never writes into that raw-results tree.  A logical replicate is identified
internally by ``run_id``; only anonymous, condition-local replicate labels are
written to ``completeness.csv``.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = PACKAGE_ROOT / "results" / "MovieLensEnv-v0" / "PPO" / "logs"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results" / "aggregated"

SCENES = ("hetero5", "homo3")
METHODS = ("fedbridge", "individual", "push_avg", "pfedme", "perfeddc", "fedavg")
METRICS = ("R_cum", "Len", "Coverage")
TAIL_EPOCHS = 10
METRIC_KEYS: Mapping[str, tuple[str, ...]] = {
    "R_cum": ("R_tra", "R_cum", "cumulative_reward"),
    "Len": ("len_tra", "Len", "interaction_length"),
    "Coverage": ("CV", "Coverage", "coverage"),
}

RUN_DIR_RE = re.compile(
    r"^\[release__(hetero5|homo3)__"
    r"(fedbridge|individual|push_avg|pfedme|perfeddc|fedavg)__([^\]]+)\]_(.+)$"
)
RANK_LOG_RE = re.compile(r"^logs_rank(\d+)\.log$")
LOG_RECORD_RE = re.compile(
    r"Epoch:\s*\[(\d+)\],\s*Rank:\s*\[(\d+)\],\s*Info:\s*(.+?)\s*$"
)
NUMPY_SCALAR_RE = re.compile(
    r"(?:(?:np|numpy)\.)?(?:float(?:16|32|64|128)?|int(?:8|16|32|64)?|"
    r"uint(?:8|16|32|64)?|bool_)\(([^()]*)\)"
)

COMPLETENESS_COLUMNS = (
    "scene",
    "method",
    "replicate_id",
    "status",
    "expected_epochs",
    "complete_epochs",
    "missing_epochs",
    "first_complete_epoch",
    "last_complete_epoch",
    "expected_ranks",
    "rank_files",
    "parsed_records",
    "malformed_records",
    "attempts_found",
)
EPOCH_COLUMNS = (
    "scene",
    "method",
    "metric",
    "epoch",
    "n",
    "mean",
    "std",
    "ci95_low",
    "ci95_high",
)
TAIL_COLUMNS = (
    "scene",
    "method",
    "metric",
    "n",
    "tail10_mean",
    "std",
    "ci95_low",
    "ci95_high",
)


@dataclass(frozen=True)
class ParsedRun:
    """Parsed values for one filesystem attempt of one logical replicate."""

    scene: str
    method: str
    run_id: str  # Internal only.  Never serialize or print this field.
    run_dir: Path  # Internal only.  Never serialize or print this field.
    epoch_metrics: Mapping[int, Mapping[str, float]]
    expected_epochs: int
    expected_ranks: int
    rank_files: int
    parsed_records: int
    malformed_records: int

    @property
    def complete_epoch_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self.epoch_metrics))

    @property
    def is_complete(self) -> bool:
        expected = set(range(1, self.expected_epochs + 1))
        return self.rank_files == self.expected_ranks and set(self.epoch_metrics) == expected


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _strip_numpy_scalars(text: str) -> str:
    """Turn scalar constructors in log reprs into literal-evaluable values."""

    previous = None
    while text != previous:
        previous = text
        text = NUMPY_SCALAR_RE.sub(r"\1", text)
    return text


def _parse_info(text: str) -> Mapping[str, object] | None:
    try:
        payload = ast.literal_eval(_strip_numpy_scalars(text))
    except (SyntaxError, ValueError):
        return None
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    return payload if isinstance(payload, dict) else None


def _extract_metrics(info: Mapping[str, object]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for metric, candidates in METRIC_KEYS.items():
        value = None
        for key in candidates:
            if key in info:
                value = _finite_float(info[key])
                if value is not None:
                    break
        if value is None:
            return None
        values[metric] = value
    return values


def parse_run(
    run_dir: Path,
    scene: str,
    method: str,
    run_id: str,
    expected_epochs: int,
    expected_ranks: int,
) -> ParsedRun:
    """Parse an attempt, then average each complete epoch across all ranks."""

    records: dict[tuple[int, int], dict[str, float]] = {}
    parsed_records = 0
    malformed_records = 0
    recognized_rank_files: set[int] = set()

    for log_path in sorted(run_dir.glob("logs_rank*.log")):
        file_match = RANK_LOG_RE.fullmatch(log_path.name)
        if file_match is None:
            continue
        file_rank = int(file_match.group(1))
        if file_rank not in range(expected_ranks):
            continue
        recognized_rank_files.add(file_rank)
        try:
            handle = log_path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            malformed_records += 1
            continue
        with handle:
            for line in handle:
                if "Epoch:" not in line:
                    continue
                record_match = LOG_RECORD_RE.search(line)
                if record_match is None:
                    malformed_records += 1
                    continue
                epoch = int(record_match.group(1))
                embedded_rank = int(record_match.group(2))
                if embedded_rank != file_rank or embedded_rank not in range(expected_ranks):
                    malformed_records += 1
                    continue
                info = _parse_info(record_match.group(3))
                metrics = _extract_metrics(info) if info is not None else None
                if metrics is None:
                    malformed_records += 1
                    continue
                # Appended/restarted logs can repeat an epoch.  The last valid
                # record is the final state and is therefore the one retained.
                records[(epoch, embedded_rank)] = metrics
                parsed_records += 1

    epoch_metrics: dict[int, dict[str, float]] = {}
    for epoch in range(1, expected_epochs + 1):
        rank_records = [records.get((epoch, rank)) for rank in range(expected_ranks)]
        if any(record is None for record in rank_records):
            continue
        epoch_metrics[epoch] = {
            metric: float(np.mean([record[metric] for record in rank_records if record is not None]))
            for metric in METRICS
        }

    return ParsedRun(
        scene=scene,
        method=method,
        run_id=run_id,
        run_dir=run_dir,
        epoch_metrics=epoch_metrics,
        expected_epochs=expected_epochs,
        expected_ranks=expected_ranks,
        rank_files=len(recognized_rank_files),
        parsed_records=parsed_records,
        malformed_records=malformed_records,
    )


def _candidate_score(run: ParsedRun) -> tuple[int, int, int, int, int]:
    """Prefer a complete attempt, then the most informative recent attempt."""

    try:
        modified_ns = run.run_dir.stat().st_mtime_ns
    except OSError:
        modified_ns = 0
    return (
        int(run.is_complete),
        len(run.epoch_metrics),
        run.rank_files,
        run.parsed_records,
        modified_ns,
    )


def discover_runs(
    logs_dir: Path,
    expected_epochs: int,
    expected_ranks: int,
    scenes: Sequence[str] = SCENES,
    methods: Sequence[str] = METHODS,
) -> tuple[dict[tuple[str, str, str], ParsedRun], dict[tuple[str, str, str], int]]:
    """Discover logical replicates and select one attempt for each run_id."""

    attempts: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for entry in sorted(logs_dir.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue
        match = RUN_DIR_RE.fullmatch(entry.name)
        if match is None:
            continue
        scene, method, run_id, _timestamp = match.groups()
        if scene not in scenes or method not in methods:
            continue
        attempts[(scene, method, run_id)].append(entry)

    selected: dict[tuple[str, str, str], ParsedRun] = {}
    attempt_counts: dict[tuple[str, str, str], int] = {}
    for key, paths in attempts.items():
        scene, method, run_id = key
        parsed_attempts = [
            parse_run(path, scene, method, run_id, expected_epochs, expected_ranks)
            for path in paths
        ]
        selected[key] = max(parsed_attempts, key=_candidate_score)
        attempt_counts[key] = len(paths)
    return selected, attempt_counts


def student_ci95(values: Iterable[float]) -> tuple[int, float, float, float, float]:
    """Return n, mean, sample std, and two-sided Student-t 95% bounds."""

    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    n = int(array.size)
    if n == 0:
        return 0, math.nan, math.nan, math.nan, math.nan
    mean = float(array.mean())
    if n == 1:
        return 1, mean, 0.0, math.nan, math.nan
    std = float(array.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=n - 1))
    half_width = critical * std / math.sqrt(n)
    return n, mean, std, mean - half_width, mean + half_width


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.8g")
    temporary.replace(destination)


def aggregate(
    selected: Mapping[tuple[str, str, str], ParsedRun],
    attempt_counts: Mapping[tuple[str, str, str], int],
    expected_epochs: int,
    expected_ranks: int,
    scenes: Sequence[str] = SCENES,
    methods: Sequence[str] = METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Build public completeness and aggregate tables without run identifiers."""

    completeness_rows: list[dict[str, object]] = []
    epoch_values: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    tail_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    strict_issues: list[str] = []

    for scene in scenes:
        for method in methods:
            condition_runs = sorted(
                (
                    (run_id, run)
                    for (run_scene, run_method, run_id), run in selected.items()
                    if run_scene == scene and run_method == method
                ),
                key=lambda pair: pair[0],
            )
            if not condition_runs:
                completeness_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "replicate_id": "",
                        "status": "missing",
                        "expected_epochs": expected_epochs,
                        "complete_epochs": 0,
                        "missing_epochs": expected_epochs,
                        "first_complete_epoch": "",
                        "last_complete_epoch": "",
                        "expected_ranks": expected_ranks,
                        "rank_files": 0,
                        "parsed_records": 0,
                        "malformed_records": 0,
                        "attempts_found": 0,
                    }
                )
                strict_issues.append(f"{scene}/{method}: no release replicate found")
                continue

            # IDs are intentionally local to a condition: their numbering does
            # not expose matching run identifiers across methods or scenes.
            for replicate_number, (run_id, run) in enumerate(condition_runs, start=1):
                complete = run.complete_epoch_numbers
                anonymous_id = f"replicate_{replicate_number:03d}"
                completeness_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "replicate_id": anonymous_id,
                        "status": "complete" if run.is_complete else "incomplete",
                        "expected_epochs": expected_epochs,
                        "complete_epochs": len(complete),
                        "missing_epochs": expected_epochs - len(complete),
                        "first_complete_epoch": min(complete) if complete else "",
                        "last_complete_epoch": max(complete) if complete else "",
                        "expected_ranks": expected_ranks,
                        "rank_files": run.rank_files,
                        "parsed_records": run.parsed_records,
                        "malformed_records": run.malformed_records,
                        "attempts_found": attempt_counts[(scene, method, run_id)],
                    }
                )
                if not run.is_complete:
                    strict_issues.append(
                        f"{scene}/{method}/{anonymous_id}: "
                        f"{len(complete)}/{expected_epochs} complete epochs"
                    )

                for epoch, metric_values in run.epoch_metrics.items():
                    for metric, value in metric_values.items():
                        epoch_values[(scene, method, metric, epoch)].append(value)
                for metric in METRICS:
                    series = [run.epoch_metrics[epoch][metric] for epoch in complete]
                    if series:
                        tail_values[(scene, method, metric)].append(
                            float(np.mean(series[-TAIL_EPOCHS:]))
                        )

    epoch_rows: list[dict[str, object]] = []
    for scene in scenes:
        for method in methods:
            for metric in METRICS:
                for epoch in range(1, expected_epochs + 1):
                    values = epoch_values.get((scene, method, metric, epoch), [])
                    if not values:
                        continue
                    n, mean, std, low, high = student_ci95(values)
                    epoch_rows.append(
                        {
                            "scene": scene,
                            "method": method,
                            "metric": metric,
                            "epoch": epoch,
                            "n": n,
                            "mean": mean,
                            "std": std,
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )

    tail_rows: list[dict[str, object]] = []
    for scene in scenes:
        for method in methods:
            for metric in METRICS:
                values = tail_values.get((scene, method, metric), [])
                if not values:
                    continue
                n, mean, std, low, high = student_ci95(values)
                tail_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "metric": metric,
                        "n": n,
                        "tail10_mean": mean,
                        "std": std,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

    completeness = pd.DataFrame(completeness_rows, columns=COMPLETENESS_COLUMNS)
    epochs = pd.DataFrame(epoch_rows, columns=EPOCH_COLUMNS)
    tails = pd.DataFrame(tail_rows, columns=TAIL_COLUMNS)
    return completeness, epochs, tails, strict_issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate 8-rank EasyRL4Rec release logs across anonymous replicates "
            "with two-sided Student-t 95% confidence intervals."
        )
    )
    parser.add_argument(
        "--logs-dir",
        "--log-root",
        dest="logs_dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="raw rank-log directory (default: package-relative results/.../logs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the three public CSVs (default: results/aggregated)",
    )
    parser.add_argument(
        "--expected-epochs",
        type=int,
        default=100,
        help="required epoch count per replicate (default: 100)",
    )
    parser.add_argument(
        "--expected-ranks",
        type=int,
        default=8,
        help="required MPI rank count (default: 8)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write aggregates from available complete epochs instead of failing",
    )
    parser.add_argument("--scene", choices=SCENES, help="Select one scene; default: both.")
    parser.add_argument("--method", choices=METHODS, help="Select one method; default: all.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    scenes = (args.scene,) if args.scene else SCENES
    methods = (args.method,) if args.method else METHODS
    if args.expected_epochs <= 0:
        parser.error("--expected-epochs must be positive")
    if args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")
    logs_dir = args.logs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not logs_dir.is_dir():
        parser.error("--logs-dir does not exist or is not a directory")
    if logs_dir == output_dir or logs_dir in output_dir.parents:
        parser.error("--output-dir must be outside the raw --logs-dir tree")

    selected, attempt_counts = discover_runs(
        logs_dir=logs_dir,
        expected_epochs=args.expected_epochs,
        expected_ranks=args.expected_ranks,
        scenes=scenes,
        methods=methods,
    )
    if not selected:
        parser.error("no directories matching the release naming convention were found")

    completeness, epochs, tails, strict_issues = aggregate(
        selected=selected,
        attempt_counts=attempt_counts,
        expected_epochs=args.expected_epochs,
        expected_ranks=args.expected_ranks,
        scenes=scenes,
        methods=methods,
    )
    if strict_issues and not args.allow_incomplete:
        print(
            f"ERROR: {len(strict_issues)} incomplete condition or replicate entries; "
            "no aggregate files were written. Re-run with --allow-incomplete "
            "only for diagnostics.",
            file=sys.stderr,
        )
        for issue in strict_issues:
            print(f"  - {issue}", file=sys.stderr)
        return 2

    _atomic_csv(completeness, output_dir / "completeness.csv")
    _atomic_csv(epochs, output_dir / "epoch_mean_ci95.csv")
    _atomic_csv(tails, output_dir / "tail10_summary_ci95.csv")

    print(
        f"Wrote completeness.csv, epoch_mean_ci95.csv, and "
        f"tail10_summary_ci95.csv ({len(selected)} anonymous replicates)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
