#!/usr/bin/env python3
# Codex-added 2026-08-23: audit fixed-tuple Siyuan jobs, application outputs,
# zero-shot checkpoint reloads and basic metric sanity without exposing seeds.
"""Check a private Siyuan job bundle prepared by ``prepare_siyuan_jobs.py``.

The hard checks establish scheduler/application completion, expected output
coverage, finite metrics, checkpoint reload agreement and domain bounds.  Soft
warnings identify constant curves but never require one algorithm to beat
another: a single seed cannot establish manuscript-level performance rankings.

Only the Python standard library is used so this command can run on a login
node after the Slurm jobs leave the queue.
"""

from __future__ import print_function

import argparse
import ast
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


N_RANKS = 8
MUJOCO_DOUBLE_POLICY_METHODS = {
    "fedbridge_singlecritic",
    "fedbridge_legacy_dualcritic",
    "pfedme",
}
MUJOCO_NO_COMMUNICATION_METHODS = {"individual"}
REC_METRIC_KEYS = {
    "R_cum": ("R_tra", "R_cum", "cumulative_reward"),
    "Len": ("len_tra", "Len", "interaction_length"),
    "Coverage": ("CV", "Coverage", "coverage"),
}
REC_LOG_RE = re.compile(
    r"Epoch:\s*\[(\d+)\],\s*Rank:\s*\[(\d+)\],\s*Info:\s*(.+?)\s*$"
)
NUMPY_SCALAR_RE = re.compile(
    r"(?:(?:np|numpy)\.)?(?:float(?:16|32|64|128)?|int(?:8|16|32|64)?|"
    r"uint(?:8|16|32|64)?|bool_)\(([^()]*)\)"
)
FATAL_LOG_RE = re.compile(
    r"Traceback \(most recent call last\)|MPI_ABORT|OUT_OF_MEMORY|"
    r"out of memory|oom-kill|Segmentation fault|ModuleNotFoundError|"
    # Codex-modified 2026-09-09: a run with finite evaluation can still have
    # non-finite optimizer diagnostics (the former zero-KL Individual path did).
    r"\bloss(?:/[A-Za-z0-9_.-]+)?\s*=\s*(?:nan|[+-]?inf)\b|"
    # Codex-modified 2026-08-23: Slurm time-limit cancellation is fatal even
    # when an EXIT trap or subprocess happens to leave a zero shell status.
    r"ImportError:|Killed process|DUE TO TIME LIMIT|\bTIMEOUT\b|"
    r"CANCELLED(?: AT|.*TIME LIMIT)",
    re.IGNORECASE,
)


class AuditError(RuntimeError):
    """A hard scientific-pipeline or scheduler acceptance check failed."""


class Report(object):
    def __init__(self):
        self.passes = []
        self.warnings = []
        self.failures = []

    def passed(self, label, message):
        self.passes.append({"label": label, "message": message})
        print("[PASS] {}: {}".format(label, message))

    def warn(self, label, message):
        self.warnings.append({"label": label, "message": message})
        print("[WARN] {}: {}".format(label, message))

    def fail(self, label, message):
        self.failures.append({"label": label, "message": message})
        print("[FAIL] {}: {}".format(label, message), file=sys.stderr)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit scheduler state and result sanity for one private Siyuan bundle."
    )
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="check only this manifest task_id; repeatable",
    )
    parser.add_argument(
        "--require-slurm-completed",
        action="store_true",
        help="also require a JOB_IDS.tsv entry and sacct COMPLETED/0:0 for every task",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="report JSON path (default: JOB_ROOT/CHECK_REPORT.json)",
    )
    return parser.parse_args(argv)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AuditError("invalid JSON artifact: {}".format(error))


def read_csv(path):
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise AuditError("missing or unreadable CSV artifact: {}".format(error))


def finite(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AuditError("{} is not numeric".format(field))
    if not math.isfinite(number):
        raise AuditError("{} is not finite".format(field))
    return number


def require(condition, message):
    if not condition:
        raise AuditError(message)


def seed_fingerprint(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def locate_mujoco_run(manifest, entry, index):
    root = Path(manifest["output_root"]) / "mujoco"
    pattern = entry["result_globs"][index]
    matches = []
    for candidate in sorted(root.glob(pattern)):
        if not candidate.is_dir():
            continue
        config_path = candidate / "run_config.json"
        if not config_path.is_file():
            continue
        try:
            config = read_json(config_path)
        except AuditError:
            continue
        if seed_fingerprint(config.get("training_seed")) == manifest["seed_sha256"]:
            matches.append(candidate)
    require(len(matches) == 1, "expected exactly one matching private MuJoCo run, found {}".format(len(matches)))
    return matches[0]


def compare_zero_shot(run_dir, primary_rows, expected_rows, expected_per_cluster):
    recomputed = read_csv(run_dir / "heldout_eval_recomputed.csv")
    require(len(recomputed) == expected_rows, "unexpected recomputed zero-shot row count")
    require(
        Counter(row.get("target_cluster") for row in recomputed)
        == Counter({"D": expected_per_cluster, "E": expected_per_cluster, "F": expected_per_cluster}),
        "recomputed zero-shot rows do not cover D/E/F exactly",
    )
    keys = ("client_id", "target_cluster", "repeat", "episode")
    expected = {
        tuple(row.get(field) for field in keys): (
            finite(row.get("return"), "launcher zero-shot return"),
            int(row.get("episode_length", 0)),
        )
        for row in primary_rows
    }
    actual = {
        tuple(row.get(field) for field in keys): (
            finite(row.get("return"), "recomputed zero-shot return"),
            int(row.get("episode_length", 0)),
        )
        for row in recomputed
    }
    require(len(expected) == len(primary_rows), "launcher zero-shot keys are duplicated")
    require(len(actual) == len(recomputed), "recomputed zero-shot keys are duplicated")
    require(set(expected) == set(actual), "launcher and checkpoint-reload zero-shot grids differ")
    for key in expected:
        require(
            abs(expected[key][0] - actual[key][0]) <= 1e-8
            and expected[key][1] == actual[key][1],
            "checkpoint-reload zero-shot result differs from inline evaluation",
        )
    summary = read_csv(run_dir / "heldout_summary_recomputed.csv")
    require(len(summary) == 3, "recomputed zero-shot summary must have three rows")
    require(
        {row.get("target_cluster") for row in summary} == {"D", "E", "F"},
        "recomputed summary does not cover D/E/F",
    )
    for row in summary:
        require(
            int(row.get("n_client_repeat_episodes", 0)) == expected_per_cluster,
            "unexpected recomputed zero-shot summary sample count",
        )
        finite(row.get("mean_return"), "zero-shot summary mean")
        finite(row.get("standard_deviation"), "zero-shot summary standard deviation")


def audit_mujoco_cell(manifest, entry, index, profile, report):
    environment = entry["conditions"][index]
    method = entry["method"]
    label = "{}/{}".format(entry["task_id"], environment)
    run_dir = locate_mujoco_run(manifest, entry, index)
    failures = list(run_dir.glob("FAILURE_rank_*.json"))
    require(not failures, "launcher wrote a rank failure artifact")
    status = read_json(run_dir / "STATUS.json")
    config = read_json(run_dir / "run_config.json")
    require(status.get("status") == "completed", "STATUS.json is not completed")
    require(config.get("environment") == environment, "run environment does not match job")
    require(config.get("method_key") == method, "run method does not match job")
    require(config.get("n_clients") == N_RANKS, "run did not use eight clients")
    require(
        seed_fingerprint(config.get("training_seed")) == manifest["seed_sha256"],
        "run seed does not belong to this private job bundle",
    )

    smoke = profile == "smoke"
    expected_steps = 64 if smoke else 3_000_000
    expected_updates = 2 if smoke else 1_500
    expected_training = 16 if smoke else 240
    communicating = method not in MUJOCO_NO_COMMUNICATION_METHODS
    expected_rounds = (2 if smoke else 300) if communicating else 0
    expected_communication = (16 if smoke else 2_400) if communicating else 8
    policies = 2 if method in MUJOCO_DOUBLE_POLICY_METHODS else 1
    expected_local = policies * (8 if smoke else 80)
    expected_heldout = policies * (24 if smoke else 2_400)
    expected_primary = 24 if smoke else 2_400
    expected_per_cluster = 8 if smoke else 800

    require(status.get("total_local_steps_per_client") == expected_steps, "unexpected local-step budget")
    require(status.get("updates_per_client") == expected_updates, "unexpected PPO update count")
    require(status.get("communication_rounds") == expected_rounds, "unexpected communication-round count")
    row_counts = status.get("row_counts", {})
    expected_counts = {
        "training_log": expected_training,
        "communication_log": expected_communication,
        "local_eval": expected_local,
        "heldout_eval": expected_heldout,
        "shift_eval": 0,
    }
    require(
        all(row_counts.get(key) == value for key, value in expected_counts.items()),
        "STATUS.json row counts do not match the public protocol",
    )

    checkpoints = sorted((run_dir / "checkpoints").glob("client_*/final_policy.pth"))
    require(len(checkpoints) == N_RANKS, "expected eight final policy checkpoints")
    require(all(path.stat().st_size > 0 for path in checkpoints), "a final checkpoint is empty")

    training = read_csv(run_dir / "training_log.csv")
    require(len(training) == expected_training, "unexpected training-log row count")
    returns = []
    metric_fields = (
        "personal_policy_loss",
        "value_loss",
        "bridge_policy_loss",
        "personal_to_bridge_regularizer",
        "bridge_to_personal_regularizer",
    )
    for row in training:
        if row.get("return_mean") not in (None, ""):
            returns.append(finite(row["return_mean"], "training return"))
        for field in metric_fields:
            if row.get(field) not in (None, ""):
                finite(row[field], field)
    require(returns, "training log contains no finite return")
    if not smoke and len({round(value, 10) for value in returns}) <= 1:
        report.warn(label, "training returns are constant; inspect the curve and environment logs")

    communication = read_csv(run_dir / "communication_log.csv")
    require(len(communication) == expected_communication, "unexpected communication-log row count")
    if communicating:
        require(
            all(finite(row.get("payload_bytes"), "payload bytes") > 0 for row in communication),
            "communicating method has a non-positive payload",
        )
        messages = sum(
            finite(row.get("messages_sent"), "messages sent")
            + finite(row.get("messages_received"), "messages received")
            for row in communication
        )
        require(messages > 0, "communicating method recorded no messages")
    else:
        require(
            all(finite(row.get("payload_bytes"), "payload bytes") == 0 for row in communication),
            "Individual PPO unexpectedly recorded a communication payload",
        )

    local_eval = read_csv(run_dir / "local_eval.csv")
    heldout = read_csv(run_dir / "heldout_eval.csv")
    shift = read_csv(run_dir / "shift_eval.csv")
    require(len(local_eval) == expected_local, "unexpected local-evaluation row count")
    require(len(heldout) == expected_heldout, "unexpected held-out row count")
    require(len(shift) == 0, "default run unexpectedly contains continuous-shift evaluation")
    for row in local_eval + heldout:
        finite(row.get("return"), "evaluation return")
        require(int(row.get("episode_length", 0)) > 0, "evaluation episode length is not positive")
    primary = [
        row
        for row in heldout
        if str(row.get("primary_eval_policy", "")).lower() == "true"
    ]
    require(len(primary) == expected_primary, "unexpected primary-policy held-out row count")
    require(
        Counter(row.get("target_cluster") for row in primary)
        == Counter({"D": expected_per_cluster, "E": expected_per_cluster, "F": expected_per_cluster}),
        "primary-policy held-out rows do not cover D/E/F exactly",
    )
    compare_zero_shot(run_dir, primary, expected_primary, expected_per_cluster)
    if smoke:
        repeat_path = run_dir / "determinism_repeat" / "heldout_eval_recomputed.csv"
        require(repeat_path.is_file(), "second deterministic checkpoint evaluation is missing")
        require(
            (run_dir / "heldout_eval_recomputed.csv").read_bytes() == repeat_path.read_bytes(),
            "repeated checkpoint evaluation is not byte deterministic",
        )
    report.passed(label, "training, checkpoints and D/E/F zero-shot checks passed")


def strip_numpy_scalars(text):
    previous = None
    while text != previous:
        previous = text
        text = NUMPY_SCALAR_RE.sub(r"\1", text)
    return text


def parse_rec_info(text):
    try:
        payload = ast.literal_eval(strip_numpy_scalars(text))
    except (SyntaxError, ValueError):
        return None
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    return payload if isinstance(payload, dict) else None


def extract_rec_metrics(info):
    output = {}
    for metric, keys in REC_METRIC_KEYS.items():
        value = None
        for key in keys:
            if key in info:
                try:
                    value = float(info[key])
                except (TypeError, ValueError):
                    value = None
                if value is not None and math.isfinite(value):
                    break
                value = None
        if value is None:
            return None
        output[metric] = value
    return output


def locate_recommender_run(manifest, entry, index):
    logs_root = (
        Path(manifest["output_root"])
        / "recommender_system"
        / "MovieLensEnv-v0"
        / "PPO"
        / "logs"
    )
    prefix = entry["result_prefixes"][index]
    matches = [
        path
        for path in sorted(logs_root.iterdir(), key=lambda value: value.name)
        if path.is_dir() and path.name.startswith(prefix)
    ] if logs_root.is_dir() else []
    require(len(matches) == 1, "expected exactly one matching recommender run, found {}".format(len(matches)))
    return matches[0]


def audit_recommender_cell(manifest, entry, index, profile, report):
    scene = entry["conditions"][index]
    method = entry["method"]
    label = "{}/{}".format(entry["task_id"], scene)
    run_dir = locate_recommender_run(manifest, entry, index)
    status = read_json(run_dir / "RUN_STATUS.json")
    expected_epochs = 1 if profile == "smoke" else 100
    require(status.get("status") == "completed", "RUN_STATUS.json is not completed")
    require(status.get("scene") == scene, "run scene does not match job")
    require(status.get("method") == method, "run method does not match job")
    require(status.get("run_id") == manifest["run_label"], "run-id does not match private bundle")
    require(status.get("expected_epochs") == expected_epochs, "unexpected expected-epoch count")
    require(status.get("final_common_epoch") == expected_epochs, "not all ranks reached the final epoch")
    require(status.get("mpi_ranks") == N_RANKS, "run did not use eight MPI ranks")
    require(
        status.get("scientific_result") == (profile == "full"),
        "scientific_result flag does not match smoke/full profile",
    )
    require(status.get("seed_values_embedded") is False, "public status unexpectedly embeds seed values")

    rank_logs = [run_dir / "logs_rank{}.log".format(rank) for rank in range(N_RANKS)]
    require(all(path.is_file() and path.stat().st_size > 0 for path in rank_logs), "expected eight non-empty rank logs")
    records = {}
    malformed = 0
    for rank, path in enumerate(rank_logs):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Epoch:" not in line:
                continue
            match = REC_LOG_RE.search(line)
            if match is None or int(match.group(2)) != rank:
                malformed += 1
                continue
            info = parse_rec_info(match.group(3))
            metrics = extract_rec_metrics(info) if info is not None else None
            if metrics is None:
                malformed += 1
                continue
            records[(int(match.group(1)), rank)] = metrics
    if malformed:
        report.warn(label, "{} epoch-like log lines were malformed but complete records remain".format(malformed))
    expected_keys = {
        (epoch, rank)
        for epoch in range(1, expected_epochs + 1)
        for rank in range(N_RANKS)
    }
    require(set(records) >= expected_keys, "rank logs do not contain every expected epoch/rank record")
    for key in expected_keys:
        metrics = records[key]
        reward = finite(metrics["R_cum"], "recommender cumulative reward")
        length = finite(metrics["Len"], "recommender interaction length")
        coverage = finite(metrics["Coverage"], "recommender coverage")
        require(0.0 <= reward <= 150.0, "cumulative reward is outside [0, 150]")
        require(0.0 < length <= 30.0, "interaction length is outside (0, 30]")
        require(0.0 <= coverage <= 1.0, "coverage is outside [0, 1]")
    if profile == "full":
        # Codex-modified 2026-09-10: keep self-contained runtime sanity checks
        # after removing historical result tables from the public repository.
        for metric in REC_METRIC_KEYS:
            means = []
            for epoch in range(1, expected_epochs + 1):
                means.append(
                    sum(records[(epoch, rank)][metric] for rank in range(N_RANKS))
                    / float(N_RANKS)
                )
            if len({round(value, 10) for value in means}) <= 1:
                report.warn(label, "{} is constant across 100 epochs; inspect this curve".format(metric))
    report.passed(label, "eight-rank inline evaluation and metric-bound checks passed")


def read_ledger(job_root):
    path = job_root / "JOB_IDS.tsv"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as error:
        raise AuditError("missing or unreadable Slurm ledger: {}".format(error))
    return {row.get("task_id"): row.get("job_id") for row in rows if row.get("task_id")}


def scheduler_record(job_id):
    completed = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,ExitCode",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if completed.returncode != 0:
        raise AuditError("sacct failed for a generated job")
    rows = [line.split("|") for line in completed.stdout.splitlines() if line.strip()]
    exact = [row for row in rows if row and row[0] == str(job_id)]
    require(len(exact) == 1 and len(exact[0]) >= 3, "no exact sacct allocation record found")
    return exact[0][1].rstrip("+"), exact[0][2]


def audit_scheduler(job_root, entries, report, required):
    ledger = read_ledger(job_root)
    if required and shutil.which("sacct") is None:
        raise AuditError("--require-slurm-completed needs sacct on a Siyuan login node")
    for entry in entries:
        label = entry["task_id"] + "/Slurm"
        job_id = ledger.get(entry["task_id"])
        if not job_id:
            if required:
                report.fail(label, "task has no JOB_IDS.tsv entry")
            else:
                report.warn(label, "scheduler state not checked (no submitted-job ledger entry)")
            continue
        if shutil.which("sacct") is not None:
            try:
                state, exit_code = scheduler_record(job_id)
                if state == "COMPLETED" and exit_code == "0:0":
                    report.passed(label, "sacct reports COMPLETED with ExitCode 0:0")
                elif required:
                    report.fail(label, "sacct state={} exit_code={}".format(state, exit_code))
                else:
                    report.warn(label, "sacct state={} exit_code={}".format(state, exit_code))
            except AuditError as error:
                if required:
                    report.fail(label, str(error))
                else:
                    report.warn(label, str(error))


def audit_job_logs(job_root, entries, report):
    ledger = read_ledger(job_root)
    logs_root = job_root / "logs"
    markers_root = job_root / "markers"
    for entry in entries:
        label = entry["task_id"] + "/logs"
        job_id = ledger.get(entry["task_id"])
        if job_id:
            paths = [
                logs_root / (entry["job_name"] + "-" + str(job_id) + ".out"),
                logs_root / (entry["job_name"] + "-" + str(job_id) + ".err"),
            ]
        else:
            paths = sorted(logs_root.glob(entry["job_name"] + "-*.out"))
            paths += sorted(logs_root.glob(entry["job_name"] + "-*.err"))
        fatal = []
        for path in paths:
            if path.is_file() and FATAL_LOG_RE.search(
                path.read_text(encoding="utf-8", errors="replace")
            ):
                fatal.append(path.name)
        if fatal:
            report.fail(label, "fatal runtime signature found in {}".format(", ".join(fatal)))
        elif paths:
            report.passed(label, "scheduler logs contain no fatal runtime signature")
        else:
            report.warn(label, "no scheduler log files were found")
        if job_id:
            markers = [
                markers_root / (entry["task_id"] + "." + str(job_id) + ".status")
            ]
            markers = [path for path in markers if path.is_file()]
        else:
            marker_pattern = entry["task_id"] + ".*.status"
            markers = sorted(markers_root.glob(marker_pattern))
        if not markers:
            report.fail(label, "missing application completion marker")
        else:
            latest = max(markers, key=lambda path: path.stat().st_mtime_ns)
            values = {}
            for line in latest.read_text(encoding="utf-8", errors="replace").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    values[key] = value
            if values.get("exit_code") != "0":
                report.fail(label, "application exit marker is not zero")
            elif values.get("status") not in (None, "completed"):
                report.fail(label, "application completion marker is not completed")


def main(argv=None):
    os.umask(0o077)
    args = parse_args(argv)
    job_root = args.job_root.expanduser().resolve()
    manifest_path = job_root / "JOB_MANIFEST.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == 1, "unsupported private job-manifest schema")
    require(Path(manifest.get("job_root", "")).resolve() == job_root, "job-root does not match manifest")
    profile = manifest.get("profile")
    require(profile in ("smoke", "full"), "manifest profile must be smoke or full")
    entries = list(manifest.get("jobs", []))
    if args.task:
        requested = set(args.task)
        known = {entry.get("task_id") for entry in entries}
        unknown = sorted(requested - known)
        require(not unknown, "unknown --task values: {}".format(", ".join(unknown)))
        entries = [entry for entry in entries if entry.get("task_id") in requested]
    require(entries, "manifest contains no selected jobs")

    report = Report()
    audit_scheduler(job_root, entries, report, args.require_slurm_completed)
    audit_job_logs(job_root, entries, report)
    checked_cells = 0
    for entry in entries:
        for index, condition in enumerate(entry.get("conditions", [])):
            label = "{}/{}".format(entry.get("task_id"), condition)
            try:
                if entry.get("component") == "mujoco":
                    audit_mujoco_cell(manifest, entry, index, profile, report)
                elif entry.get("component") == "recommender":
                    audit_recommender_cell(manifest, entry, index, profile, report)
                else:
                    raise AuditError("unknown component in private job manifest")
                checked_cells += 1
            except (AuditError, OSError, ValueError, KeyError) as error:
                report.fail(label, str(error))

    report_path = (args.report or (job_root / "CHECK_REPORT.json")).expanduser().resolve()
    try:
        report_path.relative_to(Path(manifest["release_root"]).resolve())
    except ValueError:
        pass
    else:
        raise AuditError("--report must stay outside the public release tree")
    payload = {
        "schema_version": 1,
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profile": profile,
        "run_label": manifest.get("run_label"),
        "seed_values_embedded": False,
        "selected_algorithm_jobs": len(entries),
        "checked_condition_cells": checked_cells,
        "passes": report.passes,
        "warnings": report.warnings,
        "failures": report.failures,
        "accepted": not report.failures,
        "interpretation": (
            "Hard checks establish execution and basic result sanity only; "
            "algorithm ranking requires multiple independent seeds."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report.failures:
        print(
            "SIYUAN VALIDATION FAILED: {} hard failures; {} warnings; report={}".format(
                len(report.failures), len(report.warnings), report_path
            ),
            file=sys.stderr,
        )
        return 1
    print(
        "SIYUAN VALIDATION PASSED: {} condition cells; {} warnings; report={}".format(
            checked_cells, len(report.warnings), report_path
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
