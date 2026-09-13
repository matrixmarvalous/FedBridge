#!/usr/bin/env python3
# Codex-added 2026-08-14: thin public dispatcher around preserved runners.
# Codex-modified 2026-08-18: audit the canonical BridgePPO imports.
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from experiment_spec import METHODS, N_CLIENTS, SCENES, command_args, runner_path


THIS_DIR = Path(__file__).resolve().parent
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,63}$")
EPOCH_RE = re.compile(r"Epoch:\s*\[(\d+)\],\s*Rank:\s*\[(\d+)\]")
MPI_EXPORTS = (
    "PYTHONUNBUFFERED",
    "EASYRL4REC_MOVIELENS_ROOT",
    "EASYRL4REC_DEEPFM_ROOT",
    "EASYRL4REC_OUTPUT_ROOT",
    "EASYRL4REC_RUN_STAMP",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one EasyRL4Rec scene x method x user-seed experiment."
    )
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument(
        "--seed",
        type=int,
        help="Required user-selected base seed; paper seed values are not embedded.",
    )
    parser.add_argument(
        "--run-id",
        help="Opaque public run label (letters, digits, dots or hyphens; do not put a seed here).",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=THIS_DIR,
        help="Root containing data/MovieLens and saved_models/.../DeepFM.",
    )
    parser.add_argument("--output-root", type=Path, default=THIS_DIR / "results")
    parser.add_argument("--mpi-exec", default="mpiexec")
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="release-supported execution path (CPU only)",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-asset-check", action="store_true")
    parser.add_argument(
        "--disable-categorical-validation",
        action="store_true",
        help="Opt-in diagnostic for masked-probability underflow; not a default paper setting.",
    )
    args = parser.parse_args(argv)
    if args.audit_only:
        return args
    if args.scene is None or args.method is None:
        parser.error("--scene and --method are required unless --audit-only is used")
    if args.seed is None:
        parser.error("--seed is required; this release has no default experiment seed")
    if args.seed < 0:
        parser.error("--seed must be a non-negative integer")
    if not args.run_id or not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id must match [A-Za-z0-9][A-Za-z0-9.-]{0,63}")
    if args.disable_categorical_validation and (args.scene, args.method) != (
        "homo3",
        "fedavg",
    ):
        parser.error(
            "--disable-categorical-validation is only implemented for homo3/fedavg"
        )
    return args


def asset_paths(asset_root: Path) -> tuple[Path, Path]:
    root = asset_root.expanduser().resolve()
    return (
        root / "data" / "MovieLens",
        root / "saved_models" / "MovieLensEnv-v0" / "DeepFM",
    )


def validate_assets(asset_root: Path, scene: str | None = None) -> None:
    subprocess.run(
        [
            sys.executable,
            str(THIS_DIR / "scripts" / "validate_assets.py"),
            "--asset-root",
            str(asset_root.expanduser().resolve()),
            *(["--scene", scene] if scene is not None else []),
        ],
        cwd=THIS_DIR,
        check=True,
    )


def audit_imports() -> None:
    code = """
import sys
# Codex-modified 2026-08-14: expose the two vendored packages exactly as the
# preserved runners do before importing their public PPO modules.
sys.path.insert(0, 'src/tianshou')
sys.path.insert(0, 'src/DeepCTR-Torch')
sys.path.insert(0, 'examples/policy')
from tianshou.policy import BridgePPOPolicy, PerFedDCPolicy, pFedMePolicy
from tianshou.trainer import fedbridgeonp_trainer, fedonpolicy_push_avg_trainer, perfeddc_trainer, pfedme_trainer, fedavgonpolicy_trainer
from policy_utils import get_args_all
print('EasyRL4Rec release imports: OK')
"""
    subprocess.run([sys.executable, "-c", code], cwd=THIS_DIR, check=True)


def build_command(args: argparse.Namespace) -> list[str]:
    mpi = shutil.which(args.mpi_exec) or args.mpi_exec
    command = [mpi]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        command.append("--allow-run-as-root")
    command.extend(["-n", str(N_CLIENTS)])
    for variable in MPI_EXPORTS:
        command.extend(["-x", variable])
    if args.disable_categorical_validation:
        command.extend(["-x", "EASYRL4REC_DISABLE_CATEGORICAL_VALIDATE"])
    command.extend(
        [sys.executable, "-u", str(runner_path(args.scene, args.method))]
    )
    message = f"release__{args.scene}__{args.method}__{args.run_id}"
    command.extend(command_args(args.scene, args.method))
    command.extend(
        [
            "--message", message,
            "--seed", str(args.seed),
            "--cuda", "0",
        ]
    )
    command.append("--cpu")
    if args.smoke_test:
        command.extend(
            [
                "--epoch", "1",
                "--step-per-epoch", "64",
                "--episode-per-collect", "2",
                "--training-num", "2",
                "--test-num", "2",
                "--batch-size", "64",
                "--comm_times_per_epoch", "1",
                "--quiet_progress",
                "--no_save",
            ]
        )
    return command


def find_run_dir(output_root: Path, message: str) -> Path:
    log_root = output_root / "MovieLensEnv-v0" / "PPO" / "logs"
    prefix = f"[{message}]_"
    matches = [path for path in log_root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if not matches:
        raise RuntimeError(f"no run directory was produced under {log_root}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def final_common_epoch(run_dir: Path) -> tuple[int, int]:
    per_rank: dict[int, set[int]] = {}
    for rank in range(N_CLIENTS):
        path = run_dir / f"logs_rank{rank}.log"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty rank log: {path.name}")
        epochs: set[int] = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = EPOCH_RE.search(line)
            if match and int(match.group(2)) == rank:
                epochs.add(int(match.group(1)))
        per_rank[rank] = epochs
    common = set.intersection(*(per_rank[rank] for rank in range(N_CLIENTS)))
    return (max(common) if common else 0, len(per_rank))


def write_status(run_dir: Path, args: argparse.Namespace, final_epoch: int) -> None:
    expected = 1 if args.smoke_test else 100
    payload = {
        "status": "completed" if final_epoch >= expected else "incomplete",
        "scene": args.scene,
        "method": args.method,
        "run_id": args.run_id,
        "expected_epochs": expected,
        "final_common_epoch": final_epoch,
        "mpi_ranks": N_CLIENTS,
        "seed_values_embedded": False,
        "scientific_result": not args.smoke_test,
    }
    (run_dir / "RUN_STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if final_epoch < expected:
        raise RuntimeError(
            f"run stopped at common epoch {final_epoch}; expected at least {expected}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.skip_asset_check:
        validate_assets(args.asset_root, args.scene)
    if args.audit_only:
        audit_imports()
        return 0

    command = build_command(args)
    display_command = command.copy()
    display_command[display_command.index("--seed") + 1] = "<redacted>"
    print("[easyrl4rec-release] " + shlex.join(display_command), flush=True)
    if args.dry_run:
        return 0

    movielens_root, deepfm_root = asset_paths(args.asset_root)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "EASYRL4REC_MOVIELENS_ROOT": str(movielens_root),
            "EASYRL4REC_DEEPFM_ROOT": str(deepfm_root),
            "EASYRL4REC_OUTPUT_ROOT": str(output_root),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            # Codex-modified 2026-08-14: preserved helpers inspect CUDA
            # independently of --cpu, so hide devices on the public CPU path.
            "CUDA_VISIBLE_DEVICES": "",
            # Codex-modified 2026-08-14: one parent-generated stamp keeps all
            # MPI rank logs in the same directory even when startup crosses a second.
            "EASYRL4REC_RUN_STAMP": datetime.datetime.now(
                datetime.timezone.utc
            ).strftime("%Y_%m_%d-%H_%M_%S-%f"),
        }
    )
    if args.disable_categorical_validation:
        env["EASYRL4REC_DISABLE_CATEGORICAL_VALIDATE"] = "1"
    subprocess.run(command, cwd=THIS_DIR, env=env, check=True)
    message = f"release__{args.scene}__{args.method}__{args.run_id}"
    run_dir = find_run_dir(output_root, message)
    final_epoch, ranks = final_common_epoch(run_dir)
    if ranks != N_CLIENTS:
        raise RuntimeError(f"expected {N_CLIENTS} rank logs, found {ranks}")
    write_status(run_dir, args, final_epoch)
    print(f"[easyrl4rec-release] completed: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
