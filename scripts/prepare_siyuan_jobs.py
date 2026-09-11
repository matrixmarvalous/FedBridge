#!/usr/bin/env python3
# Codex-added 2026-08-23: generate private, fixed-tuple Siyuan-1 jobs without
# embedding experiment seeds in the public release.
# Codex-modified 2026-08-25: add optional per-condition parallel jobs; the
# previous generator is backed up under .codex_hpc/backups/fedbridge_release/
# 20260825_153941/scripts/prepare_siyuan_jobs.py.
"""Prepare independent Slurm jobs for the FedBridge release on Siyuan-1.

Every generated ``.sbatch`` file fixes exactly one method and one user-supplied
seed.  By default, a method's conditions run serially in one file.  The
``--parallel-conditions`` mode instead emits one file per environment or scene,
so a full matrix becomes 30 independently schedulable jobs.  No Slurm arrays
or runtime ``--export`` values are used.  Generated files live outside the
public release because they contain the private seed selected by the runner.

This bootstrap intentionally uses only the Python standard library and remains
compatible with the older Python commonly available on login nodes.
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys


RELEASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RELEASE_ROOT.parent
MUJOCO_ROOT = RELEASE_ROOT / "mujoco"
RECOMMENDER_ROOT = RELEASE_ROOT / "recommender_system"

MUJOCO_ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v4", "Humanoid-v4")
MUJOCO_METHODS = (
    "fedbridge_singlecritic",
    "individual",
    "push_avg",
    "fedavg",
    "perfeddc",
    "pfedme",
)
RECOMMENDER_SCENES = ("hetero5", "homo3")
RECOMMENDER_METHODS = (
    "fedbridge",
    "individual",
    "pushpull",
    "pfedme",
    "perfeddc",
    "fedavg",
)

MAIN_TASKS = tuple(("mujoco", method) for method in MUJOCO_METHODS) + tuple(
    ("recommender", method) for method in RECOMMENDER_METHODS
)

LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.-]{0,47}$")
CONDA_ENV_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TIME_RE = re.compile(r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$")
MUJOCO_METHOD_SHORT = {
    "fedbridge_singlecritic": "bridge",
    "individual": "ind",
    "push_avg": "push",
    "fedavg": "avg",
    "perfeddc": "dc",
    "pfedme": "me",
    "fedbridge_legacy_dualcritic": "legacy",
}
REC_METHOD_SHORT = {
    "fedbridge": "bridge",
    "individual": "ind",
    "pushpull": "push",
    "pfedme": "me",
    "perfeddc": "dc",
    "fedavg": "avg",
}
MUJOCO_CONDITION_SHORT = {
    "HalfCheetah-v5": "hc",
    "Hopper-v4": "hop",
    "Humanoid-v4": "hum",
}
REC_CONDITION_SHORT = {"hetero5": "het", "homo3": "hom"}


class PreparationError(RuntimeError):
    """The requested private job bundle is unsafe or internally inconsistent."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed Siyuan-1 .sbatch files for one user seed; conditions "
            "are serial by default or independently schedulable when requested."
        )
    )
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="private user-selected non-negative seed (written only outside release)",
    )
    parser.add_argument(
        "--run-label",
        required=True,
        help="seed-free opaque label such as replicate-a or validation-a",
    )
    parser.add_argument(
        "--component",
        choices=("all", "mujoco", "recommender"),
        default="all",
        help=(
            "matrix subset when --task is not supplied (default: all methods; "
            "12 serial-condition jobs or 30 parallel-condition jobs)"
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="COMPONENT:METHOD",
        help=(
            "generate only this exact task; repeatable, for example "
            "mujoco:fedbridge_singlecritic"
        ),
    )
    parser.add_argument(
        "--parallel-conditions",
        action="store_true",
        help=(
            "emit one independent job per method x environment/scene (30 jobs "
            "for the complete matrix) instead of one job per method"
        ),
    )
    parser.add_argument(
        "--include-legacy-bridge-smoke",
        action="store_true",
        help=(
            "add the HalfCheetah legacy BridgePPO diagnostic; smoke profile only, "
            "not part of the 12-job manuscript matrix"
        ),
    )
    parser.add_argument(
        "--job-root",
        type=Path,
        help=(
            "new private directory for scripts/logs; default: "
            "<workspace>/.codex_hpc/fedbridge_release/PROFILE/RUN_LABEL"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "shared result root; default: <workspace>/fedbridge_hpc_results/PROFILE. "
            "Different seed bundles may share it when run labels are unique."
        ),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help=(
            "required for recommender jobs; contains data/MovieLens and "
            "saved_models/MovieLensEnv-v0/DeepFM"
        ),
    )
    parser.add_argument("--partition", default="64c512g")
    # Codex-modified 2026-09-10: public defaults match the environment files
    # and final FedBridge naming; cluster-specific names remain overridable.
    parser.add_argument("--mujoco-conda-env", default="fedbridge-mujoco")
    parser.add_argument("--recommender-conda-env", default="fedbridge-easyrl4rec")
    parser.add_argument("--conda-module", default="miniconda3/4.10.3")
    parser.add_argument("--mujoco-smoke-time", default="01:00:00")
    parser.add_argument("--mujoco-full-time", default="08:00:00")
    parser.add_argument("--recommender-smoke-time", default="01:00:00")
    parser.add_argument("--recommender-full-time", default="24:00:00")
    return parser.parse_args(argv)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def shell_quote(value):
    return shlex.quote(str(value))


def write_private(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o700 if executable else 0o600)


def validate_simple_value(value, label, pattern):
    if not pattern.fullmatch(value):
        raise PreparationError("invalid {}: {!r}".format(label, value))


def resolved_private_root(path, default):
    resolved = (path or default).expanduser().resolve()
    try:
        resolved.relative_to(RELEASE_ROOT)
    except ValueError:
        return resolved
    raise PreparationError("private generated files must be outside release: {}".format(resolved))


def ensure_no_whitespace_path(path, label):
    if any(character.isspace() for character in str(path)):
        raise PreparationError("{} may not contain whitespace: {}".format(label, path))


def parse_task(value):
    parts = value.split(":")
    if len(parts) != 2:
        raise PreparationError(
            "--task must be COMPONENT:METHOD, got {!r}".format(value)
        )
    component, method = parts
    task = (component, method)
    if task not in MAIN_TASKS:
        raise PreparationError("unsupported manuscript task: {}".format(value))
    return task


def selected_tasks(args):
    if args.task:
        tasks = [parse_task(value) for value in args.task]
    elif args.component == "all":
        tasks = list(MAIN_TASKS)
    else:
        tasks = [task for task in MAIN_TASKS if task[0] == args.component]
    if args.include_legacy_bridge_smoke:
        if args.profile != "smoke":
            raise PreparationError("--include-legacy-bridge-smoke requires --profile smoke")
        tasks.append(("mujoco", "fedbridge_legacy_dualcritic"))
    unique = []
    seen = set()
    for task in tasks:
        if task not in seen:
            unique.append(task)
            seen.add(task)
    return unique


def validate_asset_root(asset_root):
    if asset_root is None:
        raise PreparationError("recommender jobs require --asset-root")
    resolved = asset_root.expanduser().resolve()
    expected = (
        resolved / "data" / "MovieLens",
        resolved / "saved_models" / "MovieLensEnv-v0" / "DeepFM",
    )
    missing = [path for path in expected if not path.is_dir()]
    if missing:
        raise PreparationError(
            "invalid --asset-root; missing {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    return resolved


def condition_slug(condition):
    return condition.replace("-", "_").lower()


def task_id(component, method, condition=None):
    base = "{}__{}".format(component, method)
    return base if condition is None else "{}__{}".format(base, condition_slug(condition))


def job_name(component, method, run_label, condition=None):
    label = re.sub(r"[^A-Za-z0-9-]+", "-", run_label.lower()).strip("-")[:18]
    if component == "mujoco":
        pieces = ["fb-mj", MUJOCO_METHOD_SHORT[method]]
        if condition is not None:
            pieces.append(MUJOCO_CONDITION_SHORT[condition])
    else:
        pieces = ["fb-rec", REC_METHOD_SHORT[method]]
        if condition is not None:
            pieces.append(REC_CONDITION_SHORT[condition])
    pieces.append(label)
    short = "-".join(pieces)
    return short[:64].rstrip("-")


def common_header(name, wall_time, stdout_path, stderr_path):
    return [
        "#!/usr/bin/env bash",
        "# Codex-added 2026-08-23: private fixed-tuple Siyuan-1 experiment job.",
        "# PRIVATE: this generated file contains a user-selected seed; do not publish it.",
        "# One file = one algorithm x one random seed x one condition group.",
        "#SBATCH --job-name={}".format(name),
        "#SBATCH --partition=64c512g",  # Replaced below when a custom partition is used.
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=8",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --time={}".format(wall_time),
        "#SBATCH --output={}".format(stdout_path),
        "#SBATCH --error={}".format(stderr_path),
        "",
        "set -euo pipefail",
        "umask 077",
        "ulimit -c 0 || true",
        "",
    ]


def activation_lines(conda_module, conda_env):
    return [
        "module load {}".format(shell_quote(conda_module)),
        "set +u",
        "if [[ -f \"${HOME}/.bashrc\" ]]; then",
        "  source \"${HOME}/.bashrc\"",
        "fi",
        "source activate {}".format(shell_quote(conda_env)),
        "set -u",
        "if [[ \"${{CONDA_DEFAULT_ENV:-}}\" != {} ]]; then".format(
            shell_quote(conda_env)
        ),
        "  echo \"ERROR: expected Conda environment {}, got ${{CONDA_DEFAULT_ENV:-none}}\" >&2".format(
            conda_env
        ),
        "  exit 20",
        "fi",
        "echo \"[fedbridge-hpc] python=$(which python)\"",
        "python -V",
        "",
    ]


def marker_lines(marker_dir, identifier):
    marker_base = marker_dir / identifier
    return [
        "job_marker={}".format(shell_quote(str(marker_base))),
        # Codex-modified 2026-08-23: the former EXIT trap could report zero
        # when Slurm killed the batch shell at its time limit.  A completion
        # marker is now written atomically only after every application command.
        "record_success() {",
        "  marker=\"${job_marker}.${SLURM_JOB_ID:-unknown}.status\"",
        "  marker_tmp=\"${marker}.tmp.$$\"",
        "  printf 'status=completed\\nexit_code=0\\nfinished=%s\\n' \"$(date -Is)\" > \"${marker_tmp}\"",
        "  mv \"${marker_tmp}\" \"${marker}\"",
        "}",
        "",
    ]


def mujoco_script(
    args, method, name, identifier, job_root, output_root, conditions=None
):
    conditions = tuple(conditions or MUJOCO_ENVIRONMENTS)
    environment_values = " ".join(shell_quote(value) for value in conditions)
    logs = job_root / "logs"
    wall_time = args.mujoco_smoke_time if args.profile == "smoke" else args.mujoco_full_time
    lines = common_header(
        name,
        wall_time,
        logs / ("%x-%j.out"),
        logs / ("%x-%j.err"),
    )
    lines[5] = "#SBATCH --partition={}".format(args.partition)
    lines.extend(marker_lines(job_root / "markers", identifier))
    lines.extend(
        [
            "readonly COMPONENT_ROOT={}".format(shell_quote(str(MUJOCO_ROOT))),
            "readonly OUTPUT_ROOT={}".format(shell_quote(str(output_root / "mujoco"))),
            "readonly METHOD={}".format(shell_quote(method)),
            "readonly PRIVATE_EXPERIMENT_SEED={}".format(args.seed),
            "echo \"[fedbridge-hpc] component=mujoco profile={} method={} environments={} seed=redacted\"".format(
                args.profile, method, len(conditions)
            ),
            "cd \"${COMPONENT_ROOT}\"",
            "mkdir -p \"${OUTPUT_ROOT}\"",
            "export OMP_NUM_THREADS=\"${SLURM_CPUS_PER_TASK:-2}\"",
            "export OPENBLAS_NUM_THREADS=\"${SLURM_CPUS_PER_TASK:-2}\"",
            "export MKL_NUM_THREADS=\"${SLURM_CPUS_PER_TASK:-2}\"",
            "export NUMEXPR_NUM_THREADS=\"${SLURM_CPUS_PER_TASK:-2}\"",
            "export PYTHONUNBUFFERED=1",
            "export CUDA_VISIBLE_DEVICES=",
            "export I_MPI_FABRICS=shm",
            "export TMPDIR=\"${SLURM_TMPDIR:-/tmp}\"",
            "",
        ]
    )
    lines.extend(activation_lines(args.conda_module, args.mujoco_conda_env))
    lines.extend(
        [
            "command -v mpirun",
            "python -c 'import gymnasium, mpi4py, mujoco, torch; print(\"[fedbridge-hpc] MuJoCo imports OK\")'",
            "environments=({})".format(environment_values),
            "for environment in \"${environments[@]}\"; do",
            "  case \"${environment}\" in",
            "    HalfCheetah-v5) environment_slug=halfcheetah_v5 ;;",
            "    Hopper-v4) environment_slug=hopper_v4 ;;",
            "    Humanoid-v4) environment_slug=humanoid_v4 ;;",
            "    *) echo \"ERROR: unexpected environment ${environment}\" >&2; exit 22 ;;",
            "  esac",
            "  run_dir=\"${OUTPUT_ROOT}/${environment_slug}__${METHOD}__seed_${PRIVATE_EXPERIMENT_SEED}\"",
            "  if [[ -e \"${run_dir}\" ]]; then",
            "    echo \"ERROR: refusing to reuse existing ${environment}/${METHOD} result (seed redacted)\" >&2",
            "    exit 21",
            "  fi",
            "done",
            "",
            "for environment in \"${environments[@]}\"; do",
            "  case \"${environment}\" in",
            "    HalfCheetah-v5) environment_slug=halfcheetah_v5 ;;",
            "    Hopper-v4) environment_slug=hopper_v4 ;;",
            "    Humanoid-v4) environment_slug=humanoid_v4 ;;",
            "  esac",
            "  run_dir=\"${OUTPUT_ROOT}/${environment_slug}__${METHOD}__seed_${PRIVATE_EXPERIMENT_SEED}\"",
            "  echo \"[fedbridge-hpc] starting environment=${environment} method=${METHOD}\"",
            "  train_args=(",
            "    --env-name \"${environment}\"",
            "    --method \"${METHOD}\"",
            "    --seed \"${PRIVATE_EXPERIMENT_SEED}\"",
            "    --output-root \"${OUTPUT_ROOT}\"",
            "  )",
        ]
    )
    if args.profile == "smoke":
        lines.append("  train_args+=(--smoke-test)")
    lines.extend(
        [
            "  mpirun -n \"${SLURM_NTASKS:-8}\" python launcher.py \"${train_args[@]}\"",
            "  eval_args=(",
            "    --env-name \"${environment}\"",
            "    --method \"${METHOD}\"",
            "    --seed \"${PRIVATE_EXPERIMENT_SEED}\"",
            "    --run-dir \"${run_dir}\"",
            "  )",
        ]
    )
    if args.profile == "smoke":
        lines.extend(
            [
                "  eval_args+=(--episodes 1 --repeats 1 --horizon 32)",
                "  mpirun -n \"${SLURM_NTASKS:-8}\" python evaluate_mujoco.py \"${eval_args[@]}\"",
                "  mpirun -n \"${SLURM_NTASKS:-8}\" python evaluate_mujoco.py \"${eval_args[@]}\" \\",
                "    --output \"${run_dir}/determinism_repeat/heldout_eval_recomputed.csv\"",
                "  cmp \"${run_dir}/heldout_eval_recomputed.csv\" \\",
                "    \"${run_dir}/determinism_repeat/heldout_eval_recomputed.csv\"",
            ]
        )
    else:
        lines.append(
            "  mpirun -n \"${SLURM_NTASKS:-8}\" python evaluate_mujoco.py \"${eval_args[@]}\""
        )
    lines.extend(
        [
            "  echo \"[fedbridge-hpc] completed environment=${environment} method=${METHOD}\"",
            "done",
            "echo \"[fedbridge-hpc] application completed\"",
            "record_success",
            "",
        ]
    )
    return "\n".join(lines)


def recommender_script(
    args, method, name, identifier, job_root, output_root, asset_root, conditions=None
):
    conditions = tuple(conditions or RECOMMENDER_SCENES)
    scene_values = " ".join(shell_quote(value) for value in conditions)
    logs = job_root / "logs"
    wall_time = (
        args.recommender_smoke_time
        if args.profile == "smoke"
        else args.recommender_full_time
    )
    lines = common_header(
        name,
        wall_time,
        logs / ("%x-%j.out"),
        logs / ("%x-%j.err"),
    )
    lines[5] = "#SBATCH --partition={}".format(args.partition)
    lines.extend(marker_lines(job_root / "markers", identifier))
    lines.extend(
        [
            "readonly COMPONENT_ROOT={}".format(shell_quote(str(RECOMMENDER_ROOT))),
            "readonly OUTPUT_ROOT={}".format(
                shell_quote(str(output_root / "recommender_system"))
            ),
            "readonly ASSET_ROOT={}".format(shell_quote(str(asset_root))),
            "readonly METHOD={}".format(shell_quote(method)),
            "readonly RUN_ID={}".format(shell_quote(args.run_label)),
            "readonly PRIVATE_EXPERIMENT_SEED={}".format(args.seed),
            "echo \"[fedbridge-hpc] component=recommender profile={} method={} scenes={} run_id={} seed=redacted\"".format(
                args.profile, method, len(conditions), args.run_label
            ),
            "cd \"${COMPONENT_ROOT}\"",
            "mkdir -p \"${OUTPUT_ROOT}\"",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "export PYTHONUNBUFFERED=1",
            "export CUDA_VISIBLE_DEVICES=",
            "export OMPI_MCA_pml=ob1",
            "export OMPI_MCA_btl=self,sm,tcp",
            "export TMPDIR=\"${SLURM_TMPDIR:-/tmp}\"",
            "",
        ]
    )
    lines.extend(activation_lines(args.conda_module, args.recommender_conda_env))
    lines.extend(
        [
            "command -v mpiexec",
            "python -c 'import mpi4py, torch; print(\"[fedbridge-hpc] recommender imports OK\")'",
            "python scripts/validate_assets.py --asset-root \"${ASSET_ROOT}\"",
            "scenes=({})".format(scene_values),
            "for scene in \"${scenes[@]}\"; do",
            "  if find \"${OUTPUT_ROOT}/MovieLensEnv-v0/PPO/logs\" -maxdepth 1 -type d \\",
            "      -name \"[[]release__${scene}__${METHOD}__${RUN_ID}]_*\" -print -quit 2>/dev/null | grep -q .; then",
            "    echo \"ERROR: refusing to reuse existing ${scene}/${METHOD}/${RUN_ID} result\" >&2",
            "    exit 21",
            "  fi",
            "done",
            "",
            "for scene in \"${scenes[@]}\"; do",
            "  echo \"[fedbridge-hpc] starting scene=${scene} method=${METHOD}\"",
            "  run_args=(",
            "    --scene \"${scene}\"",
            "    --method \"${METHOD}\"",
            "    --seed \"${PRIVATE_EXPERIMENT_SEED}\"",
            "    --run-id \"${RUN_ID}\"",
            "    --asset-root \"${ASSET_ROOT}\"",
            "    --output-root \"${OUTPUT_ROOT}\"",
            "    --device cpu",
            "  )",
        ]
    )
    if args.profile == "smoke":
        lines.append("  run_args+=(--smoke-test)")
    lines.extend(
        [
            "  # launcher.py starts exactly eight MPI ranks itself; no outer srun/mpirun.",
            "  python launcher.py \"${run_args[@]}\"",
            "  echo \"[fedbridge-hpc] completed scene=${scene} method=${METHOD}\"",
            "done",
            "echo \"[fedbridge-hpc] application completed\"",
            "record_success",
            "",
        ]
    )
    return "\n".join(lines)


def submission_script(job_root):
    return """#!/usr/bin/env bash
# Codex-added 2026-08-23: submit independent fixed-tuple jobs (never an array).
set -euo pipefail
umask 077

job_root={job_root}
order_file="${{job_root}}/SUBMIT_ORDER.tsv"
ledger="${{job_root}}/JOB_IDS.tsv"
command -v sbatch >/dev/null || {{
  echo "ERROR: sbatch is unavailable; run this on a Siyuan login node." >&2
  exit 2
}}

if [[ ! -f "${{ledger}}" ]]; then
  printf 'job_id\\ttask_id\\tscript\\tsubmitted_at\\n' > "${{ledger}}"
fi

submitted=0
skipped=0
while IFS=$'\\t' read -r task script; do
  [[ "${{task}}" == "task_id" ]] && continue
  if awk -F '\\t' -v wanted="${{task}}" 'NR > 1 && $2 == wanted {{ found=1 }} END {{ exit(found ? 0 : 1) }}' "${{ledger}}"; then
    echo "[skip] already submitted: ${{task}}"
    skipped=$((skipped + 1))
    continue
  fi
  response="$(sbatch --parsable "${{script}}")"
  job_id="${{response%%;*}}"
  case "${{job_id}}" in
    ''|*[!0-9]*) echo "ERROR: unexpected sbatch response: ${{response}}" >&2; exit 3 ;;
  esac
  printf '%s\\t%s\\t%s\\t%s\\n' \
    "${{job_id}}" "${{task}}" "${{script}}" "$(date -Is)" >> "${{ledger}}"
  echo "[submitted] job_id=${{job_id}} task=${{task}}"
  submitted=$((submitted + 1))
done < "${{order_file}}"

echo "Submitted ${{submitted}} independent jobs; skipped ${{skipped}} already recorded jobs."
echo "Ledger: ${{ledger}}"
""".format(job_root=shell_quote(str(job_root)))


def status_script(job_root):
    return """#!/usr/bin/env bash
# Codex-added 2026-08-23: inspect scheduler state for generated fixed-tuple jobs.
set -euo pipefail

job_root={job_root}
ledger="${{job_root}}/JOB_IDS.tsv"
[[ -s "${{ledger}}" ]] || {{ echo "ERROR: no submitted-job ledger: ${{ledger}}" >&2; exit 2; }}
job_ids="$(awk -F '\\t' 'NR > 1 {{ print $1 }}' "${{ledger}}" | paste -sd, -)"
[[ -n "${{job_ids}}" ]] || {{ echo "ERROR: ledger contains no job IDs" >&2; exit 2; }}

echo "== Active or pending jobs =="
squeue -j "${{job_ids}}" -o '%.18i %.48j %.9T %.10M %.30R' || true
echo
echo "== Accounting records =="
sacct -j "${{job_ids}}" \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,MaxRSS \
  --parsable2
""".format(job_root=shell_quote(str(job_root)))


def main(argv=None):
    os.umask(0o077)
    args = parse_args(argv)
    if args.seed < 0:
        raise PreparationError("--seed must be non-negative")
    validate_simple_value(args.run_label, "--run-label", LABEL_RE)
    if str(args.seed) in args.run_label:
        raise PreparationError(
            "--run-label appears to contain the private seed; use a label such as replicate-a"
        )
    for value, label in (
        (args.mujoco_conda_env, "--mujoco-conda-env"),
        (args.recommender_conda_env, "--recommender-conda-env"),
    ):
        validate_simple_value(value, label, CONDA_ENV_RE)
    for value, label in (
        (args.mujoco_smoke_time, "--mujoco-smoke-time"),
        (args.mujoco_full_time, "--mujoco-full-time"),
        (args.recommender_smoke_time, "--recommender-smoke-time"),
        (args.recommender_full_time, "--recommender-full-time"),
    ):
        validate_simple_value(value, label, TIME_RE)
    if not CONDA_ENV_RE.fullmatch(args.partition):
        raise PreparationError("invalid --partition: {!r}".format(args.partition))

    tasks = selected_tasks(args)
    includes_recommender = any(task[0] == "recommender" for task in tasks)
    asset_root = validate_asset_root(args.asset_root) if includes_recommender else None
    job_root = resolved_private_root(
        args.job_root,
        WORKSPACE_ROOT
        / ".codex_hpc"
        / "fedbridge_release"
        / args.profile
        / args.run_label,
    )
    output_root = resolved_private_root(
        args.output_root,
        WORKSPACE_ROOT / "fedbridge_hpc_results" / args.profile,
    )
    for path, label in (
        (job_root, "job root"),
        (output_root, "output root"),
        (RELEASE_ROOT, "release root"),
    ):
        ensure_no_whitespace_path(path, label)
    if asset_root is not None:
        ensure_no_whitespace_path(asset_root, "asset root")
    if job_root.exists() and any(job_root.iterdir()):
        raise PreparationError("--job-root must be new or empty: {}".format(job_root))
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / "jobs").mkdir()
    (job_root / "logs").mkdir()
    (job_root / "markers").mkdir()
    output_root.mkdir(parents=True, exist_ok=True)

    entries = []
    submit_rows = []
    for component, method in tasks:
        if component == "mujoco":
            all_conditions = list(MUJOCO_ENVIRONMENTS)
            if method == "fedbridge_legacy_dualcritic":
                all_conditions = ["HalfCheetah-v5"]
        else:
            all_conditions = list(RECOMMENDER_SCENES)
        condition_groups = (
            [[condition] for condition in all_conditions]
            if args.parallel_conditions
            else [all_conditions]
        )
        for conditions in condition_groups:
            split_condition = conditions[0] if args.parallel_conditions else None
            identifier = task_id(component, method, split_condition)
            name = job_name(component, method, args.run_label, split_condition)
            script_path = job_root / "jobs" / (
                identifier + "__" + args.run_label + ".sbatch"
            )
            if component == "mujoco":
                result_globs = [
                    "{}__{}__seed_*".format(condition_slug(condition), method)
                    for condition in conditions
                ]
                script = mujoco_script(
                    args,
                    method,
                    name,
                    identifier,
                    job_root,
                    output_root,
                    conditions,
                )
                expected_prefixes = []
            else:
                result_globs = []
                expected_prefixes = [
                    "[release__{}__{}__{}]_".format(
                        condition, method, args.run_label
                    )
                    for condition in conditions
                ]
                script = recommender_script(
                    args,
                    method,
                    name,
                    identifier,
                    job_root,
                    output_root,
                    asset_root,
                    conditions,
                )
            write_private(script_path, script, executable=True)
            entry = {
                "component": component,
                "conditions": conditions,
                "method": method,
                "task_id": identifier,
                "job_name": name,
                "script": str(script_path),
                "result_globs": result_globs,
                "result_prefixes": expected_prefixes,
            }
            entries.append(entry)
            submit_rows.append((identifier, str(script_path)))

    seed_fingerprint = hashlib.sha256(str(args.seed).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "private_generated_bundle": True,
        "seed_is_stored_only_in_generated_sbatch_files": True,
        "seed_sha256": seed_fingerprint,
        "profile": args.profile,
        # Codex-added 2026-08-25: record whether condition cells were emitted
        # as independently schedulable jobs for maximum cluster parallelism.
        "parallel_conditions": args.parallel_conditions,
        "job_granularity": (
            "method_condition_seed" if args.parallel_conditions else "method_seed"
        ),
        "run_label": args.run_label,
        "release_root": str(RELEASE_ROOT),
        "job_root": str(job_root),
        "output_root": str(output_root),
        "recommender_asset_root": str(asset_root) if asset_root is not None else None,
        "mujoco_conda_env": args.mujoco_conda_env,
        "recommender_conda_env": args.recommender_conda_env,
        "partition": args.partition,
        "resources_per_job": {
            "nodes": 1,
            "mpi_ranks": 8,
            "cpus_per_rank": 2,
            "total_cpu_cores": 16,
            "gpus": 0,
            "exclusive": False,
        },
        "jobs": entries,
    }
    write_private(
        job_root / "JOB_MANIFEST.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    order_lines = ["task_id\tscript"] + ["{}\t{}".format(*row) for row in submit_rows]
    write_private(job_root / "SUBMIT_ORDER.tsv", "\n".join(order_lines) + "\n")
    write_private(job_root / "submit_all.sh", submission_script(job_root), executable=True)
    write_private(job_root / "status_all.sh", status_script(job_root), executable=True)

    print("Prepared {} independent fixed-tuple Slurm jobs.".format(len(entries)))
    print("Private job root: {}".format(job_root))
    print("Shared result root: {}".format(output_root))
    print("Seed: redacted (SHA-256 recorded only for bundle matching)")
    print("Submit one job: sbatch {}".format(entries[0]["script"]))
    print("Submit this bundle: bash {}".format(job_root / "submit_all.sh"))
    print("Monitor this bundle: bash {}".format(job_root / "status_all.sh"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
