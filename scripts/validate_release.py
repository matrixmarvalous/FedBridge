#!/usr/bin/env python3
# Codex-added 2026-08-21: one-command, fail-fast validation entry point for
# the isolated FedBridge release. Intended caller: repository users from any
# working directory; original experiment trees are read only.
# Codex-modified 2026-08-23: validate fixed-method/seed Siyuan job generation.
"""Validate the public FedBridge release without launching full experiments.

The default run performs packaging, unit, import, command-construction, asset
layout and (off login nodes) eight-rank MuJoCo construction checks. ``--smoke``
adds representative tiny end-to-end runs for both components and must be used
on a compute node or workstation. Full scientific training is deliberately out
of scope.

This bootstrap file stays compatible with Python 3.6 so that it can discover
and invoke the two newer, component-specific Conda interpreters itself.
"""

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time


RELEASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RELEASE_ROOT.parent
MUJOCO_ROOT = RELEASE_ROOT / "mujoco"
RECOMMENDER_ROOT = RELEASE_ROOT / "recommender_system"
N_RANKS = 8
VALIDATION_SEED = 0  # Synthetic structural-test value; never a paper-run seed.
SCENES = ("hetero5", "homo3")
METHODS = ("fedbridge", "individual", "push_avg", "pfedme", "perfeddc", "fedavg")


class ValidationError(RuntimeError):
    """A validation stage failed or could not be run reliably."""


class Runner(object):
    def __init__(self):
        self.passed = []
        self.skipped = []

    def run(self, label, command, cwd, env, quiet=False, timeout=900):
        print("[RUN ] {}".format(label), flush=True)
        started = time.monotonic()
        kwargs = {
            "cwd": str(cwd),
            "env": env,
        }
        if quiet:
            kwargs.update(
                {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "universal_newlines": True,
                }
            )
        process = subprocess.Popen(
            [str(value) for value in command], start_new_session=True, **kwargs
        )
        output = None
        try:
            output, _ = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, AttributeError):
                process.terminate()
            try:
                trailing, _ = process.communicate(timeout=5)
                if output is None:
                    output = trailing
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (OSError, AttributeError):
                    process.kill()
                trailing, _ = process.communicate()
                if output is None:
                    output = trailing
            if quiet and output:
                print(output, file=sys.stderr, end="")
            if isinstance(error, KeyboardInterrupt):
                raise
            raise ValidationError("{} timed out after {} seconds".format(label, timeout))
        elapsed = time.monotonic() - started
        if process.returncode != 0:
            if quiet and output:
                print(output, file=sys.stderr, end="")
            raise ValidationError(
                "{} failed with exit code {}".format(label, process.returncode)
            )
        self.passed.append(label)
        print("[PASS] {} ({:.1f}s)".format(label, elapsed), flush=True)

    def check(self, label, callback):
        print("[RUN ] {}".format(label), flush=True)
        started = time.monotonic()
        result = callback()
        self.passed.append(label)
        print("[PASS] {} ({:.1f}s)".format(label, time.monotonic() - started), flush=True)
        return result

    def skip(self, label, reason):
        self.skipped.append((label, reason))
        print("[SKIP] {}: {}".format(label, reason), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run fail-fast structural checks for both FedBridge release components; "
            "optionally add representative eight-rank end-to-end smoke runs."
        )
    )
    parser.add_argument("--mujoco-python", type=Path)
    parser.add_argument("--recommender-python", type=Path)
    parser.add_argument("--mujoco-mpi", type=Path)
    parser.add_argument("--recommender-mpi", type=Path)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="root containing data/MovieLens and saved_models/.../DeepFM",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="optional protected EasyRL4Rec source tree for source-manifest comparison",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        help="new empty directory in which to retain validation outputs",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="add representative tiny eight-rank rollouts (compute node/workstation only)",
    )
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-source-manifest", action="store_true")
    parser.add_argument("--skip-mpi-audit", action="store_true")
    parser.add_argument(
        "--allow-login-mpi-audit",
        action="store_true",
        help="allow construction-only MPI audit on a recognized login/data host",
    )
    return parser.parse_args(argv)


def subprocess_environment(python_path):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    executable_dir = str(Path(python_path).resolve().parent)
    env["PATH"] = executable_dir + os.pathsep + env.get("PATH", "")
    return env


def harden_process():
    """Keep validation artifacts private and disable large core dumps."""
    os.umask(0o077)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


def conda_prefixes():
    conda = shutil.which("conda")
    if not conda:
        return {}
    try:
        completed = subprocess.run(
            [conda, "env", "list", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    try:
        prefixes = json.loads(completed.stdout).get("envs", [])
    except (TypeError, ValueError):
        return {}
    return {Path(prefix).name: Path(prefix) for prefix in prefixes}


def validate_executable(path, label):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or not os.access(str(resolved), os.X_OK):
        raise ValidationError("{} is not executable: {}".format(label, resolved))
    return resolved


def python_passes_probe(candidate, probe_code):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    try:
        completed = subprocess.run(
            [str(candidate), "-c", probe_code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def resolve_python(
    override, environment_variables, preferred_names, prefixes, label, probe_code
):
    value = override
    if not value:
        for variable in environment_variables:
            value = os.environ.get(variable)
            if value:
                break
    if value:
        return validate_executable(value, label)
    for name in preferred_names:
        prefix = prefixes.get(name)
        if prefix is not None:
            candidate = prefix / "bin" / "python"
            if candidate.is_file() and python_passes_probe(candidate, probe_code):
                return validate_executable(candidate, label)
    active_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if active_name in preferred_names and python_passes_probe(sys.executable, probe_code):
        return validate_executable(sys.executable, label)
    for prefix in sorted(prefixes.values(), key=lambda path: str(path)):
        candidate = prefix / "bin" / "python"
        if candidate.is_file() and python_passes_probe(candidate, probe_code):
            return validate_executable(candidate, label)
    if python_passes_probe(sys.executable, probe_code):
        return validate_executable(sys.executable, label)
    raise ValidationError(
        "could not locate {}; create the documented Conda environment or pass {}"
        .format(label, "--" + label.replace(" ", "-").lower())
    )


def resolve_mpi(override, python_path, label):
    if override:
        return validate_executable(override, label)
    executable_dir = Path(python_path).resolve().parent
    for name in ("mpiexec.hydra", "mpiexec", "mpirun"):
        candidate = executable_dir / name
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate.resolve()
    raise ValidationError(
        "could not locate {} beside {}; pass an explicit matching MPI executable"
        .format(label, python_path)
    )


def mpi_environment(base, mpi_executable):
    """Select a single-node transport without mixing MPI installations."""
    env = base.copy()
    mpi_executable = Path(mpi_executable)
    env["PATH"] = str(mpi_executable.parent) + os.pathsep + env.get("PATH", "")
    if mpi_executable.name == "mpiexec.hydra":
        env["I_MPI_FABRICS"] = "shm"
        return env
    try:
        completed = subprocess.run(
            [str(mpi_executable), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValidationError("could not inspect MPI implementation: {}".format(error))
    version = completed.stdout or ""
    if "Open MPI" in version or "OpenRTE" in version:
        major = 4
        for token in version.replace("/", " ").split():
            if token and token[0].isdigit() and "." in token:
                try:
                    major = int(token.split(".", 1)[0])
                    break
                except ValueError:
                    pass
        env["OMPI_MCA_pml"] = "ob1"
        env["OMPI_MCA_btl"] = "self,sm,tcp" if major >= 5 else "self,vader,tcp"
    return env


def mpi_command(mpi_executable, ranks, *arguments):
    command = [Path(mpi_executable)]
    if (
        hasattr(os, "geteuid")
        and os.geteuid() == 0
        and Path(mpi_executable).name != "mpiexec.hydra"
    ):
        command.append("--allow-run-as-root")
    command.extend(["-n", str(ranks)])
    command.extend(arguments)
    return command


def looks_like_login_host():
    hostname = socket.gethostname().lower()
    return not os.environ.get("SLURM_JOB_ID") and any(
        token in hostname for token in ("login", "sydata")
    )


def require_smoke_execution_context():
    if looks_like_login_host():
        raise ValidationError(
            "--smoke is refused on a recognized login/data host; run it inside a compute allocation"
        )
    if not os.environ.get("SLURM_JOB_ID"):
        return
    try:
        nodes = os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES")
        if not nodes or int(nodes) != 1:
            raise ValidationError("--smoke requires exactly one Slurm node")
        step_tasks = os.environ.get("SLURM_STEP_NUM_TASKS")
        if step_tasks and int(step_tasks) != 1:
            raise ValidationError(
                "run one validator process, not one copy per srun task"
            )
        process_id = os.environ.get("SLURM_PROCID")
        if process_id and int(process_id) != 0:
            raise ValidationError("nested multi-task srun invocation is not supported")
        allocated_tasks = os.environ.get("SLURM_NTASKS") or os.environ.get("SLURM_NPROCS")
        if not allocated_tasks or int(allocated_tasks) < N_RANKS:
            raise ValidationError("--smoke requires at least eight allocated Slurm tasks")
    except ValueError:
        raise ValidationError("invalid integer value in Slurm allocation metadata")


def paths_overlap(left, right):
    left = Path(left).resolve()
    right = Path(right).resolve()
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def allocate_work_root(requested, protected_roots):
    temporary = requested is None
    if temporary:
        root = Path(tempfile.mkdtemp(prefix="fedbridge-release-validation-")).resolve()
    else:
        root = requested.expanduser().resolve()
    for protected in protected_roots:
        if protected is not None and paths_overlap(root, protected):
            if temporary:
                root.rmdir()
            raise ValidationError(
                "validation work root overlaps protected tree {}: {}"
                .format(Path(protected).resolve(), root)
            )
    try:
        root.relative_to(RELEASE_ROOT)
    except ValueError:
        pass
    else:
        raise ValidationError("--work-root must be outside the release tree: {}".format(root))
    if root.exists() and any(root.iterdir()):
        raise ValidationError("--work-root must be new or empty: {}".format(root))
    root.mkdir(parents=True, exist_ok=True)
    return root, temporary


def find_original_root(explicit):
    if explicit:
        resolved = explicit.expanduser().resolve()
        if not (resolved / "src").is_dir() or not (resolved / "examples").is_dir():
            raise ValidationError("invalid --source-root: {}".format(resolved))
        return resolved
    candidate = WORKSPACE_ROOT / "EasyRL4Rec"
    if (candidate / "src").is_dir() and (candidate / "examples").is_dir():
        return candidate.resolve()
    return None


def find_asset_root(explicit):
    value = (
        explicit
        or os.environ.get("FEDBRIDGE_RECOMMENDER_ASSET_ROOT")
        or os.environ.get("FEDBRIDGE_REC_ASSET_ROOT")
    )
    candidates = [Path(value).expanduser()] if value else [WORKSPACE_ROOT / "EasyRL4Rec"]
    for candidate in candidates:
        resolved = candidate.resolve()
        if (
            (resolved / "data" / "MovieLens").is_dir()
            and (resolved / "saved_models" / "MovieLensEnv-v0" / "DeepFM").is_dir()
        ):
            return resolved
    if value:
        raise ValidationError("invalid recommender asset root: {}".format(Path(value).expanduser()))
    return None


def asset_snapshot(asset_root):
    """Record metadata for every staged asset file without copying its content."""
    snapshot = {}
    bases = (
        Path(asset_root) / "data" / "MovieLens",
        Path(asset_root) / "saved_models" / "MovieLensEnv-v0" / "DeepFM",
    )
    for base in bases:
        for path in sorted(base.rglob("*")):
            if path.is_file():
                stat = path.stat()
                relative = path.relative_to(asset_root).as_posix()
                snapshot[relative] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def assert_release_clean():
    artifacts = []
    for path in RELEASE_ROOT.rglob("*"):
        if path.name in (
            "__pycache__",
            ".pytest_cache",
            ".ipynb_checkpoints",
            ".mypy_cache",
            ".ruff_cache",
        ) or path.suffix in (".pyc", ".pyo"):
            artifacts.append(path.relative_to(RELEASE_ROOT).as_posix())
    if artifacts:
        raise ValidationError("generated Python cache artifacts found: {}".format(artifacts))


def assert_files_equal(left, right):
    if Path(left).read_bytes() != Path(right).read_bytes():
        raise ValidationError("files differ: {} and {}".format(left, right))


def assert_snapshots_equal(before, after):
    if before != after:
        raise ValidationError("staged asset metadata changed during smoke")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValidationError("invalid JSON {}: {}".format(path, error))


def validate_audit_output(output_root):
    statuses = list(Path(output_root).rglob("STATUS.json"))
    if len(statuses) != 1:
        raise ValidationError("expected one MuJoCo audit STATUS.json, found {}".format(len(statuses)))
    status = read_json(statuses[0])
    if status.get("status") != "audit_only_completed":
        raise ValidationError("MuJoCo audit did not complete: {}".format(status))
    run_dir = statuses[0].parent
    config = read_json(run_dir / "run_config.json")
    contract = read_json(run_dir / "architecture_manifest.json")
    if (
        config.get("method_key") != "fedbridge_legacy_dualcritic"
        or config.get("environment") != "HalfCheetah-v5"
        or config.get("n_clients") != N_RANKS
        or contract.get("status") != "passed"
        or contract.get("method_key") != "fedbridge_legacy_dualcritic"
    ):
        raise ValidationError("MuJoCo audit metadata does not match BridgePPO")


def finite_float(value, field):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValidationError("non-numeric {} value: {!r}".format(field, value))
    if not math.isfinite(result):
        raise ValidationError("non-finite {} value: {!r}".format(field, value))
    return result


def csv_rows(path):
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise ValidationError("could not read {}: {}".format(path, error))


def validate_mujoco_smoke_output(output_root, expected_method):
    statuses = list(Path(output_root).rglob("STATUS.json"))
    if len(statuses) != 1:
        raise ValidationError("expected one MuJoCo smoke STATUS.json, found {}".format(len(statuses)))
    run_dir = statuses[0].parent
    status = read_json(statuses[0])
    if status.get("status") != "completed" or status.get("total_local_steps_per_client") != 64:
        raise ValidationError("unexpected MuJoCo smoke status: {}".format(status))
    if int(status.get("communication_rounds", 0)) <= 0:
        raise ValidationError("MuJoCo smoke performed no communication rounds")
    config = read_json(run_dir / "run_config.json")
    if (
        config.get("environment") != "HalfCheetah-v5"
        or config.get("method_key") != expected_method
        or config.get("n_clients") != N_RANKS
    ):
        raise ValidationError("unexpected MuJoCo run_config.json: {}".format(config))
    contract = read_json(run_dir / "architecture_manifest.json")
    if contract.get("status") != "passed" or contract.get("method_key") != expected_method:
        raise ValidationError("unexpected MuJoCo architecture contract: {}".format(contract))
    architecture = contract.get("architecture", {})
    if expected_method == "fedbridge_singlecritic" and (
        architecture.get("learned_critic_count") != 1
        or architecture.get("bridge_critic_count") != 0
        or contract.get("communicated_object") != "bridge_actor_only"
    ):
        raise ValidationError("single-critic architecture contract failed")
    if expected_method == "fedbridge_legacy_dualcritic" and (
        architecture.get("learned_critic_count") != 2
        or architecture.get("bridge_critic_count") != 1
        or architecture.get("checkpoint_contains_bridge_critic") is not True
        or contract.get("communicated_object") != "legacy_bridge_actor_and_critic"
    ):
        raise ValidationError("legacy BridgePPO architecture contract failed")
    checkpoints = list((run_dir / "checkpoints").glob("client_*/final_policy.pth"))
    expected_clients = {"client_{}".format(client) for client in range(N_RANKS)}
    if (
        {path.parent.name for path in checkpoints} != expected_clients
        or any(path.stat().st_size <= 0 for path in checkpoints)
    ):
        raise ValidationError("MuJoCo smoke did not produce eight non-empty checkpoints")
    communication = csv_rows(run_dir / "communication_log.csv")
    if not communication:
        raise ValidationError("MuJoCo communication_log.csv is empty")
    if {int(row["client_id"]) for row in communication} != set(range(N_RANKS)):
        raise ValidationError("MuJoCo communication rows do not cover all eight clients")
    if any(finite_float(row["payload_bytes"], "payload_bytes") <= 0 for row in communication):
        raise ValidationError("MuJoCo smoke contains a non-positive communication payload")
    messages = sum(
        finite_float(row["messages_sent"], "messages_sent")
        + finite_float(row["messages_received"], "messages_received")
        for row in communication
    )
    if messages <= 0:
        raise ValidationError("MuJoCo smoke recorded no sent or received messages")
    training = csv_rows(run_dir / "training_log.csv")
    finite_returns = [
        finite_float(row["return_mean"], "training return_mean")
        for row in training
        if row.get("return_mean") not in (None, "")
    ]
    if not finite_returns:
        raise ValidationError("MuJoCo smoke contains no finite training return")
    if {int(row["client_id"]) for row in training} != set(range(N_RANKS)):
        raise ValidationError("MuJoCo training rows do not cover all eight clients")
    metric_fields = (
        "personal_policy_loss",
        "value_loss",
        "bridge_policy_loss",
        "personal_to_bridge_regularizer",
        "bridge_to_personal_regularizer",
    )
    for row in training:
        for field in metric_fields:
            if row.get(field) not in (None, ""):
                finite_float(row[field], field)
    heldout = csv_rows(run_dir / "heldout_eval.csv")
    if len(heldout) != 48:
        raise ValidationError("expected 48 personal+bridge held-out rows, found {}".format(len(heldout)))
    if Counter(row.get("policy_type") for row in heldout) != Counter(
        {"personal": 24, "bridge": 24}
    ):
        raise ValidationError("held-out rows do not contain 24 personal and 24 bridge policies")
    heldout_keys = [
        (
            row.get("client_id"),
            row.get("policy_type"),
            row.get("target_cluster"),
            row.get("repeat"),
            row.get("episode"),
        )
        for row in heldout
    ]
    if len(set(heldout_keys)) != 48:
        raise ValidationError("MuJoCo held-out rows contain duplicate evaluation keys")
    for row in heldout:
        finite_float(row.get("return"), "heldout return")
    primary = [
        row
        for row in heldout
        if row.get("primary_eval_policy", "").lower() == "true"
    ]
    if len(primary) != 24:
        raise ValidationError("expected 24 primary held-out rows, found {}".format(len(primary)))
    if Counter(row.get("target_cluster") for row in primary) != Counter(
        {"D": 8, "E": 8, "F": 8}
    ):
        raise ValidationError("primary held-out rows do not cover D/E/F eight times each")
    return run_dir, primary


def compare_recomputed_zero_shot(run_dir, primary_rows):
    recomputed = csv_rows(run_dir / "heldout_eval_recomputed.csv")
    if len(recomputed) != 24:
        raise ValidationError("expected 24 recomputed held-out rows, found {}".format(len(recomputed)))
    if Counter(row.get("target_cluster") for row in recomputed) != Counter(
        {"D": 8, "E": 8, "F": 8}
    ):
        raise ValidationError("recomputed rows do not cover D/E/F eight times each")
    key_fields = ("client_id", "target_cluster", "repeat", "episode")
    expected = {
        tuple(row[field] for field in key_fields): finite_float(row["return"], "heldout return")
        for row in primary_rows
    }
    actual = {
        tuple(row[field] for field in key_fields): finite_float(row["return"], "recomputed return")
        for row in recomputed
    }
    if len(expected) != len(primary_rows) or len(actual) != len(recomputed):
        raise ValidationError("zero-shot rows contain duplicate client/cluster keys")
    exact_keys = {
        (str(client), cluster, "0", "0")
        for client in range(N_RANKS)
        for cluster in ("D", "E", "F")
    }
    if set(expected) != exact_keys or set(actual) != exact_keys:
        raise ValidationError("zero-shot rows do not cover the exact 8-client D/E/F grid")
    if set(expected) != set(actual):
        raise ValidationError("launcher and recomputed zero-shot row keys differ")
    method_keys = {row.get("method_key") for row in primary_rows}
    if len(method_keys) != 1:
        raise ValidationError("launcher zero-shot rows contain multiple method keys")
    expected_method = next(iter(method_keys))
    for key in expected:
        if abs(expected[key] - actual[key]) > 1e-8:
            raise ValidationError("zero-shot mismatch for {}".format(key))
    summary = csv_rows(run_dir / "heldout_summary_recomputed.csv")
    if len(summary) != 3:
        raise ValidationError("expected three recomputed held-out summary rows")
    if {row.get("target_cluster") for row in summary} != {"D", "E", "F"}:
        raise ValidationError("recomputed summary does not cover D/E/F")
    for row in summary:
        if (
            row.get("environment") != "HalfCheetah-v5"
            or row.get("method_key") != expected_method
        ):
            raise ValidationError("unexpected summary environment or method")
        if int(row.get("n_client_repeat_episodes", 0)) != N_RANKS:
            raise ValidationError("unexpected held-out summary sample count")
        finite_float(row.get("mean_return"), "summary mean_return")
        finite_float(row.get("standard_deviation"), "summary standard_deviation")
    protocol = read_json(run_dir / "heldout_protocol_recomputed.json")
    if (
        protocol.get("horizon") != 32
        or protocol.get("episodes_per_client_cluster_repeat") != 1
        or protocol.get("evaluation_repeats") != 1
        or protocol.get("fine_tuning") is not False
        or protocol.get("environment") != "HalfCheetah-v5"
        or protocol.get("method_key") != expected_method
    ):
        raise ValidationError("unexpected recomputed zero-shot protocol: {}".format(protocol))


def validate_recommender_smoke_output(output_root):
    statuses = list(Path(output_root).rglob("RUN_STATUS.json"))
    if len(statuses) != 1:
        raise ValidationError(
            "expected one recommender RUN_STATUS.json, found {}".format(len(statuses))
        )
    status = read_json(statuses[0])
    if (
        status.get("status") != "completed"
        or status.get("mpi_ranks") != N_RANKS
        or status.get("final_common_epoch") != 1
        or status.get("scientific_result") is not False
    ):
        raise ValidationError("unexpected recommender smoke status: {}".format(status))
    run_dir = statuses[0].parent
    rank_logs = [run_dir / "logs_rank{}.log".format(rank) for rank in range(N_RANKS)]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in rank_logs):
        raise ValidationError("recommender smoke did not produce eight non-empty rank logs")
    return run_dir


def validate_generated_siyuan_bundle(job_root, expected_jobs, expected_cells):
    manifest = read_json(Path(job_root) / "JOB_MANIFEST.json")
    jobs = manifest.get("jobs", [])
    if len(jobs) != expected_jobs:
        raise ValidationError(
            "expected {} generated algorithm jobs, found {}".format(
                expected_jobs, len(jobs)
            )
        )
    cells = sum(len(job.get("conditions", [])) for job in jobs)
    if cells != expected_cells:
        raise ValidationError(
            "expected {} generated condition cells, found {}".format(
                expected_cells, cells
            )
        )
    if any("condition" in job or "result_dir" in job for job in jobs):
        raise ValidationError("generated manifest uses obsolete per-condition job schema")
    if any(not Path(job.get("script", "")).is_file() for job in jobs):
        raise ValidationError("a generated Siyuan job file is missing")
    if not (Path(job_root) / "submit_all.sh").is_file():
        raise ValidationError("generated submission helper is missing")
    task_ids = [job.get("task_id") for job in jobs]
    if len(task_ids) != len(set(task_ids)):
        raise ValidationError("generated Siyuan task IDs are not unique")
    return jobs


# Codex-added 2026-08-25: verify the fast full-budget layout requested for
# Siyuan: one condition per script and one independent Slurm submission per
# condition cell. The previous validator is backed up under
# .codex_hpc/backups/fedbridge_release/20260825_153941/.
def validate_parallel_siyuan_bundle(job_root, expected_jobs, expected_cells):
    jobs = validate_generated_siyuan_bundle(
        job_root, expected_jobs, expected_cells
    )
    manifest = read_json(Path(job_root) / "JOB_MANIFEST.json")
    if manifest.get("parallel_conditions") is not True:
        raise ValidationError("parallel Siyuan manifest flag is not enabled")
    if manifest.get("job_granularity") != "method_condition_seed":
        raise ValidationError("unexpected parallel Siyuan job granularity")
    if any(len(job.get("conditions", [])) != 1 for job in jobs):
        raise ValidationError("a parallel Siyuan job contains multiple conditions")
    for job in jobs:
        text = Path(job["script"]).read_text(encoding="utf-8")
        if text.count('python launcher.py "${train_args[@]}"') > 1:
            raise ValidationError("a MuJoCo parallel script launches training repeatedly")
        if text.count('python launcher.py "${run_args[@]}"') > 1:
            raise ValidationError("a recommender parallel script launches training repeatedly")
        if job.get("component") == "mujoco":
            if text.count('python launcher.py "${train_args[@]}"') != 1:
                raise ValidationError("MuJoCo parallel script lacks one training launch")
            if text.count('python evaluate_mujoco.py "${eval_args[@]}"') != 1:
                raise ValidationError("full MuJoCo script lacks one checkpoint evaluation")
        elif job.get("component") == "recommender":
            if text.count('python launcher.py "${run_args[@]}"') != 1:
                raise ValidationError("recommender parallel script lacks one training launch")
    return jobs


def main(argv=None):
    harden_process()
    args = parse_args(argv)
    runner = Runner()
    work_root = None
    temporary_work_root = False
    asset_root = None
    assets_before = None
    success = False
    try:
        if args.smoke:
            require_smoke_execution_context()
            if args.skip_assets:
                raise ValidationError("--smoke cannot be combined with --skip-assets")
        prefixes = conda_prefixes()
        mujoco_python = resolve_python(
            args.mujoco_python,
            ("FEDBRIDGE_MUJOCO_PYTHON", "MUJOCO_PYTHON"),
            ("fedbridge-mujoco",),
            prefixes,
            "mujoco python",
            (
                "import importlib.util,sys\n"
                "if sys.version_info[:2] != (3, 9): raise RuntimeError('Python 3.9 required')\n"
                "required=('torch','gymnasium','mujoco','mpi4py')\n"
                "missing=[name for name in required if importlib.util.find_spec(name) is None]\n"
                "if missing: raise RuntimeError('missing modules: {}'.format(missing))\n"
            ),
        )
        recommender_python = resolve_python(
            args.recommender_python,
            (
                "FEDBRIDGE_RECOMMENDER_PYTHON",
                "RECOMMENDER_PYTHON",
                "REC_PYTHON",
            ),
            ("fedbridge-easyrl4rec", "easyrl4rec"),
            prefixes,
            "recommender python",
            (
                "import importlib.util,sys\n"
                "if sys.version_info[:2] != (3, 11): raise RuntimeError('Python 3.11 required')\n"
                "required=('torch','pandas','sklearn','gymnasium','logzero','mpi4py')\n"
                "missing=[name for name in required if importlib.util.find_spec(name) is None]\n"
                "if missing: raise RuntimeError('missing modules: {}'.format(missing))\n"
            ),
        )
        mujoco_env = subprocess_environment(mujoco_python)
        recommender_env = subprocess_environment(recommender_python)
        source_root = find_original_root(args.source_root)
        asset_root = find_asset_root(args.asset_root)
        if args.smoke and asset_root is None:
            raise ValidationError("--smoke requires a valid --asset-root")
        work_root, temporary_work_root = allocate_work_root(
            args.work_root, (RELEASE_ROOT, source_root, asset_root)
        )
        mpi_temp_root = work_root / "mpi-tmp"
        mpi_temp_root.mkdir(parents=True, exist_ok=True)
        mujoco_env["TMPDIR"] = str(mpi_temp_root)
        recommender_env["TMPDIR"] = str(mpi_temp_root)
        print("Validation work root: {}".format(work_root), flush=True)

        runner.check("release tree contains no Python caches", assert_release_clean)
        runner.run(
            "release manifest (all packaged files)",
            [recommender_python, RELEASE_ROOT / "scripts" / "build_release_manifest.py", "--check"],
            RELEASE_ROOT,
            recommender_env,
        )
        compile_code = (
            "from pathlib import Path\n"
            "files=sorted(Path('.').rglob('*.py'))\n"
            "[compile(p.read_bytes(), str(p), 'exec') for p in files]\n"
            "print('compiled {}/{} Python files'.format(len(files), len(files)))\n"
        )
        runner.run(
            "in-memory compilation of every Python file",
            [recommender_python, "-c", compile_code],
            RELEASE_ROOT,
            recommender_env,
        )
        shell_files = sorted(
            [path for path in RELEASE_ROOT.rglob("*.sh")]
            + [path for path in RELEASE_ROOT.rglob("*.sbatch")]
        )
        if not shell_files:
            raise ValidationError("no shell or Slurm scripts were found")
        for shell_file in shell_files:
            runner.run(
                "shell syntax {}".format(shell_file.relative_to(RELEASE_ROOT)),
                ["bash", "-n", shell_file],
                RELEASE_ROOT,
                recommender_env,
            )
        runner.run(
            "MuJoCo unit and checkpoint tests",
            [mujoco_python, "-m", "unittest", "discover", "-s", "tests", "-v"],
            MUJOCO_ROOT,
            mujoco_env,
        )
        runner.run(
            "recommender-system unit and privacy tests",
            [recommender_python, "-m", "unittest", "discover", "-s", "tests", "-v"],
            RECOMMENDER_ROOT,
            recommender_env,
        )
        runner.run(
            "recommender-system canonical import audit",
            [recommender_python, "launcher.py", "--audit-only", "--skip-asset-check"],
            RECOMMENDER_ROOT,
            recommender_env,
        )

        for scene in SCENES:
            for method in METHODS:
                runner.run(
                    "dry-run command {}/{}".format(scene, method),
                    [
                        recommender_python,
                        "launcher.py",
                        "--scene",
                        scene,
                        "--method",
                        method,
                        "--seed",
                        str(VALIDATION_SEED),
                        "--run-id",
                        "validation-{}-{}".format(scene, method),
                        "--dry-run",
                        "--skip-asset-check",
                    ],
                    RECOMMENDER_ROOT,
                    recommender_env,
                    quiet=True,
                )

        if args.skip_source_manifest:
            runner.skip("source-manifest regeneration", "disabled by --skip-source-manifest")
        elif source_root is None:
            runner.skip("source-manifest regeneration", "protected original source tree is unavailable")
        else:
            regenerated = work_root / "SOURCE_MANIFEST.csv"
            runner.run(
                "source-manifest regeneration",
                [
                    recommender_python,
                    RECOMMENDER_ROOT / "scripts" / "build_source_manifest.py",
                    "--source-root",
                    source_root,
                    "--output",
                    regenerated,
                ],
                RECOMMENDER_ROOT,
                recommender_env,
            )
            runner.check(
                "source-manifest byte comparison",
                lambda: assert_files_equal(RECOMMENDER_ROOT / "SOURCE_MANIFEST.csv", regenerated),
            )

        if args.skip_assets:
            runner.skip("recommender external-asset inventory", "disabled by --skip-assets")
        elif asset_root is None:
            runner.skip(
                "recommender external-asset inventory",
                "no legally staged asset root found; pass --asset-root to require it",
            )
        else:
            runner.run(
                "recommender external-asset inventory",
                [
                    recommender_python,
                    RECOMMENDER_ROOT / "scripts" / "validate_assets.py",
                    "--asset-root",
                    asset_root,
                ],
                RECOMMENDER_ROOT,
                recommender_env,
            )

        siyuan_job_root = work_root / "siyuan-job-generation"
        siyuan_output_root = work_root / "siyuan-generated-results"
        siyuan_command = [
            recommender_python,
            RELEASE_ROOT / "scripts" / "prepare_siyuan_jobs.py",
            "--profile",
            "smoke",
            "--seed",
            str(VALIDATION_SEED),
            "--run-label",
            "validation-a",
            "--job-root",
            siyuan_job_root,
            "--output-root",
            siyuan_output_root,
        ]
        if asset_root is None:
            siyuan_command.extend(["--component", "mujoco"])
            expected_siyuan_jobs, expected_siyuan_cells = 6, 18
        else:
            siyuan_command.extend(["--component", "all", "--asset-root", asset_root])
            expected_siyuan_jobs, expected_siyuan_cells = 12, 30
        runner.run(
            "private fixed-method/seed Siyuan job generation",
            siyuan_command,
            RELEASE_ROOT,
            recommender_env,
        )
        generated_jobs = runner.check(
            "Siyuan job manifest and matrix coverage",
            lambda: validate_generated_siyuan_bundle(
                siyuan_job_root, expected_siyuan_jobs, expected_siyuan_cells
            ),
        )
        for job in generated_jobs:
            runner.run(
                "generated Slurm syntax {}".format(job["task_id"]),
                ["bash", "-n", job["script"]],
                RELEASE_ROOT,
                recommender_env,
            )
        for helper in ("submit_all.sh", "status_all.sh"):
            runner.run(
                "generated shell syntax {}".format(helper),
                ["bash", "-n", siyuan_job_root / helper],
                RELEASE_ROOT,
                recommender_env,
            )

        # Codex-added 2026-08-25: construct and syntax-check the 30-way
        # full-budget parallel bundle without submitting compute from this
        # lightweight release validator.
        parallel_job_root = work_root / "siyuan-parallel-full-generation"
        parallel_output_root = work_root / "siyuan-parallel-full-results"
        parallel_command = [
            recommender_python,
            RELEASE_ROOT / "scripts" / "prepare_siyuan_jobs.py",
            "--profile",
            "full",
            "--parallel-conditions",
            "--seed",
            str(VALIDATION_SEED),
            "--run-label",
            "parallel-a",
            "--job-root",
            parallel_job_root,
            "--output-root",
            parallel_output_root,
            "--mujoco-full-time",
            "04:00:00",
            "--recommender-full-time",
            "12:00:00",
        ]
        if asset_root is None:
            parallel_command.extend(["--component", "mujoco"])
            expected_parallel_jobs, expected_parallel_cells = 18, 18
        else:
            parallel_command.extend(["--component", "all", "--asset-root", asset_root])
            expected_parallel_jobs, expected_parallel_cells = 30, 30
        runner.run(
            "parallel full-budget Siyuan job generation",
            parallel_command,
            RELEASE_ROOT,
            recommender_env,
        )
        parallel_jobs = runner.check(
            "parallel full-budget job granularity",
            lambda: validate_parallel_siyuan_bundle(
                parallel_job_root, expected_parallel_jobs, expected_parallel_cells
            ),
        )
        for job in parallel_jobs:
            runner.run(
                "parallel full Slurm syntax {}".format(job["task_id"]),
                ["bash", "-n", job["script"]],
                RELEASE_ROOT,
                recommender_env,
            )
        for helper in ("submit_all.sh", "status_all.sh"):
            runner.run(
                "parallel full shell syntax {}".format(helper),
                ["bash", "-n", parallel_job_root / helper],
                RELEASE_ROOT,
                recommender_env,
            )

        login_host = looks_like_login_host()
        if args.skip_mpi_audit:
            runner.skip("eight-rank MuJoCo construction audit", "disabled by --skip-mpi-audit")
        elif login_host and not args.allow_login_mpi_audit:
            runner.skip(
                "eight-rank MuJoCo construction audit",
                "recognized login/data host; rerun on a compute node or opt in explicitly",
            )
        else:
            mujoco_mpi = resolve_mpi(args.mujoco_mpi, mujoco_python, "mujoco mpi")
            mpi_env = mpi_environment(mujoco_env, mujoco_mpi)
            mpi_probe = (
                "from mpi4py import MPI\n"
                "if MPI.COMM_WORLD.Get_size() != 1: raise RuntimeError('MPI size mismatch')\n"
                "print(MPI.Get_library_version())\n"
            )
            runner.run(
                "MuJoCo MPI/mpi4py pairing preflight",
                mpi_command(mujoco_mpi, 1, mujoco_python, "-c", mpi_probe),
                MUJOCO_ROOT,
                mpi_env,
                timeout=60,
            )
            audit_root = work_root / "mujoco-audit"
            runner.run(
                "eight-rank MuJoCo construction audit",
                mpi_command(
                    mujoco_mpi,
                    N_RANKS,
                    mujoco_python,
                    "launcher.py",
                    "--env-name",
                    "HalfCheetah-v5",
                    "--method",
                    "fedbridge_legacy_dualcritic",
                    "--seed",
                    str(VALIDATION_SEED),
                    "--output-root",
                    audit_root,
                    "--audit-only",
                ),
                MUJOCO_ROOT,
                mpi_env,
                timeout=300,
            )
            runner.check("MuJoCo audit STATUS.json", lambda: validate_audit_output(audit_root))

        if args.smoke:
            mujoco_mpi = resolve_mpi(args.mujoco_mpi, mujoco_python, "mujoco mpi")
            recommender_mpi = resolve_mpi(
                args.recommender_mpi, recommender_python, "recommender mpi"
            )
            smoke_env = mpi_environment(mujoco_env, mujoco_mpi)
            recommender_smoke_env = mpi_environment(recommender_env, recommender_mpi)
            mpi_probe = (
                "from mpi4py import MPI\n"
                "if MPI.COMM_WORLD.Get_size() != 1: raise RuntimeError('MPI size mismatch')\n"
                "print(MPI.Get_library_version())\n"
            )
            runner.run(
                "MuJoCo smoke MPI/mpi4py pairing preflight",
                mpi_command(mujoco_mpi, 1, mujoco_python, "-c", mpi_probe),
                MUJOCO_ROOT,
                smoke_env,
                timeout=60,
            )
            runner.run(
                "recommender smoke MPI/mpi4py pairing preflight",
                mpi_command(recommender_mpi, 1, recommender_python, "-c", mpi_probe),
                RECOMMENDER_ROOT,
                recommender_smoke_env,
                timeout=60,
            )
            assets_before = runner.check(
                "staged-asset read-only snapshot before smoke",
                lambda: asset_snapshot(asset_root),
            )
            mujoco_smoke_root = work_root / "mujoco-smoke"
            runner.run(
                "MuJoCo FedBridge end-to-end smoke",
                mpi_command(
                    mujoco_mpi,
                    N_RANKS,
                    mujoco_python,
                    "launcher.py",
                    "--env-name",
                    "HalfCheetah-v5",
                    "--method",
                    "fedbridge_singlecritic",
                    "--seed",
                    str(VALIDATION_SEED),
                    "--output-root",
                    mujoco_smoke_root,
                    "--smoke-test",
                ),
                MUJOCO_ROOT,
                smoke_env,
                timeout=900,
            )
            run_dir, primary_rows = runner.check(
                "MuJoCo smoke artifacts and communication",
                lambda: validate_mujoco_smoke_output(
                    mujoco_smoke_root, "fedbridge_singlecritic"
                ),
            )
            evaluation_command = mpi_command(
                mujoco_mpi,
                N_RANKS,
                mujoco_python,
                "evaluate_mujoco.py",
                "--env-name",
                "HalfCheetah-v5",
                "--method",
                "fedbridge_singlecritic",
                "--seed",
                str(VALIDATION_SEED),
                "--run-dir",
                run_dir,
                "--episodes",
                "1",
                "--repeats",
                "1",
                "--horizon",
                "32",
            )
            runner.run(
                "MuJoCo independent zero-shot checkpoint reload",
                evaluation_command,
                MUJOCO_ROOT,
                smoke_env,
                timeout=600,
            )
            runner.check(
                "launcher/recomputed zero-shot agreement",
                lambda: compare_recomputed_zero_shot(run_dir, primary_rows),
            )
            repeated_output = run_dir / "determinism_repeat" / "heldout_eval_recomputed.csv"
            runner.run(
                "MuJoCo repeated deterministic zero-shot evaluation",
                evaluation_command + ["--output", repeated_output],
                MUJOCO_ROOT,
                smoke_env,
                timeout=600,
            )
            runner.check(
                "repeated zero-shot CSV byte comparison",
                lambda: assert_files_equal(
                    run_dir / "heldout_eval_recomputed.csv", repeated_output
                ),
            )

            legacy_smoke_root = work_root / "mujoco-legacy-smoke"
            runner.run(
                "BridgePPO legacy-path optimization and checkpoint smoke",
                mpi_command(
                    mujoco_mpi,
                    N_RANKS,
                    mujoco_python,
                    "launcher.py",
                    "--env-name",
                    "HalfCheetah-v5",
                    "--method",
                    "fedbridge_legacy_dualcritic",
                    "--seed",
                    str(VALIDATION_SEED),
                    "--output-root",
                    legacy_smoke_root,
                    "--smoke-test",
                ),
                MUJOCO_ROOT,
                smoke_env,
                timeout=900,
            )
            legacy_run_dir, legacy_primary_rows = runner.check(
                "BridgePPO legacy-path smoke artifacts",
                lambda: validate_mujoco_smoke_output(
                    legacy_smoke_root, "fedbridge_legacy_dualcritic"
                ),
            )
            runner.run(
                "BridgePPO trained-checkpoint evaluator reload",
                mpi_command(
                    mujoco_mpi,
                    N_RANKS,
                    mujoco_python,
                    "evaluate_mujoco.py",
                    "--env-name",
                    "HalfCheetah-v5",
                    "--method",
                    "fedbridge_legacy_dualcritic",
                    "--seed",
                    str(VALIDATION_SEED),
                    "--run-dir",
                    legacy_run_dir,
                    "--episodes",
                    "1",
                    "--repeats",
                    "1",
                    "--horizon",
                    "32",
                ),
                MUJOCO_ROOT,
                smoke_env,
                timeout=600,
            )
            runner.check(
                "BridgePPO trained zero-shot agreement",
                lambda: compare_recomputed_zero_shot(
                    legacy_run_dir, legacy_primary_rows
                ),
            )

            recommender_smoke_root = work_root / "recommender-smoke"
            runner.run(
                "recommender FedBridge end-to-end smoke",
                [
                    recommender_python,
                    "launcher.py",
                    "--scene",
                    "hetero5",
                    "--method",
                    "fedbridge",
                    "--seed",
                    str(VALIDATION_SEED),
                    "--run-id",
                    "validation-smoke",
                    "--asset-root",
                    asset_root,
                    "--output-root",
                    recommender_smoke_root,
                    "--mpi-exec",
                    recommender_mpi,
                    "--smoke-test",
                ],
                RECOMMENDER_ROOT,
                recommender_smoke_env,
                timeout=1800,
            )
            recommender_run_dir = runner.check(
                "recommender smoke RUN_STATUS.json",
                lambda: validate_recommender_smoke_output(recommender_smoke_root),
            )
            parser_code = (
                "import math,sys\n"
                "from pathlib import Path\n"
                "from scripts.aggregate_results import parse_run\n"
                "run=parse_run(Path(sys.argv[1]),'hetero5','fedbridge','validation-smoke',1,8)\n"
                "if not run.is_complete or run.complete_epoch_numbers != (1,): "
                "raise RuntimeError('incomplete recommender smoke epoch')\n"
                "if len(run.epoch_metrics[1]) != 3: "
                "raise RuntimeError('unexpected recommender metric count')\n"
                "if not all(math.isfinite(float(v)) for v in run.epoch_metrics[1].values()): "
                "raise RuntimeError('non-finite recommender smoke metric')\n"
                "print('recommender smoke aggregation parser: OK')\n"
            )
            runner.run(
                "recommender smoke log aggregation parse",
                [recommender_python, "-c", parser_code, recommender_run_dir],
                RECOMMENDER_ROOT,
                recommender_smoke_env,
            )
            runner.check(
                "staged assets remained read-only during smoke",
                lambda: assert_snapshots_equal(assets_before, asset_snapshot(asset_root)),
            )

        runner.check("release remains free of Python caches", assert_release_clean)
        runner.run(
            "final release manifest recheck",
            [recommender_python, RELEASE_ROOT / "scripts" / "build_release_manifest.py", "--check"],
            RELEASE_ROOT,
            recommender_env,
        )
        success = True
        validation_kind = "SMOKE" if args.smoke else "STRUCTURAL"
        print(
            "\n{} VALIDATION PASSED: {} stages".format(
                validation_kind, len(runner.passed)
            ),
            flush=True,
        )
        if runner.skipped:
            print("Conditional stages skipped: {}".format(len(runner.skipped)), flush=True)
            for label, reason in runner.skipped:
                print("  - {}: {}".format(label, reason), flush=True)
        if temporary_work_root:
            print("Disposable validation outputs will be removed.", flush=True)
        else:
            print("Validation outputs retained at: {}".format(work_root), flush=True)
        return 0
    except Exception as error:
        print(
            "\nVALIDATION FAILED [{}]: {}".format(type(error).__name__, error),
            file=sys.stderr,
            flush=True,
        )
        if assets_before is not None and asset_root is not None:
            try:
                assert_snapshots_equal(assets_before, asset_snapshot(asset_root))
            except (ValidationError, OSError) as asset_error:
                print(
                    "CRITICAL: post-failure asset check also failed: {}".format(asset_error),
                    file=sys.stderr,
                    flush=True,
                )
        if work_root is not None:
            print("Diagnostic outputs retained at: {}".format(work_root), file=sys.stderr, flush=True)
        return 1
    finally:
        if success and temporary_work_root and work_root is not None:
            shutil.rmtree(str(work_root))


if __name__ == "__main__":
    sys.exit(main())
