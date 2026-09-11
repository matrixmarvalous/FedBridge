#!/usr/bin/env python3
# Codex-added 2026-08-14: release configuration, privacy and parser checks.
# Codex-modified 2026-08-18: validate canonical Bridge names and structural
# source-manifest completeness after the behavior-preserving rename.
from __future__ import annotations

import ast
import csv
import hashlib
import io
import re
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


RELEASE_DIR = Path(__file__).resolve().parents[1]
PUBLIC_RELEASE_ROOT = RELEASE_DIR.parent
sys.path.insert(0, str(RELEASE_DIR))

import experiment_spec as spec  # noqa: E402
import launcher  # noqa: E402
from scripts import aggregate_results, validate_assets  # noqa: E402


SYNTHETIC_TEST_SEED = 7  # A test fixture only; not an author experiment seed.


class ReleaseTests(unittest.TestCase):
    # Codex-added 2026-08-23: guard against the rank-divergent progress padding
    # that let some clients skip the final collective and deadlock their peers.
    def test_collective_trainers_pad_progress_only_after_sampling_loop(self) -> None:
        trainer_dir = RELEASE_DIR / "src" / "tianshou" / "tianshou" / "trainer"
        for filename in ("fedavgbase.py", "perfeddcbase.py", "pfedmebase.py"):
            tree = ast.parse((trainer_dir / filename).read_text(encoding="utf-8"))
            padding_ifs = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and "t.n <= t.total" in ast.unparse(node.test)
            ]
            self.assertTrue(padding_ifs, msg=filename)
            for loop in (node for node in ast.walk(tree) if isinstance(node, ast.While)):
                nested_ids = {id(node) for child in loop.body for node in ast.walk(child)}
                self.assertFalse(
                    any(id(padding) in nested_ids for padding in padding_ifs),
                    msg=f"progress padding remains inside while loop: {filename}",
                )

    # Codex-added 2026-08-23: generated jobs must prove application success;
    # timeout text must never be accepted as a clean scheduler log.
    def test_siyuan_jobs_use_success_only_markers_and_detect_timeouts(self) -> None:
        generator = (PUBLIC_RELEASE_ROOT / "scripts" / "prepare_siyuan_jobs.py").read_text(
            encoding="utf-8"
        )
        checker = (PUBLIC_RELEASE_ROOT / "scripts" / "check_siyuan_results.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("record_success", generator)
        self.assertNotIn("trap record_exit EXIT", generator)
        self.assertIn("DUE TO TIME LIMIT", checker)
        self.assertIn("missing application completion marker", checker)
        fatal_log_re = runpy.run_path(
            str(PUBLIC_RELEASE_ROOT / "scripts" / "check_siyuan_results.py")
        )["FATAL_LOG_RE"]
        for signature in (
            "TIMEOUT",
            "CANCELLED AT node DUE TO TIME LIMIT",
            "loss=nan",
            "loss/vf=+inf",
        ):
            self.assertIsNotNone(fatal_log_re.search(signature), msg=signature)

    # Codex-added 2026-09-09: lambda_kl=0 is the Individual configuration.  All
    # KL calls and bridge optimizer steps must remain inside the coupling guard
    # so 0 * inf cannot silently yield a non-finite training loss.
    def test_zero_kl_path_skips_bridge_computation(self) -> None:
        policy_path = (
            RELEASE_DIR
            / "src"
            / "tianshou"
            / "tianshou"
            / "policy"
            / "modelfree"
            / "bridgeppo.py"
        )
        tree = ast.parse(policy_path.read_text(encoding="utf-8"))
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        guarded_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_kl = isinstance(node.func, ast.Name) and node.func.id == "kl_divergence"
            is_bridge_step = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "step"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "optim_bridge"
            )
            if not (is_kl or is_bridge_step):
                continue
            guarded = False
            ancestor = parents.get(node)
            while ancestor is not None:
                if (
                    isinstance(ancestor, ast.If)
                    and isinstance(ancestor.test, ast.Name)
                    and ancestor.test.id == "bridge_coupling"
                ):
                    guarded = True
                    break
                ancestor = parents.get(ancestor)
            guarded_calls.append(guarded)
        self.assertTrue(guarded_calls)
        self.assertTrue(all(guarded_calls))

    def test_complete_runner_matrix_exists(self) -> None:
        expected = {(scene, method) for scene in spec.SCENES for method in spec.METHODS}
        self.assertEqual(set(spec.RUNNERS), expected)
        self.assertEqual(set(spec.METHOD_ARGS), expected)
        for scene, method in expected:
            self.assertTrue(spec.runner_path(scene, method).is_file())

    def test_training_seed_has_no_default(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(
                [
                    "--scene", "hetero5",
                    "--method", "fedbridge",
                    "--run-id", "fixture",
                    "--dry-run",
                ]
            )
        audit = launcher.parse_args(["--audit-only", "--skip-asset-check"])
        self.assertTrue(audit.audit_only)

    def test_launcher_maps_public_configuration(self) -> None:
        args = launcher.parse_args(
            [
                "--scene", "hetero5",
                "--method", "fedbridge",
                "--seed", str(SYNTHETIC_TEST_SEED),
                "--run-id", "fixture",
                "--dry-run",
                "--skip-asset-check",
            ]
        )
        command = launcher.build_command(args)
        self.assertIn("run_FedBridgePPO_hetero5_seed.py", " ".join(command))
        self.assertEqual(command[command.index("--lambda_kl") + 1], "0.005")
        self.assertEqual(command[command.index("--comm_times_per_epoch") + 1], "5")
        self.assertEqual(command[command.index("--seed") + 1], str(SYNTHETIC_TEST_SEED))
        self.assertIn("release__hetero5__fedbridge__fixture", command)
        self.assertIn("--cpu", command)

        output = io.StringIO()
        with redirect_stdout(output):
            status = launcher.main(
                [
                    "--scene", "hetero5",
                    "--method", "fedbridge",
                    "--seed", str(SYNTHETIC_TEST_SEED),
                    "--run-id", "fixture",
                    "--dry-run",
                    "--skip-asset-check",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("<redacted>", output.getvalue())
        self.assertNotIn(f"--seed {SYNTHETIC_TEST_SEED}", output.getvalue())

    def test_public_launcher_rejects_unsupported_options(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(
                [
                    "--scene", "hetero5",
                    "--method", "fedbridge",
                    "--seed", str(SYNTHETIC_TEST_SEED),
                    "--run-id", "fixture",
                    "--device", "cuda",
                    "--dry-run",
                    "--skip-asset-check",
                ]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(
                [
                    "--scene", "hetero5",
                    "--method", "fedbridge",
                    "--seed", str(SYNTHETIC_TEST_SEED),
                    "--run-id", "fixture",
                    "--disable-categorical-validation",
                    "--dry-run",
                    "--skip-asset-check",
                ]
            )

    def test_asset_inventory_is_relative_and_complete(self) -> None:
        paths = validate_assets.required_relative_paths()
        self.assertEqual(len(paths), 108)
        self.assertEqual(len(set(paths)), 108)
        self.assertTrue(all(not path.is_absolute() for path in paths))
        self.assertFalse(any("matsVar" in path.parts for path in paths))

    def test_source_manifest_records_only_bounded_modifications(self) -> None:
        with (RELEASE_DIR / "SOURCE_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected_paths = {
            path.relative_to(RELEASE_DIR).as_posix()
            for path in (RELEASE_DIR / "src").rglob("*.py")
        } | {
            path.relative_to(RELEASE_DIR).as_posix()
            for path in (RELEASE_DIR / "examples" / "policy").glob("*.py")
        }
        self.assertEqual({row["release_path"] for row in rows}, expected_paths)
        self.assertEqual(
            sum(row["copy_status"] == "renamed_for_bridge_release" for row in rows),
            4,
        )
        for row in rows:
            self.assertEqual(len(row["release_sha256"]), 64)
            if row["copy_status"] == "byte_preserved":
                self.assertEqual(row["source_sha256"], row["release_sha256"])
            else:
                self.assertEqual(row["source_sha256"], "withheld")
            release_file = RELEASE_DIR / row["release_path"]
            self.assertTrue(release_file.is_file())
            # Codex-modified 2026-09-09: validate the recorded bytes, not only
            # the checksum's shape. This catches a stale source manifest after
            # a bounded release-only fix.
            self.assertEqual(
                hashlib.sha256(release_file.read_bytes()).hexdigest(),
                row["release_sha256"],
            )

    def test_public_python_uses_only_bridge_naming(self) -> None:
        pre_publication_token = "pro" + "xy"
        for path in RELEASE_DIR.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            self.assertNotIn(pre_publication_token, path.name.lower())
            self.assertNotIn(
                pre_publication_token,
                path.read_text(encoding="utf-8", errors="replace").lower(),
                msg=str(path.relative_to(RELEASE_DIR)),
            )

    def test_release_paths_are_casefold_unique(self) -> None:
        # Codex-added 2026-08-19: prevent case-insensitive filesystem clashes
        # between the canonical bridgeppo.py and any legacy standalone module.
        relative_paths = [
            path.relative_to(RELEASE_DIR).as_posix()
            for path in RELEASE_DIR.rglob("*")
            if path.is_file()
        ]
        folded = [path.casefold() for path in relative_paths]
        self.assertEqual(len(folded), len(set(folded)))
        legacy_stem = "bridge" + "double" + "ppo"
        self.assertFalse(any(legacy_stem in path.casefold() for path in relative_paths))
        self.assertTrue(
            (
                RELEASE_DIR
                / "src"
                / "tianshou"
                / "tianshou"
                / "policy"
                / "modelfree"
                / "bridgeppo.py"
            ).is_file()
        )

    def test_canonical_runners_propagate_exceptions_and_hide_old_seed_values(self) -> None:
        for filename in sorted(set(spec.RUNNERS.values())):
            text = (RELEASE_DIR / "examples" / "policy" / filename).read_text(
                encoding="utf-8"
            )
            self.assertIn("make MPI/Slurm failure status reliable", text)
            self.assertNotRegex(text, r"(?m)^\s*args\.seed\s*=\s*\d{3,}\s*$")

    def test_log_parser_requires_all_ranks_for_an_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for rank in range(8):
                lines = [
                    f"Epoch: [1], Rank: [{rank}], Info: [{{'R_tra': {rank + 1}, 'len_tra': 2, 'CV': 0.1}}]"
                ]
                if rank != 7:
                    lines.append(
                        f"Epoch: [2], Rank: [{rank}], Info: [{{'R_tra': {rank + 2}, 'len_tra': 3, 'CV': 0.2}}]"
                    )
                (run_dir / f"logs_rank{rank}.log").write_text("\n".join(lines) + "\n")
            parsed = aggregate_results.parse_run(
                run_dir, "hetero5", "fedbridge", "internal-fixture", 2, 8
            )
            self.assertEqual(parsed.complete_epoch_numbers, (1,))
            self.assertFalse(parsed.is_complete)
            self.assertAlmostEqual(parsed.epoch_metrics[1]["R_cum"], 4.5)

    def test_strict_aggregation_does_not_publish_partial_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs_dir = root / "logs"
            run_dir = logs_dir / "[release__hetero5__fedbridge__fixture]_timestamp"
            run_dir.mkdir(parents=True)
            for rank in range(8):
                (run_dir / f"logs_rank{rank}.log").write_text(
                    f"Epoch: [1], Rank: [{rank}], Info: "
                    f"[{{'R_tra': {rank + 1}, 'len_tra': 2, 'CV': 0.1}}]\n",
                    encoding="utf-8",
                )
            output_dir = root / "public"
            with redirect_stderr(io.StringIO()):
                status = aggregate_results.main(
                    [
                        "--logs-dir", str(logs_dir),
                        "--output-dir", str(output_dir),
                        "--expected-epochs", "1",
                        "--expected-ranks", "8",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertFalse(output_dir.exists())

    def test_publication_layer_contains_no_private_machine_strings(self) -> None:
        files = [
            RELEASE_DIR / "README.md",
            RELEASE_DIR / "launcher.py",
            RELEASE_DIR / "experiment_spec.py",
            RELEASE_DIR / "src" / "core" / "util" / "logger_callback.py",
            RELEASE_DIR / "src" / "DeepCTR-Torch" / "deepctr_torch" / "__init__.py",
            *sorted((RELEASE_DIR / "docs").glob("*.md")),
            *sorted((RELEASE_DIR / "slurm").glob("*")),
        ]
        private_absolute_path = re.compile(r"[\"']/(?:dssg|home|root)/")
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertIsNone(
                private_absolute_path.search(text),
                msg=f"private absolute path found in {path.name}",
            )

        deepctr_init = (
            RELEASE_DIR / "src" / "DeepCTR-Torch" / "deepctr_torch" / "__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn("EASYRL4REC_CHECK_DEEPCTR_VERSION", deepctr_init)


if __name__ == "__main__":
    unittest.main()
