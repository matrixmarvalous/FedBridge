"""Single-condition release checks using synthetic logs, not scientific results."""
from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import launcher
from scripts import aggregate_results, plot_results, validate_assets


class RepresentativeTests(unittest.TestCase):
    def test_selected_inventory_matches_published_manifest(self):
        selected = validate_assets.required_relative_paths(("hetero5",))
        other = validate_assets.required_relative_paths(("homo3",))
        self.assertEqual(len(selected), 54)
        self.assertEqual(len(other), 54)
        self.assertFalse(set(selected) & set(other))
        self.assertEqual(set(selected + other), set(validate_assets.required_relative_paths()))
        with (PACKAGE_ROOT / "docs/hetero5_asset_manifest.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["path"] for row in rows}, {p.as_posix() for p in selected})
        self.assertEqual(sum(int(row["size_bytes"]) for row in rows), 1455231928)

    def test_scene_validation_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = validate_assets.required_relative_paths(("hetero5",))
            for relative in paths:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"synthetic asset")
            manifest = root / "inventory.csv"
            validate_assets.write_manifest(manifest, root, paths)
            argv = ["validate_assets.py", "--asset-root", str(root), "--scene", "hetero5",
                    "--verify-manifest", str(manifest)]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(validate_assets.main(), 0)
            # The real launcher must not require absent homo3 assets on a hetero5 run.
            with redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main([
                    "--scene", "hetero5", "--method", "fedbridge", "--seed", "0",
                    "--run-id", "fixture", "--asset-root", str(root), "--dry-run",
                ]), 0)
            (root / paths[0]).write_bytes(b"corrupted asset")
            with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
                self.assertEqual(validate_assets.main(), 2)
            # Without a scene selector, validation must still require both scenes.
            with patch.object(sys, "argv", argv[:3]), redirect_stderr(io.StringIO()):
                self.assertEqual(validate_assets.main(), 2)

    def test_full_single_run_aggregation_and_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            run = logs / "[release__hetero5__fedbridge__fixture]_timestamp"
            run.mkdir(parents=True)
            for rank in range(8):
                lines = [
                    f"Epoch: [{epoch}], Rank: [{rank}], Info: "
                    f"[{{'R_tra': {epoch + rank}, 'len_tra': 2, 'CV': 0.1}}]"
                    for epoch in range(1, 101)
                ]
                (run / f"logs_rank{rank}.log").write_text("\n".join(lines) + "\n")
            # Unselected, incomplete runs must not contaminate the selected report.
            (logs / "[release__homo3__individual__unused]_timestamp").mkdir()
            output = root / "aggregated"
            arguments = ["--logs-dir", str(logs), "--output-dir", str(output),
                         "--scene", "hetero5", "--method", "fedbridge"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(aggregate_results.main(arguments), 0)
            epochs, tails = plot_results.load_and_validate(
                output / "epoch_mean_ci95.csv", output / "tail10_summary_ci95.csv",
                ("hetero5",), ("fedbridge",),
            )
            self.assertEqual(len(epochs), 300)
            self.assertEqual(len(tails), 3)
            self.assertTrue((epochs["n"] == 1).all())
            self.assertTrue(epochs[["ci95_low", "ci95_high"]].isna().all().all())
            reward = epochs[(epochs["metric"] == "R_cum") & (epochs["epoch"] == 1)]
            self.assertAlmostEqual(float(reward.iloc[0]["mean"]), 4.5)
            figures = root / "figures"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(plot_results.main([
                    "--input-dir", str(output), "--output-dir", str(figures),
                    "--scene", "hetero5", "--method", "fedbridge", "--dpi", "72",
                ]), 0)
            for suffix in ("png", "pdf", "svg"):
                self.assertGreater((figures / f"learning_curves.{suffix}").stat().st_size, 0)
            with self.assertRaises(ValueError):
                plot_results.load_and_validate(
                    output / "epoch_mean_ci95.csv", output / "tail10_summary_ci95.csv"
                )
            # A missing rank must still fail strict single-condition aggregation.
            (run / "logs_rank7.log").write_text("")
            incomplete_output = root / "incomplete"
            arguments[3] = str(incomplete_output)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(aggregate_results.main(arguments), 2)
            self.assertFalse(incomplete_output.exists())


if __name__ == "__main__":
    unittest.main()
