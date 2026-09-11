#!/usr/bin/env python3
# Codex-added 2026-08-14: standard-library release/configuration checks.
# Codex-modified 2026-08-18: validate the BridgePPO public naming surface.
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch


RELEASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_DIR))

import experiment_spec as spec  # noqa: E402
from evaluate_mujoco import load_agent_checkpoint  # noqa: E402
from runtime_support import build_agent, make_dynamics_env  # noqa: E402


EXPECTED_ALGORITHM_HASHES = {
    "PPO.py": "3041d2c9e06a6bfb19bd3c820a1bf1c29c2a0a742edabf63e683334f317d6e69",
    "BridgePPO.py": "76e11b935abacf633952a10e3da86f99f954e2dcac77c528113af3e57ab2be9d",
    "perfeddc_ppo.py": "f0f1b27d3dd12e127dfe420e12a8e71d2dc79c85eaa102e3d8fdd20df4185694",
    "pfedme_ppo.py": "0001b4d112ba60acd18ade8682ab14fec378f9f1a20d8516ca16e9fe3a071642",
    "single_critic_fedbridge.py": "85fe5c8946aa5ad15711b42690e70d7ed5fa1692ccc3d9f8493f6fcbd979f07f",
}


class ReleaseTests(unittest.TestCase):
    def test_private_paper_seed_matrix_is_absent(self) -> None:
        self.assertFalse(hasattr(spec, "SEEDS"))
        self.assertFalse(hasattr(spec, "JOB_MATRIX"))

    def test_current_task_specific_coefficients(self) -> None:
        self.assertEqual(
            spec.task_hyperparameters("Hopper-v4", "fedbridge_singlecritic")[
                "lambda_kl"
            ],
            0.01,
        )
        self.assertEqual(
            spec.task_hyperparameters("HalfCheetah-v5", "perfeddc")[
                "pfl_lambda_l2"
            ],
            0.003,
        )

    def test_algorithm_release_hashes(self) -> None:
        for filename, expected in EXPECTED_ALGORITHM_HASHES.items():
            payload = (RELEASE_DIR / "algorithms" / filename).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)

    def test_public_python_uses_bridge_naming(self) -> None:
        pre_publication_token = "pro" + "xy"
        pre_final_module_token = "bridge" + "double" + "ppo"
        self.assertTrue((RELEASE_DIR / "algorithms" / "BridgePPO.py").is_file())
        for path in RELEASE_DIR.rglob("*.py"):
            self.assertNotIn(pre_publication_token, path.name.lower())
            self.assertNotIn(pre_final_module_token, path.name.lower())
            self.assertNotIn(
                pre_publication_token,
                path.read_text(encoding="utf-8", errors="replace").lower(),
            )
            self.assertNotIn(
                pre_final_module_token,
                path.read_text(encoding="utf-8", errors="replace").lower(),
            )

    def test_legacy_bridge_checkpoint_round_trip(self) -> None:
        method = "fedbridge_legacy_dualcritic"
        source = build_agent(method, state_dim=11, action_dim=3, device=torch.device("cpu"))
        evaluator_target = build_agent(
            method, state_dim=11, action_dim=3, device=torch.device("cpu")
        )
        class_target = build_agent(
            method, state_dim=11, action_dim=3, device=torch.device("cpu")
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy_bridge.pt"
            source.save(checkpoint)
            load_agent_checkpoint(evaluator_target, method, checkpoint)
            class_target.load(checkpoint)
        for target in (evaluator_target, class_target):
            for branch in ("policy_priv", "policy_priv_old", "policy_bridge", "policy_bridge_old"):
                source_branch = getattr(
                    source,
                    "policy_priv_old" if branch.startswith("policy_priv") else "policy_bridge_old",
                )
                target_branch = getattr(target, branch)
                for expected, actual in zip(
                    source_branch.state_dict().values(),
                    target_branch.state_dict().values(),
                ):
                    self.assertTrue(torch.equal(expected, actual))

    def test_all_training_and_heldout_environments_construct(self) -> None:
        # This synthetic test value is not a paper experiment seed.
        synthetic_seed = 314_159_265
        factor_sets = [
            spec.TRAIN_CLUSTERS["A"],
            *[
                factors
                for environment in spec.ENVIRONMENTS
                for factors in spec.HELDOUT_CLUSTERS[environment].values()
            ],
        ]
        for environment in spec.ENVIRONMENTS:
            for factors in factor_sets:
                with self.subTest(environment=environment, factors=factors):
                    env = make_dynamics_env(environment, factors, synthetic_seed)
                    observation, _ = env.reset(seed=synthetic_seed)
                    self.assertEqual(observation.ndim, 1)
                    env.step(env.action_space.sample())
                    env.close()


if __name__ == "__main__":
    unittest.main()
