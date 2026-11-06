"""Checks for immediate group updates and actual data-pass epoch boundaries."""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = ["/root/sft-rl-experiment/exp/rl", "/root/sft-rl-experiment/verl"]
from omegaconf import OmegaConf
import single_group_trainer as sg
from trainer import RewardEarlyStopState
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
import train_rl as rl


class SingleGroupTests(unittest.TestCase):
    def test_ten_passes_count_all_900_candidates_including_filtered(self):
        ids = tuple(f"q{i}" for i in range(90))
        progress = sg.DataPassProgress(ids)
        boundaries = []
        for epoch in range(10):
            for position, name in enumerate(ids):
                updated = position % 3 == 0
                if progress.record(name, updated, None if updated else float(position % 2)):
                    boundaries.append(progress.groups_seen)
        self.assertEqual(boundaries, list(range(90, 901, 90)))
        self.assertEqual(progress.groups_seen, 900)
        self.assertEqual(progress.optimizer_updates, 300)
        self.assertEqual(progress.payload()["completed_epochs"], 10)

    def test_duplicate_before_pass_complete_is_rejected(self):
        progress = sg.DataPassProgress(("a", "b"))
        progress.record("a", True)
        with self.assertRaises(ValueError):
            progress.record("a", False, 0)

    def test_midpass_restore_preserves_seen_ids_and_update_count(self):
        progress = sg.DataPassProgress(("a", "b", "c"))
        progress.record("b", True)
        progress.record("a", False, 0)
        restored = sg.DataPassProgress.restore(progress.ids, progress.payload())
        self.assertTrue(restored.record("c", False, 1))
        self.assertEqual(restored.optimizer_updates, 1)
        self.assertEqual(restored.groups_seen, 3)

    def run_fit(self, outcomes, epochs):
        ids = ("a", "b", "c")
        trainer = object.__new__(sg.SingleGroupEpochTrainer)
        trainer.config = OmegaConf.create({"trainer": {"total_epochs": epochs, "project_name": "test",
                                                       "experiment_name": "test", "logger": ["console"]}})
        trainer.parameter_sync_step = 1
        trainer.total_training_steps = len(ids) * epochs
        trainer.global_steps = 0
        trainer.data_pass = sg.DataPassProgress(ids)
        trainer.epoch_validation_history = []
        trainer._igsm_state = RewardEarlyStopState()
        trace = []
        for method in ["_reissue_inflight_prompts", "on_train_begin", "on_train_end", "on_step_begin",
                       "_shutdown_dump_executor", "_write_run_metadata", "_compute_metrics"]:
            setattr(trainer, method, Mock())
        trainer._consume_sync_metrics = lambda: {}
        trainer._save_checkpoint = lambda: trace.append(("save", trainer.data_pass.groups_seen))

        def step(metrics, timing):
            index = trainer.global_steps - 1
            trainer.current_question_id = ids[index % len(ids)]
            trace.append(("generate", trainer.global_steps))
            value = outcomes[index]
            if value is not None:
                raise sg.SkipUniformGroup(value)
            trace.append(("optimizer", trainer.global_steps))
            return SimpleNamespace(keys=["response"], partition_id="train")

        trainer.step = step
        trainer.on_step_end = lambda: trace.append(("publish", trainer.global_steps))

        def validate():
            self.assertEqual(trainer.data_pass.groups_seen % 3, 0)
            trace.append(("validate", trainer.data_pass.groups_seen))
            trainer._igsm_state.observe(trainer.global_steps, 0.1)
        trainer._validate_epoch = validate
        with TemporaryDirectory() as folder, ExitStack() as stack:
            trainer._igsm_run_dir = Path(folder)
            for name in ["Tracking", "ValidationGenerationsLogger", "DapoFilteredRewardTableLogger"]:
                stack.enter_context(patch.object(sg, name, Mock()))
            stack.enter_context(patch.object(sg.SkipManager, "init"))
            stack.enter_context(patch.object(sg.SkipManager, "set_step"))
            stack.enter_context(patch.object(sg.tq, "kv_clear"))
            trainer.fit(object())
        return trainer, trace

    def test_updates_and_weight_publication_precede_next_question(self):
        trainer, trace = self.run_fit([None, 0, None, 1, None, 0], 2)
        self.assertLess(trace.index(("publish", 1)), trace.index(("generate", 2)))
        self.assertLess(trace.index(("publish", 3)), trace.index(("generate", 4)))
        self.assertNotIn(("optimizer", 2), trace)
        self.assertEqual([v for v in trace if v[0] == "validate"], [("validate", 3), ("validate", 6)])
        self.assertEqual(trainer.data_pass.optimizer_updates, 3)
        self.assertEqual(trainer.data_pass.groups_seen, 6)
        self.assertEqual(trainer._igsm_state.best_step, 6)

    def test_all_filtered_pass_still_completes_and_validates_without_update(self):
        trainer, trace = self.run_fit([0, 1, 0], 1)
        self.assertEqual(trainer.data_pass.optimizer_updates, 0)
        self.assertEqual(trainer.data_pass.groups_seen, 3)
        self.assertEqual([v for v in trace if v[0] == "validate"], [("validate", 3)])
        self.assertFalse(any(v[0] == "publish" for v in trace))

    def test_uniform_group_is_filtered_without_refill(self):
        refill = Mock(side_effect=AssertionError("Unexpected hidden refill"))
        buffer = sg.SingleGroupReplayBuffer(deadline_check=Mock(), trainer_mode="sync", trainer_config={},
            max_off_policy_threshold=None, max_off_policy_strategy="drop", sampler_kwargs={},
            refill_fn=refill, filter_groups_metric="reward", train_batch_size=1, gen_batch_size=1)
        buffer.finished_keys["train"] = {"q"}
        with patch.object(ReplayBuffer, "_sync_metadata_from_transfer_queue"), \
             patch.object(ReplayBuffer, "_terminal_eviction_reasons", return_value=(set(), {"q"}, set(), Counter({0.0: 1}))), \
             patch.object(ReplayBuffer, "_clear_groups"):
            with self.assertRaises(sg.SkipUniformGroup):
                buffer.sample(1, "train", 1)
        refill.assert_not_called()

    def test_actual_epoch_configuration_has_900_attempts_and_validates_every90(self):
        argv = ["rl", "--algorithm", "dapo", "--hardware", "3090-24g", "--train-data", "/tmp/pool.jsonl",
                "--output-dir", "/tmp/unused", "--train-batch-size", "1", "--mini-batch-size", "1",
                "--epoch-unit", "data-pass", "--epochs", "10"]
        with patch.object(sys, "argv", argv):
            args = rl.arguments()
        cfg = rl.build_config(args, {"train_count": 90, "train_file": "/tmp/a", "val_file": "/tmp/b"})
        self.assertEqual(cfg.trainer.total_training_steps, 900)
        self.assertEqual(cfg.trainer.test_freq, 90)
        self.assertEqual(cfg.data.gen_batch_size, 1)
        self.assertEqual(cfg.actor_rollout_ref.rollout.n, 8)
        self.assertEqual(cfg.igsm_rl.epoch_unit, "data-pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
