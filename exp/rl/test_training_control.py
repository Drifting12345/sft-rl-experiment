"""CPU checks for latest-best selection, early stopping, and the 100-step budget."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "exp/rl"), str(ROOT / "exp"), str(ROOT / "verl")]

from trainer import RewardEarlyStopState
import train_rl as rl


class TrainingControlTests(unittest.TestCase):
    def test_latest_tied_best_including_zero_reward(self):
        for reward in [0.0, 0.1]:
            with self.subTest(reward=reward):
                state = RewardEarlyStopState()
                for step in [2, 4, 6]:
                    self.assertTrue(state.observe(step, reward))
                    self.assertEqual(state.best_step, step)
                    self.assertEqual(state.best_reward, reward)
                    self.assertFalse(state.should_stop)

    def test_recovering_to_historical_best_selects_latest(self):
        state = RewardEarlyStopState()
        for step, reward in [(2, 0.2), (4, 0.1), (6, 0.2), (8, 0.1)]:
            state.observe(step, reward)
        self.assertEqual(state.best_step, 6)
        self.assertEqual(state.best_reward, 0.2)
        self.assertEqual(state.consecutive_declines, 1)
        self.assertFalse(state.should_stop)

    def test_two_strict_declines_stop_without_replacing_best(self):
        state = RewardEarlyStopState()
        for step, reward in [(2, 0.3), (4, 0.2), (6, 0.1)]:
            state.observe(step, reward)
        self.assertTrue(state.should_stop)
        self.assertEqual(state.consecutive_declines, 2)
        self.assertEqual(state.best_step, 2)

    def test_plateau_resets_declines_without_replacing_higher_best(self):
        state = RewardEarlyStopState()
        for step, reward in [(2, 0.3), (4, 0.2), (6, 0.2), (8, 0.1)]:
            state.observe(step, reward)
        self.assertEqual(state.consecutive_declines, 1)
        self.assertFalse(state.should_stop)
        self.assertEqual(state.best_step, 2)

    def test_checkpoint_restore_keeps_latest_best_and_replay_is_idempotent(self):
        state = RewardEarlyStopState()
        state.observe(2, 0.1)
        state.observe(4, 0.1)
        saved = asdict(state)
        restored = RewardEarlyStopState.restore(saved, patience=2, checkpoint_step=4)
        self.assertFalse(restored.observe(4, 0.1))
        self.assertEqual(asdict(restored), saved)
        restored.observe(6, 0.1)
        self.assertEqual(restored.best_step, 6)

    def test_explicit_budget_reaches_100_and_preserves_sampling(self):
        argv = ["rl", "--algorithm", "dapo", "--hardware", "3090-24g",
                "--train-data", "/tmp/pool.jsonl", "--output-dir", "/tmp/rl-unused",
                "--epochs", "50", "--max-steps", "100", "--validate-every", "2",
                "--save-every", "1", "--rollout-execution", "cudagraph"]
        with patch.object(sys, "argv", argv):
            args = rl.arguments()
        cfg = rl.build_config(args, {"train_count": 90, "train_file": "/tmp/train.parquet",
                                     "val_file": "/tmp/val.parquet"})
        self.assertEqual(cfg.trainer.total_training_steps, 100)
        self.assertEqual(cfg.trainer.total_epochs, 50)
        self.assertEqual((90 // cfg.data.train_batch_size) * cfg.trainer.total_epochs, 100)
        self.assertEqual(cfg.trainer.test_freq, 2)
        self.assertEqual(cfg.trainer.save_freq, 1)
        self.assertEqual(cfg.trainer.resume_mode, "disable")
        self.assertEqual(cfg.actor_rollout_ref.actor.optim.lr, 1e-6)
        self.assertEqual(cfg.actor_rollout_ref.rollout.n, 8)
        self.assertEqual(cfg.actor_rollout_ref.rollout.temperature, 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
