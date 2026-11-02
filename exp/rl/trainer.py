"""Small V1 sync trainer extension for iGSM validation and checkpoint retention.

The upstream training loop owns rollout cleanup and logging. Early stopping is
requested after validation but raised only before the next training step.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

from omegaconf import open_dict
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

if __package__:
    from .bounded_sampling import BoundedSamplingMixin
else:
    from bounded_sampling import BoundedSamplingMixin


logger = logging.getLogger(__name__)
STATE_FILENAME = "igsm_rl_state.json"
STATE_VERSION = 1
_STEP_DIRECTORY = re.compile(r"global_step_(0|[1-9][0-9]*)\Z")


@dataclass
class RewardEarlyStopState:
    """Compare adjacent validation rewards; ties reset the decline streak."""

    patience: int = 2
    previous_reward: float | None = None
    consecutive_declines: int = 0
    best_reward: float | None = None
    best_step: int | None = None
    last_validation_step: int | None = None
    should_stop: bool = False

    def observe(self, step: int, reward: float) -> bool:
        if type(step) is not int or step < 0:
            raise ValueError("Validation step must be a nonnegative integer")
        if isinstance(reward, bool) or not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError(f"Expected finite mean binary reward in [0, 1], got {reward!r}")
        if self.last_validation_step is not None:
            if step < self.last_validation_step:
                raise ValueError("Validation steps cannot go backwards")
            if step == self.last_validation_step:
                # V1 validates restored weights before its first new update.
                return False
        if self.should_stop:
            raise RuntimeError("Cannot advance an already stopped training state")
        self.consecutive_declines = (
            self.consecutive_declines + 1
            if self.previous_reward is not None and reward < self.previous_reward
            else 0
        )
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward, self.best_step = float(reward), step
        self.previous_reward = float(reward)
        self.last_validation_step = step
        self.should_stop = self.consecutive_declines >= self.patience
        return True

    @classmethod
    def restore(cls, payload: dict, *, patience: int, checkpoint_step: int):
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Invalid early-stopping state fields")
        state = cls(**payload)
        if type(state.patience) is not int or state.patience != patience or patience < 1:
            raise ValueError("Checkpoint early-stop patience does not match this run")
        if type(state.consecutive_declines) is not int or state.consecutive_declines < 0:
            raise ValueError("Invalid checkpoint decline count")
        if type(state.should_stop) is not bool or state.should_stop != (state.consecutive_declines >= patience):
            raise ValueError("Inconsistent checkpoint stop flag")
        for field in ("previous_reward", "best_reward"):
            value = getattr(state, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid checkpoint {field}")
        for field in ("best_step", "last_validation_step"):
            value = getattr(state, field)
            if value is not None and (type(value) is not int or not 0 <= value <= checkpoint_step):
                raise ValueError(f"Invalid checkpoint {field}")
        if state.last_validation_step is None:
            if any(value is not None for value in (state.previous_reward, state.best_reward, state.best_step)):
                raise ValueError("Checkpoint reward history is incomplete")
            if state.consecutive_declines:
                raise ValueError("Checkpoint decline count has no validation history")
        elif any(value is None for value in (state.previous_reward, state.best_reward, state.best_step)):
            raise ValueError("Checkpoint reward history is incomplete")
        elif state.best_step > state.last_validation_step or state.best_reward < state.previous_reward:
            raise ValueError("Checkpoint best reward history is inconsistent")
        return state


def validation_mean_reward(metrics: dict) -> tuple[str, float]:
    """Accept only this task's single-sample binary accuracy/reward mean."""
    for key in (
        "val-core/igsm_rl/acc/mean@1",
        "val-core/igsm_rl/reward/mean@1",
        "val-aux/igsm_rl/reward/mean@1",
    ):
        if key in metrics:
            value = float(metrics[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid validation binary reward: {key}={value}")
            return key, value
    raise ValueError(
        "Missing iGSM validation mean@1; require data_source=igsm_rl, "
        "val_kwargs.n=1 and binary reward/acc. Available metrics: " + ", ".join(sorted(metrics))
    )


def _atomic_replace(source, target) -> None:
    # Windows indexers can briefly hold a JSON file without delete sharing.
    # Retry only that transient platform condition; POSIX errors stay immediate.
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.01 * 2 ** attempt)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _EarlyStopRequested(Exception):
    """Raised only at the boundary after upstream has cleaned the last batch."""


class IGSMTrainer(BoundedSamplingMixin, PPOTrainerSync):
    """Keep upstream sync semantics and retain complete best/latest checkpoints."""

    def __init__(self, config):
        options = config.igsm_rl
        patience = options.get("patience", 2)
        if type(patience) is not int or patience < 1:
            raise ValueError("igsm_rl.patience must be a positive integer")
        if config.trainer.v1.trainer_mode != "sync":
            raise ValueError("IGSMTrainer requires trainer.v1.trainer_mode=sync")
        if config.trainer.get("val_only", False) and not config.trainer.get("val_before_train", True):
            raise ValueError("trainer.val_only requires trainer.val_before_train=True")
        if config.trainer.test_freq <= 0 and not config.trainer.get("val_only", False):
            raise ValueError("Validation early stopping requires trainer.test_freq > 0")
        actor_checkpoint = config.actor_rollout_ref.actor.get("checkpoint", {})
        critic_checkpoint = config.get("critic", {}).get("checkpoint", {})
        if actor_checkpoint.get("async_save", False) or critic_checkpoint.get("async_save", False):
            raise ValueError("Best/latest retention requires synchronous checkpoint writes")
        # Upstream FIFO retention cannot protect the historical best checkpoint.
        with open_dict(config.trainer):
            config.trainer.max_actor_ckpt_to_keep = None
            config.trainer.max_critic_ckpt_to_keep = None
            config.trainer.remove_previous_ckpt_in_save = False
            config.trainer.del_local_ckpt_after_load = False
        self._igsm_state = RewardEarlyStopState(patience=patience)
        self._igsm_run_dir = Path(options.run_dir).expanduser().resolve()
        self._igsm_checkpoint_root = Path(config.trainer.default_local_dir).expanduser().resolve()
        self._igsm_latest_checkpoint: Path | None = None
        self._igsm_best_checkpoint: Path | None = None
        self._igsm_last_trained_step: int | None = None
        self._igsm_validation_metrics: dict | None = None
        super().__init__(config)

    def _read_complete_checkpoint(self, path: Path) -> tuple[dict, RewardEarlyStopState]:
        match = _STEP_DIRECTORY.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            raise ValueError(f"Not a managed checkpoint directory: {path}")
        payload = json.loads((path / STATE_FILENAME).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid checkpoint completion marker: {path}")
        step = int(match.group(1))
        if (
            payload.get("schema_version") != STATE_VERSION or payload.get("complete") is not True
            or type(payload.get("checkpoint_step")) is not int or payload["checkpoint_step"] != step
            or not (path / "actor").is_dir() or not (path / "data.pt").is_file()
        ):
            raise ValueError(f"Incomplete or mismatched checkpoint: {path}")
        state = RewardEarlyStopState.restore(
            payload["early_stopping"], patience=self._igsm_state.patience, checkpoint_step=step
        )
        return payload, state

    def _complete_checkpoints(self):
        if not self._igsm_checkpoint_root.exists():
            return []
        complete = []
        for path in self._igsm_checkpoint_root.iterdir():
            if (
                _STEP_DIRECTORY.fullmatch(path.name) and not path.is_symlink()
                and (path / STATE_FILENAME).exists()
            ):
                # No marker means an interrupted/unmanaged save. An existing but
                # invalid marker is an error, never a reason to silently restart.
                payload, state = self._read_complete_checkpoint(path)
                complete.append((payload["checkpoint_step"], path, payload, state))
        return sorted(complete, key=lambda item: item[0])

    def _load_checkpoint(self):
        mode = self.config.trainer.resume_mode
        original_path = self.config.trainer.resume_from_path
        if mode == "auto":
            complete = self._complete_checkpoints()
            selected = complete[-1][1] if complete else None
        elif mode == "resume_path":
            if not isinstance(original_path, str):
                raise ValueError("trainer.resume_from_path must name a checkpoint directory")
            selected = Path(original_path).expanduser().resolve()
        elif mode == "disable":
            if self._complete_checkpoints():
                raise ValueError("Refusing to overwrite an existing run with resume_mode=disable")
            selected = None
        else:
            raise ValueError(f"Unknown resume mode: {mode}")
        if selected is None:
            # Do not let the upstream tracker pick an interrupted save lacking its sidecar.
            with open_dict(self.config.trainer):
                self.config.trainer.resume_mode = "disable"
            try:
                super()._load_checkpoint()
            finally:
                with open_dict(self.config.trainer):
                    self.config.trainer.resume_mode = mode
            return
        payload, state = self._read_complete_checkpoint(selected)
        best = None
        if state.best_step is not None:
            sibling = selected.parent / f"global_step_{state.best_step}"
            best = sibling if sibling.is_dir() else Path(payload["best_checkpoint"]).resolve()
            best_payload, _ = self._read_complete_checkpoint(best)
            if best_payload["checkpoint_step"] != state.best_step:
                raise ValueError("Best checkpoint pointer does not match saved early-stop state")
        with open_dict(self.config.trainer):
            self.config.trainer.resume_mode = "resume_path"
            self.config.trainer.resume_from_path = str(selected)
        try:
            super()._load_checkpoint()
        finally:
            with open_dict(self.config.trainer):
                self.config.trainer.resume_mode = mode
                self.config.trainer.resume_from_path = original_path
        if self.global_steps != payload["checkpoint_step"]:
            raise ValueError("Loaded model step does not match early-stop state")
        self._igsm_state = state
        self._igsm_latest_checkpoint = selected
        self._igsm_best_checkpoint = best
        self._igsm_last_trained_step = self.global_steps
        self._write_run_metadata("resumed")

    def _validate(self):
        metrics = super()._validate()
        self._igsm_validation_metrics = metrics
        return metrics

    def on_validate_end(self):
        super().on_validate_end()
        if self._igsm_validation_metrics is None:
            raise RuntimeError("Validation hook ran without completed validation metrics")
        key, score = validation_mean_reward(self._igsm_validation_metrics)
        if not self._igsm_state.observe(int(self.global_steps), score):
            return
        if self._igsm_state.best_step == self.global_steps:
            self._igsm_best_checkpoint = self._igsm_checkpoint_root / f"global_step_{self.global_steps}"
        logger.info(
            "iGSM validation step=%s %s=%s consecutive_declines=%s stop=%s",
            self.global_steps, key, score, self._igsm_state.consecutive_declines, self._igsm_state.should_stop,
        )
        # Upstream saves before validation; this also saves steps between save_freq boundaries.
        self._save_checkpoint()

    def on_step_begin(self):
        if self._igsm_state.should_stop:
            raise _EarlyStopRequested()
        return super().on_step_begin()

    def on_step_end(self):
        super().on_step_end()  # Preserve sync weight publication before validation.
        self._igsm_last_trained_step = int(self.global_steps)

    def _save_checkpoint(self):
        step = int(self.global_steps)
        directory = self._igsm_checkpoint_root / f"global_step_{step}"
        if self._igsm_latest_checkpoint != directory:
            if directory.exists() or directory.is_symlink():
                if (
                    directory.is_symlink() or not directory.is_dir()
                    or directory.resolve().parent != self._igsm_checkpoint_root
                    or (directory / STATE_FILENAME).exists()
                ):
                    raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {directory}")
                # A previous process may have died during the model write. Keep
                # that directory intact under a unique name so resumed training
                # can save this same step without deleting the partial files.
                isolated = self._igsm_checkpoint_root / f".incomplete-{directory.name}-{uuid.uuid4().hex}"
                if isolated.resolve().parent != self._igsm_checkpoint_root:
                    raise ValueError("Incomplete checkpoint isolation escaped the checkpoint root")
                directory.rename(isolated)
                logger.warning("Preserved interrupted checkpoint %s as %s", directory, isolated)
            super()._save_checkpoint()
        # Written only after all synchronous worker writes and data.pt have returned.
        payload = {
            "schema_version": STATE_VERSION,
            "complete": True,
            "checkpoint_step": step,
            "early_stopping": asdict(self._igsm_state),
            "best_checkpoint": str(self._igsm_best_checkpoint) if self._igsm_best_checkpoint else None,
        }
        _atomic_json(directory / STATE_FILENAME, payload)
        self._igsm_latest_checkpoint = directory
        self._write_run_metadata("early_stop_pending" if self._igsm_state.should_stop else "running")
        # Re-publish the standard tracker only after our complete marker exists.
        tracker = self._igsm_checkpoint_root / "latest_checkpointed_iteration.txt"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{tracker.name}.", dir=tracker.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(str(step))
                stream.flush()
                os.fsync(stream.fileno())
            _atomic_replace(temporary, tracker)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._prune_complete_checkpoints()

    def _write_run_metadata(self, status: str):
        state = asdict(self._igsm_state)
        latest = self._igsm_latest_checkpoint
        best = self._igsm_best_checkpoint
        _atomic_json(self._igsm_run_dir / "state.json", {
            "schema_version": STATE_VERSION, "status": status, "early_stopping": state,
            "latest_checkpoint": str(latest) if latest else None,
            "best_checkpoint": str(best) if best else None,
        })
        _atomic_json(self._igsm_run_dir / "latest_checkpoint.json", {
            "path": str(latest) if latest else None,
            "step": int(latest.name.removeprefix("global_step_")) if latest else None,
        })
        _atomic_json(self._igsm_run_dir / "best_checkpoint.json", {
            "path": str(best) if best else None,
            "step": self._igsm_state.best_step, "validation_mean_reward": self._igsm_state.best_reward,
        })

    def _prune_complete_checkpoints(self):
        keep = {self._igsm_latest_checkpoint, self._igsm_best_checkpoint}
        for _, directory, _, _ in self._complete_checkpoints():
            if directory in keep:
                continue
            # Both name and resolved parent are checked immediately before recursive deletion.
            if directory.is_symlink() or directory.resolve().parent != self._igsm_checkpoint_root:
                raise ValueError(f"Checkpoint deletion escaped its configured root: {directory}")
            shutil.rmtree(directory)

    def fit(self, agent_loop_manager):
        if self._igsm_state.should_stop:
            logger.info("Restored checkpoint already met the early-stop condition; no training resumed")
            self._write_run_metadata("early_stopped")
            self._shutdown_dump_executor()
            return
        # Upstream checkpoints before validating the same step. A process can
        # die in that gap, so restored weights must be evaluated before another
        # update even when this run normally skips its initial validation.
        # observe() de-duplicates checkpoints whose validation was already saved.
        trainer_config = self.config.trainer
        restored_checkpoint = self._igsm_latest_checkpoint is not None
        had_initial_validation = "val_before_train" in trainer_config
        initial_validation = trainer_config.get("val_before_train", True)
        if restored_checkpoint:
            with open_dict(trainer_config):
                trainer_config.val_before_train = True
        try:
            result = super().fit(agent_loop_manager)
        except _EarlyStopRequested:
            # The preceding step already completed upstream TQ cleanup and metric logging.
            self.on_train_end()
            self._shutdown_dump_executor()
            self._write_run_metadata("early_stopped")
            logger.info("Stopped after %s consecutive validation reward declines", self._igsm_state.patience)
            return
        finally:
            if restored_checkpoint:
                with open_dict(trainer_config):
                    if had_initial_validation:
                        trainer_config.val_before_train = initial_validation
                    else:
                        del trainer_config["val_before_train"]
        # Ensure the actual last trained step is retained even when save_freq is disabled.
        last_step = self._igsm_last_trained_step
        if last_step is not None:
            next_step = self.global_steps
            try:
                self.global_steps = last_step
                self._save_checkpoint()
            finally:
                self.global_steps = next_step
        self._write_run_metadata("early_stopped" if self._igsm_state.should_stop else "completed")
        return result
