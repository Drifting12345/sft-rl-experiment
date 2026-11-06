"""One question group at a time, with epochs defined by actual dataset passes."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import uuid

import numpy as np
from omegaconf import OmegaConf
import transfer_queue as tq

from trainer import IGSMTrainer, _atomic_json, validation_mean_reward
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.utils.skip import SkipManager
from verl.utils.tracking import Tracking, ValidationGenerationsLogger, DapoFilteredRewardTableLogger


@dataclass
class DataPassProgress:
    ids: tuple
    groups_seen: int = 0
    optimizer_updates: int = 0
    skipped_wrong: int = 0
    skipped_correct: int = 0
    current_ids: list = field(default_factory=list)

    def record(self, question_id, updated, uniform_reward=None):
        if question_id not in self.ids or question_id in self.current_ids:
            raise ValueError("Dataset pass repeated or introduced a question before covering the training pool")
        if not updated and uniform_reward not in (0.0, 1.0):
            raise ValueError("A skipped group must have a uniform binary reward")
        self.current_ids.append(question_id)
        self.groups_seen += 1
        self.optimizer_updates += int(updated)
        self.skipped_wrong += int(not updated and uniform_reward == 0)
        self.skipped_correct += int(not updated and uniform_reward == 1)
        boundary = len(self.current_ids) == len(self.ids)
        if boundary:
            assert set(self.current_ids) == set(self.ids)
            self.current_ids.clear()
        return boundary

    def payload(self):
        return dict(groups_seen=self.groups_seen, optimizer_updates=self.optimizer_updates,
                    skipped_wrong=self.skipped_wrong, skipped_correct=self.skipped_correct,
                    completed_epochs=self.groups_seen // len(self.ids),
                    epoch_progress=self.groups_seen / len(self.ids), current_ids=list(self.current_ids))

    @classmethod
    def restore(cls, ids, saved):
        state = cls(tuple(ids), **{key: saved[key] for key in
                    ["groups_seen", "optimizer_updates", "skipped_wrong", "skipped_correct", "current_ids"]})
        assert state.groups_seen == state.optimizer_updates + state.skipped_wrong + state.skipped_correct
        assert len(state.current_ids) == state.groups_seen % len(ids)
        assert len(set(state.current_ids)) == len(state.current_ids)
        assert set(state.current_ids) <= set(ids)
        assert saved["completed_epochs"] == state.groups_seen // len(ids)
        return state


class SkipUniformGroup(Exception):
    def __init__(self, reward):
        self.reward = float(reward)
        super().__init__(f"Uniform reward {reward}; no optimizer update")


class SingleGroupReplayBuffer(ReplayBuffer):
    """Never refill inside a candidate iteration; uniform groups advance the data pass."""
    def __init__(self, *, deadline_check, **kwargs):
        self.deadline_check = deadline_check
        super().__init__(**kwargs)

    def sample(self, global_steps, partition_id, batch_size):
        if partition_id == "val":
            return super().sample(global_steps, partition_id, batch_size)
        assert batch_size == 1
        debug_time = 0
        while True:
            self.deadline_check()
            self._sync_metadata_from_transfer_queue()
            inflight = len(self.pending_keys[partition_id]) + len(self.running_keys[partition_id])
            terminal = self.finished_keys[partition_id] | self.failure_keys[partition_id]
            assert len(terminal) + inflight <= 1
            if terminal and not inflight:
                if self.failure_keys[partition_id]:
                    raise RuntimeError("Single-group generation failed; incomplete answers are not training data")
                reasons = self._terminal_eviction_reasons(global_steps, partition_id)
                filtered = reasons[1]
                if filtered:
                    values = list(reasons[3])
                    assert len(filtered) == 1 and len(values) == 1 and values[0] in (0.0, 1.0)
                    self._clear_groups(partition_id, filtered)
                    raise SkipUniformGroup(values[0])
                return super().sample(global_steps, partition_id, 1)
            debug_time = self._wait_for_next_poll(partition_id, debug_time)


class SingleGroupEpochTrainer(IGSMTrainer):
    def __init__(self, config):
        manifest = json.loads((Path(config.igsm_rl.run_dir) / "run_config.json").read_text())
        ids = tuple(row["id"] for row in manifest["dataset"]["split_manifest"]["train"])
        self.data_pass = DataPassProgress(ids)
        self.epoch_validation_history = []
        self.current_question_id = None
        assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == 1
        assert config.actor_rollout_ref.rollout.n == 8
        assert config.algorithm.filter_groups.enable
        super().__init__(config)

    def _build_replay_buffer(self):
        self._igsm_generation_prompt_count = 0
        self._igsm_generation_prompt_limit = None
        self._igsm_generation_deadline = None
        def unexpected_refill(count):
            raise RuntimeError("Single-group epochs must advance the dataset instead of refilling a hidden step")
        return SingleGroupReplayBuffer(
            deadline_check=self._check_generation_deadline, trainer_mode="sync", trainer_config={},
            max_off_policy_threshold=None, max_off_policy_strategy="drop", sampler_kwargs={},
            refill_fn=unexpected_refill, filter_groups_metric=self._resolve_filter_groups_metric(),
            train_batch_size=1, gen_batch_size=1, max_inflight_gen_batches=1,
            sync_refill_failed_groups=False)

    def _fetch_one_gen_batch(self):
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch = next(self.train_dataloader_it)
        except StopIteration:
            assert not self.data_pass.current_ids, "Loader ended before a complete dataset pass"
            self.train_dataloader_it = iter(self.train_dataloader)
            batch = next(self.train_dataloader_it)
        assert len(batch["extra_info"]) == 1
        self.current_question_id = batch["extra_info"][0]["id"]
        batch["uid"] = np.array([str(uuid.uuid4())], dtype=object)
        return tu.get_tensordict(batch)

    def _progress_path(self, step):
        return self._igsm_run_dir / "data-pass-state" / f"global_step_{step}.json"

    def _save_checkpoint(self):
        # Commit progress before the native completion marker is published.
        _atomic_json(self._progress_path(self.global_steps), dict(
            progress=self.data_pass.payload(), validation_history=self.epoch_validation_history,
            checkpoint_unit="candidate_groups_seen"))
        super()._save_checkpoint()

    def _write_run_metadata(self, status):
        super()._write_run_metadata(status)
        path = self._igsm_run_dir / "state.json"
        value = json.loads(path.read_text())
        value.update(epoch_unit="data-pass", checkpoint_unit="candidate_groups_seen",
                     data_pass=self.data_pass.payload(), validation_history=self.epoch_validation_history)
        _atomic_json(path, value)

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self._igsm_latest_checkpoint is not None:
            saved = json.loads(self._progress_path(self.global_steps).read_text())
            self.data_pass = DataPassProgress.restore(self.data_pass.ids, saved["progress"])
            assert self.data_pass.groups_seen == self.global_steps
            last = self._igsm_state.last_validation_step or 0
            self.epoch_validation_history = [v for v in saved["validation_history"] if v["groups_seen"] <= last]
            self._write_run_metadata("resumed")

    def _validate_epoch(self):
        self.on_validate_begin()
        metrics = self._validate()
        _, score = validation_mean_reward(metrics)
        self.epoch_validation_history.append(dict(epoch=self.data_pass.groups_seen // len(self.data_pass.ids),
            groups_seen=self.data_pass.groups_seen, optimizer_updates=self.data_pass.optimizer_updates,
            accuracy=score))
        self.on_validate_end()  # Latest tied best, strict-decline early stopping, and checkpoint save.
        _atomic_json(self._igsm_run_dir / "epoch-validation.json", self.epoch_validation_history)
        self.logger.log(data=metrics, step=self.global_steps)

    def fit(self, agent_loop_manager):
        self.agent_loop_manager = agent_loop_manager
        SkipManager.init(self.config)
        self.logger = Tracking(project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name, default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True))
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name)
        self.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name)
        assert self.parameter_sync_step == 1
        assert self.total_training_steps == len(self.data_pass.ids) * self.config.trainer.total_epochs
        self.prev_step_profile = self.curr_step_profile = self.next_step_profile = False
        self._reissue_inflight_prompts()
        self.on_train_begin()
        events_path = self._igsm_run_dir / "question-events.jsonl"
        if events_path.exists():
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            committed = [v for v in events if v["groups_seen"] <= self.data_pass.groups_seen]
            if len(committed) != len(events):
                events_path.rename(events_path.with_suffix(f".uncommitted-{uuid.uuid4().hex}.jsonl"))
                events_path.write_text("".join(json.dumps(v) + "\n" for v in committed))
        if self.global_steps and self.global_steps % len(self.data_pass.ids) == 0:
            if (self._igsm_state.last_validation_step or 0) < self.global_steps:
                self._validate_epoch()
        while self.global_steps < self.total_training_steps and not self._igsm_state.should_stop:
            self.global_steps += 1
            SkipManager.set_step(self.global_steps)
            self.timing_raw = {}
            self.on_step_begin()
            metrics, batch, uniform_reward = {}, None, None
            try:
                with marked_timer("step", self.timing_raw):
                    batch = self.step(metrics, self.timing_raw)
                    self.on_step_end()  # Publish the updated LoRA before another training question is dispatched.
                    metrics.update(self._consume_sync_metrics())
                self._compute_metrics(batch, metrics, self.timing_raw, self.global_steps,
                                      self.data_pass.groups_seen // len(self.data_pass.ids))
                updated = True
            except SkipUniformGroup as skipped:
                uniform_reward, updated = skipped.reward, False
            boundary = self.data_pass.record(self.current_question_id, updated, uniform_reward)
            event = dict(question_id=self.current_question_id, updated=updated, uniform_reward=uniform_reward,
                         **self.data_pass.payload())
            event.pop("current_ids")
            with events_path.open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            metrics.update({"data_pass/epoch": self.data_pass.groups_seen / len(self.data_pass.ids),
                            "data_pass/groups_seen": self.data_pass.groups_seen,
                            "data_pass/optimizer_updates": self.data_pass.optimizer_updates,
                            "data_pass/group_updated": int(updated)})
            if batch is not None:
                tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
            if updated:
                self._save_checkpoint()
            if boundary:
                self._validate_epoch()
            self.logger.log(data=metrics, step=self.global_steps)
            self._write_run_metadata("running")
        self.on_train_end()
        self._save_checkpoint()
        self._write_run_metadata("early_stopped" if self._igsm_state.should_stop else "completed")
        self._shutdown_dump_executor()
