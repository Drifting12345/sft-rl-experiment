"""Bound V1 synchronous DAPO refill cost without replacing its sampling loop."""
from __future__ import annotations

import math
import time

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


class BoundedReplayBuffer(ReplayBuffer):
    """Use the upstream filtering loop, checking its deadline while polling.

    Validation keeps upstream semantics: no DAPO filtering or training budget.
    Checks surround backend calls; they do not interrupt a blocked backend RPC.
    """

    def __init__(self, *, deadline_check, **kwargs):
        self._deadline_check = deadline_check
        self._bounded_partition = None
        super().__init__(**kwargs)

    def sample(self, global_steps, partition_id, batch_size):
        previous = self._bounded_partition
        self._bounded_partition = partition_id
        try:
            if partition_id != "val":
                self._deadline_check()
            return super().sample(global_steps, partition_id, batch_size)
        finally:
            self._bounded_partition = previous

    def _sync_metadata_from_transfer_queue(self):
        bounded = self._bounded_partition is not None and self._bounded_partition != "val"
        if bounded:
            self._deadline_check()
        result = super()._sync_metadata_from_transfer_queue()
        if bounded:
            self._deadline_check()
        return result

    def _wait_for_next_poll(self, partition_id, last_debug_time):
        if partition_id != "val":
            self._deadline_check()
        result = super()._wait_for_next_poll(partition_id, last_debug_time)
        if partition_id != "val":
            self._deadline_check()
        return result


class BoundedSamplingMixin:
    """Place before PPOTrainerSync in the MRO; GRPO passes through unchanged.

    DAPO is identified by algorithm.filter_groups.enable. The configured
    advantage estimator remains grpo for both GRPO and DAPO. Prompt limits
    include the initial batch and every refill; multiply by rollout.n to obtain
    the maximum number of requested trajectories per training step.
    """

    def _dapo_filtering_enabled(self):
        algorithm = _get(self.config, "algorithm")
        groups = _get(algorithm, "filter_groups")
        return bool(_get(groups, "enable", False))

    def _sampling_limits(self):
        options = _get(self.config, "igsm_rl")
        maximum = _get(options, "max_generation_batches")
        timeout = _get(options, "generation_timeout_s")
        batch_size = _get(_get(self.config, "data"), "train_batch_size")
        if type(maximum) is not int or maximum < 1:
            raise ValueError("DAPO igsm_rl.max_generation_batches must be a positive integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("DAPO data.train_batch_size must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("DAPO igsm_rl.generation_timeout_s must be positive and finite")
        return batch_size * maximum, float(timeout)

    def _build_replay_buffer(self):
        if not self._dapo_filtering_enabled():
            return super()._build_replay_buffer()
        if self.trainer_mode != "sync":
            raise ValueError("Bounded DAPO sampling supports the synchronous trainer only")
        self._sampling_limits()
        sampler = self.config.trainer.v1.sampler
        custom = _get(sampler, "custom_sampler")
        if _get(custom, "path") or _get(custom, "name"):
            raise ValueError("Bounded DAPO sampling cannot be combined with another custom sampler")
        groups = self.config.algorithm.filter_groups
        self._igsm_generation_prompt_count = 0
        self._igsm_generation_prompt_limit = None
        self._igsm_generation_deadline = None
        return BoundedReplayBuffer(
            deadline_check=self._check_generation_deadline,
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler.max_off_policy_threshold,
            max_off_policy_strategy=sampler.max_off_policy_strategy,
            sampler_kwargs=sampler.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
            filter_groups_metric=self._resolve_filter_groups_metric(),
            sync_refill_failed_groups=bool(_get(sampler, "sync_refill_failed_groups", False)),
            train_batch_size=self.config.data.train_batch_size,
            gen_batch_size=1,
            max_inflight_gen_batches=_get(groups, "max_inflight_gen_batches", 1),
        )

    def on_step_begin(self):
        if self._dapo_filtering_enabled():
            limit, timeout = self._sampling_limits()
            self._igsm_generation_prompt_count = 0
            self._igsm_generation_prompt_limit = limit
            self._igsm_generation_deadline = time.monotonic() + timeout
        return super().on_step_begin()

    def _check_generation_deadline(self):
        deadline = getattr(self, "_igsm_generation_deadline", None)
        if deadline is None:
            raise RuntimeError("DAPO generation started before on_step_begin initialized its limits")
        if time.monotonic() >= deadline:
            submitted = self._igsm_generation_prompt_count
            raise RuntimeError(
                f"DAPO generation timed out at step {getattr(self, 'global_steps', '?')}: "
                f"submitted {submitted}/{self._igsm_generation_prompt_limit} prompts. "
                "Check reward/format validity and the supply of mixed-reward groups; "
                "no fallback to unfiltered GRPO was performed."
            )

    def _submit_batch_to_rollout(self, batch):
        if self._dapo_filtering_enabled():
            self._check_generation_deadline()
            requested = len(batch)
            submitted = self._igsm_generation_prompt_count
            limit = self._igsm_generation_prompt_limit
            if submitted + requested > limit:
                raise RuntimeError(
                    f"DAPO generation prompt budget exhausted at step {getattr(self, 'global_steps', '?')}: "
                    f"submitted={submitted}, requested={requested}, limit={limit}. "
                    "Not enough informative groups within the budget; "
                    "no extra prompts were dispatched and no reward was fabricated."
                )
            self._igsm_generation_prompt_count += requested
        return super()._submit_batch_to_rollout(batch)
