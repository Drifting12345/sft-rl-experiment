"""Strict iGSM accuracy reward, preserving real EOS before verl decodes it away.

Set reward.reward_manager.source=importlib, name=IGSMRewardManager and
module.path to this file, plus reward.custom_reward_function.path to this file
and name=compute_score. Rollout must use ignore_eos=false. The class is lazily
imported so the pure scoring function can be checked without a training runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from eval_qwen3_igsm import parse_answer


def completion_metadata(response_ids, valid_response_len: int, eos_token_ids, response_limit: int) -> dict:
    """An unfinished suffix such as answer:2 from answer:22 always scores zero."""
    if not 0 <= valid_response_len <= len(response_ids):
        raise ValueError("Invalid response length in reward input")
    finished = bool(valid_response_len and int(response_ids[valid_response_len - 1]) in eos_token_ids)
    truncated = not finished and valid_response_len >= response_limit
    return {"finished": finished, "valid_response_len": valid_response_len,
            "generation_has_eos": finished, "generation_truncated": truncated,
            "finish_reason": "stop" if finished else "length" if truncated else "incomplete"}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Canonical binary reward: mean reward is accuracy, with no shaping terms."""
    if data_source != "igsm_rl":
        raise ValueError(f"Unexpected reward data source: {data_source!r}")
    label = str(ground_truth).strip()
    if not re.fullmatch(r"[0-9]{1,2}", label) or not 0 <= int(label) <= 22:
        raise ValueError("iGSM ground truth must be an integer in 0..22")
    info = extra_info or {}
    # Fail closed when completion metadata was not supplied by our manager.
    reason = "stop" if info.get("generation_has_eos") is True and info.get("finished") is True else (
        "length" if info.get("generation_truncated") is True else "incomplete")
    predicted, _ = parse_answer(solution_str, reason)
    return float(predicted is not None and predicted == int(label))


def _eos_ids(model_path: Path, tokenizer) -> set[int]:
    config = json.loads((model_path / "generation_config.json").read_text(encoding="utf-8"))
    ids = set()
    for value in (config.get("eos_token_id"), tokenizer.eos_token_id):
        values = value if isinstance(value, list) else [value]
        for item in values:
            if type(item) is not int or item < 0:
                raise ValueError("Model and tokenizer must declare valid EOS token IDs")
            ids.add(item)
    if not ids:
        raise ValueError("No EOS token IDs declared by the model/tokenizer")
    return ids


def __getattr__(name):
    if name != "IGSMRewardManager":
        raise AttributeError(name)
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

    class IGSMRewardManager(NaiveRewardManager):
        """Keep Naive's reward plumbing while checking undecoded terminal tokens."""

        def __init__(self, config, tokenizer, compute_score, **kwargs):
            super().__init__(config, tokenizer, compute_score, **kwargs)
            rollout = config.actor_rollout_ref.rollout
            if rollout.get("ignore_eos", False):
                raise ValueError("IGSM reward requires rollout.ignore_eos=false")
            self._response_limit = int(rollout.response_length)
            self._eos_token_ids = _eos_ids(Path(config.actor_rollout_ref.model.path), tokenizer)

        async def run_single(self, data):
            data = data[-1:]
            item = data[0]
            response_ids = item.batch["responses"]
            width = response_ids.shape[-1]
            valid_length = int(item.batch["attention_mask"][-width:].sum().item()) if width else 0
            recorded_length = item.non_tensor_batch.get("response_len")
            if recorded_length is not None and int(recorded_length) != valid_length:
                raise ValueError("response_len disagrees with the reward attention mask")
            metadata = completion_metadata(response_ids, valid_length, self._eos_token_ids, self._response_limit)
            import numpy as np
            # Replace row dictionaries instead of mutating shared prompt metadata.
            for key in ("extra_info", "tool_extra_fields"):
                previous = data.non_tensor_batch.get(key)
                value = dict(previous[0]) if previous is not None and previous[0] is not None else {}
                value.update(metadata)
                data.non_tensor_batch[key] = np.array([value], dtype=object)
            return await super().run_single(data)

    IGSMRewardManager.__qualname__ = "IGSMRewardManager"
    globals()[name] = IGSMRewardManager
    return IGSMRewardManager
