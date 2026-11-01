"""Default rule-based reward for the iGSM text-training subset."""
from .igsm import compute_score


def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)


_default_compute_score = default_compute_score


def get_default_compute_score(reward_name=None):
    return default_compute_score


__all__ = ["default_compute_score", "get_default_compute_score"]
