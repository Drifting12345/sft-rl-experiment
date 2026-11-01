"""Exact integer-answer reward for iGSM; this does not verify the reasoning."""
import re

FINAL = re.compile(r"(?m)^####[ \t]+([+-]?\d+)[ \t]*\s*\Z")
INTEGER = re.compile(r"[+-]?\d+\Z")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "igsm":
        raise ValueError(f"Expected data_source='igsm', got {data_source!r}")
    expected = str(ground_truth).strip()
    if not INTEGER.fullmatch(expected):
        raise ValueError("iGSM ground_truth must be an integer")
    # Do not allow an answer list to earn credit by including a correct number.
    markers = re.findall(r"(?m)^####", solution_str)
    match = FINAL.search(solution_str)
    correct = len(markers) == 1 and match is not None and int(match[1]) == int(expected)
    return {"score": float(correct), "acc": float(correct)}
