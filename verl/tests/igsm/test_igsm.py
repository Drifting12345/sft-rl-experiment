"""CPU tests for the task-specific reward and data interface."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward = load("igsm_reward", "verl/utils/reward_score/igsm.py")
data = load("igsm_data", "examples/igsm/prepare_data.py")


class RewardTests(unittest.TestCase):
    def test_correct_answer(self):
        self.assertEqual(reward.compute_score("igsm", "Reasoning.\n#### 7", "7")["score"], 1.0)

    def test_wrong_answer(self):
        self.assertEqual(reward.compute_score("igsm", "Reasoning.\n#### 8", "7")["score"], 0.0)

    def test_format_is_required(self):
        for answer in ("7", "Reasoning contains 7", "#### 7.0", "#### 7 or 8", "#### 7\nextra text"):
            with self.subTest(answer=answer):
                self.assertEqual(reward.compute_score("igsm", answer, "7")["score"], 0.0)

    def test_multiple_answers_do_not_earn_credit(self):
        self.assertEqual(reward.compute_score("igsm", "#### 1\n#### 7", "7")["score"], 0.0)

    def test_invalid_gold_rejected(self):
        with self.assertRaises(ValueError):
            reward.compute_score("igsm", "#### 7", "invalid")


class DataTests(unittest.TestCase):
    def test_no_reference_solution_in_prompt_or_metadata(self):
        row = {"problem": "How many widgets?", "solution": "PRIVATE_REFERENCE_REASONING", "answer": "918273"}
        record = data.make_record(row, 0, "train")
        self.assertNotIn("PRIVATE_REFERENCE_REASONING", str(record))
        self.assertNotIn("918273", str(record["prompt"]))
        self.assertEqual(record["reward_model"]["ground_truth"], "918273")

    def test_deduplicated_disjoint_deterministic_split(self):
        rows = [{"problem": f"problem {i}", "answer": str(i)} for i in range(10)]
        rows += [dict(rows[0])]
        train, val = data.split_records(rows)
        self.assertEqual(len(train) + len(val), 10)
        self.assertEqual((train, val), data.split_records(rows))
        self.assertFalse({r["extra_info"]["problem_sha256"] for r in train} & {r["extra_info"]["problem_sha256"] for r in val})

    def test_conflicting_answers_rejected(self):
        with self.assertRaises(ValueError):
            data.split_records([{"problem": "same", "answer": "1"}, {"problem": "same", "answer": "2"}])

    def test_invalid_or_unverified_rows_rejected(self):
        for row in ({"problem": "a", "answer": "1.5"}, {"problem": "a", "answer": "1", "verified": False}, {"answer": "1"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                data.make_record(row, 0, "train")


if __name__ == "__main__":
    unittest.main()
