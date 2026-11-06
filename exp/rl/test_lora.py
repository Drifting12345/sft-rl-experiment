"""CPU regression checks for RL LoRA construction, groups and adapter export."""
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "exp/rl"), str(ROOT / "exp"), str(ROOT / "verl")]

import torch
from peft import PeftModel
from transformers import Qwen3Config, Qwen3ForCausalLM
from omegaconf import OmegaConf
import train_rl as rl
from sft import train_sft as sft
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.model_merger.base_model_merger import BaseModelMerger

torch.set_num_threads(1)


def sft_defaults():
    with patch.object(sys, "argv", ["sft", "--train-data", "/tmp/pool.jsonl",
                                  "--output-dir", "/tmp/sft-unused"]):
        return sft.arguments()


def rl_arguments(algorithm, hardware):
    with patch.object(sys, "argv", ["rl", "--algorithm", algorithm, "--hardware", hardware,
                                  "--train-data", "/tmp/pool.jsonl",
                                  "--output-dir", "/tmp/rl-unused"]):
        return rl.arguments()


def make_model(dropout=0.05, base_dtype=torch.float32):
    defaults = sft_defaults()
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, tie_word_embeddings=False)
    module = Qwen3ForCausalLM(config).to(dtype=base_dtype)
    settings = SimpleNamespace(lora_rank=defaults.lora_r, lora_alpha=defaults.lora_alpha,
        target_modules=defaults.lora_target_modules, target_parameters=None,
        exclude_modules=None, lora_adapter_path=None, lora={"dropout": dropout})
    # Invoke the real engine construction method without initializing a GPU mesh.
    return FSDPEngine._build_lora_module(SimpleNamespace(model_config=settings), module), config


class LoRAContractTests(unittest.TestCase):
    def test_all_profiles_match_sft_and_generate_eight_per_prompt(self):
        expected = sft_defaults()
        for algorithm in ("grpo", "dapo"):
            for hardware in ("3090-24g", "4090-48g"):
                with self.subTest(algorithm=algorithm, hardware=hardware):
                    args = rl_arguments(algorithm, hardware)
                    cfg = rl.build_config(args, {"train_count": 90,
                        "train_file": "/tmp/train.parquet", "val_file": "/tmp/val.parquet"})
                    m = cfg.actor_rollout_ref.model
                    self.assertEqual(m.lora_rank, expected.lora_r)
                    self.assertEqual(m.lora_alpha, expected.lora_alpha)
                    self.assertEqual(m.lora.rank, expected.lora_r)
                    self.assertEqual(m.lora.alpha, expected.lora_alpha)
                    self.assertEqual(m.lora.dropout, expected.lora_dropout)
                    self.assertEqual(list(m.target_modules), expected.lora_target_modules)
                    self.assertEqual(list(m.lora.target_modules), expected.lora_target_modules)
                    self.assertEqual(args.samples_per_prompt, 8)
                    self.assertEqual(cfg.actor_rollout_ref.rollout.n, 8)
                    self.assertEqual(cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n, 256)

    def test_only_adapters_update_and_export_preserves_trained_policy(self):
        torch.manual_seed(42)
        model, config = make_model()
        trained = {n: p for n, p in model.named_parameters() if p.requires_grad}
        frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
        self.assertTrue(trained and frozen)
        self.assertTrue(all("lora_" in n for n in trained))
        self.assertTrue(all("lora_" not in n for n in frozen))
        dropout_layers = [m.lora_dropout["default"].p for m in model.modules()
                          if hasattr(m, "lora_dropout") and "default" in m.lora_dropout]
        self.assertEqual(len(dropout_layers), 7)
        self.assertTrue(all(p == 0.05 for p in dropout_layers))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        optimizer = torch.optim.AdamW(list(trained.values()), lr=0.01)
        loss = model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        self.assertTrue(all(p.grad is None for p in frozen.values()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in trained.values()))
        optimizer.step()
        self.assertTrue(all(torch.equal(before[n], p) for n, p in frozen.items()))
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in trained.items()))
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=tokens).logits
        meta = FSDPCheckpointManager._get_lora_train_meta(None, model)
        self.assertEqual((meta["r"], meta["lora_alpha"], meta["lora_dropout"]), (16, 32, 0.05))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            actor, exported = folder / "actor", folder / "exported"
            actor.mkdir()
            (actor / "lora_train_meta.json").write_text(json.dumps(meta))
            merger = SimpleNamespace(config=SimpleNamespace(local_dir=str(actor), target_dir=str(exported)))
            merger._load_lora_train_meta = MethodType(BaseModelMerger._load_lora_train_meta, merger)
            weights = {n: p.detach().clone() for n, p in model.state_dict().items()}
            adapter = BaseModelMerger.save_lora_adapter(merger, weights)
            saved = json.loads((Path(adapter) / "adapter_config.json").read_text())
            self.assertEqual((saved["r"], saved["lora_alpha"], saved["lora_dropout"]), (16, 32, 0.05))
            base = Qwen3ForCausalLM(config)
            base.load_state_dict(weights)
            reloaded = PeftModel.from_pretrained(base, adapter, torch_device="cpu",
                                                 device_map={"": "cpu"})
            reloaded.eval()
            with torch.no_grad():
                torch.testing.assert_close(reloaded(input_ids=tokens).logits, expected, atol=1e-5, rtol=1e-5)
                merged = reloaded.merge_and_unload(safe_merge=True)
                merged.eval()
                torch.testing.assert_close(merged(input_ids=tokens).logits, expected, atol=1e-5, rtol=1e-5)

    def test_fsdp_wrapped_names_do_not_drop_adapter_tensors(self):
        from verl.utils.fsdp_utils import get_fsdp_peft_model_state_dict
        from peft.utils.save_and_load import get_peft_model_state_dict

        class Wrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self._fsdp_wrapped_module = module

        model, _ = make_model()
        expected = get_peft_model_state_dict(model)
        layer = model.base_model.model.model.layers[0]
        layer.self_attn.q_proj.lora_A["default"] = Wrapper(layer.self_attn.q_proj.lora_A["default"])
        layer.self_attn.q_proj.lora_B["default"] = Wrapper(layer.self_attn.q_proj.lora_B["default"])
        model.base_model.model.model.layers[0] = Wrapper(layer)
        state = {name.replace("_fsdp_wrapped_module.", ""): value
                 for name, value in model.state_dict().items()}
        actual = get_fsdp_peft_model_state_dict(model, state_dict=state)
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(len(actual), 14)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        self.assertTrue(any("_fsdp_wrapped_module." in name for name, _ in model.named_modules()))

    def test_bf16_base_keeps_fp32_lora_and_adam_states(self):
        torch.manual_seed(42)
        model, _ = make_model(base_dtype=torch.bfloat16)
        trained = {n: p for n, p in model.named_parameters() if p.requires_grad}
        frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
        self.assertTrue(all(p.dtype == torch.float32 for p in trained.values()))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in frozen.values()))
        with torch.no_grad():
            for name, parameter in trained.items():
                if "lora_B" in name:
                    parameter.fill_(0.01)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(trained.values(), lr=1e-6)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in trained.items() if "lora_A" in n))
        self.assertTrue(all(torch.equal(before[n], p) for n, p in frozen.items()))
        for state in optimizer.state.values():
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)

    def test_fsdp_merger_preserves_fp32_lora_entries(self):
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger
        weights = {"base_model.model.layer.lora_A.default.weight":
                       torch.tensor([[0.012345678, 0.023456789]], dtype=torch.float32),
                   "base_model.model.layer.weight": torch.tensor([[0.12345]], dtype=torch.float32)}
        with tempfile.TemporaryDirectory() as folder:
            torch.save(weights, Path(folder) / "model_world_size_1_rank_0.pt")
            merger = object.__new__(FSDPModelMerger)
            merger.config = SimpleNamespace(local_dir=folder)
            merged = merger._load_and_merge_state_dicts(1, 1, (1,), ("fsdp",))
            adapter_key = "base_model.model.layer.lora_A.default.weight"
            self.assertEqual(merged[adapter_key].dtype, torch.float32)
            torch.testing.assert_close(merged[adapter_key], weights[adapter_key], rtol=0, atol=0)
            self.assertEqual(merged["base_model.model.layer.weight"].dtype, torch.bfloat16)

    def test_legacy_metadata_without_dropout_still_exports(self):
        model, _ = make_model()
        meta = FSDPCheckpointManager._get_lora_train_meta(None, model)
        meta.pop("lora_dropout")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "lora_train_meta.json").write_text(json.dumps(meta))
            merger = SimpleNamespace(config=SimpleNamespace(local_dir=str(folder), target_dir=str(folder / "out")))
            merger._load_lora_train_meta = MethodType(BaseModelMerger._load_lora_train_meta, merger)
            weights = {n: p.detach().clone() for n, p in model.state_dict().items()}
            adapter = BaseModelMerger.save_lora_adapter(merger, weights)
            saved = json.loads((Path(adapter) / "adapter_config.json").read_text())
            self.assertEqual(saved["lora_dropout"], 0.0)
            self.assertEqual(saved["lora_alpha"], 32)


if __name__ == "__main__":
    unittest.main()
