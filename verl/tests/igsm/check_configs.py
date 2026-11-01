"""Compose both training variants without loading Ray, models or CUDA."""
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
searchpath = "file://" + (ROOT / "verl/trainer/config").as_posix()
for algorithm, critic, n in (("grpo", "false", 4), ("gae", "true", 1)):
    overrides = [
        f"hydra.searchpath=[{searchpath}]",
        f"algorithm.adv_estimator={algorithm}", f"critic.enable={critic}",
        f"actor_rollout_ref.rollout.n={n}",
        "actor_rollout_ref.model.path=/tmp/model", "critic.model.path=/tmp/model",
        "data.train_files=/tmp/train.parquet", "data.val_files=/tmp/val.parquet",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "examples/igsm/config"), version_base=None):
        c = compose(config_name="igsm", overrides=overrides)
    # Required values in disabled optional components can remain ??? in upstream.
    OmegaConf.to_container(c, resolve=True, throw_on_missing=False)
    assert c.algorithm.adv_estimator == algorithm and c.critic.enable == (critic == "true")
    assert c.actor_rollout_ref.rollout.n == n
    assert c.actor_rollout_ref.ref.strategy == "fsdp"
    assert c.actor_rollout_ref.actor.fsdp_config.param_offload
    assert c.critic.fsdp.param_offload
    assert not c.distillation.enabled and not c.reward.reward_model.enable
    assert c.actor_rollout_ref.rollout.agent.default_agent_loop == "single_turn_agent"
    print(f"CONFIG_OK algorithm={algorithm} critic={c.critic.enable} rollout_n={n}")
