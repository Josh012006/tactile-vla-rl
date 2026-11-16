"""src/algos/rl_config.py and the YAMLs of configs/algo/rl.yaml, configs/experiment/rl/: the defaults are the paper's (PA-RL, Appendix B.1) and the group's decisions of docs/algos/rl.md,
a method cannot be given what it does not do, and a wrong key or value is an error."""
import dataclasses
from pathlib import Path

import pytest
import yaml

from src.algos import rl_config as rc
from src.algos import tracking

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted((ROOT / "configs" / "experiment" / "rl").glob("*.yaml"))


def test_the_yaml_defaults_are_the_papers_and_the_groups():
    c = rc.load_rl_config()
    assert (c.num_candidates, c.keep_top, c.local_steps, c.local_lr) == (32, 10, 10, 3e-4)                              # paper B.1
    assert (c.critic_lr, c.critic_batch_size, c.gamma, c.cal_ql_alpha, c.mixing_ratio) == (3e-4, 256, 0.99, 0.01, 0.5)   # paper B.1
    assert c.critic_hidden == (512, 512, 512) and c.random_shift_px == 4 and c.step_reward == -1.0 and not c.hl_gauss
    assert (c.utd, c.tau, c.envs_per_batch, c.total_episodes) == (1, 0.005, 50, 500)                                    # group
    assert (c.warmup_episodes, c.warmup_critic_steps, c.critic_image_size, c.critic_cameras) == (50, 5000, 128, ("exterior", "wrist"))
    assert (c.distill_lr, c.distill_batch_size, c.distill_epochs, c.eval_episodes) == (2.5e-5, 64, 1, 100)
    assert (c.rollout_selection, c.distill_selection, c.eval_selection) == ("softmax", "softmax", "argmax") and not c.critic_touch.enabled
    assert c.rounds == 10 and c.distills and c.variant == "parl_global_local"


def test_the_yaml_has_exactly_the_keys_of_the_dataclass_and_the_dataclass_defaults_are_the_yamls():
    raw = yaml.safe_load(rc.CONFIG_PATH.read_text())
    fields = {f.name for f in dataclasses.fields(rc.RLConfig)}
    assert set(raw) == fields
    assert rc.config_from_dict(raw) == rc.RLConfig()                                                                     # no default lives in two places with two values


def test_the_wandb_modes_are_the_trackings():
    assert tuple(rc.WANDB_MODES) == tuple(tracking.MODES)


def test_the_three_experiments_are_the_three_methods():
    found = {}
    for f in EXPERIMENTS:
        exp, cfg = rc.load_rl_experiment(f)
        found[cfg.variant] = (exp["name"], cfg)
        assert exp["model"] == "pi05" and f.stem == exp["name"].replace("peg_insert_", "") and cfg.output_root == "runs/rl/peg_insert" and cfg.seed == 42
    assert set(found) == {"best_of_n", "parl_global", "parl_global_local"}
    best, a, b = (found[k][1] for k in ("best_of_n", "parl_global", "parl_global_local"))
    assert (best.keep_top, best.local_steps, best.rollout_selection, best.distills) == (1, 0, "argmax", False)
    assert (a.keep_top, a.local_steps, a.distills) == (10, 0, True) and (b.keep_top, b.local_steps, b.distills) == (10, 10, True)
    assert all(c.num_candidates == 32 for c in (best, a, b))                                                             # k = 32 for the three


def test_the_command_line_wins_over_the_experiment():
    _, cfg = rc.load_rl_experiment(ROOT / "configs/experiment/rl/pi05_parl_global.yaml", {"local_steps": 5, "num_candidates": 16})
    assert cfg.local_steps == 5 and cfg.num_candidates == 16 and cfg.variant == "parl_global_local"


@pytest.mark.parametrize("overrides, message", [
    ({"method": "ppo"}, "method must be"),
    ({"keep_top": 40}, "cannot exceed"),
    ({"total_episodes": 975}, "multiple of envs_per_batch"),
    ({"warmup_episodes": 30}, "multiple of envs_per_batch"),
    ({"distill_batch_size": 60, "distill_micro_batch_size": 16}, "multiple of distill_micro_batch_size"),
    ({"critic_cameras": ["exterior", "exterior"]}, "critic_cameras"),
    ({"critic_cameras": ["tactile"]}, "critic_cameras"),
    ({"critic_cameras": "wrist"}, "is a list"),
    ({"mixing_ratio": 1.0}, "mixing_ratio"),
    ({"gamma": 0.0}, "gamma and tau"),
    ({"rollout_selection": "greedy"}, "rollout_selection"),
    ({"wandb_mode": "sometimes"}, "wandb_mode"),
    ({"utd": 0}, "utd"),
    ({"n_candidates": 32}, "unknown RL keys"),
])
def test_a_wrong_key_or_value_is_an_error(overrides, message):
    with pytest.raises(ValueError, match=message):
        rc.load_rl_config(overrides=overrides)


def test_the_critics_touch_is_the_systems_touch_spec():
    from src.touch.features import TouchSpec
    c = rc.load_rl_config(overrides={"critic_use_force": True, "critic_use_tactile": True})
    assert c.critic_touch == TouchSpec(use_tactile=True, use_force=True) and c.critic_touch.dim == 60
    assert rc.load_rl_config().critic_touch.dim == 0
    with pytest.raises(ValueError, match="critic.use_tactile"):
        rc.load_rl_config(overrides={"critic_use_tactile": "false"})                                   # a string would be truthy: refused
    with pytest.raises(ValueError, match="no effect"):
        rc.load_rl_config(overrides={"critic_force_mode": "proxy"})
    proxy = rc.load_rl_config(overrides={"critic_use_force": True, "critic_force_mode": "proxy"}).critic_touch
    assert proxy.needs_previous_target and proxy.dim == 7
    assert rc.load_rl_config(overrides={"critic_use_force": True, "critic_use_tactile": True, "critic_force_mode": "proxy"}).critic_touch.dim == 63


def test_best_of_n_cannot_be_given_what_it_does_not_do():
    ok = {"method": "best_of_n", "keep_top": 1, "local_steps": 0, "rollout_selection": "argmax"}
    assert rc.load_rl_config(overrides=ok).variant == "best_of_n"
    for bad, message in (({"keep_top": 10}, "keep_top must be 1"), ({"local_steps": 3}, "local_steps must be 0"), ({"rollout_selection": "softmax"}, "must be argmax")):
        with pytest.raises(ValueError, match=message):
            rc.load_rl_config(overrides={**ok, **bad})


def write(tmp_path, text):
    f = tmp_path / "exp.yaml"
    f.write_text(text)
    return f


def test_an_experiment_file_is_checked(tmp_path):
    exp = rc.load_experiment(write(tmp_path, "name: a\nmodel: pi05\nstart: {run: peg_insert_pi05_lora_teleop, step: 1200}\n"))
    assert exp["start"] == {"run": "peg_insert_pi05_lora_teleop", "step": 1200}
    for text, message in (("model: pi05\n", "`name` is required"), ("name: a\n", "`model` is required"), ("name: a\nmodel: pi05\nfull: true\n", "unknown keys"),
                          ("name: a\nmodel: pi05\nset: [1]\n", "`set` is a mapping"), ("name: a\nmodel: pi05\nstart: {run: x}\n", "`start` is"), ("- a\n", "a mapping")):
        with pytest.raises(ValueError, match=message):
            rc.load_experiment(write(tmp_path, text))
    with pytest.raises(ValueError, match="no experiment file"):
        rc.load_experiment(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="unknown RL keys"):                                                              # a key the algorithm YAML does not have is refused when the config is built
        rc.load_rl_experiment(write(tmp_path, "name: a\nmodel: pi05\nset: {local_step: 3}\n"))


def test_the_evaluation_rounds_are_the_warm_up_every_five_and_the_last():
    assert rc.load_rl_config().eval_rounds == (0, 5, 10)
    assert rc.load_rl_config(overrides={"total_episodes": 150, "eval_every_rounds": 2}).eval_rounds == (0, 2, 3)            # 3 rounds: 0, 2 and the last
    assert rc.load_rl_config(overrides={"eval_every_rounds": 50}).eval_rounds == (0, 10)
    with pytest.raises(ValueError, match="eval_every_rounds"):
        rc.load_rl_config(overrides={"eval_every_rounds": 0})
