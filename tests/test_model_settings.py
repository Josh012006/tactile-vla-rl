"""The per-component settings of the model YAMLs (src/models/settings.py) and their application to openpi (src/models/openpi_config.py).
The first tests need nothing but the standard library + numpy; the ones that build openpi models (abstractly, nothing is allocated) need
the `vla` dependency group and are skipped otherwise."""
import dataclasses
from pathlib import Path

import pytest
import yaml

from src.models import openpi_config as oc
from src.models import settings as S
from src.models.settings import ComponentSettings, parse_model_settings
from src.touch.features import TouchSpec

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAMLS = sorted((ROOT / "configs" / "model").glob("*.yaml"))
OFF = TouchSpec()


def comp(use_lora=True, rank=16, touch=OFF):
    return ComponentSettings(use_lora, rank, touch)


# -- parsing ---------------------------------------------------------------------------------------------------------------------------
def test_defaults_are_openpis_and_touch_is_off():
    s = parse_model_settings({"family": "pi05"})
    assert s.vlm == comp(True, 16) and s.action_head == comp(True, 32) and s.active_touch == {} and s.required_obs_keys == ()
    f = parse_model_settings({"family": "pi0_fast"})
    assert f.vlm == comp(True, 16) and f.action_head is None and list(f.components) == ["vlm"]


def test_each_component_has_its_own_keys():
    cfg = {"family": "pi05", "vlm": {"use_lora": False, "lora_rank": 8}, "action_head": {"lora_rank": 64, "use_tactile": True}}
    s = parse_model_settings(cfg)
    assert s.vlm == comp(False, 8)
    assert s.action_head == comp(True, 64, TouchSpec(use_tactile=True))
    assert s.any_lora and list(s.active_touch) == ["action_head"] and s.required_obs_keys == ("extra/tactile",)
    assert parse_model_settings({"family": "pi05", "vlm": {"use_lora": False}, "action_head": {"use_lora": False}}).any_lora is False
    assert parse_model_settings({"family": "pi05", "vlm": {"lora_rank": 4}}).action_head.lora_rank == 32          # the other keeps its default


@pytest.mark.parametrize("cfg", [
    {"family": "pi05", "vlm": {"use_lora": "yes"}},
    {"family": "pi05", "vlm": {"use_lora": 1}},
    {"family": "pi05", "vlm": 16},
    {"family": "pi05", "vlm": {"lora_rank": 0}},
    {"family": "pi05", "vlm": {"lora_rank": True}},
    {"family": "pi05", "vlm": {"lora_rank": "8"}},
    {"family": "pi05", "action_head": {"lora_rank": 8.0}},
    {"family": "pi05", "vlm": {"rank": 8}},                                   # unknown key
    {"family": "pi05", "vlm": {"use_tactile": "false"}},                        # a string would be truthy
    {"family": "pi05", "action_head": {"use_force": True, "force_mode": "guess"}},
    {"family": "pi0_fast", "action_head": {"lora_rank": 32}},                 # pi0-FAST has no action expert
    {"family": "pi0_fast", "action_head": {}},
])
def test_bad_values_are_refused(cfg):
    with pytest.raises(ValueError):
        parse_model_settings(cfg)


def test_the_force_proxy_is_refused_at_parse_time_before_anything_is_built():
    with pytest.raises(NotImplementedError, match="proxy"):
        parse_model_settings({"family": "pi05", "action_head": {"use_force": True, "force_mode": "proxy"}})


@pytest.mark.parametrize("old", [{"use_lora": True}, {"lora_rank": {"vlm": 16}}])
def test_the_old_global_keys_are_refused_with_the_new_spelling(old):
    with pytest.raises(ValueError, match="one block per component"):
        parse_model_settings({"family": "pi05", **old})


def test_variant_names():
    assert S.variant_name("gemma_2b", True, 16) == "gemma_2b_lora"             # openpi's own names for openpi's ranks
    assert S.variant_name("gemma_300m", True, 32) == "gemma_300m_lora"
    assert S.variant_name("gemma_2b", True, 8) == "gemma_2b_lora_r8"
    assert S.variant_name("gemma_300m", True, 16) == "gemma_300m_lora_r16"       # 16 is the VLM's default, not the action expert's
    assert S.variant_name("gemma_2b", False, 8) == "gemma_2b"
    s = parse_model_settings({"family": "pi05", "vlm": {"use_lora": False}, "action_head": {"lora_rank": 8}})
    assert (S.component_variant(s, "vlm"), S.component_variant(s, "action_head")) == ("gemma_2b", "gemma_300m_lora_r8")


def test_describe_is_one_line_per_component():
    text = S.describe(parse_model_settings({"family": "pi05", "action_head": {"use_force": True}}))
    assert text == "vlm: LoRA rank 16, touch: none (task only); action_head: LoRA rank 32, touch: force[measured] (ft_force 3 + gripper_effort 1) = 4 values"


@pytest.mark.parametrize("path", MODEL_YAMLS, ids=lambda p: p.name)
def test_the_shipped_yamls_are_consistent(path):
    cfg = yaml.safe_load(path.read_text())
    s = parse_model_settings(cfg)
    if s.any_lora:
        assert cfg.get("load") == "weight_loader", f"{path.name}: a LoRA model cannot be loaded strictly from a checkpoint without LoRA"
    assert not s.active_touch, f"{path.name}: touch must be off in the shipped YAML (it is the task-only baseline; configs/touch will switch it on)"
    for name, c in s.components.items():            # the touch switches are written out, so that turning one on is a one-word edit
        assert set(cfg[name]) == set(S.BLOCK_FIELDS), f"{path.name}: block {name} should list {list(S.BLOCK_FIELDS)}"


# -- on openpi's models -------------------------------------------------------------------------------------------------------------
@pytest.fixture
def openpi(monkeypatch):
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    import openpi.models.gemma as gemma
    import openpi.models.gemma_fast as gemma_fast
    monkeypatch.setattr(gemma, "get_config", gemma.get_config)                 # whatever the tests install is undone afterwards
    monkeypatch.setattr(gemma_fast, "get_config", gemma_fast.get_config)
    return gemma, gemma_fast


@dataclasses.dataclass(frozen=True)
class FakeTrainConfig:
    model: object
    freeze_filter: object = None


def pi05(vlm="gemma_2b_lora", head="gemma_300m_lora"):
    from openpi.models import pi0_config
    return FakeTrainConfig(pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16, paligemma_variant=vlm, action_expert_variant=head))


def fast():
    from openpi.models import pi0_fast
    return FakeTrainConfig(pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, max_token_len=180))


def counts(train_cfg):
    """(total, trainable, lora, {lora key: shape}) in millions of parameters, from an abstract model: nothing is allocated."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax
    import numpy as np
    model = nnx.eval_shape(train_cfg.model.create, jax.random.key(0))
    flat = lambda state: {"/".join(map(str, k)): v for k, v in tu.flatten_dict(state.to_pure_dict()).items()}      # noqa: E731
    size = lambda d: sum(int(np.prod(v.shape)) for v in d.values()) / 1e6                                           # noqa: E731
    total = flat(nnx.state(model, nnx.Param))
    trainable = flat(nnx.state(model, nnx.All(nnx.Param, nnx.Not(train_cfg.freeze_filter))))
    lora = {k: v.shape for k, v in total.items() if "lora" in k}
    return size(total), size(trainable), size({k: total[k] for k in lora}), lora


def settings_of(**blocks):
    return parse_model_settings({"family": "pi05", **blocks})


def test_default_settings_are_exactly_openpis(openpi):
    base = pi05()
    got = oc.apply_settings(base, settings_of())
    assert (got.model.paligemma_variant, got.model.action_expert_variant) == ("gemma_2b_lora", "gemma_300m_lora")
    total, trainable, lora, _ = counts(got)
    assert lora == pytest.approx(50.0, abs=0.1)
    # openpi's freeze filter: LoRA + SigLIP (415 M) + projections (2.2 M) are trainable, the Gemma weights are not (kept as is, on purpose)
    assert trainable == pytest.approx(467.0, abs=0.5)
    assert total == pytest.approx(3403.4, abs=0.5)
    assert counts(dataclasses.replace(base, freeze_filter=base.model.get_freeze_filter()))[1] == pytest.approx(trainable)   # same as openpi's own


def test_no_lora_anywhere_is_full_fine_tuning_of_the_base_architecture(openpi):
    got = oc.apply_settings(pi05(), settings_of(vlm={"use_lora": False}, action_head={"use_lora": False}))
    assert (got.model.paligemma_variant, got.model.action_expert_variant) == ("gemma_2b", "gemma_300m")
    total, trainable, lora, lora_keys = counts(got)
    assert lora == 0 and not lora_keys
    assert trainable == total == pytest.approx(3353.4, abs=0.5)


def test_the_components_are_independent(openpi):
    """LoRA on the VLM only / on the action expert only: openpi's freeze filter handles each combination."""
    vlm_only = oc.apply_settings(pi05(), settings_of(action_head={"use_lora": False}))
    head_only = oc.apply_settings(pi05(), settings_of(vlm={"use_lora": False}))
    assert (vlm_only.model.paligemma_variant, vlm_only.model.action_expert_variant) == ("gemma_2b_lora", "gemma_300m")
    assert (head_only.model.paligemma_variant, head_only.model.action_expert_variant) == ("gemma_2b", "gemma_300m_lora")
    both = counts(oc.apply_settings(pi05(), settings_of()))
    t_vlm, t_head = counts(vlm_only), counts(head_only)
    assert 0 < t_vlm[2] < both[2] and 0 < t_head[2] < both[2] and t_vlm[2] + t_head[2] == pytest.approx(both[2], rel=1e-6)      # LoRA parameters add up
    assert t_vlm[1] > 0 and t_head[1] > 0
    assert all("llm" in k for k in t_vlm[3]) and all("_1" not in k.split("PaliGemma/llm")[1] for k in t_vlm[3])      # VLM LoRA only, in the VLM
    assert all("_1" in k.split("PaliGemma/llm")[1] for k in t_head[3])                                               # action expert LoRA only


def test_ranks_reach_the_lora_matrices(openpi):
    gemma, _ = openpi
    got = oc.apply_settings(pi05(), settings_of(vlm={"lora_rank": 8}, action_head={"lora_rank": 64}))
    assert (got.model.paligemma_variant, got.model.action_expert_variant) == ("gemma_2b_lora_r8", "gemma_300m_lora_r64")
    _, _, lora, keys = counts(got)
    vlm = [s for k, s in keys.items() if "PaliGemma/llm" in k and "_1" not in k.split("PaliGemma/llm")[1]]
    head = [s for k, s in keys.items() if "PaliGemma/llm" in k and "_1" in k.split("PaliGemma/llm")[1]]
    assert vlm and head
    assert all(8 in s for s in vlm) and all(64 in s for s in head), (vlm[:2], head[:2])
    cfg = gemma.get_config("gemma_2b_lora_r8")
    assert cfg.lora_configs["attn"].rank == 8 and cfg.lora_configs["attn"].alpha == 8.0 and cfg.lora_configs["ffn"].rank == 8
    assert gemma.get_config("gemma_2b_lora").lora_configs["attn"].rank == 16            # openpi's variants are untouched
    assert gemma.get_config("gemma_300m_lora").lora_configs["ffn"].rank == 32
    # LoRA parameters are linear in the rank: VLM 8 (half of 16) + action head 64 (twice 32)
    import numpy as np
    d_keys = counts(oc.apply_settings(pi05(), settings_of()))[3]
    n = lambda shapes: sum(int(np.prod(s)) for s in shapes) / 1e6                       # noqa: E731
    d_vlm = n(s for k, s in d_keys.items() if "_1" not in k.split("PaliGemma/llm")[1])
    d_head = n(s for k, s in d_keys.items() if "_1" in k.split("PaliGemma/llm")[1])
    assert lora == pytest.approx(d_vlm / 2 + d_head * 2, rel=1e-6)


def test_the_default_rank_does_not_install_anything(openpi):
    gemma, gemma_fast = openpi
    before = (gemma.get_config, gemma_fast.get_config)
    oc.apply_settings(pi05(), settings_of())
    assert (gemma.get_config, gemma_fast.get_config) == before


def test_the_family_and_the_openpi_model_must_agree(openpi):
    with pytest.raises(ValueError, match="action expert"):
        oc.apply_settings(pi05(), parse_model_settings({"family": "pi0_fast"}))
    with pytest.raises(ValueError, match="has none"):
        oc.apply_settings(fast(), settings_of())


def test_pi0_fast_has_only_the_vlm(openpi):
    _, gemma_fast = openpi
    fast_settings = lambda **b: parse_model_settings({"family": "pi0_fast", **b})            # noqa: E731
    got = oc.apply_settings(fast(), fast_settings(vlm={"lora_rank": 8}))
    assert got.model.paligemma_variant == "gemma_2b_lora_r8"
    cfg = gemma_fast.get_config("gemma_2b_lora_r8")
    assert cfg.lora_configs.attn.rank == 8 and cfg.lora_configs.ffn.alpha == 8.0
    assert cfg.variant == "gemma_2b_lora"                                               # the Module only knows the base name
    off = oc.apply_settings(fast(), fast_settings(vlm={"use_lora": False}))
    assert off.model.paligemma_variant == "gemma_2b"


def test_pi0_fast_lora_is_openpis_own_and_trains_only_lora_and_the_non_gemma_parts(openpi):
    """The default: openpi's `gemma_2b_lora` (rank 16) through Pi0FASTConfig.get_freeze_filter."""
    base_total = counts(oc.apply_settings(fast(), parse_model_settings({"family": "pi0_fast", "vlm": {"use_lora": False}})))
    got = oc.apply_settings(fast(), parse_model_settings({"family": "pi0_fast"}))
    assert got.model.paligemma_variant == "gemma_2b_lora"
    total, trainable, lora, keys = counts(got)
    assert keys and lora > 0 and 0 < trainable < total and total == pytest.approx(base_total[0] + lora, rel=1e-6)
    assert trainable == pytest.approx(lora + 415.0, abs=20)             # LoRA + whatever openpi leaves unfrozen outside the Gemma (SigLIP): the llm itself is frozen
    assert base_total[1] == base_total[0]                               # no LoRA: nothing frozen, as in openpi
