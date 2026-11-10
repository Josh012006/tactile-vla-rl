"""`use_lora` / `lora_rank` of the model YAMLs (src/models/openpi_config.py). The first tests need nothing but the standard library;
the ones that build openpi models (abstractly, nothing is allocated) need the `vla` dependency group and are skipped otherwise."""
import dataclasses
from pathlib import Path

import pytest
import yaml

from src.models import openpi_config as oc

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAMLS = sorted((ROOT / "configs" / "model").glob("*.yaml"))


# -- parsing ---------------------------------------------------------------------------------------------------------------------------
def test_defaults_are_openpis():
    assert oc.parse_lora_settings({"family": "pi05"}) == oc.LoraSettings(True, 16, 32)
    assert oc.parse_lora_settings({"family": "pi0_fast"}) == oc.LoraSettings(True, 16, None)


def test_values_come_from_the_yaml():
    cfg = {"family": "pi05", "use_lora": False, "lora_rank": {"vlm": 8, "action_head": 64}}
    assert oc.parse_lora_settings(cfg) == oc.LoraSettings(False, 8, 64)
    assert oc.parse_lora_settings({"family": "pi05", "lora_rank": {"vlm": 4}}) == oc.LoraSettings(True, 4, 32)      # the other keeps its default


@pytest.mark.parametrize("cfg", [
    {"family": "pi05", "use_lora": "yes"},
    {"family": "pi05", "use_lora": 1},
    {"family": "pi05", "lora_rank": 16},
    {"family": "pi05", "lora_rank": {"vlm": 0}},
    {"family": "pi05", "lora_rank": {"vlm": True}},
    {"family": "pi05", "lora_rank": {"vlm": "8"}},
    {"family": "pi05", "lora_rank": {"vlm": 8.0}},
    {"family": "pi05", "lora_rank": {"expert": 8}},
    {"family": "pi0_fast", "lora_rank": {"action_head": 32}},
])
def test_bad_values_are_refused(cfg):
    with pytest.raises(ValueError):
        oc.parse_lora_settings(cfg)


def test_variant_names():
    assert oc.variant_name("gemma_2b", True, 16) == "gemma_2b_lora"             # openpi's own names for openpi's ranks
    assert oc.variant_name("gemma_300m", True, 32) == "gemma_300m_lora"
    assert oc.variant_name("gemma_2b", True, 8) == "gemma_2b_lora_r8"
    assert oc.variant_name("gemma_300m", True, 16) == "gemma_300m_lora_r16"       # 16 is the VLM's default, not the action expert's
    assert oc.variant_name("gemma_2b", False, 8) == "gemma_2b"


@pytest.mark.parametrize("path", MODEL_YAMLS, ids=lambda p: p.name)
def test_the_shipped_yamls_are_consistent(path):
    cfg = yaml.safe_load(path.read_text())
    settings = oc.parse_lora_settings(cfg)
    if settings.use_lora:
        assert cfg.get("load") == "weight_loader", f"{path.name}: a LoRA model cannot be loaded strictly from a checkpoint without LoRA"


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


def test_default_settings_are_exactly_openpis(openpi):
    base = pi05()
    got = oc.apply_lora_settings(base, oc.parse_lora_settings({"family": "pi05"}))
    assert (got.model.paligemma_variant, got.model.action_expert_variant) == ("gemma_2b_lora", "gemma_300m_lora")
    total, trainable, lora, _ = counts(got)
    assert lora == pytest.approx(50.0, abs=0.1)
    # openpi's freeze filter: LoRA + SigLIP (415 M) + projections (2.2 M) are trainable, the Gemma weights are not (kept as is, on purpose)
    assert trainable == pytest.approx(467.0, abs=0.5)
    assert total == pytest.approx(3403.4, abs=0.5)
    assert counts(dataclasses.replace(base, freeze_filter=base.model.get_freeze_filter()))[1] == pytest.approx(trainable)   # same as openpi's own


def test_use_lora_false_is_full_fine_tuning_of_the_base_architecture(openpi):
    got = oc.apply_lora_settings(pi05(), oc.parse_lora_settings({"family": "pi05", "use_lora": False}))
    assert (got.model.paligemma_variant, got.model.action_expert_variant) == ("gemma_2b", "gemma_300m")
    total, trainable, lora, lora_keys = counts(got)
    assert lora == 0 and not lora_keys
    assert trainable == total == pytest.approx(3353.4, abs=0.5)


def test_ranks_reach_the_lora_matrices(openpi):
    gemma, _ = openpi
    got = oc.apply_lora_settings(pi05(), oc.parse_lora_settings({"family": "pi05", "lora_rank": {"vlm": 8, "action_head": 64}}))
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
    d_keys = counts(oc.apply_lora_settings(pi05(), oc.parse_lora_settings({"family": "pi05"})))[3]
    n = lambda shapes: sum(int(np.prod(s)) for s in shapes) / 1e6                       # noqa: E731
    d_vlm = n(s for k, s in d_keys.items() if "_1" not in k.split("PaliGemma/llm")[1])
    d_head = n(s for k, s in d_keys.items() if "_1" in k.split("PaliGemma/llm")[1])
    assert lora == pytest.approx(d_vlm / 2 + d_head * 2, rel=1e-6)


def test_the_default_rank_does_not_install_anything(openpi):
    gemma, gemma_fast = openpi
    before = (gemma.get_config, gemma_fast.get_config)
    oc.apply_lora_settings(pi05(), oc.parse_lora_settings({"family": "pi05"}))
    assert (gemma.get_config, gemma_fast.get_config) == before


def test_pi0_fast_has_only_the_vlm_rank(openpi):
    _, gemma_fast = openpi
    from openpi.models import pi0_fast
    base = FakeTrainConfig(pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, max_token_len=180))
    got = oc.apply_lora_settings(base, oc.parse_lora_settings({"family": "pi0_fast", "lora_rank": {"vlm": 8}}))
    assert got.model.paligemma_variant == "gemma_2b_lora_r8"
    cfg = gemma_fast.get_config("gemma_2b_lora_r8")
    assert cfg.lora_configs.attn.rank == 8 and cfg.lora_configs.ffn.alpha == 8.0
    assert cfg.variant == "gemma_2b_lora"                                               # the Module only knows the base name
    off = oc.apply_lora_settings(base, oc.parse_lora_settings({"family": "pi0_fast", "use_lora": False}))
    assert off.model.paligemma_variant == "gemma_2b"
