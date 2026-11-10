"""Static checks that run everywhere (no Isaac Sim, no GPU): every source file compiles, the packages that the learner
imports stay free of simulator imports, and the bank cache key behaves."""
import ast
import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "envs"
SIM_SIDE = {"forge_env.py", "forge_backend.py", "forge_probe.py", "start_bank.py", "server.py", "robot_assets.py", "robots.py"}
FORBIDDEN_ON_LEARNER_SIDE = {"torch", "isaaclab", "isaacsim", "omni", "isaaclab_tasks", "pxr", "warp"}


def imports_of(path):
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_every_source_file_compiles(path):
    compile(path.read_text(), str(path), "exec")


@pytest.mark.parametrize("path", sorted(p for p in SRC.glob("*.py") if p.name not in SIM_SIDE), ids=lambda p: p.name)
def test_learner_side_files_import_no_simulator_or_torch(path):
    bad = imports_of(path) & FORBIDDEN_ON_LEARNER_SIDE
    assert not bad, f"{path.name} is imported by the learner (JAX) process but imports {sorted(bad)}"


def test_bank_digest_depends_on_physics_fields_only():
    pytest.importorskip("torch")
    from src.envs.start_bank import bank_digest
    cfg = {"task_name": "t", "num_envs": 8, "robot": "r", "bank_size": 16, "bank_seed": 1, "camera_size": 224}
    d = bank_digest(cfg)
    assert d == bank_digest({**cfg, "server_port": 9999, "max_steps_per_episode": 3, "auto_reset": False,
                             "bank_cache_dir": "elsewhere", "env_seed": 5, "success_hold_steps": 4})
    for key, value in (("num_envs", 16), ("robot", "x"), ("bank_size", 32), ("bank_seed", 2), ("camera_size", 128)):
        assert bank_digest({**cfg, key: value}) != d, key
