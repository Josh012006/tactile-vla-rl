"""Static checks that run everywhere (no Isaac Sim, no GPU): every source file compiles and the files that the learner
imports stay free of simulator imports."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "envs"
SIM_SIDE = {"forge_env.py", "forge_backend.py", "server.py", "robot_assets.py", "robots.py"}
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
