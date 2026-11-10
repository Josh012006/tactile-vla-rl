"""Static checks that run everywhere (no Isaac Sim, no GPU): every source file compiles, the simulator-independent files that the
learner imports stay free of simulator imports, and nothing outside a simulator subpackage imports one."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "envs"
SIMULATOR_PACKAGES = ("isaaclab",)             # src/envs/<name>/: the code that imports a simulator (one subpackage per simulator)
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


def relative_imports_of(path):
    """Names imported relatively (`from .x import y`, `from . import x`): the first module name after the dots."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            out.add((node.module or "").split(".")[0] or node.names[0].name)
    return out


GENERIC = sorted(SRC.glob("*.py"))
SIMULATOR_SIDE = sorted(p for pkg in SIMULATOR_PACKAGES for p in (SRC / pkg).glob("*.py"))


def test_there_are_files_to_check():
    assert GENERIC and SIMULATOR_SIDE, "src/envs/*.py or src/envs/<simulator>/*.py not found: the layout changed, update this file"


@pytest.mark.parametrize("path", GENERIC + SIMULATOR_SIDE, ids=lambda p: str(p.relative_to(SRC)))
def test_every_source_file_compiles(path):
    compile(path.read_text(), str(path), "exec")


@pytest.mark.parametrize("path", GENERIC, ids=lambda p: p.name)
def test_simulator_independent_files_import_no_simulator_or_torch(path):
    bad = imports_of(path) & FORBIDDEN_ON_LEARNER_SIDE
    assert not bad, f"{path.name} is imported by the learner (JAX) process but imports {sorted(bad)}"


@pytest.mark.parametrize("path", GENERIC, ids=lambda p: p.name)
def test_simulator_independent_files_do_not_import_a_simulator_package(path):
    bad = relative_imports_of(path) & set(SIMULATOR_PACKAGES)
    assert not bad, f"{path.name} imports the simulator package(s) {sorted(bad)}: the dependency must go the other way"


@pytest.mark.parametrize("path", [p for p in SIMULATOR_SIDE if p.name == "__init__.py"], ids=lambda p: str(p.relative_to(SRC)))
def test_simulator_package_init_imports_nothing_heavy(path):
    bad = imports_of(path) & FORBIDDEN_ON_LEARNER_SIDE
    assert not bad, f"{path.relative_to(SRC)} must stay free of simulator imports (importing the package must be free), has {sorted(bad)}"
