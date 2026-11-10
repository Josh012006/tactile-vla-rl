"""The VLA models come from the openpi fork pinned as the submodule third_party/openpi. What can be checked without JAX (CPU):

  * every configs/model/*.yaml has the keys the code will read and names an openpi TrainConfig that exists in the pinned fork
    (a typo, or a config that the pinned commit does not have, fails here and not an hour into a job);
  * the submodule is checked out at the commit this repository records and has no local change: what ran is what git says ran.

Skipped when the submodule is not checked out (git submodule update --init third_party/openpi).
"""
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / "third_party" / "openpi"
CONFIG_FILES = [OPENPI / "src/openpi/training/config.py", OPENPI / "src/openpi/training/misc/polaris_config.py"]
MODEL_YAMLS = sorted((ROOT / "configs" / "model").glob("*.yaml"))
REQUIRED_KEYS = ("name", "family", "openpi_config", "checkpoint", "action_dim", "action_horizon", "output_semantics")

pytestmark = pytest.mark.skipif(not CONFIG_FILES[0].exists(), reason="submodule third_party/openpi is not checked out")


def train_config_names():
    """Names of the TrainConfig(...) declared in the fork (a text scan: importing openpi needs the whole JAX stack)."""
    names = set()
    for f in CONFIG_FILES:
        if f.exists():
            names |= set(re.findall(r'TrainConfig\(\s*(?:#[^\n]*\s*)*name="([^"]+)"', f.read_text()))
    return names


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def test_the_scan_finds_the_configs_it_is_meant_to_find():
    names = train_config_names()
    assert {"pi05_droid", "pi0_fast_droid", "pi0_fast_droid_jointpos_polaris"} <= names, sorted(names)


@pytest.mark.parametrize("path", MODEL_YAMLS, ids=lambda p: p.name)
def test_model_yaml_is_complete_and_names_an_existing_openpi_config(path):
    cfg = yaml.safe_load(path.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    assert not missing, f"{path.name}: missing {missing}"
    assert cfg["name"] == path.stem, f"{path.name}: `name` must be the file name"
    assert cfg["openpi_config"] in train_config_names(), f"{path.name}: openpi has no TrainConfig {cfg['openpi_config']!r} at the pinned commit"
    assert str(cfg["checkpoint"]).startswith("gs://openpi-assets/"), path.name


def test_there_is_at_least_one_model_yaml():
    assert MODEL_YAMLS, "configs/model/ is empty"


def test_the_submodule_is_at_the_recorded_commit_and_clean():
    try:
        entry = git("ls-files", "-s", "third_party/openpi")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    if not entry.startswith("160000"):
        pytest.skip("third_party/openpi is not recorded as a submodule yet (git submodule add ...)")
    recorded = entry.split()[1]
    actual = git("rev-parse", "HEAD", cwd=OPENPI)
    assert actual == recorded, f"third_party/openpi is at {actual[:10]} but this repository records {recorded[:10]}: git submodule update --init"
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=OPENPI)
    assert not dirty, f"third_party/openpi has local changes:\n{dirty}"
