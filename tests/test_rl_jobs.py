"""jobs/job_rl_best_of_n.sh and jobs/job_rl_parl.sh: what they pass to `python -m scripts.rl` (RL_JOB_PRINT_ARGS=1 prints the command and stops before anything else), and the refusals."""
import os
import shutil
import subprocess

import pytest

from src.algos.rl_config import load_rl_experiment
from src.models.interface import ROOT

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
BON, PARL = "jobs/job_rl_best_of_n.sh", "jobs/job_rl_parl.sh"
EXP = "configs/experiment/rl/"


def run(script, *args):
    env = {**os.environ, "RL_JOB_PRINT_ARGS": "1", "SLURM_SUBMIT_DIR": str(ROOT)}
    done = subprocess.run(["bash", str(ROOT / script), *args], capture_output=True, text=True, env=env, cwd="/")
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def test_best_of_n_runs_its_experiment_with_the_options_given():
    assert run(BON, "--start", "run", "4000") == (0, f"python -m scripts.rl --experiment {EXP}pi05_best_of_n.yaml --start run 4000", "")
    assert run(BON, "--start", "run", "4000", "--run-name", "probe", "--set", "total_episodes=100")[1] == (
        f"python -m scripts.rl --experiment {EXP}pi05_best_of_n.yaml --start run 4000 --run-name probe --set total_episodes=100")


def test_a_resume_carries_no_experiment_and_another_best_of_n_experiment_replaces_the_default(tmp_path):
    assert run(BON, "--resume", "runs/rl/peg_insert/x")[1] == "python -m scripts.rl --resume runs/rl/peg_insert/x"
    other = tmp_path / "bon.yaml"
    other.write_text("name: x\nset:\n  method: best_of_n\n")
    assert run(BON, "--experiment", str(other), "--start", "r", "1")[1] == f"python -m scripts.rl --experiment {other} --start r 1"


def test_best_of_n_refuses_a_parl_experiment_and_no_arguments():
    code, _, err = run(BON, "--experiment", f"{EXP}pi05_parl_global.yaml", "--start", "r", "1")
    assert code == 1 and "not a best_of_n experiment" in err
    assert run(BON)[0] == 1


@pytest.mark.parametrize("variant, experiment", [("A", "pi05_parl_global"), ("a", "pi05_parl_global"), ("global", "pi05_parl_global"),
                                                 ("B", "pi05_parl_global_local"), ("b", "pi05_parl_global_local"), ("global_local", "pi05_parl_global_local")])
def test_each_parl_variant_runs_its_experiment_and_the_rest_goes_through(variant, experiment):
    assert run(PARL, "--variant", variant, "--start", "run", "4000", "--set", "total_episodes=100")[1] == (
        f"python -m scripts.rl --experiment {EXP}{experiment}.yaml --start run 4000 --set total_episodes=100")
    assert run(PARL, f"--variant={variant}", "--run-name", "p")[1] == f"python -m scripts.rl --experiment {EXP}{experiment}.yaml --run-name p"


def test_the_two_experiment_files_are_the_two_versions_of_the_method():
    assert load_rl_experiment(f"{EXP}pi05_parl_global.yaml")[1].variant == "parl_global"
    assert load_rl_experiment(f"{EXP}pi05_parl_global_local.yaml")[1].variant == "parl_global_local"
    assert load_rl_experiment(f"{EXP}pi05_best_of_n.yaml")[1].variant == "best_of_n"


def test_parl_needs_a_variant_unless_it_resumes_and_refuses_what_does_not_fit():
    code, _, err = run(PARL, "--start", "r", "1")
    assert code == 1 and "--variant A" in err
    assert run(PARL, "--resume", "runs/rl/peg_insert/x")[1] == "python -m scripts.rl --resume runs/rl/peg_insert/x"
    assert "unknown --variant" in run(PARL, "--variant", "C", "--start", "r", "1")[2]
    assert "choose one" in run(PARL, "--variant", "A", "--experiment", f"{EXP}pi05_parl_global.yaml")[2]
    assert "for a new run" in run(PARL, "--variant", "A", "--resume", "x")[2]
    code, _, err = run(PARL, "--experiment", f"{EXP}pi05_best_of_n.yaml", "--start", "r", "1")
    assert code == 1 and "not a parl experiment" in err
    assert run(PARL, "--experiment", f"{EXP}pi05_parl_global.yaml", "--start", "r", "1")[0] == 0
    assert run(PARL)[0] == 1


@pytest.mark.parametrize("script, name, hours", [(BON, "rl_bon", "06:00:00"), (PARL, "rl_parl", "12:00:00")])
def test_the_scripts_parse_and_ask_for_one_l40s_and_the_sigterm_before_the_limit(script, name, hours):
    assert subprocess.run(["bash", "-n", str(ROOT / script)]).returncode == 0
    text = (ROOT / script).read_text()
    for line in (f"#SBATCH --job-name={name}", f"#SBATCH --time={hours}", "#SBATCH --gres=gpu:l40s:1", "#SBATCH --signal=B:TERM@300", "#SBATCH --no-requeue", "#SBATCH --mail-type=ALL"):
        assert line in text
    assert not (ROOT / "jobs" / "job_rl.sh").exists()


def test_the_parl_job_keeps_the_jax_memory_fraction_the_environment_gives_it_and_defaults_to_half():
    text = (ROOT / PARL).read_text()
    assert "export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}" in text
    script = "\n".join(line for line in text.splitlines() if line.startswith("export XLA_PYTHON_CLIENT_MEM_FRACTION"))
    for given, expected in ((None, "0.5"), ("0.6", "0.6")):
        env = {k: v for k, v in os.environ.items() if k != "XLA_PYTHON_CLIENT_MEM_FRACTION"}
        if given:
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = given
        done = subprocess.run(["bash", "-c", script + '\necho "$XLA_PYTHON_CLIENT_MEM_FRACTION"'], capture_output=True, text=True, env=env)
        assert done.stdout.strip() == expected
