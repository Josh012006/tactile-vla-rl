# Tactile-vla-rl

Code for studying **how tactile feedback helps reinforcement-learning fine-tuning of vision-language-action (VLA) models**, in a controlled setup: one task, one simulator, one
starting policy per demonstration type, and the same RL algorithm run with and without touch.

- **Task**: FORGE peg insertion in Isaac Lab (`Isaac-Forge-PegInsert-Direct-v0`), Franka arm + Robotiq 2F-85 gripper with 56 simulated taxels, 150 steps at 15 Hz.
- **Model**: pi0.5 (flow matching, openpi, LoRA). pi0-FAST is a supervised baseline only; the RL code is pi0.5 only.
- **Methods**: supervised fine-tuning (SFT) on demonstrations, then online RL with **Best-of-N** (frozen VLA, critic picks the best of k candidate action chunks) or **PA-RL** (the VLA is also
  distilled on the critic's best actions; variant A: global selection, variant B: global selection + local gradient ascent on Q).
- **Where touch enters**: the critic only (taxels, measured force or a force proxy). The VLA never sees touch, so the SFT checkpoint is the same for every run and the comparison is the same
  algorithm with and without touch in the critic.

PA-RL follows *Policy Agnostic RL* (Mark et al., arXiv 2412.06685). Every value of the RL code says whether it comes from the paper or is a decision of this repository
(`docs/algos/rl.md`).

## Pipeline

```
demonstrations (scripted / noisy / teleop, 50 episodes each)         docs/demos.md
      |
  SFT of pi0.5 / pi0-FAST ---> checkpoints ---> SFT evaluation (success-rate curve)   docs/algos/sft.md, docs/eval/sft_eval.md
      |  start checkpoint (--start <SFT run> <step>)
  online RL                                                                           docs/algos/rl.md
      |  every round keeps its critic (and, for PA-RL, its VLA)
  evaluation of the saved rounds (fixed episodes) ---> curves, paired comparisons     scripts/eval_rl.py, scripts/plot_rl_curves.py
```

## Repository layout

| Path | Content |
|---|---|
| `src/envs/` | env protocol, WebSocket + msgpack client/server, episode manager, demo storage; `src/envs/isaaclab/`: the Isaac Lab FORGE environment, robots (Franka + Robotiq, taxels), server |
| `src/models/` | model interface, YAML settings, openpi config and loading, `OpenPIPolicy` (pi0.5, pi0-FAST) |
| `src/algos/` | `sft.py`, `demos.py` (SFT); `rl_*.py` (candidate sampler and cache, critic, replay buffer, action optimisation, critic update, rollouts, distillation, the run loop); `tracking.py` (wandb) |
| `src/eval/` | rollout loop over K environments, episode seeds and logs, sweep over checkpoints, curves, videos |
| `src/touch/` | which touch signals a component sees and the vector built from them |
| `configs/` | `algo/{sft,rl}.yaml` (defaults), `model/{pi05,pi0_fast}.yaml`, `task/peg_insert_forge.yaml`, `experiment/{sft,rl}/` (one file = one run) |
| `scripts/` | command-line entry points: `sft.py`, `rl.py`, `eval_sft.py`, `eval_rl.py`, `plot_rl_curves.py`, `compare_eval_paired.py`, `fetch_models.py`, `setup_isaac_venv.sh`, `demos/`, probes (`probe_*.py`) |
| `jobs/` | Slurm scripts (`job_sft_lora.sh`, `job_rl_best_of_n.sh`, `job_rl_parl.sh`, `job_eval_sft.sh`, `job_eval_rl.sh`, `job_demos.sh`, `env_server.sh`, probes) |
| `docs/` | design and measurements, one document per part (see below) |
| `tests/` | CPU tests with fakes; Isaac tests behind `--isaac` |
| `assets/forge_demo_norm/` | normalisation statistics of the demonstrations |
| `third_party/` | `openpi` and `IsaacLab` (git submodules, pinned); `robotiq/` (NVIDIA assets, copied by hand, git-ignored) |

## Installation

Written for a Slurm cluster with NVIDIA GPUs (L40S 46 GB for LoRA SFT, RL and evaluation; A100 80 GB for full fine-tuning), `uv`, and Python >= 3.11. Internet is needed on the login node
only: the jobs run offline.

```bash
git clone --recurse-submodules <repository URL> && cd tactile-vla-rl

# 1. Learner environment: JAX, openpi (editable, from the submodule), matplotlib
uv sync --group vla && source .venv/bin/activate
python -m scripts.fetch_models            # checkpoints, tokenizers and normalisation statistics, into openpi's cache

# 2. Simulator environment: Isaac Sim 5.1 + Isaac Lab (separate venv, 10-20 min, once)
bash scripts/setup_isaac_venv.sh

# 3. Robotiq assets (not in the repository): copy NVIDIA's files into third_party/robotiq/
#      robotiq_usd/franka_robotiq_2f_85_flattened.usd, robotiq_usd/TSF_85.usd, robotiq-tsf85-contact-twin-main/

# 4. Create folders for the data and the runs results
mkdir runs
mkdir demos

# 5. Experiment tracking (optional: --set wandb_mode=offline | disabled)
wandb login
```

The Slurm scripts of the training and demonstration jobs carry a `--mail-user` address: change it. Details of the simulator (its start-up, what is reproducible, what it costs) are in
`docs/envs/isaaclab_env.md`.

## Usage

Everything is configured by YAML (no Hydra): `configs/algo/*.yaml` holds the defaults, an experiment file in `configs/experiment/` names one run, and the command line wins over both
(`--set KEY=VALUE` for the algorithm, `--model-set KEY=VALUE` for the model). Write floats with a dot in the YAML and on the command line (`3.0e-4`: `3e-4` is read as a string).

### 1. Demonstrations

The three demonstration sets (`scripted`, `noisy`, `teleop`, 50 episodes each) are collected, replayed and converted with `jobs/job_demos.sh`; see `docs/demos.md`.

### 2. SFT and its evaluation

```bash
python -m scripts.sft --experiment configs/experiment/sft/pi05_lora_teleop.yaml --dry-run      # login node, no GPU
sbatch jobs/job_sft_lora.sh --experiment configs/experiment/sft/pi05_lora_teleop.yaml          # a main run: runs/sft/peg_insert/<name>/
sbatch jobs/job_sft_lora.sh --resume runs/sft/peg_insert/peg_insert_pi05_lora_teleop           # after a stop or a failure
sbatch jobs/job_eval_sft.sh --run runs/sft/peg_insert/peg_insert_pi05_lora_teleop --n-episodes 100   # success rate of step 0 and every checkpoint
```

Six experiment files (pi0.5 and pi0-FAST x three demonstration types) are in `configs/experiment/sft/`. Full fine-tuning: `jobs/job_sft_full.sh`.

### 3. Online RL

The run starts from an SFT checkpoint (`--start <SFT run> <step>`). A run is a warm-up (episodes of the base VLA, then critic updates on them) followed by rounds of 50 episodes collected with the
critic; each round ends with critic updates (one per environment step) and, for PA-RL, the distillation of the VLA and a refresh of the candidate cache.

```bash
# Best-of-N, without touch in the critic (the default experiment). The warm-up (600 episodes, 120 000 critic updates),
# the 20 rounds of 50 episodes and an evaluation of the run every 5 rounds are written in the experiment file
sbatch jobs/job_rl_best_of_n.sh --start peg_insert_pi05_lora_teleop 1600 --run-name bon_teleop_s1600

# the same with the taxels and the measured force in the critic
sbatch jobs/job_rl_best_of_n.sh --experiment configs/experiment/rl/pi05_best_of_n_touch.yaml \
  --start peg_insert_pi05_lora_teleop 1600 --run-name bon_touch_teleop_s1600

# PA-RL, variant A (global) or B (global + local ascent)
sbatch jobs/job_rl_parl.sh --variant B --start peg_insert_pi05_lora_teleop 1600 --run-name parl_b_teleop

# what a run would do (no GPU, no model, no simulator), and how to continue a run that stopped
python -m scripts.rl --experiment configs/experiment/rl/pi05_best_of_n.yaml --start peg_insert_pi05_lora_teleop 1600 --dry-run
sbatch jobs/job_rl_best_of_n.sh --resume runs/rl/peg_insert/<run name>
```

Settings that matter most (`configs/algo/rl.yaml`; the defaults are the paper's, and the Best-of-N experiment files set the warm-up, the number of rounds and the evaluation schedule):

| Key | Default | Meaning |
|---|---|---|
| `num_candidates` / `keep_top` | 32 / 10 | k candidates sampled from the VLA per state; the top m by Q are kept (Best-of-N: m = 1, argmax) |
| `local_steps` / `local_lr` | 10 / 3.0e-4 | variant B: gradient steps on Q per kept chunk and their size |
| `total_episodes`, `envs_per_batch` | 500, 50 | online episodes (rounds of `envs_per_batch` episodes) |
| `warmup_episodes`, `warmup_critic_steps` | 50, 5000 | base-VLA episodes and critic updates before the first round |
| `utd` | 1 | critic updates per environment step |
| `cal_ql_alpha` | 0.01 | weight of the Cal-QL regulariser of the critic loss |
| `reward_mode` | `step_cost` | `step_cost` (-1 per plan, 0 at success), `sparse_01`, `sparse_pm1` (terminal success / time-out rewards) |
| `critic_use_tactile`, `critic_use_force`, `critic_force_mode` | false, false, `measured` | touch in the critic; `measured` or `proxy` (joint tracking residual) force |
| `distill_lr`, `distill_selection` | 2.5e-5, `softmax` | PA-RL distillation |
| `eval_every_rounds`, `eval_episodes` | 5, 100 | evaluations made during the run (round 0 = after the warm-up) |

### 4. Evaluating the saved rounds, curves and comparisons

Every round keeps its critic (`critics/round_<r>.msgpack`) and, for PA-RL, its VLA (`vlas/round_<r>/`), so any round can be evaluated again on the same fixed episodes.

```bash
python -m scripts.eval_rl --run runs/rl/peg_insert/<run> --dry-run                          # what is evaluated, what is done
sbatch jobs/job_eval_rl.sh --run runs/rl/peg_insert/<run> --rounds 0 5 10 --n-episodes 500 \
  --output-dir runs/rl/peg_insert/<run>/eval_rl_0_5_10                                      # results/<round>.json in that folder
python -m scripts.eval_rl --run <run> --rounds 0 5 --selection vla --output-dir <dir>       # the VLA alone (no critic) of those rounds
```

Use a separate `--output-dir` for each concurrent job (the settings of the evaluation are part of the protocol of an output folder, and two jobs must not write in the same one), then plot all the folders of a run at once:

```bash
python -m scripts.plot_rl_curves \
  --curve "no touch=runs/rl/peg_insert/<run a>/eval_rl_*/results/*.json" \
  --curve "touch=runs/rl/peg_insert/<run b>/eval_rl_*/results/*.json" \
  --reference "VLA alone (SFT step 1600)=runs/sft/peg_insert/<SFT run>/eval/results/1600.json" \
  --out runs/rl/peg_insert/curves --title "Best-of-N, teleop demonstrations"
python -m scripts.plot_rl_curves --relative --curve ... --out ...     # change since each run's own round 0, same episodes
python -m scripts.compare_eval_paired --a <files of run a> --b <files of run b>     # episode-by-episode comparison (several files = a plateau)
```

`plot_rl_curves` writes `curve.png`, `curve.pdf`, `curve.json` (every number of the figure) and `table.md`; up to four curves, and with exactly two a panel of their paired difference.

## Evaluation protocol

- A success rate is measured on **fixed episodes** (seed 42): the same starting states for every checkpoint, round and run. The error bars are one standard error of a proportion,
  sqrt(p (1 - p) / n): about 5 points for 100 episodes, 2.2 for 500.
- The standard error says how well the episodes pin a value down. It is **not** the variability between two training runs: a configuration run once is one draw of the training.
- Evaluations are **comparable, not bit-reproducible**: the camera rendering is not reproducible, and only the starting states of the first batch of 100 episodes are guaranteed identical from one
  evaluation to another. The paired tools (`compare_eval_paired.py`, the paired panel of `plot_rl_curves.py`) say when starts are not bit-identical. Keep one kind of GPU for a whole curve.
- The evaluations made during a run (`<run>/eval/`, 100 episodes) and those of `eval_rl.py` (`<run>/eval_rl*/`, any number of episodes) are kept apart and never mixed.
- Run `--dry-run` first, on the login node: it reads the configuration and the files and loads no model.

## Compute

Measured or estimated on an L40S (see `docs/algos/rl.md` section 5 and the headers of the job scripts for the details):

| What | Time |
|---|---|
| pi0.5 LoRA SFT, 4001 steps | about 10 h (7 h on an A100) |
| Best-of-N, warm-up of 600 episodes and 120 000 updates, then 20 rounds of 50 episodes and 5 in-run evaluations | about 6.5 h (calculated: warm-up about 142 min, 20 rounds of about 9.7 min, evaluations about 45 min); a round is about 300 s of collection and 6 400 critic updates at 0.041 s |
| PA-RL, 10 rounds of 50 episodes | about 7.5 h (estimate) |
| evaluation of one round of an RL run, 500 episodes | about 45 min |

One job needs one GPU: the env server, the VLA and the critic share it, so several jobs run in parallel on several GPUs.

## Tests

```bash
uv sync --group dev                 # pytest
python -m pytest -q                 # CPU tests, with fakes for the model and the simulator
python -m pytest -q --isaac         # also the tests that start real Isaac Lab servers (a GPU node)
```

Some tests need the `vla` group (`tests/test_openpi_policy.py` runs the real transforms) and are meant to be run on the cluster. The tests never start a real SFT or RL run.

## Documentation

| Document | Content |
|---|---|
| `docs/demos.md` | collecting, replaying, checking and converting the demonstrations |
| `docs/envs/isaaclab_env.md` | the Isaac Lab environment: robots, taxels, interface, cost, what is and is not reproducible |
| `docs/models/pi05.md`, `docs/models/pi0_fast.md` | the two models: inputs, outputs, parameters, LoRA, settings |
| `docs/algos/sft.md` | the SFT: data, normalisation statistics, loop, checkpoints, resume, measurements |
| `docs/eval/sft_eval.md` | the evaluation protocol: fixed episodes, K environments per server, determinism, resume |
| `docs/algos/rl.md` | Best-of-N and PA-RL: algorithm, every value with its source (paper / decision / choice), metrics, costs, open points, implementation |

## Status

- The code of SFT, evaluation, Best-of-N and PA-RL (variants A and B) is complete and tested.
- Best-of-N has been run on the cluster, with and without touch in the critic, together with controlled tests of the critic's reward scale and of the warm-up budget.
- PA-RL has been run as probes. Its hyperparameters (the distillation learning rate and, for variant B, the step of the local ascent) are still being calibrated, so no main PA-RL run is reported.
- Results (tables and figures) are produced with the scripts above and are not stored in the repository.

## Acknowledgements

[Mila](https://mila.quebec), which provides the compute, [openpi](https://github.com/Physical-Intelligence/openpi) (pi0.5, pi0-FAST; a pinned fork is the `third_party/openpi` submodule), [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and its FORGE
task (a fork is the `third_party/IsaacLab` submodule), and *Policy Agnostic RL* (Mark et al., arXiv 2412.06685). The demonstrations and the first SFT recipe come from the earlier `expo-ft` repository,
which this one replaces for the new experiments.
