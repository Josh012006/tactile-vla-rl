# Isaac Lab environment: FORGE PegInsert with a Robotiq 2F-85 and taxels

This document describes the simulation environment of `tactile-vla-rl`: what it contains, how to start and drive it, what it costs in time and GPU
memory when the number of environments grows, and above all **what is guaranteed and what is not** regarding reproducibility.

Every figure here comes from measurements made with the tests of the repository (`tests/test_isaac_*.py`) on an **NVIDIA L40S (46 GB), Isaac Sim 5.1,
Isaac Lab 2.3.0, torch 2.7.0**. Nothing is claimed beyond what was measured; what was not measured is listed in section 9.

---

## 1. In short

| | |
|---|---|
| Task | `Isaac-Forge-PegInsert-Direct-v0` (FORGE), instruction: "Insert the peg into the socket." |
| Default robot | Franka + Robotiq 2F-85 + **56 taxels** (2 fingers × 7 × 4), profile `robotiq_2f85_tsf` |
| Actions | 8 values: 7 **absolute** joint targets (rad) + gripper command (0 open, 1 closed) |
| Observations | 2 images 224×224, joints, gripper, force/torque, peg–socket offset, gripper effort, **tactile (2, 7, 4) N** |
| Rate | 15 Hz control (decimation 8 at 120 Hz physics), episodes of at most 150 steps |
| Parallelism | N independent environments in **one** scene and **one** process, driven in "rounds" |
| Interface | WebSocket + msgpack server, one round trip per step, `VectorEnvClient` client (numpy only) |

**Reproducibility in one table** (details in section 7):

| Guaranteed (measured) | Not guaranteed (measured) |
|---|---|
| The **same sequence** of `reset` / `step`, replayed from a fresh server, gives **bit-identical** observations (tactile included) | The start of a `(seed, round)` **after another episode** (Robotiq only): about 1 environment in 3 starts slightly elsewhere, median 0.01 to 0.06 mm on the peg, rarely 1 to 11 mm |
| The environments of a round are **exactly independent** of one another | The start also depends on `num_envs` and on the environment index |
| The reset is bit-identical **up to the closing of the gripper** on the peg | The images (rendering) were not compared bit for bit |
| With the Franka, the start does **not** depend on the previous episode | Another GPU, driver or Isaac version was not tested |

---

## 2. What the environment contains

**Task.** FORGE's peg insertion (Isaac Lab), with FORGE's real reset logic: it acts on **all** the environments and advances the physics, which
forces the round-based operation (section 5).

**Robots** (`robot` field of the YAML):

| Profile | Content |
|---|---|
| `franka` | Franka + FORGE's Franka Hand, unmodified |
| `robotiq_2f85` | Franka arm + Robotiq 2F-85 (NVIDIA RoboLab USD), no taxel |
| `robotiq_2f85_tsf` | Same + 7×4 tactile zones per finger (56 taxels), sends `extra/tactile` in newtons |

The Robotiq USD is built from `third_party/robotiq/` at the server's **first start** (`src/envs/isaaclab/robot_assets.py`, result in
`third_party/robotiq/_build/`). The Robotiq profile carries `finger_effort = 6.0` and `fingerpad_length = 0.0056`.

**Sensors always sent.** The server sends everything the robot can measure (tactile, force, gripper effort); the consumer chooses what to use. A single
server therefore serves the runs with and without touch.

**Dynamics.** Arm PD gains 400 / 60 (FORGE sets them to zero), Isaac Lab's timeout disabled (episode tracking is done on the server side, per
environment), deterministic GPU PhysX solver (`enhanced_determinism: true`).

**Episodes.** Sparse reward 1.0 where FORGE says "inserted" (xy < 2.5 mm and z below its threshold), `terminated` = success (held for
`success_hold_steps` steps), `truncated` = timeout at `max_steps_per_episode`. By default the gripper does not open (`gripper_can_open: false`:
`action[:, 7]` is ignored, the gripper stays at the grasp width of the reset), because the zero-shot pi05_droid_jointpos tends to open and drop the
peg.

**The three FORGE tasks.** The server runs the task of its YAML (`task_name`); nothing else in the code is specific to one of them. The scripts take the YAML as a parameter
(`ENV_CONFIG=<yaml> bash jobs/env_server.sh`, `--task-config`), an SFT run records its task (`task:` of its experiment file) and the evaluation and the RL read it back.

| Task | YAML | `task_name` | Prompt | `max_steps_per_episode` (FORGE's 10 / 20 / 30 s at 15 Hz) | Success height (z below, with xy < 2.5 mm) |
|---|---|---|---|---|---|
| PegInsert | `configs/task/peg_insert_forge.yaml` | `Isaac-Forge-PegInsert-Direct-v0` | "Insert the peg into the socket." | 150 | `fixed.height * success_threshold` |
| GearMesh | `configs/task/gear_mesh_forge.yaml` | `Isaac-Forge-GearMesh-Direct-v0` | "Mesh the gear with the other gears." | 300 | `fixed.height * success_threshold` |
| NutThread | `configs/task/nut_thread_forge.yaml` | `Isaac-Forge-NutThread-Direct-v0` | "Screw the nut onto the bolt." | 450 | `fixed.thread_pitch * success_threshold` (`src/envs/task_geometry.py`) |

Everything else in the two new YAMLs is PegInsert's and **was not measured for them**: the Robotiq grip (`finger_effort`, `fingerpad_length`, chosen so that the *peg* is neither crushed nor lost),
the PD gains, `reset_passes`, the camera. The observation key `peg_socket_offset` is kept for the three tasks (held object minus target pose), so that the demonstrations, the policy and
the clients keep one format. The server tells its task in `spec["robot"]["task"]` (the teleop client reads it). Prompt, grip and the reset of the held object are to be checked on the
cluster with the new YAMLs (`pytest tests/test_isaac_fidelity.py --isaac --isaac-config configs/task/<task>_forge.yaml -s`, and a few resets seen from the teleop client).

### Where the code is

```
src/envs/                     logic shared by all the environments (no simulator or torch import)
  protocol.py                 server/client contract: rounds, observation keys, EnvSpec, StepResult
  wire.py  client.py          msgpack + VectorEnvClient (one or several servers seen as a single vector)
  collect.py                  Collector (steps / episodes modes) and run_fixed_rounds (evaluation on a fixed set)
  episode_manager.py  episode_tracker.py  backend.py  ws_server.py     rounds, episode tracking, WebSocket server
src/envs/isaaclab/            everything specific to Isaac Lab (imported on the simulator side only)
  server.py                   entry point: python -m src.envs.isaaclab.server --config ...
  forge_backend.py            simulator adapter: reset, step, observe, debug tools, PhysX overrides
  forge_env.py                ForgeEnvJointPosPi05: action space, Robotiq adapter, taxels, reset_lockstep
  robots.py  robot_assets.py  robot profiles, USD construction
configs/task/peg_insert_forge.yaml    the task configuration (the only file the server reads)
jobs/env_server.sh  jobs/job_isaac_tests.sh  scripts/setup_isaac_venv.sh
tests/                        CPU tests (fakes) and Isaac tests (--isaac, GPU)
```

The files of `src/envs/` never import `src/envs/isaaclab/`: `tests/test_static.py` checks it. The learning side (JAX) therefore needs neither Isaac Sim
nor torch.

---

## 3. Installation

Two separate virtual environments:

| Venv | Role | Creation |
|---|---|---|
| `.venv` | learning side + client (numpy, msgpack, websockets, pyyaml) | `uv sync` |
| `.venv-isaaclab` | simulator: Isaac Sim 5.1, torch 2.7.0 (cu128), Isaac Lab (submodule `third_party/IsaacLab`) | `bash scripts/setup_isaac_venv.sh` (once, 10 to 20 min, ~10 GB, no GPU needed) |

The script is idempotent; it initialises the `third_party/IsaacLab` submodule if needed. `third_party/robotiq/` is **not** a submodule: these are NVIDIA's
original files, copied by hand and listed in `.gitignore` (never committed): `robotiq_usd/franka_robotiq_2f_85_flattened.usd`,
`robotiq_usd/TSF_85.usd` and `robotiq-tsf85-contact-twin-main/`. Without them the server stops with a `FileNotFoundError` that lists them.

CPU tests, no simulator:

```bash
uv run pytest tests -q          # the Isaac tests are skipped without --isaac
```

---

## 4. Starting a server

On a GPU node (L40S):

```bash
jobs/env_server.sh                                          # configs/task/peg_insert_forge.yaml, 8 environments, port 8102
jobs/env_server.sh --set num_envs=16 --set server_port=8103  # override a field of the YAML
ENV_CONFIG=other.yaml jobs/env_server.sh                     # another configuration file
```

The script activates `.venv-isaaclab`, exports `OMNI_KIT_ACCEPT_EULA=YES` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (needed for reproducibility, see
section 7), then runs `python -m src.envs.isaaclab.server`. The server prints `[env_server] ready: N environments, port ...` when it accepts connections.
Measured start-up: 25 to 55 s (N = 1 to 64), plus the USD construction on the very first start.

- **Several servers** (one per GPU): start the first one alone and wait for `ready`: the first start writes the USD into `third_party/robotiq/_build/`,
  and two first starts at the same time would write it simultaneously.
- Clean shutdown on SIGINT / SIGTERM.
- The client waits for a starting server only at the first connection.

### YAML fields (`configs/task/peg_insert_forge.yaml`)

| Field | Default | Effect |
|---|---|---|
| `num_envs` | 8 | environments simulated by this server |
| `clone_in_fabric` | false | **must stay false**: otherwise cameras and contact sensors see only one environment |
| `robot` | `robotiq_2f85_tsf` | `franka` / `robotiq_2f85` / `robotiq_2f85_tsf` |
| `env_seed`, `auto_reset` | 0, true | default sequence of rounds; next round started automatically |
| `camera_size` | 224 | side of the images (square); the VLA input is 224×224 |
| `max_steps_per_episode`, `control_hz` | 150, 15 | per-environment timeout; `control_hz` is informative (actual: decimation 8 at 120 Hz) |
| `success_hold_steps` | 1 | consecutive "inserted" steps required to declare success (≥ 1) |
| `gripper_can_open`, `gripper_obs` | false, `normalized` | gripper frozen or not; `gripper_position` = `1 − width/0.04` (0 open, 1 closed, ~0.9 holding the peg), or `closed` / `raw` |
| `arm_stiffness`, `arm_damping` | 400, 60 | arm PD gains |
| `disable_sleep` | false | (no measured effect on reproducibility) |
| `enhanced_determinism` | true | deterministic GPU PhysX solver; **identical for all the data of an experiment** |
| `reset_passes` | 2 | number of chained resets at each `reset` (each costs a real reset) |
| `finger_stiffness`, `finger_damping`, `finger_effort`, `fingerpad_length`, `robot_usd`, `tactile_exclude_zone_joints`, `reset_grip_open_width` | (commented out) | optional Robotiq settings |
| `physx_<field>` (via `--set`) | | any field of Isaac Lab's `PhysxCfg`, converted to the field's type |
| `server_host`, `server_port` | `0.0.0.0`, 8102 | |

The dynamics and reproducibility fields require a server restart.

---

## 5. Using the environment

### Round protocol

FORGE's reset acts on all the environments and advances the physics: the environments of a server therefore **start their episodes together** (a *round*,
with FORGE's real reset). Each one then plays its own episode (success or timeout at its own pace). An environment whose episode is finished is **held
in place** (`acting = False`) until the others have finished; the next round then starts.

- `reset(seed, round)`: starts a round. `seed` fixes the whole sequence of rounds, `round` the position in the sequence (simulation seed `seed + round`).
  With no argument: next round. With `seed` alone: round 0.
- `step(actions)`: advances **all** the environments; actions `(num_envs, 8)` float32. When the last running environment finishes and `auto_reset` is on,
  **the next round starts in the same call**: `obs` is the first observation of the new round, and the real last observation of each episode that ended
  at this step is in `final_obs` (rows `final_ids`). Nothing is lost or stale.
- `auto_reset=False`: when all have finished, nothing restarts by itself, `step` has no effect (`acting` false everywhere) until the client's `reset`. This is
  the mode of the evaluation on a fixed set.
- A step costs **a single** network round trip. No silent retry: if a call fails, the exception propagates (restart from a checkpoint rather than hide a
  lost episode).

Fields of a `StepResult`: `obs`, `reward`, `terminated` (success), `truncated` (timeout, not a terminal state for bootstrapping), `success`, `t`,
`acting`, `done`, `all_done`, `round_id`, `final_obs`, `final_ids`.

### Observations

| Key | Shape (one environment) | Type | Content |
|---|---|---|---|
| `observation/exterior_image_1_left` | (224, 224, 3) | uint8 | exterior camera |
| `observation/wrist_image_left` | (224, 224, 3) | uint8 | wrist camera |
| `observation/joint_position` | (7,) | float32 | arm joints (rad) |
| `observation/gripper_position` | (1,) | float32 | according to `gripper_obs` |
| `extra/ft_force` | (3,) | float32 | force of FORGE's F/T sensor |
| `extra/force_threshold` | () | float32 | FORGE's force threshold |
| `extra/peg_socket_offset` | (3,) | float32 | peg–socket offset |
| `extra/gripper_effort` | () | float32 | finger motor effort (Robotiq) |
| `extra/tactile` | (2, 7, 4) | float32 | N; `[0]` left finger, `[1]` right, row 0 = fingertip (taxel profile only) |

The keys are those of the DROID / pi05 convention (image + state) and of the old server (`extra/*`).

### Example

```python
from src.envs.client import VectorEnvClient
from src.envs.collect import Collector, run_fixed_rounds

env = VectorEnvClient([("localhost", 8102)])             # or several servers: [("node1", 8102), ("node2", 8102)]
res = env.reset(seed=0)                                  # round 0
while True:
    res = env.step(policy(res.obs))                      # policy: obs (N, ...) -> actions (N, 8) float32
    # res.done: episodes finished at this step (last obs in res.final_obs); res.acting false = environment waiting

# Collection
collector = Collector(env, policy, mode="steps")         # Rollout (T, N, ...); or mode="episodes" -> list of Trajectory
data = collector.collect(80)                             # 80 steps in EACH environment
update(data); collector.version += 1                     # during the update, the environments are not advanced

# Evaluation on a fixed set of starts (auto_reset=False, no episode cut or repeated)
results = run_fixed_rounds(env, policy, rounds=range(8), seed=0)
# one dict per (round, environment): round, env, success, terminated, truncated, length, episode_return
```

What the `Collector` guarantees: the `next_obs` of a transition that ends an episode is the **last** observation of that episode (never the first of the
next round); `terminated` and `truncated` are separate; every transition carries the policy `version` that chose the action and its `round`; rows with
`acting == False` are not transitions (the `steps` mode keeps them so that the arrays stay rectangular, the `episodes` mode never stores them); episodes in
progress at the end of a call continue at the next call. `total_idle_steps` counts the steps spent waiting: this is the price of synchronous rounds.

Beware in `episodes` mode: stopping in the middle of a round keeps the episodes that finished first, hence over-represents the short ones (the successes).
Ask for a multiple of `num_envs`, or use `run_fixed_rounds` to evaluate.

---

## 6. Costs and gains with several environments

Measurements: `COST_NUM_ENVS=1,4,16,64 SLURM_SUBMIT_DIR=$PWD bash jobs/job_isaac_tests.sh tests/test_isaac_cost.py` (L40S, profile
`robotiq_2f85_tsf`, 150 steps per episode, rollout T = 80 steps). Steps are measured on episodes of 30 steps and projected to the real length. JSON report:
`reports/env_cost_<timestamp>.json`.

### Cost of one more environment

| N | VRAM (MiB) | step (ms) | of which physics | of which observation | reset (s) | env steps / s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8273 | 519.9 | 505.9 | 11.4 | 8.19 | 2 |
| 4 | 8114 | 545.2 | 532.1 | 9.8 | 8.44 | 7 |
| 16 | 8453 | 578.4 | 559.7 | 12.0 | 9.22 | 28 |
| 64 | 9807 | 674.8 | 624.3 | 18.6 | 9.89 | 95 |

Linear fits on these four points:

| Quantity | Fixed part | Per environment | R² |
|---|---:|---:|---:|
| VRAM | 8103 MiB | 26.3 MiB | 0.98 |
| step | 530.8 ms | 2.29 ms | 0.98 |
| of which physics | 519.6 ms | 1.69 ms | 0.94 |
| of which observation | 10.2 ms | 0.13 ms | 0.95 |
| reset | 8.41 s | 0.025 s | 0.87 |

Reading: **the cost is mostly fixed**. At N = 1, physics is 97 % of the step and this cost hardly depends on N; VRAM is dominated by ~8.1 GiB fixed
(the VRAM measured at N = 4 is slightly below that of N = 1: this is measurement noise, not a saving). Adding environments to the same process is
therefore almost free, whereas launching several small servers would pay ~8 GiB and ~0.5 s per step each time (a conclusion drawn from the model, not
measured with several servers).

### Gain

Update every T = 80 steps of each environment (`update_per: step`):

| N | VRAM (MiB) | update (s) | samples / update | samples / s | gain | efficiency | reset share | gain per extra GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8273 | 45.9 | 80 | 1.74 | 1.00 | 100 % | 10 % | — |
| 4 | 8114 | 48.2 | 320 | 6.63 | 3.81 | 95 % | 9 % | — |
| 16 | 8453 | 51.4 | 1280 | 24.91 | 14.30 | 89 % | 10 % | 75.6 |
| 64 | 9807 | 65.1 | 5120 | 78.64 | 45.15 | 71 % | 8 % | 29.5 |

One update per round of N complete episodes of 150 steps (`update_per: episode`, target 64 episodes):

| N | VRAM (MiB) | update (s) | episodes / update | episodes / h | time for 64 episodes (s) | gain | efficiency | gain per extra GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8273 | 86.0 | 1 | 41.9 | 5505 | 1.00 | 100 % | — |
| 4 | 8114 | 90.8 | 4 | 158.6 | 1453 | 3.79 | 95 % | — |
| 16 | 8453 | 94.8 | 16 | 607.4 | 379 | 14.51 | 91 % | 76.8 |
| 64 | 9807 | 111.5 | 64 | 2066.4 | 111 | 49.37 | 77 % | 32.3 |

"Efficiency" = gain / N. At N = 64: ×45 samples per second, ×49 episodes per hour, for +1.5 GiB of VRAM compared with N = 1. Episodes are counted at 150
steps: in reality, an environment that succeeds early waits for the others (idle steps), which these tables do not model; `Collector.total_idle_steps`
measures it on real data.

### Limits of these figures

- **Measured only at N = 1, 4, 16, 64.** Nothing beyond 64; the gain per GiB is already falling (76 → 32), do not extrapolate to large N without
  measuring (`COST_NUM_ENVS=...,128`).
- `max_envs_in_gpu_memory = 1268` is the extrapolation of the VRAM fit with a 10 % margin: **it is not a measurement**, and the step time would also
  increase.
- A single machine (L40S), a single robot profile, 224×224 cameras; changing `camera_size` or the robot changes the costs.
- Several real servers (several GPUs) were not measured; the multi-server client is tested on the CPU side with fake servers.
- Test durations: fidelity ≈ 20 min, cost (4 sizes) ≈ 11 min, one finger-sweep setting ≈ 7 min; an interactive `salloc` is killed at its time limit
  (≈ 25 to 35 min observed), hence `jobs/job_isaac_tests.sh` (4 h).

---

## 7. Reproducibility: what is guaranteed and what is not

There are **two** different properties, often confused:

1. *"The same sequence of calls gives the same results."* **Guaranteed.**
2. *"The start of `(seed, round)` is the same whatever ran before."* **Not guaranteed** with the Robotiq.

### 7.1 Guaranteed

**G1. Same sequence, same bits.** Replayed twice from a **freshly started** server (same YAML, same machine), the same sequence of `reset` and `step`
gives bit-identical observations (difference 0.0 on the 7 non-image keys, tactile included; the resets of the rounds are included). Test:
`test_the_same_sequence_on_a_fresh_server_is_bit_exact` (4 environments, 20 steps, 3 rounds). The "floor" of the full state (365 tensors: joints, links,
peg, velocities, accelerations…) for two identical histories is 0 everywhere. Conditions: `enhanced_determinism: true` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (set by `jobs/env_server.sh` and by the tests).

*Scope.* Tested over 20 steps; `test_long_run_is_stable` checks over ~330 steps that nothing blows up (finite values, peg that does not fly away) but **not**
bit-for-bit equality over long sequences. The mechanism (deterministic GPU PhysX) is the same, but it is not measured beyond that.

**G2. Independence of the environments of a round.** Driving environment 0 violently changes **nothing** (difference 0.0) in the other environments, on
joints, force, offset, tactile, gripper effort, peg. Test: `test_the_environments_of_a_round_are_independent`.

**G3. A clean reset up to the grasp.** The physical reset is bit-identical up to the closing of the gripper on the peg (physics step ~36 to 42 of the first
pass, out of 134 reset steps); before that, floor 0. With the **Franka**, another episode before the reset changes nothing (0 difference on 361 tensors).

**G4. Valid and varied starts.** Every start remains a valid FORGE start; two different `(seed, round)` give different starts (~2.6 cm of offset difference
between two rounds, test `test_start_states_are_valid_and_diverse`); the next round is indeed the next seed (`test_next_round_is_the_next_seed`).

**G5. Episode semantics.** Nothing lost, nothing stale at the boundary between rounds; correct `final_obs`; `terminated` and `truncated` separate
(`test_a_round_ends_and_the_next_one_starts_by_itself`, CPU tests of the manager, the collector and the protocol).

### 7.2 Not guaranteed

**N1. The start of a `(seed, round)` after another episode (Robotiq only).** Comparison of the start of `(seed, round 0)` after a simple reset against the
same start after 24 steps of another seed, 4 environments × 6 starts = 24 "environment starts" (default setting):

| Measure | Value |
|---|---|
| starts that differ | 8 out of 24 (6 to 8 out of 24 depending on the settings tried) |
| median difference on the peg, when there is one | 0.01 to 0.06 mm |
| worst difference observed | 5.2 mm (default setting); 1 to 11 mm depending on the settings |
| maximum difference on the joints | up to 1.4e-2 rad |

Concrete consequences:
- the start obtained remains a valid FORGE start;
- two evaluations of the "same round" made with different histories start from an **almost** identical state, not strictly identical; `run_fixed_rounds`
  therefore fixes the mapping `(seed, round) → start` **up to that difference**;
- the trajectories that follow such a start are not guaranteed identical (contacts: the difference can grow);
- a bit-for-bit replay is guaranteed only if the **whole sequence** (resets, steps, actions) is identical from a fresh server (G1), which, for a training
  run, also supposes identical actions.

**N2. The start depends on `(seed, round, num_envs, i)`.** Changing `num_envs` changes the starts; the start of environment `i` of a server with N
environments is not that of a server with a single environment. Compare runs at the same `num_envs`. (With several servers, the seed of server *k* is
shifted by `k × SEED_STRIDE`, 1 000 003.)

**N3. The images.** The sequence test excludes the images; the rendering was not compared bit for bit between two servers. Do not assume it.

**N4. Other hardware or version.** All the measurements: L40S, Isaac Sim 5.1, Isaac Lab 2.3.0, torch 2.7.0. Another GPU, driver or version may change the
bits.

**N5. Any change to the sequence or the configuration.** One more `reset` or `step`, another `reset_passes`, `enhanced_determinism`, a PD gain, a finger
setting: this changes the following states. The server's `debug` operations (`state_dump`, `trace_next_reset`, etc.) are test tools: do not call them
during data collection.

**N6. Effect on the science, not measured.** A difference of the order of a hundredth of a millimetre in the median is very small compared with the
dispersion of FORGE's starts; the effect on a success rate over hundreds of episodes **was not measured**.

### 7.3 Cause (located, not fixed)

- The reset is identical up to the closing of the gripper on the peg; there, float-level differences (1e-9 to 1e-6) are amplified by a factor 100 to 1000
  per step during 3 to 4 steps, then saturate around 1e-4; the second reset pass erases them in most environments (hence `reset_passes: 2`).
- It is a **hidden state in PhysX's contact solver**, neither readable nor resettable from Python.
- It is not: body sleeping (`disable_sleep` has no effect), the taxels (same differences with or without tactile zones in the old repository), finger
  damping or stiffness, the PhysX friction parameters tested.
- The Robotiq grasp saturates the finger motor at 6.0 N·m during the grasp: stiffness and damping therefore do not change the clamping force.

### 7.4 What was tried (`tests/test_isaac_finger_sweep.py`, 4 environments × 6 starts)

| Attempt | Result |
|---|---|
| `reset_passes` 2 instead of 1 | 1.5 → 0.2 mm on one seed (measurement from the old repository); a 3rd pass brings nothing; adopted |
| `disable_sleep` | no effect |
| `finger_damping` 20 / 100 / 500, `finger_stiffness` 2000 | still 6 to 8 starts out of 24 |
| `physx_friction_correlation_distance` 0.0005, `physx_friction_offset_threshold` 0.002 | still 6 to 8 out of 24 |
| `finger_effort` 3 | 7/24, worst difference 1.2 mm, but tactile force divided by ~2 and pegs lost: changes the task, rejected |
| `physx_solver_type` 0 (PGS) | breaks the grasp (peg dropped in 18 cases out of 24) |
| `physx_friction_correlation_distance` 5e-5 | the server stops (exit code 1), cause not investigated |

What would remain: the mimic joints or the collision shapes of the Robotiq asset, or the inside of PhysX. Judged not worth it: it would force redoing the
demonstrations and the SFT.

### 7.5 Usage recommendations

1. **Compare conditions on the same sequence of rounds, from a fresh server, with the same `num_envs`.**
2. Keep `enhanced_determinism`, `reset_passes` and the whole YAML identical for **all** the data of an experiment; record with each run the YAML, the git
   commit and the Isaac Sim / Isaac Lab versions.
3. For an evaluation, `run_fixed_rounds` with the same `rounds` and the same `seed`; state the result as "same `(seed, round)`", **not** "same states bit
   for bit".
4. If independence from the history were essential (touch aside), the `franka` profile has it; it has no taxels and therefore does not answer the research
   question.
5. Do not call the `debug` operations during a collection.

---

## 8. Tests

| Command | What it does |
|---|---|
| `uv run pytest tests -q` | CPU tests (the ones that need Isaac are skipped): protocol, client/server, manager, tracker, collector, cost, static architecture tests, demonstrations |
| `SLURM_SUBMIT_DIR=$PWD bash jobs/job_isaac_tests.sh tests/test_isaac_fidelity.py` | fidelity: what survives a reset, reset divergence, dependence on the previous episode, valid starts, independence, long run, bit-exact sequence |
| `COST_NUM_ENVS=1,4,16,64 SLURM_SUBMIT_DIR=$PWD bash jobs/job_isaac_tests.sh tests/test_isaac_cost.py` | time and VRAM as a function of N (other variables: `COST_STEPS`, `COST_RESETS`, `COST_MEASURED_EPISODE`, `COST_ROLLOUT_STEPS`, `COST_EPISODE_STEPS`) |
| `SLURM_SUBMIT_DIR=$PWD SWEEP="default;finger_damping=50" SWEEP_TAG=x bash jobs/job_isaac_tests.sh tests/test_isaac_finger_sweep.py` | sweep of finger / PhysX settings against the residual (3 settings per job, ≈ 7 min each); report `reports/finger_sweep_<tag>.json` |

The Isaac tests accept `--isaac-config` and `--isaac-set KEY=VALUE` to change the YAML or a setting. The static tests check among other things that no file of
`src/envs/` imports a simulator or torch, nor the `isaaclab` subpackage.

---

## 9. What was not measured

- N > 64 (and the real VRAM / time at large N); several servers on several GPUs with Isaac.
- Bit-for-bit equality over long sequences (> 20 steps) and of the image rendering.
- Another GPU, driver or version of Isaac Sim / Isaac Lab.
- The effect of the start residual (N1) on a success rate or on the comparison of two policies.
- The cause of the server crash with `physx_friction_correlation_distance=5e-5`.
- GearMesh and NutThread in this environment: the YAMLs exist, nothing was run (no Isaac Sim where they were written). The Isaac task ids, the grip on the gear / the nut, the length of the episodes,
  the success test (`task_geometry`), the reset passes, whether `success_hold_steps: 5` is reachable for the nut.
- The loss / crushing of the peg over hundreds of seeds for the settings of the sweep (the criterion that had fixed `finger_effort = 6`).
