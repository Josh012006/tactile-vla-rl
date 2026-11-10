"""Probe the VLA models: what they receive, what they return, what they cost in GPU memory, and a short rollout on FORGE.

    python -m src.models.probe --model pi05     --port 8102          # env server running (jobs/env_server.sh, num_envs=1)
    python -m src.models.probe --model pi0_fast --port 8102
    python -m src.models.probe --model pi0_fast --synthetic          # no env server: fake observation (shapes, timing, VRAM only)

Run it through jobs/job_probe_vla.sh. One model per process (a model that is deleted does not give its memory back to the next one
in the same process, so the numbers of the second would be wrong). openpi is imported inside the functions that need it: this file
imports without JAX, so its logic is tested on CPU (tests/test_probe.py).

What is reported (JSON in reports/probe_<model>_<time>.json and printed):
  * the observation the model receives (openpi's DROID format: the env's keys + `prompt`) and what is left of it after openpi's own
    input transforms (normalised state, tokens: how many of max_token_len are used);
  * the action chunk the model returns, next to the current joint state, with an indicator of what it looks like (offsets from the
    state, or absolute targets) to compare with `output_semantics` of the model YAML: the YAML is what the rollout uses, the indicator
    only tells whether it is plausible;
  * GPU memory: parameters (bytes, by dtype), JAX bytes in use after the load, after the first inference (compile) and its peak,
    this process's usage and the whole GPU in nvidia-smi. Weights are loaded in bfloat16 (openpi's create_trained_policy);
    inference only: training (optimizer, activations, a second copy for a target) needs much more;
  * timing of the first inference (includes the compilation) and of the following ones;
  * a rollout of one episode per environment on the FORGE server: the chunk is executed `--replan` steps at a time, then the model is
    called again from the new observation.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml

from src.envs import protocol as P

ROOT = Path(__file__).resolve().parents[2]
SEMANTICS = ("offsets_from_chunk_start_state", "absolute_targets")
MIB = 2 ** 20


# -- model YAML ---------------------------------------------------------------------------------------------------------------
def load_model_cfg(name: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "model" / f"{name}.yaml").read_text())
    if cfg["output_semantics"] not in SEMANTICS:
        raise ValueError(f"configs/model/{name}.yaml: output_semantics must be one of {SEMANTICS}, got {cfg['output_semantics']!r}")
    return cfg


# -- observation in, actions out (pure numpy) -----------------------------------------------------------------------------------
def to_openpi_obs(obs: Dict[str, np.ndarray], env: int, prompt: str) -> dict:
    """The observation of environment `env` in the format openpi's DROID policies read (droid_policy.DroidInputs): the five keys it
    needs, the prompt included. The env keys already are DROID's (images 224 x 224 uint8, 7 joints, 1 gripper)."""
    return {
        "observation/exterior_image_1_left": obs[P.EXTERIOR_IMAGE][env],
        "observation/wrist_image_left": obs[P.WRIST_IMAGE][env],
        "observation/joint_position": np.asarray(obs[P.JOINT_POSITION][env], dtype=np.float32),
        "observation/gripper_position": np.asarray(obs[P.GRIPPER_POSITION][env], dtype=np.float32).reshape(-1),
        "prompt": prompt,
    }


def chunk_to_targets(chunk: np.ndarray, q_plan: np.ndarray, semantics: str) -> np.ndarray:
    """Model output (H, >= 8) -> env actions (H, 8): 7 ABSOLUTE joint targets + gripper.
    offsets_from_chunk_start_state: the 7 joints are offsets from the joint state at the moment of the plan (q_plan): add it back once.
    absolute_targets: the model already returns absolute targets. The gripper (column 7) is passed through."""
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] < P.ACTION_DIM:
        raise ValueError(f"expected a chunk (H, >= {P.ACTION_DIM}), got {chunk.shape}")
    out = np.zeros((len(chunk), P.ACTION_DIM), np.float32)
    out[:, :7] = chunk[:, :7]
    if semantics == "offsets_from_chunk_start_state":
        out[:, :7] += np.asarray(q_plan, dtype=np.float32)[:7]
    elif semantics != "absolute_targets":
        raise ValueError(semantics)
    out[:, 7] = chunk[:, 7]
    return out


def looks_like(raw_chunk: np.ndarray, q_plan: np.ndarray) -> dict:
    """Indicator only: is the raw chunk closer to zero (offsets) or to the current joint state (absolute targets)?"""
    raw = np.asarray(raw_chunk, dtype=np.float64)[:, :7]
    q = np.asarray(q_plan, dtype=np.float64)[:7]
    d_off, d_abs = float(np.abs(raw).mean()), float(np.abs(raw - q).mean())
    return {"mean_abs_raw": d_off, "mean_abs_raw_minus_state": d_abs, "looks_like": "offsets" if d_off < d_abs else "absolute"}


def describe(x) -> dict:
    a = np.asarray(x)
    d = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if a.size and a.dtype.kind in "fiu":
        d.update(min=float(a.min()), max=float(a.max()), mean=float(a.astype(np.float64).mean()))
    return d


# -- rollout ----------------------------------------------------------------------------------------------------------------------
def run_rollout(env, infer: Callable[[dict], dict], model_cfg: dict, steps: int, replan: int, seed: int = 0) -> dict:
    """One round (one episode per environment) driven by `infer`: openpi obs dict -> {"actions": (H, >= 8) [, "policy_timing"]}.
    `env` is a VectorEnvClient. The chunk is executed `replan` steps, then the model is called again from the new observation.
    Stops at `steps` steps or when every episode has ended. auto_reset is off: nothing starts by itself."""
    n, prompt = env.num_envs, env.spec.prompt
    semantics = model_cfg["output_semantics"]
    replan = max(1, min(int(replan), int(model_cfg["action_horizon"])))
    res = env.reset(seed=seed, round=0)
    actions = np.zeros((n, P.ACTION_DIM), np.float32)         # rows of environments that ended are ignored by the server (not acting)
    chunk: List[Optional[np.ndarray]] = [None] * n
    used = [0] * n
    ended = np.zeros(n, bool)
    plans: List[dict] = []
    track = [[] for _ in range(n)]              # |target - joint state after the step| (rad, max over the 7 joints), per step
    first_q = np.array(res.obs[P.JOINT_POSITION], dtype=np.float32)
    t = 0
    while t < steps and not res.all_done:
        for e in range(n):
            if ended[e]:
                continue
            if chunk[e] is None or used[e] >= replan:
                q_plan = np.asarray(res.obs[P.JOINT_POSITION][e], dtype=np.float32)
                t_call = time.monotonic()
                out = infer(to_openpi_obs(res.obs, e, prompt))
                wall_ms = 1e3 * (time.monotonic() - t_call)           # openpi's own `infer_ms` stops before the result is fetched
                raw = np.asarray(out["actions"], dtype=np.float32)
                chunk[e], used[e] = chunk_to_targets(raw, q_plan, semantics), 0
                plans.append({"step": t, "env": e, "raw_shape": list(raw.shape), "q_plan": q_plan.tolist(),
                              "raw_first": raw[0].tolist(), "raw_last": raw[-1].tolist(),
                              "infer_ms": float(out.get("policy_timing", {}).get("infer_ms", float("nan"))), "wall_ms": round(wall_ms, 1),
                              **looks_like(raw, q_plan)})
            actions[e] = chunk[e][used[e]]
            used[e] += 1
        res = env.step(actions, auto_reset=False)
        for e in range(n):
            if not ended[e]:
                track[e].append(float(np.abs(actions[e, :7] - res.obs[P.JOINT_POSITION][e][:7]).max()))
        ended |= res.done
        t += 1
    per_env = [{"env": e, "success": bool(res.success[e]), "terminated": bool(res.terminated[e]), "truncated": bool(res.truncated[e]),
                "steps": len(track[e]), "tracking_err_max_rad": max(track[e]) if track[e] else None,
                "tracking_err_mean_rad": float(np.mean(track[e])) if track[e] else None,
                "joint_motion_rad": float(np.abs(np.asarray(res.obs[P.JOINT_POSITION][e]) - first_q[e]).max())} for e in range(n)]
    return {"steps_run": t, "replan": replan, "semantics_used": semantics, "all_done": bool(res.all_done), "envs": per_env, "plans": plans}


# -- GPU memory ---------------------------------------------------------------------------------------------------------------------
def gpu_memory() -> dict:
    """JAX allocator state of device 0 (bytes in use / peak, MiB) and nvidia-smi (this process, whole GPU). Best effort: a missing
    source is reported as an `*_error` entry, never raised."""
    out: dict = {}
    try:
        import jax
        ms = jax.local_devices()[0].memory_stats() or {}
        for k, name in (("bytes_in_use", "jax_in_use_mib"), ("peak_bytes_in_use", "jax_peak_in_use_mib"), ("bytes_limit", "jax_limit_mib")):
            if k in ms:
                out[name] = round(ms[k] / MIB, 1)
    except Exception as e:  # noqa: BLE001
        out["jax_error"] = repr(e)
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()
        mine = [int(m) for pid, m in (a.split(",") for a in apps if a.strip()) if int(pid) == os.getpid()]
        out["process_mib"] = sum(mine) if mine else None          # None: pid not visible (container): use gpu_used_mib
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()[0].split(",")
        out["gpu_used_mib"], out["gpu_total_mib"] = int(gpu[0]), int(gpu[1])
    except Exception as e:  # noqa: BLE001
        out["nvidia_smi_error"] = repr(e)
    return out


def param_bytes(policy) -> dict:
    """Size of the model parameters actually on the device, by dtype."""
    try:
        import flax.nnx as nnx
        import jax
        by: Dict[str, int] = {}
        for x in jax.tree.leaves(nnx.state(policy._model, nnx.Param)):   # noqa: SLF001
            by[str(x.dtype)] = by.get(str(x.dtype), 0) + int(x.nbytes)
        return {"total_mib": round(sum(by.values()) / MIB, 1), "by_dtype_mib": {k: round(v / MIB, 1) for k, v in by.items()}}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


# -- inspecting (needs openpi; loading itself is in src/models/loading.py) -------------------------------------------------------------------------------------------
def trainable_counts(policy, train_cfg) -> dict:
    """Parameters of the model, and the ones the training would update under the freeze filter of the config (millions)."""
    try:
        import flax.nnx as nnx
        import flax.traverse_util as tu
        import numpy as np

        def count(state, pred=lambda path: True):
            return sum(int(np.prod(v.shape)) for k, v in tu.flatten_dict(state.to_pure_dict()).items() if pred(k))

        model = policy._model                                                                      # noqa: SLF001
        total = nnx.state(model, nnx.Param)
        trainable = nnx.state(model, nnx.All(nnx.Param, nnx.Not(train_cfg.freeze_filter)))
        is_lora = lambda path: any("lora" in str(part) for part in path)                            # noqa: E731
        return {"total_m": round(count(total) / 1e6, 1), "trainable_m": round(count(trainable) / 1e6, 1),
                "lora_m": round(count(total, is_lora) / 1e6, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def inspect_transforms(policy, openpi_obs: dict) -> dict:
    """What is left of the observation after openpi's input transforms (what the model really receives), per leaf."""
    import flax.traverse_util as tu
    out = {}
    try:
        flat = tu.flatten_dict(policy._input_transform(copy.deepcopy(openpi_obs)), sep="/")      # noqa: SLF001
        out = {k: describe(v) for k, v in flat.items()}
        mask = flat.get("tokenized_prompt_mask")
        if mask is not None:
            out["_tokens_used"] = int(np.asarray(mask).sum())
            out["_tokens_capacity"] = int(np.asarray(mask).shape[-1])
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def synthetic_obs(prompt: str) -> dict:
    rng = np.random.default_rng(0)
    return {"observation/exterior_image_1_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, np.float32), "observation/gripper_position": np.zeros(1, np.float32),
            "prompt": prompt}


def save_images(obs: dict, out_dir: Path, tag: str) -> List[str]:
    from PIL import Image
    paths = []
    for key, name in (("observation/exterior_image_1_left", "exterior"), ("observation/wrist_image_left", "wrist")):
        p = out_dir / f"probe_{tag}_{name}.png"
        Image.fromarray(np.asarray(obs[key])).save(p)
        paths.append(str(p))
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="configs/model/<model>.yaml: pi05 | pi0_fast")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--steps", type=int, default=40, help="control steps of the rollout (max_steps of the task is 150)")
    ap.add_argument("--replan", type=int, default=8, help="chunk steps executed before the model is called again")
    ap.add_argument("--warm", type=int, default=3, help="extra inferences after the first one, to time and to see the peak memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-lora", choices=["true", "false"], default=None, help="override `use_lora` of the model YAML (false: the exact base function)")
    ap.add_argument("--lora-rank-vlm", type=int, default=None, help="override lora_rank.vlm of the model YAML")
    ap.add_argument("--lora-rank-action-head", type=int, default=None, help="override lora_rank.action_head of the model YAML")
    ap.add_argument("--synthetic", action="store_true", help="no env server: a fake observation, no rollout (shapes, timing, VRAM)")
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    a = ap.parse_args(argv)

    cfg = load_model_cfg(a.model)
    if a.use_lora is not None:
        cfg["use_lora"] = a.use_lora == "true"
    for key, val in (("vlm", a.lora_rank_vlm), ("action_head", a.lora_rank_action_head)):
        if val is not None:
            cfg["lora_rank"] = {**(cfg.get("lora_rank") or {}), key: val}
    from src.models.openpi_config import parse_lora_settings
    lora = parse_lora_settings(cfg)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}{'' if lora.use_lora else '_nolora'}_{time.strftime('%Y%m%d_%H%M%S')}"
    report: dict = {"model": a.model, "yaml": cfg, "lora": dataclasses.asdict(lora), "synthetic": a.synthetic, "memory": {"before_load": gpu_memory()}}
    try:
        import jax
        report["device"] = str(jax.local_devices()[0])
    except Exception as e:  # noqa: BLE001
        report["device"] = repr(e)

    from src.models.loading import load_policy
    t0 = time.monotonic()
    policy, train_cfg = load_policy(cfg)
    report["load_s"] = round(time.monotonic() - t0, 1)
    report["memory"]["after_load"] = gpu_memory()
    report["params"] = param_bytes(policy)
    m = train_cfg.model
    report["openpi_model"] = {"type": str(m.model_type), "action_dim": m.action_dim, "action_horizon": m.action_horizon,
                              "max_token_len": getattr(m, "max_token_len", None),
                              "paligemma_variant": getattr(m, "paligemma_variant", None),
                              "action_expert_variant": getattr(m, "action_expert_variant", None)}
    report["load_mode"] = cfg.get("load", "strict")
    report["trainable"] = trainable_counts(policy, train_cfg)

    env = None
    if a.synthetic:
        first = synthetic_obs("Insert the peg into the socket.")
    else:
        from src.envs.client import VectorEnvClient
        env = VectorEnvClient([(a.host, a.port)])
        res = env.reset(seed=a.seed, round=0)
        first = to_openpi_obs(res.obs, 0, env.spec.prompt)
        report["env_spec"] = {"num_envs": env.num_envs, "action_dim": env.spec.action_dim, "prompt": env.spec.prompt,
                              "obs": {k: [list(s), d] for k, (s, d) in env.spec.obs.items()}}
    report["images"] = save_images(first, out_dir, tag)
    report["observation_sent"] = {k: (describe(v) if k != "prompt" else v) for k, v in first.items()}
    report["observation_after_transforms"] = inspect_transforms(policy, first)

    timings = []
    t0 = time.monotonic()
    out = policy.infer(first)
    timings.append(round(1e3 * (time.monotonic() - t0), 1))
    report["memory"]["after_first_inference"] = gpu_memory()
    chunk = np.asarray(out["actions"])
    report["first_chunk"] = {**describe(chunk), "rows": np.round(chunk, 4).tolist(),
                             **looks_like(chunk, first["observation/joint_position"])}
    for _ in range(a.warm):
        t0 = time.monotonic()
        policy.infer(first)
        timings.append(round(1e3 * (time.monotonic() - t0), 1))
    report["infer_ms_wall"] = {"first_with_compile": timings[0], "next": timings[1:]}
    report["memory"]["after_warm"] = gpu_memory()

    if env is not None:
        report["rollout"] = run_rollout(env, policy.infer, cfg, a.steps, a.replan, seed=a.seed)
        report["memory"]["after_rollout"] = gpu_memory()
        env.close()

    path = out_dir / f"probe_{tag}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print_summary(report, path)
    return 0


def print_summary(r: dict, path: Path) -> None:
    mem = r["memory"]
    print(f"\n=== {r['model']}  ({r['device']})   {'SYNTHETIC observation' if r['synthetic'] else 'FORGE observation'}")
    print(f"openpi config {r['yaml']['openpi_config']}  checkpoint {r['yaml']['checkpoint']}")
    print(f"model: {r['openpi_model']}   load {r['load_s']} s   mode: {r['load_mode']}")
    print(f"lora settings: {r['lora']}   parameters (M): {r['trainable']}")
    print("-- observation sent to the model")
    for k, v in r["observation_sent"].items():
        print(f"   {k:42s} {v}")
    print("-- after openpi's input transforms (what the model sees)")
    for k, v in r["observation_after_transforms"].items():
        print(f"   {k:42s} {v}")
    fc = r["first_chunk"]
    print(f"-- first chunk {fc['shape']} {fc['dtype']}   range [{fc.get('min', 0):.3f}, {fc.get('max', 0):.3f}]   looks like: {fc['looks_like']} "
          f"(mean|raw| {fc['mean_abs_raw']:.3f}, mean|raw - state| {fc['mean_abs_raw_minus_state']:.3f}); YAML says {r['yaml']['output_semantics']}")
    for i, row in enumerate(fc["rows"][:3]):
        print(f"   row {i}: {row}")
    print(f"-- inference wall time (ms): first (compile) {r['infer_ms_wall']['first_with_compile']}, next {r['infer_ms_wall']['next']}")
    print("-- GPU memory (MiB)")
    print(f"   parameters on device: {r['params']}")
    for name, v in mem.items():
        print(f"   {name:22s} {v}")
    base = mem["before_load"].get("jax_in_use_mib", 0) or 0
    after = mem["after_warm"].get("jax_in_use_mib")
    peak = mem["after_warm"].get("jax_peak_in_use_mib")
    if after is not None and peak is not None:
        print(f"   => JAX in use after warm-up {after - base:.0f} MiB, peak {peak - base:.0f} MiB (above the state before the load)")
    ro = r.get("rollout")
    if ro:
        print(f"-- rollout: {ro['steps_run']} steps, replan {ro['replan']}, semantics used {ro['semantics_used']}")
        for e in ro["envs"]:
            print(f"   env {e['env']}: success {e['success']} terminated {e['terminated']} truncated {e['truncated']} steps {e['steps']} "
                  f"| tracking error max {e['tracking_err_max_rad']} mean {e['tracking_err_mean_rad']} rad | joint motion {e['joint_motion_rad']:.3f} rad")
        walls = [p["wall_ms"] for p in ro["plans"]]
        print(f"   inference wall time per plan (ms): first {walls[0]}, median {float(np.median(walls)):.0f}, max {max(walls)}")
        votes = [p["looks_like"] for p in ro["plans"]]
        print(f"   {len(votes)} plans, indicator: {votes.count('offsets')} offsets / {votes.count('absolute')} absolute (YAML: {ro['semantics_used']})")
    print(f"report: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
