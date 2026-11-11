"""Probe the VLA models: what they receive, what they return, what they cost in GPU memory, and a short rollout on FORGE.
A measuring tool, not part of the model code: the model lives in src/models, the rollout and its video in src/eval.

    python -m scripts.probe_vla --model pi05     --port 8102          # env server running (jobs/env_server.sh, num_envs=1)
    python -m scripts.probe_vla --model pi0_fast --port 8102
    python -m scripts.probe_vla --model pi0_fast --synthetic          # no env server: fake observation (shapes, timing, VRAM only)
    python -m scripts.probe_vla --model pi05 --obs-file reports/obs_<tag>.npz   # no env server: replay a saved observation
    python -m scripts.probe_vla --model pi05 --no-lora                # the exact base function (use_lora=false in every component)
    python -m scripts.probe_vla --model pi05 --set action_head.lora_rank=8     # any key of the model YAML: <component>.<key>=<value>

Run it through jobs/job_probe_vla.sh. One model per process (a model that is deleted does not give its memory back to the next one
in the same process, so the numbers of the second would be wrong). openpi is imported inside the functions that need it: this file
imports without JAX, so its logic is tested on CPU (tests/test_probe_vla.py).

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
  * the check of the common interface (src/models/interface.py + openpi_policy.py) on the real model: the batched path
    (`OpenPIPolicy.plan`) and openpi's own `Policy.infer` return the same chunk for the same observation and the same random key;
  * a rollout of one episode per environment on the FORGE server (src/eval/rollout.py), driven through that interface: the chunk is
    executed `--replan` steps at a time, then the model is called again from the new observation. It records the task progress
    (peg-socket xy / z distance, contact force: src/eval/progress.py) and a video (exterior | wrist cameras + that text)
    reports/probe_<tag>_env<k>.mp4;
  * the first observation, saved as reports/obs_<tag>.npz and replayable with --obs-file: two models (LoRA or not, ...) can then be
    compared on exactly the same input with `python -m scripts.compare_reports`.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from src.envs import protocol as P
from src.eval import video
from src.eval.progress import XY_INSERTED_MM, progress_sample
from src.eval.rollout import looks_like, run_rollout
from src.models.interface import ROOT, apply_overrides, load_model_cfg, to_openpi_obs
from src.models.settings import describe as describe_settings
from src.models.settings import parse_model_settings

MIB = 2 ** 20


def describe(x) -> dict:
    a = np.asarray(x)
    d = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if a.size and a.dtype.kind in "fiu":
        d.update(min=float(a.min()), max=float(a.max()), mean=float(a.astype(np.float64).mean()))
    return d


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


def save_obs(obs: dict, path: Path) -> None:
    """The openpi observation dict (arrays + `prompt`) in one .npz: replay it with --obs-file."""
    np.savez_compressed(path, **{k.replace("/", "__"): (np.asarray(v) if k != "prompt" else np.asarray(str(v))) for k, v in obs.items()})


def load_obs(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k.replace("__", "/"): (str(z[k]) if k == "prompt" else z[k]) for k in z.files}


def obs_sha256(obs: dict) -> str:
    """Same observation <=> same hash (keys, dtypes, shapes, bytes, prompt): what `scripts.compare_reports` checks before comparing two models."""
    h = hashlib.sha256()
    for k in sorted(obs):
        v = obs[k]
        h.update(k.encode())
        if k == "prompt":
            h.update(str(v).encode())
        else:
            a = np.ascontiguousarray(v)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def env_obs_of(row: dict) -> Dict[str, np.ndarray]:
    """The env-format observation (N = 1) of an openpi observation dict: what `VLAPolicy.plan` takes."""
    return {P.EXTERIOR_IMAGE: np.asarray(row["observation/exterior_image_1_left"])[None], P.WRIST_IMAGE: np.asarray(row["observation/wrist_image_left"])[None],
            P.JOINT_POSITION: np.asarray(row["observation/joint_position"], np.float32)[None],
            P.GRIPPER_POSITION: np.asarray(row["observation/gripper_position"], np.float32).reshape(1, -1)}


def interface_check(model, row: dict, seed: int = 7) -> dict:
    """`model.plan` (batched path of src/models/openpi_policy.py) against openpi's `Policy.infer` on the same observation and the same random
    key. Both give the model's chunk after openpi's output transforms. Expected: identical, or bfloat16 noise at most."""
    import jax
    model.seed(seed)
    model.policy._rng = jax.random.key(seed)                                      # noqa: SLF001
    ref = np.asarray(model.policy.infer(copy.deepcopy(row))["actions"], dtype=np.float32)
    got = model.plan(env_obs_of(row), row["prompt"]).raw[0, 0][:, :ref.shape[1]]
    return {"shape_reference": list(ref.shape), "shape_batched": list(got.shape),
            "max_abs_diff": float(np.abs(ref - got).max()) if ref.shape == got.shape else None}


def parse_sets(items: List[str]) -> Dict[str, object]:
    """`--set vlm.use_lora=false --set action_head.lora_rank=8` -> {"vlm.use_lora": False, "action_head.lora_rank": 8} (values read as YAML)."""
    out: Dict[str, object] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects COMPONENT.KEY=VALUE, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
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
    ap.add_argument("--set", action="append", default=[], metavar="COMPONENT.KEY=VALUE",
                    help="override a key of the model YAML, e.g. vlm.use_lora=false, action_head.lora_rank=8 (repeatable). "
                         "No LoRA anywhere = the exact base function")
    ap.add_argument("--no-lora", action="store_true", help="shortcut: use_lora=false for every component of the model (the exact base function)")
    ap.add_argument("--synthetic", action="store_true", help="no env server: a fake observation, no rollout (shapes, timing, VRAM)")
    ap.add_argument("--obs-file", default=None, help="no env server: replay an observation saved by an earlier run (reports/obs_<tag>.npz), no rollout")
    ap.add_argument("--no-video", action="store_true", help="do not record the rollout video")
    ap.add_argument("--video-envs", type=int, default=1, help="record the first N environments")
    ap.add_argument("--fps", type=int, default=15, help="frame rate of the video file (the env's control rate is not read: DROID's 15 Hz by default)")
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    a = ap.parse_args(argv)

    if a.synthetic and a.obs_file:
        ap.error("--synthetic and --obs-file are exclusive")
    cfg = load_model_cfg(a.model)
    overrides = parse_sets(a.set)
    if a.no_lora:
        overrides = {**{f"{c}.use_lora": False for c in parse_model_settings(cfg).components}, **overrides}
    cfg = apply_overrides(cfg, overrides)
    settings = parse_model_settings(cfg)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}{'' if settings.any_lora else '_nolora'}_{time.strftime('%Y%m%d_%H%M%S')}"
    report: dict = {"model": a.model, "yaml": cfg, "settings": dataclasses.asdict(settings), "settings_text": describe_settings(settings),
                           "memory": {"before_load": gpu_memory()}}
    try:
        import jax
        report["device"] = str(jax.local_devices()[0])
    except Exception as e:  # noqa: BLE001
        report["device"] = repr(e)

    from src.models.openpi_policy import OpenPIPolicy
    t0 = time.monotonic()
    model = OpenPIPolicy.load(cfg)
    policy, train_cfg = model.policy, model.train_cfg                  # openpi's own Policy and TrainConfig, for the measurements below
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
    if a.obs_file:
        first, source = load_obs(a.obs_file), f"file:{a.obs_file}"
    elif a.synthetic:
        first, source = synthetic_obs("Insert the peg into the socket."), "synthetic"
    else:
        from src.envs.client import VectorEnvClient
        env = VectorEnvClient([(a.host, a.port)])
        res = env.reset(seed=a.seed, round=0)
        first, source = to_openpi_obs(res.obs, 0, env.spec.prompt), "env"
        report["env_spec"] = {"num_envs": env.num_envs, "action_dim": env.spec.action_dim, "prompt": env.spec.prompt,
                              "obs": {k: [list(s), d] for k, (s, d) in env.spec.obs.items()}}
    report["observation_source"] = source
    report["observation_sha256"] = obs_sha256(first)
    if not a.obs_file:
        report["observation_file"] = str(out_dir / f"obs_{tag}.npz")
        save_obs(first, out_dir / f"obs_{tag}.npz")
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
    report["interface_check"] = interface_check(model, first)

    if env is not None:
        frames = {} if a.no_video else {e: [] for e in range(min(a.video_envs, env.num_envs))}

        def on_step(t, res, plans):
            for e, fl in frames.items():
                fl.append(video.compose_frame(res.obs, e, t, plans[e], progress_sample(res.obs, e), bool(res.success[e])))

        model.seed(a.seed)
        report["rollout"] = run_rollout(env, model, a.steps, a.replan, seed=a.seed, on_step=on_step if frames else None)
        report["videos"] = [video.write_video(fl, out_dir / f"probe_{tag}_env{e}.mp4", a.fps) for e, fl in frames.items()]
        report["memory"]["after_rollout"] = gpu_memory()
        env.close()

    path = out_dir / f"probe_{tag}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print_summary(report, path)
    return 0


def print_summary(r: dict, path: Path) -> None:
    mem = r["memory"]
    print(f"\n=== {r['model']}  ({r['device']})   observation: {r['observation_source']}   sha256 {r['observation_sha256'][:12]}")
    print(f"openpi config {r['yaml']['openpi_config']}  checkpoint {r['yaml']['checkpoint']}")
    print(f"model: {r['openpi_model']}   load {r['load_s']} s   mode: {r['load_mode']}")
    print(f"settings: {r['settings_text']}   parameters (M): {r['trainable']}")
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
    ic = r.get("interface_check", {})
    print(f"-- interface check (model.plan vs openpi Policy.infer, same key): {ic}")
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
            g = e["progress"]
            if g.get("samples"):
                print(f"   env {e['env']} peg-socket: xy {g['xy_mm_start']:.1f} -> {g['xy_mm_end']:.1f} mm (min {g['xy_mm_min']:.1f} at step {g['step_of_xy_min']}), "
                      f"z {g['z_mm_start']:.1f} -> {g['z_mm_end']:.1f} mm (min {g['z_mm_min']:.1f}), steps within {XY_INSERTED_MM} mm xy: "
                      f"{g['steps_xy_within_tolerance']}, max force {g['force_n_max']:.1f} N")
        walls = [p["wall_ms"] for p in ro["plans"]]
        print(f"   inference wall time per call (ms): first {walls[0]}, median {float(np.median(walls)):.0f}, max {max(walls)}")
        votes = [p["looks_like"] for p in ro["plans"]]
        print(f"   {len(votes)} plans, indicator: {votes.count('offsets')} offsets / {votes.count('absolute')} absolute (YAML: {ro['semantics_used']})")
    for v in r.get("videos", []):
        print(f"video: {v}")
    if r.get("observation_file"):
        print(f"observation saved (replay with --obs-file): {r['observation_file']}")
    print(f"report: {path}")


if __name__ == "__main__":
    raise SystemExit(main())

