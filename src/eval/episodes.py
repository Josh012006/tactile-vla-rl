"""Evaluation of a policy on FIXED episodes, one environment at a time, deterministic by construction and resumable at the episode.

What makes an evaluation reproducible, and what this file does about each part:

  * the START of an episode: episode i is `env.reset(seed=seeds[i], round=0)`, with `seeds` generated ONCE from a master seed (default 42,
    the way expo-ft does it: `np.random.default_rng(master).integers(0, 2**31 - 1, n)`), so every checkpoint sees the same episodes and a
    longer evaluation extends a shorter one (the first n seeds do not change). One environment per server: the start of an episode
    depends on (seed, round, num_envs, env index) (docs/envs/isaaclab_env.md N2), and the model is always called with a batch of 1, so the
    shapes of the jitted model (and therefore its arithmetic) never depend on how many environments are still running.
  * the POLICY's randomness: `policy.seed(seed)` before every episode. pi0-FAST decodes greedily (no randomness at all); pi0.5 draws its
    flow-matching noise from the policy's key, which is split at every call and therefore depends on the history: restarting the key at the
    episode's seed makes the noise a function of the episode alone, whatever ran before it, in which order, or in which thread.
  * the ENVIRONMENT: bit-identical only for an identical sequence of calls from a fresh server on the same machine (G1 of
    docs/envs/isaaclab_env.md); the start of an episode after other episodes is almost, not strictly, identical (N1). Each record therefore
    carries `start_sha` (the first observation) and `traj_sha` (every joint / gripper / peg-socket reading of the episode): two records
    with the same `traj_sha` ARE the same episode bit for bit, and `summarize` folds them into one `eval_sha` per checkpoint.

Resume. Each finished episode is one line of `results/<label>.jsonl` (flushed and fsynced). A restart reads the file, skips the episodes
that are there, runs the others and recomputes everything from the file: the numbers are those of an evaluation that was never cut.
A line cut by a kill is dropped. A checkpoint evaluated in several attempts is statistically equivalent to one evaluated in one go, not
bit-identical (N1): `attempt` is recorded in every line and `single_attempt` in the summary.

numpy only (no JAX, no matplotlib): it works with any `VLAPolicy`, openpi or a test double.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.envs import protocol as P
from src.eval.rollout import run_rollout

DEFAULT_SEED = 42
SEED_BOUND = 2 ** 31 - 1
SEEDS_FILE = "episode_seeds.json"
HASHED_KEYS = (P.JOINT_POSITION, P.GRIPPER_POSITION, P.PEG_SOCKET_OFFSET)       # what a trajectory hash covers: no image (N3: the rendering was not compared)


class EvalAborted(Exception):
    """The evaluation was told to stop (SIGTERM, SIGINT) in the middle of an episode: the episode is dropped, not recorded."""


# -- files -------------------------------------------------------------------------------------------------------------------------
def atomic_write_json(path: Path, obj: Any) -> None:
    """Write `obj` as JSON so that a reader never sees half a file: a temporary file next to it, then a rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# -- the fixed episodes ------------------------------------------------------------------------------------------------------------------
def make_seeds(master_seed: int, n: int) -> List[int]:
    """The n episode seeds of `master_seed`, as expo-ft (scripts/eval_curve.py) draws them."""
    return [int(s) for s in np.random.default_rng(int(master_seed)).integers(0, SEED_BOUND, size=int(n))]


def load_or_create_seeds(output_dir: Path, master_seed: int, n: int) -> List[int]:
    """The n seeds of the evaluation, kept in <output_dir>/episode_seeds.json so that a later run (more checkpoints, more episodes) uses the same ones.
    A file made with another master seed is refused; one made with another n is fine (the first seeds of a longer list are those of a shorter one)."""
    path = Path(output_dir) / SEEDS_FILE
    seeds = make_seeds(master_seed, n)
    if path.exists():
        saved = json.loads(path.read_text())
        if int(saved["master_seed"]) != int(master_seed):
            raise ValueError(f"{path} was made with --seed {saved['master_seed']}, not {master_seed}: the episodes would not be the ones the earlier results used "
                             f"(use another --output-dir for another seed)")
        k = min(len(saved["seeds"]), n)
        if saved["seeds"][:k] != seeds[:k]:
            raise ValueError(f"{path} does not hold the seeds this numpy generates for --seed {master_seed} (another numpy version?): refusing to mix them")
        if len(saved["seeds"]) >= n:
            return seeds
    atomic_write_json(path, {"master_seed": int(master_seed), "n": int(n), "seeds": seeds})
    return seeds


# -- one episode ------------------------------------------------------------------------------------------------------------------------
class TrajectoryHash:
    """Running sha256 of the readings of an episode (HASHED_KEYS, environment 0). `start` is the hash of the first observation alone."""

    def __init__(self):
        self._h = hashlib.sha256()
        self.start: Optional[str] = None

    def update(self, obs: Mapping[str, np.ndarray]) -> None:
        step = hashlib.sha256()
        for key in HASHED_KEYS:
            if key in obs:
                step.update(np.ascontiguousarray(np.asarray(obs[key])[0]).tobytes())
        digest = step.digest()
        if self.start is None:
            self.start = digest.hex()[:16]
        self._h.update(digest)

    def hexdigest(self) -> str:
        return self._h.hexdigest()[:16]


def run_episode(env, policy, *, episode: int, seed: int, replan: int, max_steps: int, stop: Optional[threading.Event] = None,
                video_path: Optional[Path] = None, fps: int = 15) -> dict:
    """One episode of `episode`'s `seed` on `env` (a VectorEnvClient with ONE environment), driven by `policy` (src/models/interface.py). The policy's random stream
    is restarted at `seed` first. Returns the record that goes in the JSONL (see `summarize` for what is read back). Raises EvalAborted when `stop` is set
    during the episode, and RuntimeError when the episode did not end within `max_steps` (it is never recorded as a failure it did not make)."""
    if env.num_envs != 1:
        raise ValueError(f"an evaluation episode runs in a server with ONE environment, this one has {env.num_envs}")
    policy.seed(seed)
    decode = getattr(policy, "decode_stats", None)
    if decode is not None:
        decode.reset()
    hasher = TrajectoryHash()
    frames: Optional[list] = [] if video_path is not None else None

    def on_step(t, res, plans):
        if stop is not None and stop.is_set():
            raise EvalAborted(f"episode {episode} stopped at step {t}")
        hasher.update(res.obs)
        if frames is not None:
            from src.eval import video
            from src.eval.progress import progress_sample
            frames.append(video.compose_frame(res.obs, 0, t, plans[0], progress_sample(res.obs, 0), bool(res.success[0])))

    t0 = time.monotonic()
    out = run_rollout(env, policy, steps=max_steps, replan=replan, seed=seed, on_step=on_step)
    wall_s = time.monotonic() - t0
    if not out["all_done"]:
        raise RuntimeError(f"episode {episode} (seed {seed}) did not end within {max_steps} steps: is the server's max_steps_per_episode larger than the protocol's?")
    e = out["envs"][0]
    walls = [p["wall_ms"] for p in out["plans"]]
    record = {"episode": int(episode), "seed": int(seed), "success": bool(e["success"]), "terminated": bool(e["terminated"]), "truncated": bool(e["truncated"]),
              "steps": int(e["steps"]), "start_sha": hasher.start, "traj_sha": hasher.hexdigest(), "plans": len(out["plans"]),
              "plan_ms_median": float(np.median(walls)) if walls else None, "tracking_err_max_rad": e["tracking_err_max_rad"],
              "progress": {k: v for k, v in e["progress"].items() if k != "success"}, "wall_s": round(wall_s, 2),
              "decode": decode.summary() if decode is not None and decode.chunks else None, "video": None}
    if frames:
        from src.eval import video
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        record["video"] = video.write_video(frames, Path(video_path), fps)
    return record


# -- the append-only log of a checkpoint ------------------------------------------------------------------------------------------------------
class EpisodeLog:
    """<label>.jsonl: one JSON line per finished episode, appended, flushed and fsynced. One writer per file (a lock inside the process; the sweep's own lock between processes)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.duplicates = 0
        self._lock = threading.Lock()

    def load(self) -> Dict[int, dict]:
        """The records by episode index. A last line cut by a kill (no newline at its end) is dropped and the file is cut back to the last whole line, so that the next
        append starts on a clean one. A whole line that is not JSON is an error (the file was edited or damaged: nothing here would produce it). An episode that appears twice
        (two runs on the same folder) keeps its first record, and `duplicates` counts the others."""
        self.duplicates = 0
        if not self.path.exists():
            return {}
        raw = self.path.read_bytes()
        end = raw.rfind(b"\n") + 1
        if end < len(raw):
            with self._lock, open(self.path, "r+b") as f:
                f.truncate(end)
        records: Dict[int, dict] = {}
        for n, line in enumerate(raw[:end].splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                episode = int(rec["episode"])
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"{self.path}, line {n}: not an episode record ({e!r})") from e
            if episode in records:
                self.duplicates += 1
            else:
                records[episode] = rec
        return records

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(dict(record), default=str) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())


# -- numbers -----------------------------------------------------------------------------------------------------------------------------
def wilson95(k: int, n: int) -> Tuple[float, float]:
    """Wilson's 95 % interval of a proportion k / n (still informative at 0 % and 100 %, where the standard error is 0)."""
    if n <= 0:
        return (0.0, 1.0)
    z, p = 1.959964, k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(label: str, step: int, records: Mapping[int, dict], seeds: Sequence[int]) -> dict:
    """The result of a COMPLETE checkpoint (an episode for every seed), recomputed from its records only. `eval_sha` fingerprints the whole evaluation: two evaluations
    of the same weights with the same eval_sha are bit-identical episode by episode."""
    n = len(seeds)
    missing = [i for i in range(n) if i not in records]
    if missing:
        raise ValueError(f"checkpoint {label}: {len(missing)} episodes missing (first: {missing[:5]})")
    recs = [records[i] for i in range(n)]
    for i, r in enumerate(recs):
        if int(r["seed"]) != int(seeds[i]):
            raise ValueError(f"checkpoint {label}: episode {i} was run with seed {r['seed']}, the evaluation's is {seeds[i]}: the log belongs to another evaluation")
    successes = [bool(r["success"]) for r in recs]
    k = sum(successes)
    p = k / n
    attempts = sorted({int(r.get("attempt", 1)) for r in recs})
    decode = [r["decode"] for r in recs if r.get("decode")]
    won = [r["steps"] for r in recs if r["success"]]
    shas = hashlib.sha256("".join(str(r.get("traj_sha")) for r in recs).encode()).hexdigest()[:16]
    return {"label": label, "step": int(step), "n_episodes": n, "n_success": k, "success_rate": p, "success_se": math.sqrt(p * (1 - p) / n),
            "wilson95": list(wilson95(k, n)), "successes": successes, "seeds": [int(s) for s in seeds],
            "steps_mean": float(np.mean([r["steps"] for r in recs])), "steps_mean_success": float(np.mean(won)) if won else None,
            "attempts": attempts, "single_attempt": len(attempts) == 1, "gpus": sorted({str(r.get("gpu", "unknown")) for r in recs}),
            "wall_s_total": round(sum(r["wall_s"] for r in recs), 1), "eval_sha": shas,
            "decode_chunks": sum(d["chunks"] for d in decode) if decode else None, "decode_failed": sum(d["failed"] for d in decode) if decode else None,
            "episodes": [{"episode": r["episode"], "seed": r["seed"], "success": r["success"], "steps": r["steps"], "start_sha": r["start_sha"],
                          "traj_sha": r["traj_sha"], "attempt": r.get("attempt", 1)} for r in recs]}


# -- the protocol: what may not change between two runs on the same output folder ---------------------------------------------------------
PROTOCOL_FILE = "eval_protocol.json"
FROZEN_PROTOCOL = ("master_seed", "replan", "max_steps", "num_envs", "model_cfg", "task", "recipe")


def check_protocol(saved: Mapping[str, Any], now: Mapping[str, Any]) -> None:
    """Refuse a run whose protocol differs from the one the earlier results of the folder were made with (it would mix episodes that are not comparable).
    The number of episodes, the checkpoints, the number of workers and the videos are free to change."""
    diff = [f"{k}: {saved.get(k)!r} in the folder, {now.get(k)!r} now" for k in FROZEN_PROTOCOL
            if json.loads(json.dumps(saved.get(k), default=str)) != json.loads(json.dumps(now.get(k), default=str))]
    if diff:
        raise ValueError("this --output-dir holds results made with another protocol (use another --output-dir, or put the settings back):\n  " + "\n  ".join(diff))
