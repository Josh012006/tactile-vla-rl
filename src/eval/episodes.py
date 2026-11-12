"""Evaluation of a policy on FIXED episodes, K environments in ONE server, comparable between checkpoints by construction and resumable at the episode.

What makes two checkpoints comparable, and what this file does about each part:

  * the START of an episode: a server holds K environments (the protocol's `num_envs`) that start together, with FORGE's real reset, seeded with the master seed (default 42): the start
    state of environment e is a function of (master seed, K, e) alone (docs/envs/isaaclab_env.md N2). Episode i is environment i % K of batch i // K: batch 0 is `reset(seed=master, round=0)`
    from a FRESH server (every checkpoint gets one), so with at most K episodes every checkpoint starts from the very same K states, bit for bit on the same GPU model, and no episode's start
    depends on what an earlier episode did (the history dependence of N1 only exists from batch 1 on: more than K episodes). The environments that are not needed (K larger than the number of
    episodes, or episodes already done) are held where they are, so the starts of the others do not depend on which ones play.
  * the POLICY's randomness: episode i has a seed (`episode_seeds.json`, drawn ONCE from the master seed the way expo-ft does: `np.random.default_rng(master).integers(0, 2**31 - 1, n)`, so a
    longer evaluation extends a shorter one). The flow noise of pi0.5 for its k-th chunk of episode i is a function of (that seed, k) alone (`plan_seed`, `VLAPolicy.plan(row_seeds=...)`): not
    of the other episodes in the batch, of how many are still running, or of what ran before. pi0-FAST decodes greedily: no randomness at all. The model is always called with K rows
    (`pad_to`): one shape, whatever the number of episodes still running.
  * what is NOT reproducible: the cameras. Two fresh servers render slightly different images from the very first frame (about 0.5 / 255 on average, sensor-level noise;
    docs/eval/sft_eval.md section 3), so two evaluations of the same weights give different trajectories. This is noise of the same kind for every checkpoint: the comparison is statistical
    (same starts, same episode seeds, standard error), not bit for bit. Each record carries `start_sha` (the first observation's joints, gripper and peg-socket offset: identical for the
    same episode on any fresh server) and `traj_sha` (the same readings at every step, which the image noise makes differ from one evaluation to the next).

Resume. Each finished episode is one line of `results/<label>.jsonl` (flushed and fsynced the moment its episode ends, even while the others of its batch still run). A restart reads the
file, runs the batch again from a fresh server with only the missing episodes playing, and recomputes everything from the file: the numbers are those of an evaluation that was never cut.
A line cut by a kill is dropped. `attempt` is recorded in every line and `single_attempt` in the summary.

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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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


def plan_seed(episode_seed: int, k: int) -> int:
    """The seed of the sampling noise of the k-th chunk (0-based) of the episode with `episode_seed`: a function of the two alone (32 bits, what `VLAPolicy.plan(row_seeds=...)` takes)."""
    return int(np.random.SeedSequence([int(episode_seed), int(k)]).generate_state(1, np.uint32)[0])


# -- one batch of episodes --------------------------------------------------------------------------------------------------------------
class TrajectoryHash:
    """Running sha256 of the readings of an episode (HASHED_KEYS, environment `env_index`). `start` is the hash of the first observation alone."""

    def __init__(self, env_index: int = 0):
        self._h = hashlib.sha256()
        self.start: Optional[str] = None
        self.env_index = int(env_index)

    def update(self, obs: Mapping[str, np.ndarray]) -> None:
        step = hashlib.sha256()
        for key in HASHED_KEYS:
            if key in obs:
                step.update(np.ascontiguousarray(np.asarray(obs[key])[self.env_index]).tobytes())
        digest = step.digest()
        if self.start is None:
            self.start = digest.hex()[:16]
        self._h.update(digest)

    def hexdigest(self) -> str:
        return self._h.hexdigest()[:16]


def run_batch(env, policy, episodes: Mapping[int, Tuple[int, int]], *, master_seed: int, batch: int, replan: int, max_steps: int,
              on_episode: Callable[[dict], None], stop: Optional[threading.Event] = None, pad_to: Optional[int] = None,
              videos: Optional[Mapping[int, Path]] = None, fps: int = 15) -> dict:
    """One batch: the real reset of round `batch` of `master_seed` in every environment of `env` (a VectorEnvClient with K environments), then the episodes in `episodes` -- {environment index:
    (episode index, episode seed)} -- played together, the other environments held. `on_episode(record)` is called with the record of each episode the moment it ends (the JSONL line,
    minus what the caller adds). The policy's noise for the k-th chunk of an episode is `plan_seed(its seed, k)` (see the top of the file) and the model is called with `pad_to` rows
    (default: all K). `videos`: {environment index: path} of the episodes to film. Raises EvalAborted when `stop` is set (what has ended has been reported; the rest is dropped) and
    RuntimeError when an episode did not end within `max_steps`. Returns {"wall_s", "steps_run", "plans"} of the batch."""
    k = env.num_envs
    bad = [e for e in episodes if not 0 <= int(e) < k]
    if not episodes or bad:
        raise ValueError(f"episodes must name environments of this server (0..{k - 1}), got {sorted(episodes)}")
    seeds = {int(e): int(seed) for e, (_, seed) in episodes.items()}
    hashers = {int(e): TrajectoryHash(int(e)) for e in episodes}
    finished = np.zeros(k, bool)
    films: Dict[int, list] = {int(e): [] for e in (videos or {}) if int(e) in hashers}
    if pad_to is None:
        pad_to = k
    t0 = time.monotonic()

    def on_step(t, res, n_plans):
        if stop is not None and stop.is_set():
            raise EvalAborted(f"batch {batch} stopped at step {t}")
        for e, h in hashers.items():
            if not finished[e]:
                h.update(res.obs)
        for e, frames in films.items():
            if not finished[e]:
                from src.eval import video
                from src.eval.progress import progress_sample
                frames.append(video.compose_frame(res.obs, e, t, n_plans[e], progress_sample(res.obs, e), bool(res.success[e])))
        finished[:] = finished | res.done

    done_records: Dict[int, dict] = {}

    def on_env_done(e, summ):
        if e not in hashers:
            return
        episode, seed = episodes[e]
        walls = summ["plan_wall_ms"]
        record = {"episode": int(episode), "seed": int(seed), "batch": int(batch), "env": int(e), "success": bool(summ["success"]), "terminated": bool(summ["terminated"]),
                  "truncated": bool(summ["truncated"]), "steps": int(summ["steps"]), "start_sha": hashers[e].start, "traj_sha": hashers[e].hexdigest(), "plans": len(walls),
                  "plan_ms_median": float(np.median(walls)) if walls else None, "tracking_err_max_rad": summ["tracking_err_max_rad"],
                  "progress": {key: v for key, v in summ["progress"].items() if key != "success"}, "wall_s": round(time.monotonic() - t0, 2),
                  "decode": summ.get("decode"), "video": None}
        if e in films and films[e]:
            from src.eval import video
            Path(videos[e]).parent.mkdir(parents=True, exist_ok=True)
            record["video"] = video.write_video(films[e], Path(videos[e]), fps)
        done_records[e] = record
        on_episode(record)

    active = np.zeros(k, bool)
    active[list(hashers)] = True
    out = run_rollout(env, policy, steps=max_steps, replan=replan, seed=int(master_seed), round_id=int(batch), active=active,
                      row_seeds=lambda e, j: plan_seed(seeds.get(e, 0), j), pad_to=pad_to, on_step=on_step, on_env_done=on_env_done)
    if not out["all_done"]:
        missing = sorted(set(hashers) - set(done_records))
        raise RuntimeError(f"episodes {[episodes[e][0] for e in missing]} (batch {batch}) did not end within {max_steps} steps: is the server's max_steps_per_episode larger than the protocol's?")
    return {"wall_s": round(time.monotonic() - t0, 2), "steps_run": out["steps_run"], "plans": len(out["plans"]), "decode": decode_totals(done_records.values())}


def decode_totals(records) -> Optional[dict]:
    """pi0-FAST: the chunks planned for these episodes and how many could not be decoded (None for a model without token strings)."""
    parts = [r["decode"] for r in records if r.get("decode")]
    if not parts:
        return None
    chunks, failed = sum(d["chunks"] for d in parts), sum(d["failed"] for d in parts)
    return {"chunks": chunks, "failed": failed, "failed_rate": failed / chunks if chunks else 0.0}


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


def wall_total(recs: Sequence[Mapping[str, Any]]) -> float:
    """Seconds of simulation of these records. The episodes of a batch run together and each one's `wall_s` is the time from the batch's start to its end: a batch lasted as long as its
    last episode. A record without `batch` (one environment per server) is a batch of its own."""
    longest: Dict[Any, float] = {}
    for r in recs:
        key = r.get("batch", ("episode", r["episode"]))
        longest[key] = max(longest.get(key, 0.0), float(r["wall_s"]))
    return sum(longest.values())


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
            "wall_s_total": round(wall_total(recs), 1), "eval_sha": shas,
            "decode_chunks": sum(d["chunks"] for d in decode) if decode else None, "decode_failed": sum(d["failed"] for d in decode) if decode else None,
            "episodes": [{"episode": r["episode"], "seed": r["seed"], "success": r["success"], "steps": r["steps"], "start_sha": r["start_sha"],
                          "traj_sha": r["traj_sha"], "attempt": r.get("attempt", 1)} for r in recs]}


# -- the protocol: what may not change between two runs on the same output folder ---------------------------------------------------------
PROTOCOL_FILE = "eval_protocol.json"
FROZEN_PROTOCOL = ("master_seed", "replan", "max_steps", "num_envs", "model_batch", "model_cfg", "norm_stats_sha", "task", "recipe")


def check_protocol(saved: Mapping[str, Any], now: Mapping[str, Any]) -> None:
    """Refuse a run whose protocol differs from the one the earlier results of the folder were made with (it would mix episodes that are not comparable).
    The number of episodes, the checkpoints, the number of workers and the videos are free to change."""
    diff = [f"{k}: {saved.get(k)!r} in the folder, {now.get(k)!r} now" for k in FROZEN_PROTOCOL
            if json.loads(json.dumps(saved.get(k), default=str)) != json.loads(json.dumps(now.get(k), default=str))]
    if diff:
        raise ValueError("this --output-dir holds results made with another protocol (use another --output-dir, or put the settings back):\n  " + "\n  ".join(diff))
