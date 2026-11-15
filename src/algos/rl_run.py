"""The run of an RL experiment: warm-up, collection rounds with the critic, its updates, the evaluations, and the resume (docs/algos/rl.md section 3 and 7). Best-of-N is complete here (the VLA stays frozen);
PA-RL's distillation and its cache refresh are the next step and refused for now.

    run = RLRun(cfg, policy, run_dir, open_env, log, max_steps=150)       # open_env(num_envs, label) -> a context manager giving a VectorEnvClient (a server started and stopped by it)
    summary = run.run()                                                   # a stage at a time; `run.stop` (threading.Event) ends it after the current stage

A STAGE is the unit of progress and of resume. Stage 0 is the warm-up: `warmup_episodes` episodes of the base VLA alone (its first candidate is executed), then `warmup_critic_steps` critic updates on them. Stage r >= 1 is
collection round r: `envs_per_batch` episodes with the critic (the rollout selection of the method), then `utd` x (their environment steps) critic updates. The rounds are numbered so that no start is played twice:
the warm-up uses the rounds 0 .. W - 1 of the server's reset sequence (`reset(seed, round)`), the online round r the round W + r - 1. After each stage the run writes, in this order,
    episodes/stage_<r>.npz     the stage's episodes with the VLA's candidates of their states (the replay buffer and the cache are rebuilt from these files, in order),
    resume/critic.msgpack      the critic, its delayed copy and Adam's state,
    resume/state.json          the stage, the update count, the random generators: the file that says the stage is finished.
A run that is cut resumes after the last finished stage (`--resume`): the stage in progress is played again from its start (its episodes are the ones that were not saved).
The evaluations (`eval_rounds`: the warm-up, every `eval_every_rounds`, the last) are the 100 fixed episodes of the SFT evaluations (src/eval/episodes.py) with the critic's argmax over the VLA's candidates, on a FRESH
server (same starts as every SFT evaluation on the same GPU); each is eval/results/<round>.json (the format of the SFT evaluations: the curve and the paired comparisons read it) and the critic of that round is kept in
critics/round_<r>.msgpack. Only one env server runs at a time (Isaac Sim, the VLA and the critic share one GPU): the collection server is stopped for an evaluation and started again after it.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, ContextManager, Dict, List, Optional

import flax.serialization
import jax
import numpy as np

from src.algos import rl_critic as rc
from src.algos.rl_buffer import Episode, ReplayBuffer
from src.algos.rl_candidates import CandidateCache
from src.algos.rl_config import RLConfig
from src.algos.rl_critic_update import CriticTrainer, run_updates
from src.algos.rl_rollout import Collected, RLActor, collect_round, evaluate, store_round
from src.eval import episodes as E
from src.models.interface import Candidates

STATE_FILE = "resume/state.json"
CRITIC_FILE = "resume/critic.msgpack"
CRITIC_LOG_EVERY = 500                  # updates between two `critic` lines of metrics.jsonl (not sent to wandb: the round line carries the mean)
OpenEnv = Callable[[int, str], ContextManager]


# -- files -------------------------------------------------------------------------------------------------------------------------------------
def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_episodes(path: Path, collected: List[Collected]) -> None:
    """The episodes of a stage and the VLA's candidates of their states (T per episode, T + 1 for a time-out), in one .npz."""
    eps = [c.episode for c in collected]
    arrays = {"exterior": np.concatenate([e.exterior for e in eps]), "wrist": np.concatenate([e.wrist for e in eps]), "joints": np.concatenate([e.joints for e in eps]),
              "gripper": np.concatenate([e.gripper for e in eps]), "actions": np.concatenate([e.actions for e in eps]),
              "plans": np.array([e.plans for e in eps]), "success": np.array([e.success for e in eps]), "prompts": np.array([e.prompt for e in eps]),
              "env": np.array([c.env for c in collected]), "round_id": np.array([c.round_id for c in collected]), "steps": np.array([c.steps for c in collected]),
              "n_candidates": np.array([len(c.candidates.normalized) for c in collected]),
              "cand_normalized": np.concatenate([c.candidates.normalized for c in collected]), "cand_states": np.concatenate([c.candidates.states for c in collected]),
              "cand_targets": np.concatenate([c.candidates.targets for c in collected])}
    if eps[0].touch is not None:
        arrays["touch"] = np.concatenate([e.touch for e in eps])
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_episodes(path: Path) -> List[Collected]:
    with np.load(path) as z:
        plans, n_cand = z["plans"], z["n_candidates"]
        rows, acts, cands = np.concatenate([[0], np.cumsum(plans + 1)]), np.concatenate([[0], np.cumsum(plans)]), np.concatenate([[0], np.cumsum(n_cand)])
        out = []
        for i in range(len(plans)):
            r, a, c = slice(rows[i], rows[i + 1]), slice(acts[i], acts[i + 1]), slice(cands[i], cands[i + 1])
            episode = Episode(exterior=z["exterior"][r], wrist=z["wrist"][r], joints=z["joints"][r], gripper=z["gripper"][r], actions=z["actions"][a], success=bool(z["success"][i]),
                              prompt=str(z["prompts"][i]), touch=z["touch"][r] if "touch" in z else None)
            out.append(Collected(episode=episode, candidates=Candidates(normalized=z["cand_normalized"][c], states=z["cand_states"][c], targets=z["cand_targets"][c], infer_ms=0.0),
                                 env=int(z["env"][i]), round_id=int(z["round_id"][i]), steps=int(z["steps"][i])))
    return out


def critic_bytes(critic, trainer_opt_state=None) -> bytes:
    state = {"params": critic.params, "target_params": critic.target_params}
    if trainer_opt_state is not None:
        state["opt_state"] = trainer_opt_state
    return flax.serialization.to_bytes(state)


class RLRun:
    def __init__(self, cfg: RLConfig, policy, run_dir: Path, open_env: OpenEnv, log, max_steps: int, stop: Optional[threading.Event] = None, eval_seed: int = E.DEFAULT_SEED,
                 title: Optional[str] = None):
        if cfg.distills:
            raise NotImplementedError("PA-RL's distillation and cache refresh are the next step: for now only method: best_of_n runs")
        if cfg.warmup_episodes < 1:
            raise ValueError("warmup_episodes must be at least one batch: the critic's batches mix warm-up and online data and it is trained on the warm-up first")
        self.cfg, self.policy, self.run_dir, self.open_env, self.log = cfg, policy, Path(run_dir), open_env, log
        self.max_steps, self.stop, self.eval_seed, self.title = int(max_steps), stop or threading.Event(), eval_seed, title
        horizon = policy.info.action_horizon
        self.replan = max(1, min(cfg.replan_steps or horizon // 2, horizon))
        self.trainer = CriticTrainer(rc.CriticNet.from_config(cfg), cfg, horizon)
        self.actor = RLActor(policy, cfg, self.trainer.optimiser)
        plans = -(-self.max_steps // self.replan)
        self.buffer = ReplayBuffer((cfg.warmup_episodes + cfg.total_episodes) * (plans + 1), horizon, cfg.gamma, cfg.step_reward, touch_dim=cfg.critic_touch.dim)
        self.cache: Optional[CandidateCache] = None                       # built with the first candidates (their state dimension)
        self.rng = np.random.default_rng(cfg.seed)                        # the critic's batches
        self.stage = -1                                                   # the last finished stage (-1: none)
        self._env = None
        self._env_cm = None
        self.seeds = E.make_seeds(eval_seed, cfg.eval_episodes)
        self.eval_dir = self.run_dir / "eval"

    # -- stage of the warm-up and of a round ------------------------------------------------------------------------------------------------------
    @property
    def warmup_rounds(self) -> int:
        return self.cfg.warmup_episodes // self.cfg.envs_per_batch

    def _collection_env(self):
        if self._env is None:
            self._env_cm = self.open_env(self.cfg.envs_per_batch, "collect")
            self._env = self._env_cm.__enter__()
        return self._env

    def _close_collection(self) -> None:
        if self._env_cm is not None:
            cm, self._env_cm, self._env = self._env_cm, None, None
            cm.__exit__(None, None, None)

    def _store(self, collected: List[Collected], warmup: bool) -> None:
        if self.cache is None:
            self.cache = CandidateCache(self.cfg.num_candidates, self.policy.info.action_horizon, collected[0].candidates.states.shape[1], capacity=self.buffer.capacity)
        store_round(self.buffer, self.cache, collected, warmup=warmup, version=0)             # Best-of-N never refreshes the cache: the VLA never changes

    def _update_critic(self, steps: int, stage: int) -> Dict[str, float]:
        def log_window(update: int, means: Dict[str, float]) -> None:
            self.log.write(event="critic", stage=stage, update=update, **means)

        return run_updates(self.trainer, self.buffer, self.cache, self.rng, steps=steps, log_every=CRITIC_LOG_EVERY, log=log_window)

    def _warmup(self) -> None:
        cfg, started = self.cfg, time.monotonic()
        collected: List[Collected] = []
        for w in range(self.warmup_rounds):
            part, stats = collect_round(self._collection_env(), self.actor, mode="vla", seed=cfg.seed, round_id=w, replan=self.replan, max_steps=self.max_steps, pad_to=cfg.envs_per_batch)
            collected += part
            self.log.write(event="warmup", step=0, round_id=w, **stats)
        self._store(collected, warmup=True)
        updates = self._update_critic(cfg.warmup_critic_steps, 0)
        self.log.write(event="warmup", step=0, summary=True, episodes=len(collected), success_rate=float(np.mean([c.episode.success for c in collected])), env_steps=sum(c.steps for c in collected),
                       buffer_rows=len(self.buffer), critic_updates=cfg.warmup_critic_steps, critic=updates, wall_s=round(time.monotonic() - started, 1))
        self._commit(0, collected)

    def _round(self, r: int) -> None:
        cfg, started = self.cfg, time.monotonic()
        round_id = self.warmup_rounds + r - 1
        collected, stats = collect_round(self._collection_env(), self.actor, mode=cfg.rollout_selection, params=self.trainer.critic.params, seed=cfg.seed, round_id=round_id,
                                         key=jax.random.fold_in(jax.random.key(cfg.seed + 1), r), replan=self.replan, max_steps=self.max_steps, pad_to=cfg.envs_per_batch)
        self._store(collected, warmup=False)
        n_updates = cfg.utd * stats["env_steps"]
        updates = self._update_critic(n_updates, r)
        online = [e for e in self.buffer.episodes if not e["warmup"]]
        self.log.write(event="round", step=r, round_id=round_id, **stats, critic_updates=n_updates, critic=updates, buffer_rows=len(self.buffer), online_episodes=len(online),
                       online_success_rate=float(np.mean([e["success"] for e in online])), round_wall_s=round(time.monotonic() - started, 1))
        self._commit(r, collected)

    # -- the end of a stage and the resume ---------------------------------------------------------------------------------------------------------
    def _commit(self, stage: int, collected: List[Collected]) -> None:
        save_episodes(self.run_dir / "episodes" / f"stage_{stage:03d}.npz", collected)
        _atomic(self.run_dir / CRITIC_FILE, critic_bytes(self.trainer.critic, self.trainer.opt_state))
        state = {"stage": stage, "updates": self.trainer.updates, "rng": self.rng.bit_generator.state, "critic_key": jax.random.key_data(self.trainer.key).tolist()}
        _atomic(self.run_dir / STATE_FILE, json.dumps(state).encode())          # last: the stage is finished only when this file says so
        self.stage = stage

    def resume(self) -> bool:
        """Rebuild the buffer, the cache, the critic and the random generators from the files of the last finished stage. False when there is nothing to resume."""
        path = self.run_dir / STATE_FILE
        if self.stage >= 0:                                              # already rebuilt (or already run): never twice
            return True
        if not path.exists():
            return False
        state = json.loads(path.read_text())
        for stage in range(state["stage"] + 1):
            self._store(load_episodes(self.run_dir / "episodes" / f"stage_{stage:03d}.npz"), warmup=stage == 0)
        template = {"params": self.trainer.critic.params, "target_params": self.trainer.critic.target_params, "opt_state": self.trainer.opt_state}
        loaded = flax.serialization.from_bytes(template, (self.run_dir / CRITIC_FILE).read_bytes())
        self.trainer.critic = rc.CriticState(params=loaded["params"], target_params=loaded["target_params"])
        self.trainer.opt_state, self.trainer.updates = loaded["opt_state"], int(state["updates"])
        self.trainer.key = jax.random.wrap_key_data(np.array(state["critic_key"], dtype=np.uint32))
        self.rng.bit_generator.state = state["rng"]
        self.stage = int(state["stage"])
        return True

    # -- evaluation ---------------------------------------------------------------------------------------------------------------------------------
    def result_path(self, r: int) -> Path:
        return self.eval_dir / "results" / f"{r}.json"

    def _evaluate(self, r: int) -> dict:
        """The fixed episodes with the critic's argmax over the VLA's candidates, on a fresh server (the collection server is stopped: one server at a time on the GPU)."""
        cfg = self.cfg
        self._close_collection()
        _atomic(self.run_dir / "critics" / f"round_{r:03d}.msgpack", critic_bytes(self.trainer.critic))
        started = time.monotonic()
        with self.open_env(cfg.eval_episodes, f"eval_{r}") as env:
            result = evaluate(self.actor, self.trainer.critic.params, env, self.seeds, replan=self.replan, max_steps=self.max_steps, master_seed=self.eval_seed, label=str(r), step=r,
                              mode=cfg.eval_selection)
        result.update(round=r, critic_updates=self.trainer.updates, buffer_rows=len(self.buffer))
        _atomic(self.result_path(r), json.dumps(result, default=float, indent=1).encode())
        self.log.write(event="eval", step=r, **{k: v for k, v in result.items() if k not in ("episodes", "seeds", "successes", "label", "step")}, eval_wall_s=round(time.monotonic() - started, 1))
        try:
            from src.eval.curve import write_curve
            write_curve(self.eval_dir / "results", self.eval_dir, self.title, name=self.run_dir.name, x_label="collection round (0 = after the warm-up)")
        except Exception as error:                                       # noqa: BLE001  a plot never stops the run
            print(f"the curve could not be drawn: {error!r}", flush=True)
        return result

    # -- the run ------------------------------------------------------------------------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        resumed = self.resume()
        self.log.write(event="start", variant=cfg.variant, rounds=cfg.rounds, eval_rounds=list(cfg.eval_rounds), resumed_after_stage=self.stage if resumed else None,
                       critic_params=rc.num_parameters(self.trainer.critic.params), buffer_capacity=self.buffer.capacity, replan=self.replan, max_steps=self.max_steps)
        stopped = False
        try:
            for r in range(cfg.rounds + 1):
                if r > self.stage:
                    if self.stop.is_set():
                        stopped = True
                        break
                    self._warmup() if r == 0 else self._round(r)
                if r in cfg.eval_rounds and not self.result_path(r).exists():
                    if self.stop.is_set():
                        stopped = True
                        break
                    self._evaluate(r)
        finally:
            self._close_collection()
        summary = {"stage": self.stage, "stopped": stopped, "rounds": cfg.rounds, "updates": self.trainer.updates, "evaluated": sorted(int(p.stem) for p in (self.eval_dir / "results").glob("*.json"))}
        self.log.write(event="stopped" if stopped else "done", **summary)
        return summary
