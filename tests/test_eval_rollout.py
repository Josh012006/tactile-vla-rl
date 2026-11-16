"""src/eval/rollout.py: the rollout loop (driven over a real socket against the fake simulator by a fake model, through the policy
interface of src/models/interface.py) and the indicator. The observation handed to openpi and the conversion of a chunk into env actions
are tested in tests/test_interface.py."""
import numpy as np
import pytest

from fakes import tag_of
from src.envs import protocol as P
from src.envs.client import VectorEnvClient
from src.eval import rollout
from src.models.interface import SEMANTICS, CallablePolicy, StaticPolicy
from test_client_server import Running, servers  # noqa: F401  (the `servers` fixture)

# What openpi's DroidInputs reads at inference (third_party/openpi/src/openpi/policies/droid_policy.py).
DROID_KEYS = {"observation/exterior_image_1_left", "observation/wrist_image_left", "observation/joint_position",
              "observation/gripper_position", "prompt"}


def test_the_indicator_tells_offsets_from_absolute_targets():
    q = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 2.0, 0.8], np.float32)
    assert rollout.looks_like(np.tile(0.02 * np.ones(8), (10, 1)), q)["looks_like"] == "offsets"
    assert rollout.looks_like(np.tile(q + 0.02, (10, 1)), q)["looks_like"] == "absolute"


class FakeModel:
    """Returns a chunk of `h` rows that drives joint 0 to 1.0 (the fake simulator's 'inserted'); counts its calls."""

    def __init__(self, semantics, h=10):
        self.semantics, self.h, self.calls = semantics, h, []

    def __call__(self, obs):
        self.calls.append(obs)
        q0 = float(obs["observation/joint_position"][0])
        chunk = np.zeros((self.h, 8), np.float32)
        chunk[:, 0] = (1.0 - q0) if self.semantics == "offsets_from_chunk_start_state" else 1.0
        return {"actions": chunk, "policy_timing": {"infer_ms": 12.5}}


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_rollout_succeeds_with_either_reading_when_the_yaml_matches_the_model(servers, semantics):  # noqa: F811
    s = servers(num_envs=2, max_steps=20, seed=0)
    model = FakeModel(semantics)
    cfg = {"output_semantics": semantics, "action_horizon": 10}
    with VectorEnvClient([s.endpoint]) as env:
        out = rollout.run_rollout(env, CallablePolicy(model, cfg), steps=12, replan=4)
    assert all(e["success"] and e["terminated"] for e in out["envs"]), out["envs"]
    assert out["all_done"] and out["semantics_used"] == semantics
    assert {frozenset(c) for c in model.calls} == {frozenset(DROID_KEYS)}
    assert all(p["infer_ms"] == 12.5 * p["batch"] and p["raw_shape"] == [10, 8] for p in out["plans"])
    assert out["plans"][0]["batch"] == 2                      # both environments planned in ONE call
    assert out["plans"][0]["looks_like"] in ("offsets", "absolute")


def test_rollout_calls_the_model_every_replan_steps_and_clips_replan_to_the_horizon(servers):  # noqa: F811
    s = servers(num_envs=1, max_steps=50, seed=0)

    class Idle:                                    # never inserts: holds joint 0 where it is
        calls = 0

        def __call__(self, obs):
            Idle.calls += 1
            chunk = np.zeros((6, 8), np.float32)
            chunk[:, 0] = float(obs["observation/joint_position"][0])
            return {"actions": chunk}

    with VectorEnvClient([s.endpoint]) as env:
        out = rollout.run_rollout(env, CallablePolicy(Idle(), {"output_semantics": "absolute_targets", "action_horizon": 6}), steps=24, replan=100)
    assert out["replan"] == 6 and Idle.calls == 4 and out["steps_run"] == 24      # 24 steps, a new chunk every 6
    assert not out["envs"][0]["success"] and out["envs"][0]["steps"] == 24
    assert out["plans"][0]["step"] == 0 and [p["step"] for p in out["plans"]] == [0, 6, 12, 18]


def test_rollout_is_the_same_episode_for_the_same_seed(servers):  # noqa: F811
    s = servers(num_envs=2, max_steps=20, seed=3)
    cfg = {"output_semantics": "absolute_targets", "action_horizon": 10}
    with VectorEnvClient([s.endpoint]) as env:
        a = rollout.run_rollout(env, CallablePolicy(FakeModel("absolute_targets"), cfg), steps=12, replan=4, seed=3)
        b = rollout.run_rollout(env, CallablePolicy(FakeModel("absolute_targets"), cfg), steps=12, replan=4, seed=3)
    assert [e["steps"] for e in a["envs"]] == [e["steps"] for e in b["envs"]]
    assert [p["q_plan"] for p in a["plans"]] == [p["q_plan"] for p in b["plans"]] and tag_of(3, 0) == 9


def test_rollout_records_the_progress_and_calls_on_step_after_the_reset_and_every_step(servers):  # noqa: F811
    s = servers(num_envs=2, max_steps=20, seed=0)
    seen = []
    with VectorEnvClient([s.endpoint]) as env:
        out = rollout.run_rollout(env, CallablePolicy(FakeModel("absolute_targets"), {"output_semantics": "absolute_targets", "action_horizon": 10}),
                                steps=12, replan=4, on_step=lambda t, res, plans: seen.append((t, list(plans))))
    assert seen[0] == (0, [0, 0])                                         # after the reset, before any plan
    assert [t for t, _ in seen] == list(range(len(seen)))                 # then once per step
    assert seen[-1][1][0] >= 1                                            # the plan counter follows
    for e in out["envs"]:
        g = e["progress"]
        assert g["samples"] == e["steps"] + 1 == len(e["progress_series"])        # the state after the reset + one per step
        assert set(e["progress_series"][0]) == {"t", "xy_mm", "z_mm", "force_n"}


def test_a_static_policy_holds_the_arm_through_the_same_rollout(servers):  # noqa: F811
    """The algorithms' dry run: no model, the arm is told to stay where it is (the fake simulator's joints 1 and 2 carry its clock and
    tag, so only the chunks are compared with the state, not the state with itself)."""
    s = servers(num_envs=2, max_steps=20, seed=0)
    with VectorEnvClient([s.endpoint]) as env:
        out = rollout.run_rollout(env, StaticPolicy(action_horizon=6), steps=12, replan=6)
    assert [e["steps"] for e in out["envs"]] == [12, 12] and not any(e["success"] for e in out["envs"])
    assert out["plans"] and all(np.allclose(p["raw_first"][:7], p["q_plan"]) for p in out["plans"])


class Seeded:
    """Inserts (joint 0 -> 1.0) and records, per call, the rows it was asked for, their seeds and the batch it was given."""

    def __init__(self, h=10):
        self.h, self.calls = h, []
        self.info = type("I", (), {"output_semantics": "absolute_targets", "action_horizon": h})()
        self.last_valid = None

    def plan(self, obs, prompt, num_samples=1, *, row_seeds=None, pad_to=None):
        from src.models.interface import Plan
        n = len(obs[P.JOINT_POSITION])
        self.calls.append((n, row_seeds, pad_to))
        raw = np.zeros((n, 1, self.h, 8), np.float32)
        raw[..., 0] = 1.0
        return Plan(targets=raw.copy(), raw=raw, infer_ms=1.0)


def test_inactive_environments_are_held_and_never_planned_for_and_the_others_do_not_notice(servers):  # noqa: F811
    s = servers(num_envs=3, max_steps=20, seed=5)
    with VectorEnvClient([s.endpoint]) as env:
        everyone = rollout.run_rollout(env, Seeded(), steps=12, replan=4, seed=5)
        policy = Seeded()
        only = rollout.run_rollout(env, policy, steps=12, replan=4, seed=5, active=[False, True, False], pad_to=8, row_seeds=lambda e, k: 100 * e + k)
    assert [n for n, _, _ in policy.calls] == [1] and all(pad == 8 for _, _, pad in policy.calls)     # one row: the one that plays
    assert policy.calls[0][1] == [100]                                                                   # environment 1, its first plan
    assert only["envs"][1]["success"] and only["all_done"]
    assert only["envs"][1]["steps"] == everyone["envs"][1]["steps"]                                      # the same start: the held environments change nothing for it
    assert only["envs"][0]["steps"] == 0 or not only["envs"][0]["success"]                               # a held one does not succeed by itself


def test_on_env_done_is_called_once_per_episode_when_it_ends_with_its_own_summary(servers):  # noqa: F811
    s = servers(num_envs=3, max_steps=20, seed=5)
    done = []
    with VectorEnvClient([s.endpoint]) as env:
        out = rollout.run_rollout(env, Seeded(), steps=20, replan=4, seed=5, on_env_done=lambda e, summ: done.append((e, summ["steps"], summ["success"])))
    assert sorted(e for e, _, _ in done) == [0, 1, 2]
    assert {(e, st, ok) for e, st, ok in done} == {(e["env"], e["steps"], e["success"]) for e in out["envs"]}


def test_the_last_commanded_target_is_given_only_to_a_policy_that_asks_for_it(servers):  # noqa: F811
    """The force proxy (src/touch/features.py) reads `client/previous_joint_target`: zero residual (the joints themselves) at the first plan, then the target of the last step sent.
    A policy without a proxy component (every SFT evaluation) gets the observation exactly as the server sent it."""
    from src.touch.features import TouchSpec
    s = servers(num_envs=1, max_steps=50, seed=0)
    keys = (P.PREVIOUS_TARGET, P.JOINT_POSITION)

    def model(obs):
        chunk = np.zeros((6, 8), np.float32)
        chunk[:, 0] = float(obs[P.JOINT_POSITION][0]) + 0.01
        return {"actions": chunk}

    def run(touch):
        seen = []

        class Recording(CallablePolicy):
            def plan(self, obs, prompt, num_samples=1, **kwargs):
                seen.append({k: np.array(v) for k, v in obs.items() if k in keys})
                return super().plan(obs, prompt, num_samples, **kwargs)

        policy = Recording(model, {"output_semantics": "absolute_targets", "action_horizon": 6})
        policy.touch = touch
        with VectorEnvClient([s.endpoint]) as env:
            rollout.run_rollout(env, policy, steps=12, replan=6)
        return policy, seen

    plain, seen_plain = run({"action_head": TouchSpec(use_tactile=False)})
    assert not plain.uses_previous_target and len(seen_plain) == 2 and all(P.PREVIOUS_TARGET not in o for o in seen_plain)
    asking, (first, second) = run({"action_head": TouchSpec(use_force=True, force_mode="proxy")})
    assert asking.uses_previous_target
    assert first[P.PREVIOUS_TARGET].shape == (1, 7) and np.array_equal(first[P.PREVIOUS_TARGET][0], np.asarray(first[P.JOINT_POSITION][0], np.float32)[:7])
    assert not np.array_equal(second[P.PREVIOUS_TARGET], np.asarray(second[P.JOINT_POSITION], np.float32)[:, :7])
    assert np.isclose(second[P.PREVIOUS_TARGET][0, 0], first[P.JOINT_POSITION][0][0] + 0.01, atol=1e-5)             # the last target of the first chunk: joint 0 at plan time + 0.01
