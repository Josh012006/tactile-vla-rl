"""src/algos/rl_candidates.py (the sampler, the cache, the refresh) around a fake policy, and the two new methods of OpenPIPolicy (sample_candidates, normalized_to_targets) on a stub: no GPU, no JAX.
The check of those two methods against the real transforms is in tests/test_openpi_policy.py (needs the `vla` group)."""
import types

import numpy as np
import pytest

from src.algos import rl_candidates as rcand
from src.algos.rl_config import RLConfig
from src.envs import protocol as P
from src.models.interface import Candidates, Plan
from src.models.openpi_policy import OpenPIPolicy

H, DS = 4, 6


def obs_of(n, offset=0):
    """Environment i's first joint is its own number, so a candidate says where it came from."""
    q = np.zeros((n, 7), np.float32)
    q[:, 0] = np.arange(n) + offset
    return {P.JOINT_POSITION: q, P.GRIPPER_POSITION: np.zeros((n, 1), np.float32), P.EXTERIOR_IMAGE: np.zeros((n, 2, 2, 3), np.uint8), P.WRIST_IMAGE: np.zeros((n, 2, 2, 3), np.uint8)}


def fake_candidates(joints, seeds, k):
    """Candidate s of an environment = (its joint 0, its seed, s) written in the first three numbers: deterministic and tells the row, the seed and the sample."""
    n = len(joints)
    norm = np.zeros((n, k, H, 8), np.float32)
    for i in range(n):
        for s in range(k):
            norm[i, s, 0, :3] = (joints[i], seeds[i] % 1000, s)
    return Candidates(normalized=norm, states=np.repeat(np.asarray(joints, np.float32)[:, None], DS, axis=1), targets=norm.copy(), infer_ms=1.0)


class FakePolicy:
    def __init__(self):
        self.calls = []

    def sample_candidates(self, obs, prompt, num_samples, *, row_seeds=None, pad_to=None):
        joints = obs[P.JOINT_POSITION][:, 0]
        self.calls.append({"n": len(joints), "prompt": list(prompt) if not isinstance(prompt, str) else prompt, "k": num_samples, "seeds": list(row_seeds), "pad_to": pad_to})
        return fake_candidates(joints, row_seeds, num_samples)


def cfg(**kw):
    return RLConfig(**{"num_candidates": 3, "keep_top": 2, "rows_per_sample_call": 4, **kw})


def test_the_sampler_calls_the_model_in_fixed_slices_and_keeps_the_order_and_each_rows_seed():
    policy = FakePolicy()
    c = rcand.CandidateSampler(policy, cfg()).sample(obs_of(10), "insert", list(range(100, 110)))
    assert [x["n"] for x in policy.calls] == [4, 4, 2] and {x["pad_to"] for x in policy.calls} == {4}        # one shape for every call: one compilation
    assert c.normalized.shape == (10, 3, H, 8) and c.states.shape == (10, DS) and c.infer_ms == 3.0
    assert np.array_equal(c.normalized[:, 0, 0, 0], np.arange(10)) and np.array_equal(c.normalized[:, 0, 0, 1], np.arange(100, 110) % 1000)     # row i keeps its state and ITS seed
    assert [x["seeds"] for x in policy.calls][2] == [108, 109] and all(x["k"] == 3 for x in policy.calls)


def test_prompts_one_per_environment_follow_their_slice_and_wrong_lengths_fail():
    policy = FakePolicy()
    sampler = rcand.CandidateSampler(policy, cfg())
    sampler.sample(obs_of(6), [f"p{i}" for i in range(6)], list(range(6)))
    assert [x["prompt"] for x in policy.calls] == [["p0", "p1", "p2", "p3"], ["p4", "p5"]]
    with pytest.raises(ValueError, match="row_seeds"):
        sampler.sample(obs_of(3), "p", [1, 2])
    with pytest.raises(ValueError, match="prompts"):
        sampler.sample(obs_of(3), ["a"], [1, 2, 3])


def test_the_cache_stores_by_id_grows_and_refuses_what_is_not_there():
    cache = rcand.CandidateCache(3, H, DS, capacity=2)
    first = fake_candidates([10.0, 11.0], [1, 2], 3)
    cache.put([0, 1], first, version=0)
    cache.put([5], fake_candidates([15.0], [3], 3), version=0)                                       # beyond the capacity: it grows, ids 2-4 stay empty
    assert cache.size == 6 and cache.capacity >= 6 and cache.has([0, 1, 2, 5]).tolist() == [True, True, False, True]
    norm = cache.get([5, 0])
    assert norm[0, 0, 0, 0] == 15.0 and norm[1, 0, 0, 0] == 10.0 and cache.states_of([1])[0, 0] == 11.0
    with pytest.raises(KeyError, match="no candidates"):
        cache.get([0, 3])
    assert cache.versions_of([0, 5]).tolist() == [0, 0] and cache.nbytes > 0
    for bad in ([0, 0], [-1, 1], [0]):
        with pytest.raises(ValueError, match="ids must be"):
            cache.put(bad, first, 0)
    with pytest.raises(ValueError, match="do not fit"):
        cache.put([0, 1], fake_candidates([1.0, 2.0], [1, 2], 5), 0)


def test_stale_ids_are_the_old_versions_and_the_holes_and_a_put_replaces():
    cache = rcand.CandidateCache(3, H, DS)
    cache.put([0, 1, 2], fake_candidates([0.0, 1.0, 2.0], [1, 2, 3], 3), version=0)
    cache.put([4], fake_candidates([4.0], [4], 3), version=0)
    assert cache.stale_ids(0).tolist() == [3]                                                         # only the hole
    assert cache.stale_ids(1).tolist() == [0, 1, 2, 3, 4]                                               # a new VLA: everything is old
    cache.put([1], fake_candidates([1.0], [9], 3), version=1)
    assert cache.stale_ids(1).tolist() == [0, 2, 3, 4] and cache.get([1])[0, 0, 0, 1] == 9


def test_refresh_recomputes_the_asked_states_with_seeds_of_their_id_and_version_and_reads_the_buffer_in_chunks():
    policy, reads = FakePolicy(), []
    sampler = rcand.CandidateSampler(policy, cfg())
    cache = rcand.CandidateCache(3, H, DS)

    def source(ids):
        reads.append(list(ids))
        return obs_of(len(ids), offset=int(ids[0])), "insert"

    ticks = iter(range(100))
    seen = []
    out = rcand.refresh(cache, sampler, source, [9, 8, 7, 6, 5, 4], version=2, seed=11, chunk=4, progress=lambda done, total: seen.append((done, total)), clock=lambda: next(ticks))
    assert reads == [[4, 5, 6, 7], [8, 9]] and seen == [(4, 6), (6, 6)] and out["states"] == 6 and out["seconds"] == 1 and out["model_ms"] == 2.0       # sorted, in chunks
    assert cache.has(range(4, 10)).all() and not cache.has([3]).any() and (cache.versions_of(range(4, 10)) == 2).all()
    seed_of_7 = rcand.candidate_seed(11, 7, 2) % 1000
    assert cache.get([7])[0, 0, 0, 1] == seed_of_7
    again = rcand.CandidateCache(3, H, DS)
    rcand.refresh(again, sampler, source, range(4, 10), version=2, seed=11, chunk=3)
    assert np.array_equal(again.normalized[4:10], cache.normalized[4:10])                             # repeatable, whatever the chunking
    assert rcand.candidate_seed(11, 7, 2) != rcand.candidate_seed(11, 7, 3) != rcand.candidate_seed(12, 7, 3)    # a new version draws new noise


def test_a_snapshot_of_the_cache_restores_it_with_its_versions_in_a_cache_of_any_capacity():
    cache = rcand.CandidateCache(3, H, DS)
    cache.put([0, 1, 2], fake_candidates([0.0, 1.0, 2.0], [5, 6, 7], 3), version=0)
    cache.put([1], fake_candidates([1.0], [9], 3), version=4)
    snap = cache.snapshot()
    assert snap["normalized"].shape[0] == snap["version"].shape[0] == cache.size and snap["version"].tolist() == [0, 4, 0]
    for capacity in (1, 100):                                                                          # smaller than the snapshot (it grows) and larger
        other = rcand.CandidateCache(3, H, DS, capacity=capacity)
        other.restore(**snap)
        assert other.size == cache.size and other.versions_of(range(3)).tolist() == [0, 4, 0] and np.array_equal(other.get(range(3)), cache.get(range(3)))
        assert np.array_equal(other.snapshot()["states"], snap["states"])
    with pytest.raises(ValueError, match="does not fit"):
        rcand.CandidateCache(5, H, DS).restore(**snap)                                                 # another k


def test_concat_refuses_nothing_to_join():
    with pytest.raises(ValueError):
        rcand.concat_candidates([])


# -- the two methods of OpenPIPolicy, on a stub ---------------------------------------------------------------------------------------
def stub_policy(native=True, model_dim=12):
    """An object with what the two methods read, and the real methods bound to it."""
    stub = types.SimpleNamespace(_native_samples=native, last_sample=None, _noise_shape=(H, model_dim), info=types.SimpleNamespace(output_semantics="offsets_from_chunk_start_state"),
                                 policy=types.SimpleNamespace(_output_transform=lambda d: {"actions": np.asarray(d["actions"])[:, :8] * 2.0 + float(d["state"][0])}))
    stub.plan = lambda obs, prompt, s, row_seeds=None, pad_to=None: _plan(stub, obs, s, model_dim)
    return stub


def _plan(stub, obs, s, model_dim):
    n = len(obs[P.JOINT_POSITION])
    out = np.arange(n * s * H * model_dim, dtype=np.float32).reshape(n * s, H, model_dim)                 # (environment, sample)-major, wider than the env's 8 dimensions
    state = np.repeat(np.arange(1, n + 1, dtype=np.float32)[:, None], DS, axis=1)
    stub.last_sample = (out, state)
    raw = np.stack([[out[i * s + k][:, :8] * 2.0 + state[i, 0] for k in range(s)] for i in range(n)]).astype(np.float32)
    q = obs[P.JOINT_POSITION]
    targets = raw.copy()
    targets[..., :7] += q[:, None, None, :7]
    return Plan(targets=targets, raw=raw, infer_ms=2.5)


def test_sample_candidates_returns_the_normalised_chunks_in_environment_sample_order_and_plans_own_targets():
    stub = stub_policy()
    obs = obs_of(3)
    c = OpenPIPolicy.sample_candidates(stub, obs, "p", 2, row_seeds=[1, 2, 3], pad_to=4)
    out = stub.last_sample[0]
    assert c.normalized.shape == (3, 2, H, 8) and c.states.shape == (3, DS) and c.targets.shape == (3, 2, H, 8) and c.infer_ms == 2.5
    assert np.array_equal(c.normalized[2, 1], out[2 * 2 + 1][:, :8]) and np.array_equal(c.normalized[0, 1], out[1][:, :8])         # row j = environment * S + sample, first 8 dimensions
    assert c.states[1, 0] == 2.0


def test_normalized_to_targets_is_the_inverse_of_what_plan_did_for_the_same_chunks():
    stub = stub_policy()
    obs = obs_of(3, offset=1)
    c = OpenPIPolicy.sample_candidates(stub, obs, "p", 2)
    back = OpenPIPolicy.normalized_to_targets(stub, c.normalized, c.states, obs[P.JOINT_POSITION])
    assert np.allclose(back, c.targets)                                                                # the round trip of an unchanged chunk gives plan's own targets
    moved = OpenPIPolicy.normalized_to_targets(stub, c.normalized + 0.5, c.states, obs[P.JOINT_POSITION])
    assert np.allclose(moved - back, 1.0)                                                              # a displacement in the normalised space is the output transform's scale (2 here) in the env's
    with pytest.raises(ValueError, match="expected"):
        OpenPIPolicy.normalized_to_targets(stub, c.normalized[..., :5], c.states, obs[P.JOINT_POSITION])


def test_sample_candidates_refuses_pi0_fast():
    with pytest.raises(ValueError, match="not pi0-FAST"):
        OpenPIPolicy.sample_candidates(stub_policy(native=False), obs_of(1), "p", 2)
