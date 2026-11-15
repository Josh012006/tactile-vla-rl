# RL fine-tuning of the VLA: Best-of-N and PA-RL (design, decisions, open points)

Status (15 Nov. 2026): design settled; implementation steps 1-4 of 7 done (section 7: the YAML defaults and loader; the candidate sampler and cache; the critic network and the replay buffer; the action optimisation). No critic update, rollout, distillation or job code yet. Every value below says where it comes from:
**paper** = Policy Agnostic RL (Mark et al., arXiv 2412.06685v1), with the section / page; **group** = decided with Josué (15 Nov. 2026) after the exchange with Jesse;
**proposal** = chosen by Claude because the paper is silent, to be confirmed before the step that needs it; **open** = to decide.
Rule: the defaults of the YAMLs are the paper's; where the paper has no value we ask before choosing.

## 1. What is built

Three methods, all on pi0.5 (scope below), all with ONE online-trained critic and UTD = 1, no ensemble, no pessimism (Jesse: a conservatism bias is likely harmful):

| method | VLA | action executed in rollouts | action executed in evaluation |
|---|---|---|---|
| Best-of-N | frozen | argmax of Q over the k candidates (group) | same |
| PA-RL, variant A (global) | distilled online | a_t ~ pi^Opt over the top-m (paper, Alg. 2 line 7) | argmax of pi^Opt (paper, B.1; group) |
| PA-RL, variant B (global + local) | distilled online | same, after T gradient steps on Q | same |

Best-of-N is Algorithm 2 of the paper without the distillation step, restricted to global optimisation (group). Variant A is Jesse's "distil after best-of-N, no gradient ascent".
Tactile / force goes to the CRITIC ONLY (the SFT checkpoint is the same for every run and has not learned touch); the comparison is the same algorithm with the critic with and without touch.
Evaluation is the paper's: the VLA WITH action optimisation at inference (so the critic, hence touch, is also used at evaluation). Group decision, 15 Nov. 2026.

**Scope: pi0.5 only (decision of 15 Nov. 2026).** pi0-FAST is not part of these methods, and nothing of the RL code is wired to it. Two reasons measured or read: it decodes greedily as we run it (no sampling temperature is set, see src/models/interface.py), so the k candidates of a state would be identical unless a sampling temperature is added; and 32 candidates cost far too much (about 4 s per state, extrapolated from a probe with synthetic observations, 15 Nov. 2026: about 36 h of model for 1000 episodes of rollouts alone). pi0-FAST stays a baseline of the SFT part (docs/algos/sft.md).

## 2. Algorithm (paper, section 4, Alg. 1 and 2, pp. 7-8)

- Stage I, action optimisation at a state s: sample k actions (chunks) from the VLA; **global**: keep the top m by Q; **local**: T steps `a <- a + alpha * grad_a Q(s, a)` on each kept action, a step being kept only if it
  raises Q (otherwise the loop of that action stops: Alg. 1 lines 5-9). The dataset action is NOT added to the set (group; the paper includes it in Alg. 1 line 3 but says it is only "sometimes helpful").
- pi^Opt(a|s,m) = softmax over the optimised set of Q (temperature 1) (Eq. 4.3). In evaluation, and in Best-of-N, the argmax.
- Stage II, distillation (PA-RL only): annotate every state of the buffer (warm-up data included, Eq. 4.4) with one action of pi^Opt, and train the VLA on it with its own loss (flow matching for pi0.5: the existing
  `OpenPIPolicy.training_batch` / `loss`). Candidate to imitate: softmax sample or argmax, BOTH paths exist (YAML switch), chosen by the variance of the candidate Q-values (paper B.1, Tables 5-6:
  CALVIN with std 0.02 is better with argmax, the kitchens with std 1.6-11.5 with softmax; no threshold is given). **The standard deviation of the candidates' Q is logged at every iteration** (group).
- Critic target: the Bellman backup uses actions of pi^Opt at s' (paper, p. 9 and Fig. 16: using the base policy there makes fine-tuning unstable). The VLA's raw candidates of every state are CACHED with the transition
  (paper 5.2.2 and Appendix E); the optimisation (global ranking, local steps) is done on the fly with the current critic. The cache of Best-of-N never expires (the VLA never changes);
  PA-RL recomputes it after every distillation epoch (paper App. E: "the actions cache is updated after every epoch of policy training").
- Loop (Alg. 2, online, no offline data): collect with pi^Opt -> critic updates (UTD 1) -> distillation epoch -> refresh the cache.

## 3. Values

| item | value | source |
|---|---|---|
| candidates k | 32 | paper B.1; group |
| kept m (PA-RL) | 10 | paper B.1 |
| Best-of-N | k = 32, argmax (m = 1), no local step | group |
| local steps T / step alpha | 10 / 3e-4, in the model's normalised chunk space, no clipping | paper B.1; group (the paper gives no bounding) |
| a step is kept only if Q rises | yes, per action | paper Alg. 1 |
| dataset action in the candidate set | no | group |
| critic count / UTD | 1 critic / 1 update per environment step (not per plan: 8 per plan) | group |
| critic | image encoder + MLP 3 x 512, the action concatenated to the input of every layer, random shift 4 px, Adam, lr 3e-4, batch 256 | paper B.1 (image domains, Kumar et al. 2023) |
| critic inputs (state) | both cameras (exterior + wrist) resized to 128 x 128, a small CNN encoder trained from scratch with the critic (random shift 4 px), concatenated with the 7 joint positions and the gripper position, then the MLP above; the action input is the whole chunk (pi0.5: 16 x 8, normalised space). The tactile variant ADDS the touch vector after the encoders; nothing else changes, so the comparison with / without touch is clean. Fallback if the Q values do not learn from ~20 000 transitions: a frozen pre-trained encoder (ResNet-18), not before | group (15 Nov. 2026); paper: from scratch with random shift, no camera or resolution given |
| critic touch | the same three switches as every component of the system (`critic_use_tactile`, `critic_use_force`, `critic_force_mode`; `src/touch/features.py`): the vector [ft_force (3) \| gripper_effort (1) \| tactile (56)], each entry sign(x) * log1p(\|x\|), concatenated to the state feature, no history | repo convention (src/touch/features.py), applied to the critic |
| critic architecture | one CNN per camera (4 convolutions 3x3, stride 2, 32-64-128-128 channels, GroupNorm + ReLU, average pooling to 4x4, Dense 256, LayerNorm), proprioception as it is, MLP 3 x 512 with LayerNorm + ReLU and the action at every layer, scalar Q (about 2.5 M parameters); the images are stored at 224 and resized on the device | proposal (the paper gives only the MLP shape and the random shift) |
| critic loss | Cal-QL (Eq. 3.3): TD in MSE with pi^Opt targets + alpha * (E[max(Q(s,a~VLA), V^mu(s))] - E[Q(s,a_data)]), alpha = 0.01 | paper B.1 (alpha 0.01 for CALVIN and the real robot; MSE there, HL-Gauss only when Q explodes); group: stay as close as possible to the paper (Jesse) |
| Cal-QL mixing ratio | 0.5: half of each critic batch from the warm-up data, half from the online buffer | paper B.1 ("rest" = CALVIN and real robot) |
| Cal-QL regulariser actions a ~ pi | the VLA's cached candidates of the state (no extra sampling) | proposal (the paper only says "sampled from the learned policy", Eq. 3.3) |
| Cal-QL reference value V^mu(s) | Monte-Carlo return-to-go of EVERY stored trajectory (warm-up and online), computed at the end of each episode with the plan-level reward and gamma; the regulariser applies to every state of the batch | group (15 Nov. 2026); our reading of Cal-QL, the paper is silent on the online phase |
| time-out at the 150-step horizon | not terminal: the target bootstraps from Q of the last state (one extra candidate plan per truncated episode); only success is terminal (target 0) | group |
| critic target network | a delayed copy of the same critic (not a second Q), tau = 0.005 | paper Eq. 3.3 ("delayed target Q"); tau: group (standard SAC / CQL value, the paper gives none) |
| reward / discount | -1 per plan until success, 0 at success; gamma = 0.99 per plan | paper B.1 (bias -1, gamma 0.99); group (a plan = a transition) |
| collection | synchronous batches: K environments play a batch of K episodes with the critic and the VLA frozen; then UTD = 1 critic updates (as many as the batch's environment steps), then (PA-RL) distillation and cache refresh | group; Alg. 2 of the paper |
| parallel environments K | 50 per batch (about 20 rounds for 1000 episodes; simulator alone about 32 min for 1000 episodes, estimate from docs/envs) | group |
| budget | 1000 online episodes in total (paper: 1000 in simulation) = 20 rounds of 50 episodes; PA-RL distils ONE epoch over the whole buffer per round, then refreshes the cache | paper (1000 episodes); group (1 epoch per round) |
| which critic ranks and optimises | the CURRENT critic Q_theta ranks the candidates (global), does the gradient ascent (local) and picks the pi^Opt action at s'; the delayed target network Q_bar only EVALUATES that retained action in the TD target (y = r + gamma * Q_bar(s', a'_Opt)). Distillation annotation and rollouts use Q_theta | group; paper Alg. 1 and Eq. 3.3 (the paper does not say which of the two ranks) |
| warm-up | 50 episodes of the base VLA, then 5 000 critic gradient steps | group (paper: 50 rollouts for OpenVLA, 5e5 steps of offline pre-training) |
| distillation | lr 2.5e-5 constant, batch 64, the LoRA of the SFT, one epoch over the whole buffer per round (about 30 steps in the first round to about 330 in the last, about 3 400 in total, about 8 h at 8.7 s per step: estimate); Adam state kept from one round to the next (one continuous fine-tuning cut in 20 rounds) | group (15 Nov. 2026); paper: 5e-5 (1e-5 kitchens), "reduced" for OpenVLA without a value; same loop as src/algos/sft.py |
| code stack of the critic | JAX / Flax + optax in the same process as the VLA (openpi is JAX: the critic, the candidates and jax.grad of Q with respect to the action share the card, no framework conversion, all under jit) | group (15 Nov. 2026); the paper is silent |
| cache refresh (PA-RL) | FULL after every round, as the paper (every buffer state recomputed with the updated VLA); about 10-12 h for pi0.5 (measured per state, extrapolated). Target: the whole run under 10 h, revisited once everything is built (fewer steps, more parallelism) | paper (App. E); group (15 Nov. 2026) |
| distributional critic (HL-Gauss) | not at first; added only if Q explode | paper B.1 |
| evaluation | 100 episodes, the same starting states as the SFT evaluations, with optimisation at inference | group |

## 4. Metrics logged at every iteration

Candidate Q standard deviation (mean over states, and the distribution), Q mean / min / max and Q minus the Monte-Carlo discounted return of the visited states (over- and under-estimation, as the paper's Fig. 14), TD loss,
success rate of the rollouts, the share of states where the argmax differs from the VLA's first sample, local optimisation: Q gain, share of accepted steps, L1 norm of the displacement (paper Fig. 13), distillation loss,
Jesse's warning: value-estimation error (over- or under-estimation) drives many collapses, so the Q statistics are watched as much as the success rate.

## 5. Costs

MEASURED on 15 Nov. 2026 (L40S, policy limited to half of the card, `scripts/probe_batch_cost.py`, synthetic observations, 5 timed calls per row, log 11716915):

| model | rows x samples per call | median per call | per state |
|---|---|---|---|
| pi0.5 | 1 x 1 | 101 ms | 101 ms |
| pi0.5 | 8 x 1 | 678 ms | 85 ms |
| pi0.5 | 50 x 1 | 4 319 ms | 86 ms |
| pi0.5 | 1 x 32 | 159 ms | 159 ms |
| pi0.5 | 8 x 32 | 1 529 ms | 191 ms |
| pi0.5 | 50 x 32 | out of memory (30 GiB asked, 24 GiB available) | - |

- pi0.5: 32 candidates per state cost little more than the state itself (the prefix is computed once; 159-191 ms against 85-101 ms). The images' prefix is the floor (about 85 ms per state, about 5 h over the whole refresh even with one candidate).
  Calls of 50 environments x 32 samples do not fit: the rollouts split them (`rows_per_sample_call = 8` in `configs/algo/rl.yaml`, so 7 calls of 8 for 50 environments). One round of 50 episodes: about 182 s of model (19 plans x 50 x 191 ms) + about 97 s of simulator = about 280 s, so about 93 min for 1000 episodes (estimate).
- Cache refresh of pi0.5 (every state of the buffer after every round, about 218 500 recomputations over the run): 9.6 h (calls of 1 row) to 11.6 h (calls of 8 rows), linear extrapolation. Decision: full refresh as the paper (section 3); the total cost of a run is to be brought back under 10 h AFTER everything is built (fewer steps or more parallelism), not before.
- Distillation: about 8 h (3 400 steps at 8.7 s), estimate. Critic updates (UTD 1: about 150 000 updates of a small CNN + MLP) not measured yet.

## 6. Open points (no reference in the paper), to decide one at a time before the step that needs them

1. Cost of a run: the target is under 10 h in total (group); the measured pieces (section 5) add up to more (refresh about 10-12 h, distillation about 8 h, rollouts about 1.5 h). To revisit once the implementation is complete: fewer distillation / refresh steps, more parallelism.

Settled on 15 Nov. 2026 (see section 3): budget 1000 episodes with one distillation epoch per round, which critic ranks (Q_theta) and evaluates (Q_bar), the critic's inputs and touch (the repo's convention), the distillation recipe, the code stack (JAX / Flax) and the cache refresh policy (full).

## 7. Implementation steps (one at a time, each with its tests, no job launched without a go)

1. DONE (15 Nov. 2026): YAML defaults (`configs/algo/rl.yaml`, one experiment per method in `configs/experiment/rl/`) and their loader (`src/algos/rl_config.py`, `tests/test_rl_config.py`). Open: the SFT checkpoint each run starts from (`start: {run, step}` of the experiment, optional until the launch script exists).
2. DONE (15 Nov. 2026): candidate sampler and cache (`src/algos/rl_candidates.py`, `tests/test_rl_candidates.py`; `OpenPIPolicy.sample_candidates` / `normalized_to_targets` and `Candidates` in `src/models/`, checked against the real transforms in `tests/test_openpi_policy.py`, which needs the `vla` group and was NOT run here).
   - The critic and the local ascent work in the model's NORMALISED chunk space (first 8 dimensions of the model output, before the output transforms): `sample_candidates` returns those chunks with the state rows, `normalized_to_targets` brings a chunk (chosen or moved) back to the env's space to be executed or distilled.
   - `CandidateSampler`: pi0.5 in calls of `rows_per_sample_call` = 8 environments, every call padded to 8 (one compilation).
   - `CandidateCache`: by state id, with the version of the VLA that drew the candidates (Best-of-N: one version for ever; PA-RL: a new one after each distillation epoch; `stale_ids`), `refresh` recomputes them reading the buffer in chunks, with seeds that are a function of (seed, state id, version). About 16 KiB per state for pi0.5 (20 000 states: about 330 MB).
3. DONE (15 Nov. 2026): critic network and replay buffer (`src/algos/rl_critic.py`, `src/algos/rl_buffer.py`, `tests/test_rl_critic.py`, `tests/test_rl_buffer.py`; JAX / Flax, tested on CPU). The network: `q_values(net, params, state, actions)` gives Q of N chunks per state (a state is encoded once), the target is a delayed copy (`soft_update`), touch is a switch of the config. The buffer: one row per state, the rows of an episode contiguous (the next state of row i is row i + 1, the ids are the candidate cache's), reward -1 per plan and 0 at success (the only terminal), V^mu = Monte-Carlo return-to-go of every episode, a time-out (any end without success) bootstraps from its last state (`needs_candidates`), batches that mix warm-up and online rows by `mixing_ratio`. About 300 KB per state (6.3 GB for the 1050 episodes of a run, allocated lazily).
4. DONE (15 Nov. 2026): action optimisation (`src/algos/rl_action_opt.py`, `tests/test_rl_action_opt.py`; JAX, tested on CPU on a toy critic with closed-form answers, then on the real network at small size). `ActionOptimiser(encode, score, keep_top, local_steps, local_lr)` (`for_critic(net, cfg)`): global top-m by Q of the k cached candidates, then T steps of `a <- a + alpha * grad_a Q` per kept chunk, a step kept only if Q rises strictly (a NaN / inf Q is refused). The ascent is deterministic, so "stop the loop of this action" (Alg. 1) and "refuse every non-rising step" give the same result and the code does the latter, with no per-action mask. `pick(result, mode, rng, temperature)` is pi^Opt: softmax sample or argmax of the optimised set (the mode is `rollout_selection`, `distill_selection` or `eval_selection`; the caller gives the rng). Best-of-N = keep_top 1, no local step, argmax. The critic was split into `rl_critic.encode_state` (CNNs, once per state) and `rl_critic.score` (MLP), so the T + 1 evaluations of the ascent run the MLP only: without that, a critic update of 256 states would re-run the CNNs on 512 images at every ascent step (about 150 000 updates per run). `summarise` gives the section 4 metrics (candidate Q std mean / p10 / p50 / p90, Q gain, accepted share, L1 displacement, share of states where the chosen chunk differs from the VLA's first sample). Not decided here, for step 5: which selection mode produces the a' of the TD target (rollout_selection is the paper-faithful reading: a sample of pi^Opt).
5. Critic update with pi^Opt targets and the warm-up phase; measure the real cost.
6. Best-of-N rollouts and its evaluation (environment servers of the evaluation).
7. Distillation (pi0.5) and the cache refresh: PA-RL variants A and B.
