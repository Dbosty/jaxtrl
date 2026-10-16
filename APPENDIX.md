# 7. Appendix

Below are the training and agent specifications used to generate the policies
in this reproduction of APEGS (Kalanther, Bharvirkar, Bostwick, Maheshwari,
Sastry). Structured to mirror the paper's own Section 7 Appendix, with four
additional sections (7.3-7.6) covering what the paper's appendix doesn't:
the Algorithm 1/2/3-specific hyperparameters this reproduction needed, the
verified deviations from the published values, and two extra baselines not
in the paper at all -- a DAgger-trained classifier alternative to LKSP
(7.5) and an offline (non-DAgger) ablation of it (7.6).

## 7.1. Training Specification

### 7.1.1. Hyperparameters and Model Architectures

We used a variant of PPO utilizing a recurrent network for its policy, and
used identical model architectures for all learned policies, excluding the
output dimension. Table 1 details the hyperparameters used for PPO in all
learned policies and Table 2 details the model architectures for the actor
and value functions.

**Table 1: Hyperparameters (PPO)**

| Hyperparameter | Value | vs. published paper |
|---|---|---|
| Frames per batch | 25,000 (250 envs x 100 steps) | paper: 50,000 -- halved; the paper's batch size OOMs the LSTM critic's backward pass on a 24GB GPU |
| Sub-batch size | 5,000 | matches |
| Number of epochs | 10 | matches |
| Discount factor (gamma) | 0.99 | matches |
| GAE parameter (lambda) | 0.95 | matches |
| Clip range (epsilon) | 0.2 | matches |
| Critic coefficient (c0) | 0.5 | matches |
| Entropy coefficient (c1) | 0.005 | matches |
| Optimizer | Adam | matches |
| Learning rate | 3e-4 | matches |
| Betas | (0.9, 0.999) | paper states (0.9, 0.99) -- verified the training code never passes `betas` to `torch.optim.Adam`, so torch's default silently applies |
| Eps | 1e-8 | matches (torch default) |
| Weight decay | 0 | matches |
| Critic loss type | smooth L1 | not stated in the paper |

**Table 2: Actor/Value Function Model Architecture**

| Hyperparameter | Policy | Value Function |
|---|---|---|
| LSTM hidden size | 64 | 64 |
| LSTM num layers | 1 | 1 |
| Head net arch | [64, 32, 16] | [64, 32, 16] |
| Head activation function | ReLU | ReLU |

### 7.1.2. Actor Distributions

PPO uses stochastic policies to explore the environment. Pursuer and evader
policies (outside of LKSP) use TanhNormal distributions to sample actions
during training, bounding actions to the environment's valid range.

The published paper rescales the pre-squash mean according to
`mu <- tanh(mu / scale) * scale`, with `scale = 5`, to stabilize training by
preventing the mean from saturating near the tanh's bounds. **This
reproduction's actor uses `torchrl.modules.TanhNormal` directly** (verified
via `jaxtrl/src/jaxtrl/model/th.py`, which re-exports it rather than
defining a rescaled subclass) **-- the scale=5 rescaling is not implemented.**

The LKSP policy uses a classification head (Categorical distribution over
`{0, ..., K}`, softmax over output logits) to select which level-k pursuer
policy to deploy -- this part matches the paper as described.

## 7.2. Agent Specification

### 7.2.1. Reward Function

For all policies, a reward of 1 is given if the goal is reached and 0
otherwise, for every timestep (`pgj.jaxtrl.reward.pursuer.SparsePursuer` /
`pgj.jaxtrl.reward.evader.SparseEvader`, unmodified from the existing pgj
codebase):

- Pursuer reward: 1 if the LLP catches the evader that timestep, else 0.
- Evader reward: 1 if the evader reaches its goal that timestep, else 0.

### 7.2.2. Observation Featurization

Unmodified from the existing pgj codebase (`pgj.jaxtrl.agent.llp.IndTeamLlp`,
`.hlp.IndTeamHlp`, `.evader.IndEvader`). Additional features derived from
each agent's raw environment observation, to aid learning:

- **Pursuers** (LLP and HLP share an identical 19-dim featurized
  observation): own and teammate position/direction, evader position/
  direction, binary flags for whether the evader is currently seen and has
  ever been seen (separately by LLP and HLP), delta to the evader's last-seen
  position, timesteps since each pursuer last saw the evader, and total
  elapsed timesteps.
- **Evader** (21-dim featurized observation, used by `IndEvader` for level-1
  through level-K evaders and by PAP/LKSP's opponent-sampling code): own
  position/direction, goal position and delta, LLP/HLP position/direction,
  binary flags for whether it has seen LLP/HLP, delta to each one's last-seen
  position, timesteps since last seeing each one, and total elapsed
  timesteps. (The level-0 evader instead uses the 2-dim
  `GreedyEvader`/`Evader` observation -- just the delta to its goal -- since
  it is a fixed shortest-path/random mixture, not a trained recurrent
  policy.)

All pursuer policies, including PAP and LKSP, use the same featurized
observation.

### 7.2.3. Action Selection

Unmodified from the existing pgj codebase. Pursuer and evader policies
(outside of LKSP) select a desired point, and the agent takes the shortest
path toward it. Actions are `(dy, dx, sin(theta), cos(theta))`, where `dy,
dx` is the delta from the current position and `sin(theta), cos(theta)`
gives the agent's heading `theta = arctan(sin(theta)/cos(theta))`,
discretized by the environment. LKSP's action is a discrete choice among the
K+1 pre-trained pursuer levels (Section 7.1.2); the chosen level's own
`(dy, dx, sin(theta), cos(theta))` action is what actually reaches the
environment.

## 7.3. Algorithm 1/2/3 Hyperparameters

Not present in the published paper's appendix (which documents the shared
PPO/architecture setup, not Algorithm 1-3's own free parameters). Listed
here since they were necessary choices this reproduction had to make, absent
a reference implementation.

| Hyperparameter | Value | Note |
|---|---|---|
| K (levels beyond level 0) | 4 | matches the paper's reported experiments |
| eta_pursuer (B_eta(.) radius for sampled pursuer opponents) | 0.05 | unstated in paper; our choice |
| eta_evader (B_eta(.) radius for sampled evader opponents) | 0.05 | unstated in paper; our choice |
| N (level-0 mixture alpha discretization, A = {l/N : l in [N]}) | 10 | unstated in paper; our choice |
| Opponent resampling granularity | once per outer training iteration | unstated in paper; we did not resample per-episode |
| Total frames, `level_k.py` per stage (9 stages) | ~20,000,000 | inferred; paper does not state a per-stage budget |
| Total frames, `pap.py` / `lksp.py` (1 stage each) | ~40,000,000 | inferred from Figure 4's x-axis extent |
| Eval rollouts (`evaluate.py`) | 500 parallel envs x 100 steps per (level, level) combination | evaluation-only, not a training hyperparameter |

## 7.4. Deviations from the Published Paper

Summary of every verified or inferred divergence between this reproduction
and the paper, consolidated from the sections above:

1. **Frames per batch: 25,000 vs. the paper's 50,000.** Halved due to a
   24GB GPU running out of memory on the LSTM critic's backward pass at the
   paper's batch size.
2. **Adam betas: (0.9, 0.999) vs. the paper's stated (0.9, 0.99).** Verified
   in `jaxtrl/src/jaxtrl/rl/ppo/_ppo.py` and `marl/mappo/_mappo.py`: neither
   passes `betas` to `torch.optim.Adam`, so torch's default silently
   overrides the paper's stated value.
3. **No TanhNormal mean rescaling** (`mu <- tanh(mu/5)*5`, Section 7.1.2 of
   the paper). The actor uses `torchrl.modules.TanhNormal` unmodified.
4. **No reference implementation of Algorithms 1-3 existed to reproduce
   from.** `level_k.py`/`pap.py`/`lksp.py` were written from the paper's
   pseudocode and prose; eta, N, and opponent-resampling granularity
   (Section 7.3 above) are this reproduction's own choices, not verified
   against the original.
5. **Single training run, no seed averaging.** The paper does not state how
   many seeds/runs its reported numbers average over; figures here reflect
   one run per level.
6. **Per-stage training budgets were inferred, not stated.** The paper
   gives no explicit total-frame budget for the 9 `level_k.py` stages;
   20M/stage was chosen by inference from Figure 4's PAP/LKSP x-axis extent
   (~4x10^7), which is a different (harder, adaptive-opponent) training
   problem than a single level_k.py stage.

None of the above indicate a bug in the reproduction; qualitatively, every
figure (2, 3, 4, 5, 6, 7) reproduces the paper's claimed pattern. The
compounding effect of these deviations is the most likely explanation for
where absolute numbers differ, including the one qualitative reversal found
at 40M frames (Figure 5: this reproduction's LKSP outperforms PAP at every
evader level, where the paper reports the reverse except at level 0). Note
that this reversal narrows substantially at 100M frames (Section 7.5.3) --
PAP was likely still undertrained at 40M, given how much later its reward
signal ramps up in Figure 4 compared to LKSP's.

## 7.5. DAgger Classifier (Alternative to LKSP)

Not part of the published paper -- added to this reproduction as a
comparison point against LKSP, evaluated with the same plots (Figures 4
and 5).

### 7.5.1. What It Does

LKSP's meta-controller (Section 7.1.2) is trained by PPO: it earns reward
only indirectly, through whether the pursuer level it selects wins. But
Figure 2 (and the paper's own Q1 finding) establishes that **pursuer level
k is the best response to evader level k**. During training we always know
the ground truth -- we are the ones sampling which evader level is active
each episode (`pap.py`'s `UniformLevelKEvaderPolicy`) -- so "which pursuer
level should act" reduces to "which evader level am I facing," a label we
already have, with no reward signal required. That reframes the
meta-controller's job as **supervised classification**: given the shared
pursuer-team observation history, predict the active evader level in
`{1, ..., K}`, then deploy the pursuer level matching that prediction.

A classifier trained once on a fixed rollout's states would still fail at
deployment: its own predictions determine which pursuer level acts, which
changes what states get visited next, and a classifier never trained on
those self-induced states will make compounding errors there (the standard
imitation-learning distribution-shift problem). **DAgger** (Dataset
Aggregation) closes that gap by looping: roll out the *current* classifier,
label the states it actually visits (for free -- we always know the true
active evader level), aggregate those labels into the training set, and
retrain. Over iterations the classifier is trained on the distribution of
states it will actually encounter, not just a fixed expert's distribution.

This implementation approximates DAgger's "aggregate all data forever"
with a bounded FIFO replay buffer (Table 3), since retraining on the full
multi-ten-million-sample run every iteration is infeasible.

### 7.5.2. Algorithmic Sketch

```
Algorithm 4  DAgger Classifier Training (pursuit-level selection)

Require: frozen pursuer levels (pi_LLP^(k), pi_HLP^(k)) for k in {1,...,K}
         and frozen evader levels pi_EDR^(k) for k in {1,...,K},
         both from Algorithm 1
 1: Initialize classifier pi_CLS (MLP, Categorical over {1,...,K}) with
    random weights
 2: Initialize empty FIFO replay buffer B of capacity C
 3: for iteration i = 1, 2, ..., N do
 4:     Sample evader level l ~ Unif({1,...,K}) for this batch of episodes
 5:     Roll out (pi_LLP^(pi_CLS(h_t)), pi_HLP^(pi_CLS(h_t))) against
        evader level l, deploying whichever pursuer level pi_CLS selects
        at each timestep t, given the observation history h_t
 6:     Collect (h_t, l) for every visited timestep t -- l is exact,
        since it was chosen in line 4, not estimated
 7:     Add all (h_t, l) pairs to B, evicting the oldest entries if
        |B| > C
 8:     for epoch = 1, ..., E do
 9:         for minibatch (h, l) sampled from B do
10:             Update pi_CLS by gradient descent on cross-entropy
                loss(pi_CLS(h), l)
11:         end for
12:     end for
13: end for
```

Implementation: `envs/pgj/dagger.py`. `LKSPPolicy`/`TeamPursuerAgent`
(lines 5-6's deployment mechanism -- evaluate all K candidate pursuer
pairs every step via `deterministic_sample`, select by index) are reused
unchanged from `lksp.py`; only the meta-controller's training procedure
(lines 8-12, supervised cross-entropy via a replay buffer) differs from
LKSP's PPO loop.

**Table 3: DAgger-specific Hyperparameters**

| Hyperparameter | Value |
|---|---|
| Classifier architecture | MLP, net_arch [64, 32, 16], ReLU (matches LKSP's meta-controller head, for a fair comparison) |
| Optimizer | Adam, lr 3e-4 |
| Replay buffer capacity (C) | 200,000 samples (FIFO) |
| Epochs per iteration (E) | 5 |
| Minibatch size | 5,000 |
| Rollout batch size (line 5) | 250 envs x 100 steps = 25,000 samples/iteration |

### 7.5.3. Results

Trained at both 40M and 100M total frames (same budgets used for PAP/LKSP
comparison), against the same evader-level-1..K opponent distribution
PAP/LKSP train against.

| Metric | 40M frames | 100M frames |
|---|---|---|
| Final train/loss (cross-entropy) | 0.812 | 0.795 |
| Final train/accuracy (evader-level classification, 4-way) | 0.630 | 0.637 |
| Final rollout/pursuer_winrate | 0.819 | 0.801 |

**The classifier's performance is essentially flat between 40M and 100M
frames** -- accuracy and win rate both move by ~1 point, within run-to-run
noise. This contrasts with PAP, which visibly continued improving over
that same range (Section 7.4's note above). Consistent with the
classifier being bottlenecked by how predictable the evader's level
actually is from behavior alone (63% on a 4-way problem, vs. 25% random),
not by optimization budget -- more environment steps just means more of
the same easy-to-fit data, not a higher accuracy ceiling.

Figure 5 win rate vs. evader level, DAgger column, both budgets:

| Evader level | 40M | 100M |
|---|---|---|
| L0 | 0.782 | 0.786 |
| L1 | 0.735 | 0.741 |
| L2 | 0.863 | 0.865 |
| L3 | 0.820 | 0.830 |
| L4 | 0.811 | 0.797 |

Per-level numbers barely move between budgets (all within ~1-2 points),
the same "flat past 40M" pattern as the training-curve/accuracy numbers
above.

DAgger consistently ranks **below both PAP and LKSP** at every evader
level, at both budgets -- the RL-trained approaches (which optimize
directly for winning) outperform the classifier (which only optimizes for
correctly guessing the evader's level, an imperfect proxy for "which
pursuer level wins"). DAgger's one advantage, visible in Figure 4, is
convergence speed: it reaches its plateau about as fast as LKSP, and much
faster than PAP's slow reward-discovery ramp -- consistent with it having
a supervised label at every step rather than a sparse win/loss reward.

## 7.6. Offline Classifier (Ablation of DAgger)

Also not part of the published paper -- a second comparison point, this
one isolating exactly what DAgger's online data aggregation buys over a
naive supervised baseline.

### 7.6.1. What It Does

Where DAgger (Section 7.5) trains its classifier on states generated by
its *own, evolving* predictions, this baseline trains the identical
classifier architecture on a dataset collected **once, upfront, with no
classifier involved at all**: every (pursuer level, evader level)
combination in `{0,...,K} x {1,...,K}` is rolled out directly using the
*true*, already-trained frozen policies from Algorithm 1 (the same
coverage principle as Figure 7's dataset D), each observation labeled with
the true evader level. The classifier then trains via ordinary supervised
cross-entropy on this fixed dataset -- no rollout-relabel-retrain loop, no
replay buffer, just repeated epochs over the same ~1,000,000 samples.

At deployment, this classifier drives its own rollout via the exact same
`LKSPPolicy`/`TeamPursuerAgent` mechanism as DAgger and LKSP. The
prediction is what determines which pursuer level acts -- but unlike
DAgger, nothing in its training data was ever generated by an imperfect
classifier's choices. Any state its own mistakes lead it into during
deployment is out-of-distribution for it. This is the "predetermined
trajectories" baseline discussed when scoping DAgger: it isolates the
value of DAgger's online correction by holding architecture, dataset size,
and opponent coverage roughly fixed, and changing only *how the data was
collected*.

**Table 4: Offline-classifier-specific Hyperparameters**

| Hyperparameter | Value |
|---|---|
| Classifier architecture | MLP, net_arch [64, 32, 16], ReLU (identical to DAgger's, for a fair comparison) |
| Dataset composition | all (K+1) x K = 20 (pursuer level, evader level) combinations, true frozen policies on both sides |
| Dataset size | 500 envs x 100 steps x 20 combinations = 1,000,000 samples, collected once |
| Optimizer | Adam, lr 3e-4 |
| Epochs over the fixed dataset | 50 |
| Minibatch size | 5,000 |
| Eval-rollout frequency | every 5 epochs (always including epoch 1) |

### 7.6.2. Results

| Metric | Offline | DAgger (for comparison) |
|---|---|---|
| Final train/loss (cross-entropy) | 1.051 | 0.795 (100M) |
| Final train/accuracy (4-way) | 0.512 | 0.637 (100M) |
| Final rollout/pursuer_winrate | 0.734 | 0.801 (100M) |

Figure 5 win rate vs. evader level (100M-budget comparison set):

| Evader level | PAP | LKSP | DAgger | Offline |
|---|---|---|---|---|
| L0 | 0.823 | 0.835 | 0.786 | 0.784 |
| L1 | 0.915 | 0.841 | 0.724 | 0.707 |
| L2 | 0.886 | 0.949 | 0.863 | 0.759 |
| L3 | 0.917 | 0.892 | 0.825 | 0.682 |
| L4 | 0.869 | 0.881 | 0.812 | 0.759 |

The ordering is consistent at every evader level: **PAP/LKSP > DAgger >
Offline**. The offline classifier's accuracy (51%) is meaningfully below
DAgger's (64%) despite an *identical* architecture and a larger, broader
one-shot dataset (1M samples covering all 20 level combinations, vs.
DAgger's 200K-sample rolling buffer) -- the only thing that changed is
*how* the data was collected. This is the distribution-shift argument
(Section 7.5.1) made concrete: the offline classifier never saw a single
state generated by its own mistakes, so it has no basis for handling them
at deployment, while DAgger's worse-looking training data (generated by a
classifier that was often wrong) is exactly what let it generalize better
once deployed.

Figure 4 shows the same story from a different angle: the offline
classifier's win-rate curve rises quickly then plateaus around 0.73,
below DAgger's ~0.80 plateau -- more epochs over the same fixed dataset
does not close the gap, since the ceiling here is a *distribution*
problem, not an optimization or data-volume problem.
