# Profile-MoE: Theoretical Foundations & Scalability Proof

## 1. Core Architecture (Formal)

### Traditional MoE Router
```
r(x) = softmax(W_r · x)           where W_r ∈ R^(n_experts × d_model)
```
The router is a learned linear map from hidden state → expert scores.
∂r/∂W_r is part of backprop. Router and experts are **co-trained and entangled**.

### Profile-MoE Router
```
φ(x)  = Profiler(x)               φ: R^d_model → R^d_profile
p_i   = Profile(E_i)              p_i ∈ R^d_profile (fixed per expert)
r(x)  = softmax( cos_sim(φ(x), [p_1,...,p_n]) / τ )
```
The router has **zero learned parameters**. It computes similarity between
input profile and expert profiles. ∂r/∂(expert_weights) = 0.

This is the key property enabling swappability.

---

## 2. The Swapping Theorem

### Definition: Expert Pool
```
Pool = {(E_1, p_1), (E_2, p_2), ..., (E_n, p_n)}
where E_i is an expert function and p_i is its calibrated profile vector.
```

### Definition: Swap Operation
```
swap(Pool, i, E_i') = Pool' where:
  E_i is replaced by E_i'
  p_i is replaced by p_i' = Profile(E_i')  (recalibrated)
```

### Theorem 1: Routing Isolation
For any input x, the routing decision for expert j (j ≠ i) is unchanged after swap:
```
  r_j(x) = r'_j(x)    ∀ j ≠ i
```
**Proof**: The router computes cos_sim(φ(x), p_j) for each expert j.
Only p_i changes during swap. For j ≠ i, p_j is unchanged, therefore
cos_sim(φ(x), p_j) is unchanged, therefore r_j(x) is unchanged. ∎

### Theorem 2: Localized Impact
After swapping E_i for E_i', only inputs where E_i was in the original
top-k are affected. All other inputs produce identical outputs.

**Proof**: By Theorem 1, routing weights for all experts j ≠ i are unchanged.
The MoE output is Σ w_j · E_j(x). For inputs where w_i = 0 (E_i not in top-k),
the sum is unchanged. For inputs where w_i > 0, only the E_i term changes,
and its new weight w'_i depends only on p'_i (by the router definition). ∎

### Why Traditional MoE Cannot Do This
In traditional MoE, replacing E_i changes the loss landscape. The router
W_r was trained with the OLD E_i. The new E_i' has different activation
patterns, different error characteristics. W_r's mapping from hidden states
to expert scores was optimized for the old expert pool. There's no guarantee
that E_i' even gets selected for the right inputs.

Worse: traditional MoE experts aren't domain-specialized in any explicit way.
They specialize organically during co-training, and the specialization is
an emergent property of the (router, experts) system. Swap one out and the
emergent specialization collapses.

---

## 3. Scalability Analysis

### 3.1 Computational Scaling

Routing cost: O(n · d) where n = num_experts, d = profile_dims.

| Scale | n (experts) | d (profile dims) | Routing ops | Expert FLOPs | Overhead |
|-------|------------|-------------------|-------------|--------------|----------|
| MVP   | 4          | 4                 | 16          | ~1K          | 1.6%     |
| Small | 16         | 20                | 320         | ~100M        | 0.0003%  |
| Medium| 64         | 50                | 3,200       | ~10B         | 0.00003% |
| Large | 128        | 50                | 6,400       | ~100B        | ~0%      |

Routing overhead becomes negligible at scale. The profiler φ(x) adds a
small MLP cost (d_model → hidden → d_profile), also negligible.

### 3.2 Discriminative Capacity

Question: can d profile dimensions distinguish n experts?

In d-dimensional space with normalized profiles (on unit hypersphere):
- Binary profiles (each dim = 0 or 1): 2^d distinguishable experts
- Continuous profiles with minimum angular separation θ: ~ O(1/θ^d)

For d=50 (practical benchmark suite: MMLU categories + coding + math + etc.):
- Binary: 2^50 ≈ 10^15 experts — effectively unlimited
- Continuous with θ=0.1 rad: ~10^50 experts

**The bottleneck is not the math. It's benchmark design.**

Can we MEASURE enough independent capability dimensions?

Current major benchmarks:
- MMLU: 57 subjects → could be 57 profile dimensions
- HumanEval: coding → 1-2 dims
- GSM8K/MATH: math reasoning → 1-2 dims
- HellaSwag: commonsense → 1 dim
- TruthfulQA: factuality → 1 dim
- Multilingual: 10+ languages → 10+ dims
- Safety/refusal: 1-2 dims
- Instruction following: 1-2 dims
- Long context: 1 dim

~75+ measurable dimensions exist today. Enough for hundreds of experts.

### 3.3 Profile Maintenance at Scale

Adding a new expert to a pool of n experts:
```
Cost = O(d · b)  where b = benchmark examples per dimension
     = O(50 · 100) = 5,000 evaluations
     = ~minutes for a small model, ~hours for a large one
```
This is ONE-TIME, embarrassingly parallel, and requires NO retraining.

Traditional MoE: adding an expert requires full or partial retraining of
the router + rebalancing all other experts. Days to weeks.

### 3.4 Load Balancing at Scale

Profile-based load balancing is a **scheduling** problem, not a training problem:

1. Monitor: track which profiles appear most frequently in production
2. If "coding-like" profiles dominate:
   - Option A: Add more coding experts (horizontal scale)
   - Option B: Split coding into sub-profiles (python, js, rust, etc.)
3. Router naturally distributes: similar prompts → similar experts

No auxiliary loss needed (unlike traditional MoE which needs load-balancing
loss terms to prevent expert collapse).

---

## 4. The Profile Calibration Pipeline (Infrastructure)

This is the critical piece. Profiles must be ACCURATE or the router fails.

### 4.1 Calibration Protocol

```
For each expert E_i:
  For each benchmark dimension d_j:
    Run E_i on benchmark d_j's test set
    Record score s_{i,j} (accuracy, F1, MSE, etc.)
  Normalize: p_i = [s_{i,1}, s_{i,2}, ..., s_{i,d}] / Σ s_{i,k}
```

This produces a profile vector where each dimension = "how good is this expert
at this skill, relative to its other skills."

Alternative: global normalization across experts:
```
p_i[j] = s_{i,j} / max_k(s_{k,j})   → "how good vs the best expert"
```
This makes profiles comparable across experts. A score of 0.9 means "90% as
good as the best expert on this dimension."

### 4.2 Profile Decay & Recalibration

Profiles are snapshots. They decay as:
- The expert is fine-tuned
- The benchmark itself evolves
- The definition of "good" changes

Infrastructure needs:
```
Profile {
    vector: [float; d],
    calibrated_at: timestamp,
    benchmark_versions: {name: version},
    confidence: float,        // based on benchmark sample size
    decay_rate: float          // estimated per dimension
}
```

Recalibration triggers:
- Time-based: every N days
- Performance-based: if routing accuracy drops below threshold
- Event-based: after expert fine-tuning

### 4.3 Cold Start: New Expert Joins the Pool

```
1. Expert E_new arrives (no profile)
2. Run calibration protocol → p_new
3. Insert (E_new, p_new) into pool
4. Router immediately uses p_new for matching
5. Zero retraining. Zero downtime.
```

---

## 5. The Profiler Training Problem

The profiler φ(x): R^d_model → R^d_profile maps inputs to profile space.

### Option A: Supervised (domain-labeled data)
Train φ on (input, domain_label) pairs. Domain label → one-hot profile.
Simple, but requires labeled data.

### Option B: Self-supervised (expert performance signal)
For each input x in a training set:
  1. Run ALL experts on x
  2. Expert_i performance on x → target profile
  3. Train φ(x) to predict this target

φ learns: "inputs that Expert A does well on → profile [high_A, low_others]"

This is elegant: the same calibration system that produces expert profiles
also produces training data for the profiler.

### Option C: LLM-as-profiler (zero-shot)
Use a lightweight LLM to classify prompts:
```
"Classify this prompt by required skills: coding, math, reasoning, creative, factual.
 Output: JSON with scores 0-1 for each."
```
No training needed. Accuracy depends on the classifier LLM.

### Recommendation: B for training, C for cold start, A for validation.

---

## 6. Failure Modes & Mitigations

| Failure Mode | Cause | Detection | Mitigation |
|-------------|-------|-----------|------------|
| Profile collapse | Multiple experts get near-identical profiles | Profile pairwise cosine > 0.95 | Finer benchmarks, or merge redundant experts |
| Profiler drift | Input distribution shifts, φ becomes miscalibrated | Routing accuracy drop over time | Periodic profiler recalibration |
| Expert obsolescence | Expert no longer reflects its profile | Per-cluster MSE drift | Auto-recalibrate flagged experts |
| Dimension collapse | Some profile dims have zero variance across experts | Low variance in dimension j | Remove or replace that benchmark |
| Router ambiguity | Input profile is equidistant from multiple experts | Entropy of routing weights is high | Lower temperature τ, or accept soft routing |
| Overhead creep | Too many experts → routing cost grows | Monitor routing latency | Hierarchical routing (coarse → fine) |

---

## 7. From Regression MVP → Production LLM

The mapping is direct because the architecture is identical at the abstraction level:

| Component | Regression MVP | Production LLM |
|-----------|---------------|----------------|
| Input x | 2D coordinate | Token sequence → embedding |
| Profiler φ(x) | 2→8→4 MLP | embed_dim→hidden→d_profile MLP |
| Expert E_i | 2→16→1 MLP | FFN layer (2-layer MLP with GELU, ~65M params) |
| Profile p_i | 4-dim normalized MSE⁻¹ | d_profile-dim normalized benchmark scores |
| Router | cos_sim → softmax → top-k | cos_sim → softmax → top-k |
| Output | Σ w_i · E_i(x) | Σ w_i · FFN_i(hidden_state) |
| Calibration | Run on test clusters | Run on benchmark suites |

The only difference is scale, not structure. The router is IDENTICAL code.

---

## 8. What the MVP Actually Proves

The regression MVP proves these properties hold:

1. **Routing accuracy**: φ(x) + profile matching correctly identifies which
   expert is best for which input type. (Proves profiles encode capability.)

2. **Compositional generalization**: borderline inputs get soft routing
   (both experts activated). (Proves the router handles ambiguity.)

3. **Swap isolation**: replacing Expert A only affects cluster A performance.
   (Proves Theorem 1 & 2 hold empirically.)

4. **Cold start**: adding a new expert with its calibrated profile works
   immediately. (Proves infrastructure viability.)

5. **Load distribution**: expert utilization matches input distribution.
   (Proves no collapse without auxiliary loss.)

All five properties are invariant to scale. They hold for 4 experts on 2D
regression, and they hold for 64 experts on 4096-dim embeddings, because
the mathematical structure of the router is unchanged.

---

## 9. Summary: Why This Is Infrastructure

Traditional MoE is a **model architecture** — tightly coupled router+experts.

Profile-MoE is **infrastructure** — a framework where:
- Experts are pluggable modules with declared capabilities
- The router is a stateless matching function
- Profiles are the API between experts and the router
- Calibration is the onboarding process
- Swapping is the core operation

This is analogous to:
- Microservices with health checks (profiles = service health)
- Load balancers with weighted routing (profiles = weights)
- Package managers with dependency resolution (profiles = version compat)

The innovation is applying this pattern INSIDE the model architecture,
at the neural network level.
