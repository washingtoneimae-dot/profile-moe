# Profile-MoE: Technical Documentation

How the system works. For formal proofs, see THEORY.md. For architecture rationale, see PLAN.md.

---

## 1. The Routing Algorithm

The core of Profile-MoE is a three-step routing decision made for every input.

### Step 1: Profile the Input
```
φ(x): R^d_input → R^d_profile

Input (x) → Profiler → input_profile = [0.92, 0.03, 0.02, 0.03]
```
The profiler is a trained classifier. In the regression MVP, it's an sklearn MLPClassifier that maps 2D coordinates to cluster probabilities. In the transformer, it's a learned linear projection `φ(x) = softmax(W_φ · x)`. The output is always a normalized probability vector over profile dimensions.

### Step 2: Compare to Expert Profiles
```
similarities[i] = cos_sim(input_profile, expert_profiles[i])

Expert A profile: [0.99, 0.02, 0.01, 0.02]  →  similarity = 0.998
Expert B profile: [0.03, 0.97, 0.03, 0.02]  →  similarity = 0.052
Expert C profile: [0.02, 0.03, 0.96, 0.04]  →  similarity = 0.041
Expert D profile: [0.03, 0.02, 0.03, 0.95]  →  similarity = 0.058
```
Cosine similarity between the input profile and each expert's calibrated profile. Both vectors are normalized to unit length first. No learned parameters. Pure math.

### Step 3: Weighted Top-K Selection
```
weights = softmax(similarities / τ)        τ = temperature (default 0.1)
top_k = argsort(weights)[-k:]              k = 2 (default)
output = Σ(w_i · Expert_i(x))              weighted combination
```
Temperature controls routing sharpness. Low τ (0.01) → near one-hot selection. High τ (1.0) → soft blending. Selected expert weights are renormalized to sum to 1.

### Why This Is Swappable
The router reads `expert_profiles[i]` — an array. It never reads `Expert_i.weights`. Swap Expert A for Expert A', update `expert_profiles[0]` to A's new calibrated profile. The router doesn't notice. The similarity math doesn't change. That's the whole trick.

---

## 2. Expert Calibration

Each expert starts with no profile. Calibration builds it.

```
For each expert:
  For each benchmark domain d:
    Run expert.predict() on domain d's test set
    Record MSE_d = mean((y_true - y_pred)^2)
  
  skill_d = 1 / (MSE_d + ε)        # invert error → capability
  profile[d] = skill_d / Σ(skill)   # normalize to sum=1
```

An expert that scores well on the code benchmark gets a high `profile[code]` value. An expert that fails math gets a near-zero `profile[math]` value. The profile is an honest capability report, not a learned embedding.

### Calibration Cost
- O(n_experts × n_domains × n_test_samples) forward passes
- Embarrassingly parallel (each expert independently)
- Seconds for toy models, minutes for real LLMs

---

## 3. The Swap Operation

What actually happens when you swap Expert_code:

```
Before:
  experts = [Expert_code(old), Expert_math, Expert_creative, Expert_reasoning]
  expert_profiles = [[0.99, 0.02, 0.01, 0.02], ...]

After:
  1. Train/receive new Expert_code(new)
  2. Calibrate: expert_profiles[0] = calibrate(Expert_code(new))
     → [0.91, 0.05, 0.01, 0.03]  (different function, different profile)
  3. Replace: experts[0] = Expert_code(new)
  4. Done.
```

The router's next call to `cos_sim(input_profile, expert_profiles)` automatically uses the new profile. No weights updated. No training step. No downtime.

In traditional MoE: `W_r` was trained to route based on the OLD Expert_code's behavior. After swap, `W_r · x` still routes as if the old expert is there. The new expert receives wrong inputs or is ignored. Retraining required.

---

## 4. Profile Versioning

When adding a new domain (e.g., "law") to an existing pool:

### v1 Space (4-dim)
```
Profile dims: [code, creative, math, reasoning]
Law expert calibrated on v1: [0.57, 0.13, 0.06, 0.24]
```
The law expert looks mediocre at everything. Its true law capability is UNMEASURED. The router can't route law queries to it because it doesn't know law exists.

### v2 Space (5-dim) — The Upgrade
```
1. Add "law" benchmark to the framework
2. Recalibrate ALL experts on law benchmark:
   - Expert_code on law: MSE=98.7  →  profile[law] ≈ 0.01  (honest: bad at law)
   - Expert_law on law: MSE=0.05  →  profile[law] ≈ 0.99  (honest: great at law)
3. Retrain profiler φ(x) to recognize "law" as a 5th class
4. All experts now carry 5-dim v2 profiles
5. Law expert joins the pool
```

Existing experts score near-zero on the new dimension — honest. The profiler learns to classify law inputs. The router now has enough information to route correctly.

---

## 5. Project Structure

```
profile-moe/
│
├── mvp.py                          Regression proof (no PyTorch needed)
│   ├── Expert class                Tiny MLP + profile vector
│   ├── PromptProfiler              Classifier: input → profile
│   ├── ProfileRouter               cos_sim + softmax + top-k
│   ├── ProfileMoE                  Combines profiler + router + experts
│   ├── evaluate()                  Per-cluster MSE + routing accuracy
│   ├── SwapReport                  9-section before/after comparison
│   ├── analyze_temperature()       τ sweep
│   └── analyze_expert_scaling()    2/3/4 expert comparison
│
├── versioning_demo.py              Adding a 5th expert (no PyTorch)
│   ├── Profile v1 → v2 upgrade
│   ├── Recalibration cascade
│   ├── Law expert routing accuracy
│   └── Speed comparison vs agentic swarm
│
├── comparison_benchmark.py         Profile-MoE vs DeepSeek-style (no PyTorch)
│   ├── DeepSeekStyleRouter         Learned W_r · x → top-k
│   ├── DeepSeekMoE                 Full DeepSeek-style system
│   ├── ProfileMoE                  Full profile-based system
│   └── Head-to-head evaluation
│
├── transformer_benchmark.py        nanoGPT MoE architecture (numpy only)
│   ├── Multi-domain character data
│   ├── TransformerMoE              Attention + MoE FFN layers
│   ├── LearnedRouter vs ProfileRouter
│   └── Forward pass timing
│
├── transformer_training.py         Full training benchmark (needs PyTorch)
│   ├── MoETransformer              GPT-style with MoE layers
│   ├── MoETransformerLayer          Attention + vectorized expert dispatch
│   ├── train_model()               Full backprop training loop
│   └── evaluate_model()            Per-domain perplexity
│
├── generate_graphs.py              Publication charts (5 PNGs)
├── export_findings.py              Master workbook (7-sheet Excel)
│
├── THEORY.md                       Formal proofs
├── PLAN.md                         Architecture spec + prior art
├── DEEPSEEK_REFERENCE.md           DeepSeek paper reference
└── README.md                       Overview + verification guide
```

---

## 6. Data Flow: A Single Prediction

For a regression input `(x=0.2, y=0.1)`:

```
1. Profiler.classify([0.2, 0.1])
   → "This looks like cluster 'code'" with 92% confidence
   → input_profile = [0.92, 0.03, 0.02, 0.03]

2. For each expert, cos_sim(input_profile, expert.profile):
   Expert_code:      cos_sim([0.92,0.03,0.02,0.03], [0.99,0.02,0.01,0.02]) = 0.998
   Expert_math:      cos_sim([0.92,0.03,0.02,0.03], [0.03,0.97,0.03,0.02]) = 0.048
   Expert_creative:  cos_sim(..., [0.02,0.03,0.96,0.04]) = 0.009
   Expert_reasoning: cos_sim(..., [0.03,0.02,0.03,0.95]) = 0.018

3. Softmax([0.998, 0.048, 0.009, 0.018] / 0.1)
   → weights = [0.9999, 0.0001, 0.0000, 0.0000]

4. Top-2: Expert_code (w=0.9999), Expert_math (w=0.0001)

5. Expert_code.predict([0.2, 0.1]) = 0.148
   Expert_math.predict([0.2, 0.1])  = -0.695

6. Output = 0.9999 × 0.148 + 0.0001 × (-0.695) = 0.148
```

---

## 7. The Transformer Architecture

The transformer training benchmark uses a GPT-style decoder with MoE FFN layers.

```
Token IDs → Token Embed + Position Embed
    ↓
[Layer 0]
    ├── LayerNorm → Causal Self-Attention → Residual
    └── LayerNorm → MoE FFN → Residual
        ├── Router picks top-2 experts
        ├── Expert_0(hidden_state) + Expert_3(hidden_state)
        └── Weighted sum → output
    ↓
[Layer 1] (same structure)
    ↓
LayerNorm → LM Head → logits
```

Expert dispatch is vectorized: tokens are grouped by assigned expert, processed in batches, then scattered back with weights. No per-token Python loops.

### Learned Router Variant
```
MoE FFN with Learned Router:
  logits = hidden_state @ W_r          # (B, S, n_experts)
  probs = softmax(logits / 0.1)
  top_k_idx, top_k_weights = topk(probs, 2)
  → forward to selected experts (same dispatch)
```

### Profile Router Variant
```
MoE FFN with Profile Router:
  input_profiles = softmax(profiler(hidden_state))    # φ(x)
  similarities = cos_sim(input_profiles, expert_profiles)
  top_k_idx, top_k_weights = topk(similarities, 2)
  → forward to selected experts (same dispatch)
```

Both variants share identical expert FFNs. Only the routing mechanism differs.

---

## 8. Key Design Decisions

### Temperature (τ)
Controls routing sharpness. Low τ → confident routing (one expert dominates). High τ → soft blending. Default 0.1 was chosen because:
- 0.01-0.10: sharp routing, lowest MSE, 1 expert effectively active
- 0.50: transitions, 1.8 experts active, MSE increases
- 1.00+: soft routing, 2 experts active, MSE significantly higher

### Top-K (k=2)
Two experts selected per input. One primary (high weight), one secondary (near-zero weight for well-separated inputs, meaningful weight for ambiguous boundary inputs). This is standard MoE practice (DeepSeek, Mixtral both use k=2).

### Profile Dimension Count
4 dimensions for regression (one per cluster). 4 for transformer (one per domain). At production scale: 20-50 dimensions from standard benchmarks. The router's discriminative capacity grows exponentially with dimensions (2^d for binary profiles), so 50 dimensions can separate more experts than anyone will ever build.

---

## 9. Router Z-Loss: Why Profile-MoE Doesn't Need It

### What Router Z-Loss Is

In large-scale MoE training, the router produces raw scores called **logits** before applying softmax. In the learned router:

```
logits = W_r · x        ← unbounded real numbers, can be very large
probs = softmax(logits)  ← exp(large_number) → numerical overflow
```

When logits become too large (e.g., 80+), `exp(80)` exceeds floating-point range. Even with float32 precision, the softmax computation destabilizes the entire training run. DeepSeek-V3, GShard, and Switch Transformers all document this problem.

The **router z-loss** (from ST-MoE, Zoph et al. 2022) solves it:

```
L_z = (1/B) · Σ log²( Σ_j exp(x_i · W_r[:,j]) )
```

It penalizes large logit magnitudes. Added as an auxiliary loss term alongside the main language modeling loss. It keeps logits numerically stable but adds complexity: another loss term to tune, another coefficient to balance, and it slightly interferes with the main training objective.

DeepSeek-V3 still uses it (with a very small coefficient) even alongside their auxiliary-loss-free bias balancing. It's considered necessary infrastructure.

### Why Profile-MoE Doesn't Need It

Profile-MoE routing never produces unbounded logits:

```
similarities = cos_sim(input_profile, expert_profiles)  ← bounded to [-1, 1]
weights = softmax(similarities / τ)                      ← max exp(1/τ), stable
```

Cosine similarity is inherently bounded to [-1, 1]. With default τ=0.1, the maximum value fed into `exp()` is `1/0.1 = 10`. `exp(10) ≈ 22,026` — perfectly safe in float32. There is no scenario where the routing computation overflows.

This stability comes from architecture, not from an auxiliary loss term. The bounded input space is a property of using cosine similarity instead of learned linear projections. We get numerical stability for free.

### Comparison

| | Learned Router | Profile Router |
|---|---|---|
| Logit range | Unbounded (−∞, +∞) | Bounded [−1, 1] |
| Max exp() input | Can exceed 80 | ≤ 10 (at τ=0.1) |
| Numerical stability | Requires z-loss | Guaranteed by design |
| Auxiliary loss terms | Z-loss + load balance loss | None required |
| Training complexity | 2 extra hyperparameters | 1 (temperature only) |

---

## 10. Reproducibility

Every script is deterministic given a seed. Random states are fixed at module level or passed explicitly. Results will vary slightly between machines due to floating-point differences in sklearn/PyTorch, but routing accuracy and swap isolation ratios should be within 1% of reported values.

To regenerate all results from scratch:
```bash
bash run_all.sh    # hackathon branch
# or individually:
python mvp.py
python versioning_demo.py
python comparison_benchmark.py
python transformer_training.py
python generate_graphs.py
python export_findings.py
```
