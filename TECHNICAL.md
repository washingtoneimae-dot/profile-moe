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

### Per-Token vs Per-Prompt Routing

A critical architectural difference that surfaces at scale:

**DeepSeek routes PER TOKEN.** Every token independently goes through `W_r @ x`:

```
Prompt: "generate me a website for my umbrella company"

Token "generate"  → W_r @ x  →  Expert_42 (87%)
Token "website"   → W_r @ x  →  Expert_156 (91%)    ← different expert
Token "umbrella"  → W_r @ x  →  Expert_201 (73%)    ← different expert again
Token "company"   → W_r @ x  →  Expert_88 (68%)     ← yet another
```

Each token picks its own experts. There is no unified "this is a web development request" decision. The routing is microscopic — emergent from gradient signals, not declared. Nobody can inspect the router and say why Expert_201 was chosen for "umbrella."

**Profile-MoE profiles the ENTIRE PROMPT.** The profiler φ(x) reads the full input and outputs one profile vector shared across all MoE layers:

```
φ("generate me a website for my umbrella company")
  →  [web_dev: 0.72, business: 0.14, design: 0.08, general: 0.04, math: 0.01, law: 0.01]

Router at every layer:
  cos_sim(φ(prompt), expert_profiles)
  → Expert_web_dev (0.96), Expert_business (0.04)
```

The web dev expert handles the coding. The business expert adds context about companies. Both contribute. The decision is traceable: `web_dev: 0.72` matched the web expert's profile of `web: 0.93`.

**What this means for the profiler at scale:** The profiler needs training data — prompts labeled by required expertise. This is engineering, not research. A dataset like:

```
"Write a Python function to sort a list"    → [coding: 0.9, general: 0.1]
"Draft a privacy policy for my app"         → [law: 0.6, business: 0.3]
"Generate me a website for my umbrella co"  → [web_dev: 0.72, business: 0.14]
```

The router and experts are ready. The profiler is the only component that needs scale-specific training data. Everything else is proven and architecture-independent.

---

## 8. Key Design Decisions

### Temperature (τ)
Controls routing sharpness. Low τ → confident routing (one expert dominates). High τ → soft blending. Default 0.1 was chosen because:
- 0.01-0.10: sharp routing, lowest MSE, 1 expert effectively active
- 0.50: transitions, 1.8 experts active, MSE increases
- 1.00+: soft routing, 2 experts active, MSE significantly higher

### Adaptive Temperature (Boundary-Aware Routing)
When handling ambiguous inputs at decision boundaries, fixed-τ routing produces sharp, all-or-nothing weights even when two experts are similarly suited. The **adaptive temperature** mechanism softens routing at boundaries — when the top-2 experts have similar cosine similarities, τ increases so both contribute meaningfully rather than one dominating at 99%+.

**Important limitation:** τ cannot change *which* expert is selected — `softmax(s/τ)` preserves ranking for all τ > 0. Adaptive τ softens weights at boundaries but cannot fix a wrong routing decision. It is a blending aid, not a correction mechanism.

```python
top_gap = sims[0] - sims[1]
tau = base_tau + (1.0 - base_tau) * (1.0 - top_gap)
```

On well-separated inputs (gap ≈ 1.0): τ stays near base_tau — sharp routing. On boundary inputs (gap → 0): τ approaches 1.0 — both experts contribute. Enable with: `ProfileRouter(temperature=0.1, adaptive=True)`

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

DeepSeek-V3 still uses it (with a very small coefficient) even alongside their auxiliary-loss-free bias balancing. DeepSeek-V4 (April 2026) retains the V3 MoE framework with only minor activation function changes — the logit stability problem remains, and z-loss or equivalent numerical stabilization is still required. It's considered necessary infrastructure at every DeepSeek scale.

### Why Profile-MoE Doesn't Need It

Profile-MoE routing never produces unbounded logits:

```
similarities = cos_sim(input_profile, expert_profiles)  ← bounded to [-1, 1]
weights = softmax(similarities / τ)                      ← max exp(1/τ), stable
```

Cosine similarity is inherently bounded to [-1, 1]. With default τ=0.1, the maximum value fed into `exp()` is `1/0.1 = 10`. `exp(10) ≈ 22,026` — perfectly safe in float32. There is no scenario where the routing computation overflows.

This stability comes from architecture, not from an auxiliary loss term. The bounded input space is a property of using cosine similarity instead of learned linear projections. We get numerical stability for free.

> **Aside: Bias as expert health monitor.** If Profile-MoE adopts DeepSeek-V3's auxiliary-loss-free bias mechanism (`top-k = softmax(cos_sim + b_i)`), the bias values become a built-in monitoring dashboard. A bias that keeps dropping means the router is avoiding an expert despite its profile — the expert may be stale or miscalibrated. A bias that keeps rising means the router favors an expert beyond what its profile claims — the profile understates its capability. A bias hovering near zero means profile and reality match. No learned router can provide this because it has no explicit "expected vs actual" capability — only opaque weights.

### Comparison

| | Learned Router | Profile Router |
|---|---|---|
| Logit range | Unbounded (−∞, +∞) | Bounded [−1, 1] |
| Max exp() input | Can exceed 80 | ≤ 10 (at τ=0.1) |
| Numerical stability | Requires z-loss | Guaranteed by design |
| Auxiliary loss terms | Z-loss + load balance loss | None required |
| Training complexity | 2 extra hyperparameters | 1 (temperature only) |

---

## 10. Developer Observability: Seeing Inside the Router

Profile-MoE provides full visibility into every routing decision. Traditional MoE provides none.

### Verbose Mode (per-prediction trace)

Every prediction can be traced. `mvp.py` outputs:

```
INPUT:        (-0.26, 1.49)
INPUT PROFILE: {'code': '0.9877', 'creative': '0.0119', 'math': '0.0002', 'reasoning': '0.0003'}

EXPERT PROFILES:
  Expert_code:      [code=0.995, creative=0.002, math=0.001, reasoning=0.002] ← SELECTED
  Expert_creative:  [code=0.008, creative=0.990, math=0.001, reasoning=0.000]
  Expert_math:      [code=0.044, creative=0.036, math=0.919, reasoning=0.001] ← SELECTED
  Expert_reasoning: [code=0.017, creative=0.001, math=0.001, reasoning=0.981]

COSINE SIMILARITIES:
  Expert_code: 1.0000
  Expert_creative: 0.0205
  Expert_math: 0.0480
  Expert_reasoning: 0.0176

ROUTER: top-2 experts
  Expert_code: weight=0.9999, output=1.6845
  Expert_math: weight=0.0001, output=-0.6949
FINAL OUTPUT: 1.6844
```

A developer can see:
1. What the profiler thought the input was (code, 98.8% confidence)
2. Every expert's declared capability profile
3. The exact cosine similarity score for each expert
4. Which experts were selected and their final weights
5. Each expert's individual output before combination
6. The final weighted output

Given this trace, a developer can answer: "Why was Expert_code chosen?" → Because its profile matched the input profile with cosine similarity 1.0000. "Why was Expert_creative NOT chosen?" → Because its profile says it's bad at code (code=0.008), and the input looks like code.

### Swap Report (9-section before/after comparison)

When an expert is swapped, the `SwapReport` generates:

1. **Profile Comparison** — which capability dimensions changed most
2. **Safety** — per-domain MSE before/after with isolation ratio
3. **Efficacy** — target domain deep dive with % change
4. **Routing Behavior** — per-domain routing accuracy delta
5. **Confidence** — router weights on the swapped expert, cross-domain leak check
6. **Utilization Shift** — expert load redistribution
7. **Latency** — mean/P50/P99 comparison
8. **Edge Cases** — boundary inputs at 25%/50%/75% between domains
9. **Verdict** — pass/fail with specific issues

### What Traditional MoE Cannot Show

In a learned router, the only answer to "Why was Expert_42 chosen for token 'umbrella'?" is: "Because `W_r[42,:] @ x_umbrella` produced a high logit." You cannot inspect W_r to understand why. The knowledge is distributed across thousands of floating-point weights with no semantic meaning.

In Profile-MoE, the answer is: "Because the profiler scored this prompt as `web_dev: 0.72` and Expert_web_dev's calibrated profile shows `web_dev: 0.93`. Cosine similarity: 0.96." Every number has a name and a meaning.

---

## 11. Reproducibility

mvp.py, versioning_demo.py and comparison_benchmark.py are deterministic (fixed random states, verified across runs). transformer_training.py is NOT seeded: PPL and speed vary run-to-run (observed ranges: learned 9.5–10.4, profile 9.2–10.6; speed ratio 0.89×–1.03×). The committed `transformer_results.json` is the reference run. What is stable within 1% across runs: 99.88% routing accuracy, 38.4× swap isolation, 96.0% law routing.

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

---

## 12. Known Limitations & Honest Risks

This section exists so nobody else has to find these.

### 12.1 PPL Advantage May Not Scale

**Severity:** Medium. **Condition:** Training data > 1T tokens.

Our transformer benchmark shows Profile-MoE beating learned routing on perplexity (reference run: 9.3 vs 10.1, −8.1%). But the benchmark is not seeded (PPL observed across runs: 9.2–10.6 vs 9.5–10.4 — profile wins ~2 of 3), and it was measured with 8 epochs on a tiny multi-domain dataset. Declared profiles act as a structural prior — they give the model useful information that the learned router must discover from gradients. On small data, this is an advantage. On TB-scale training data with millions of gradient steps, the learned router may close or reverse this gap.

**Resolution:** Train both routing mechanisms on a 1B+ token corpus with a 100M+ parameter model. Until then, claim "comparable accuracy ceiling" rather than "better accuracy."

### 12.2 Profiler Training Data Does Not Exist

**Severity:** High. **Condition:** Real-world deployment.

The profiler φ(x) needs training data: prompts labeled by required expertise. No public dataset exists for this. The profiler is the single largest gap between proof-of-concept and production.

**Resolution:** Build a prompt→skills dataset using existing LLMs as labelers, or derive profile labels from benchmark performance. This is engineering, not research, but substantial.

### 12.3 Adaptive Temperature Sample Size

**Severity:** Low. **Condition:** Publishing the 32% figure.

Validated on 3 geometric outlier samples. The mechanism is sound (gap-based τ adjustment), but the specific percentage is anecdotal, not statistical.

**Resolution:** Run with 10+ random seeds to generate a meaningful sample of boundary flips.

### 12.4 Per-Prompt vs Per-Token Routing

**Severity:** Low. **Condition:** Tasks requiring token-level expert granularity.

DeepSeek routes per token, per layer. We route per prompt, shared across layers. For most use cases, prompt-level profiling is sufficient. For tasks where different tokens need different expertise, add per-layer or per-token bias adjustment.

### 12.5 Overlapping Expert Domains

**Severity:** Low. **Condition:** Two experts share the same benchmark score.

Router cannot distinguish identical profiles. The bias mechanism (or random tie-breaking) determines selection. Resolve with finer benchmarks or deliberate bias use (A/B testing, cost-aware routing).

### 12.6 DeepSeek-V4 Hash Routing Not Compared

**Severity:** Low. **Condition:** Comparing parameter counts to V4.

V4 uses hash routing for first 3 layers — our estimate overcounts by ~5-10%. The architectural advantage (zero learned routing parameters) is unchanged.

### 12.7 Profile Staleness After Fine-Tuning

**Severity:** Medium. **Condition:** Continuous expert improvement.

Profiles reflect calibration-time capability. Detectable via bias monitor, requires recalibration discipline.

### 12.8 Speed Measured on CPU, Not GPU

**Severity:** Low. **Condition:** Production GPU inference.

Measured on 177K-param CPU model. GPU tensor cores optimize matrix multiplies differently. Needs GPU benchmark at production scale.


## 13. Reproducibility
