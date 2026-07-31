# Profile-MoE MVP: Minimal Prediction Calculator

## Philosophy
Don't build a smart MoE. Build the smallest possible functional unit that proves:
1. Profile-based routing works
2. Experts are swappable without retraining
3. We can observe every decision (which expert, why, how fast)

## The Setup: Synthetic Regression

### Data: 4 Gaussian clusters in 2D space
```
Cluster A ("code-like")   → centered at (0, 0), label = f(x,y) = x² + y
Cluster B ("math-like")   → centered at (5, 0), label = f(x,y) = sin(x) * y
Cluster C ("creative")    → centered at (0, 5), label = f(x,y) = x * cos(y)
Cluster D ("reasoning")   → centered at (5, 5), label = f(x,y) = sqrt(x² + y²)
```

Each cluster = a different underlying function. An expert trained on one cluster
will perform well on that cluster and poorly on others. This mimics domain specialization.

### Profile Dimensions (3-dim, for simplicity)
```
[cluster_A_proximity, cluster_B_proximity, cluster_C_proximity, cluster_D_proximity]
```
Each expert's profile = how well it performs on each cluster (normalized).

Or simpler: each expert's profile = [is_good_at_code_like, is_good_at_math_like, is_good_at_creative, is_good_at_reasoning]

## Architecture

```
INPUT (x, y) ──┬──► Prompt Profiler ──► input_profile [0.8, 0.1, 0.05, 0.05]
                │        (tiny MLP)
                │
                ├──► Expert A (trained on cluster A)
                ├──► Expert B (trained on cluster B)
                ├──► Expert C (trained on cluster C)
                └──► Expert D (trained on cluster D)
                         │
                     profiles:        Router: cosine_sim(input_profile, expert_profile)
                     A: [1,0,0,0]              ↓
                     B: [0,1,0,0]         softmax → top-2
                     C: [0,0,1,0]              ↓
                     D: [0,0,0,1]         output = w_a·E_a(x,y) + w_b·E_b(x,y)
```

## Observable Outputs (every prediction)

```
INPUT:        (0.2, 0.1)
TRUE LABEL:   0.14  (cluster A function: x²+y)
INPUT PROFILE:[0.92, 0.03, 0.02, 0.03]  ← profiler says: "this looks like cluster A"

EXPERT PROFILES:
  Expert A [code]:    [1.00, 0.05, 0.02, 0.01]  ← trained on A
  Expert B [math]:    [0.03, 0.97, 0.04, 0.02]
  Expert C [creative]:[0.02, 0.03, 0.96, 0.03]
  Expert D [reason]:  [0.04, 0.02, 0.03, 0.94]

COSINE SIMILARITIES:
  A: 0.998  ← WINNER
  B: 0.045
  C: 0.038
  D: 0.052

ROUTER: selected Expert A (weight: 0.97), Expert D (weight: 0.03)
OUTPUT:  0.15  (A contributed 0.148, D contributed 0.002)

TIME: 0.3ms routing + 0.7ms expert computation = 1.0ms total
```

## The Swap Test

After training:
1. Replace Expert A (cluster A) with Expert A' (trained on cluster A but different function)
2. Update Expert A's profile to reflect its new performance
3. Run the same input through — router automatically uses A' for cluster A inputs
4. Zero retraining of router or other experts

## What We Benchmark

| Metric | What | Baseline |
|--------|------|----------|
| Routing accuracy | % of inputs routed to correct expert | 25% random |
| MSE by cluster | Per-cluster error | Single model, random routing |
| Swap fidelity | MSE before/after swap (should only affect swapped cluster) | N/A |
| Expert utilization | % of inputs each expert handles | 25% each ideal |
| Routing overhead | Time spent in profiler+router vs expert computation | Profiled |

## Implementation

### Expert: tiny MLP
```python
class Expert(nn.Module):
    def __init__(self, input_dim=2, hidden=16):
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.profile = None  # set after training

    def calibrate_profile(self, test_data):
        """Run expert on each cluster's test set, build profile from MSE"""
        mse_per_cluster = []
        for cluster_data in test_data:
            pred = self(cluster_data.x)
            mse = F.mse_loss(pred, cluster_data.y)
            mse_per_cluster.append(mse)
        # Invert MSE → skill score (lower MSE = higher skill)
        skills = 1.0 / (torch.tensor(mse_per_cluster) + 1e-6)
        self.profile = skills / skills.sum()  # normalize
```

### Router: pure similarity math
```python
class ProfileRouter:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def route(self, input_profile, expert_profiles, k=2):
        # Cosine similarity between input and each expert
        sims = F.cosine_similarity(
            input_profile.unsqueeze(0), 
            expert_profiles
        )
        weights = F.softmax(sims / self.temperature, dim=0)
        top_k = torch.topk(weights, k)
        return top_k.indices, top_k.values / top_k.values.sum()
```

### Prompt Profiler: trainable but simple
```python
class PromptProfiler(nn.Module):
    def __init__(self, input_dim=2, hidden=8, profile_dim=4):
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Softmax(dim=-1)
        )
```

## Expected Results (Hypothesis)

1. Router achieves >85% accuracy (vs 25% random) on clean cluster inputs
2. Borderline inputs (between clusters) get soft routing (both experts activated)
3. Swap test: replacing Expert A only affects cluster A's MSE; other clusters unchanged
4. Routing overhead is negligible (~5-10% of total compute)
5. Profile calibration (running test sets) takes seconds, not hours of retraining

## File Structure
```
profile-moe/
├── PLAN.md
├── README.md
├── mvp.py              # Single file: everything in ~300 lines
├── benchmark.py        # Run the benchmarks
├── visualize.py        # Plot: clusters, routing decisions, profiles
└── results/            # Output charts and logs
```

## Why This Proves the Concept

This IS the basic functional unit. An LLM is just this scaled up:
- Instead of 2D input → text tokens embedded in high-dim space
- Instead of 4 experts → 8/16/64 experts
- Instead of regression → next-token prediction
- Instead of cluster functions → domain knowledge (code, math, law, etc.)

The routing mechanism — profile matching — is identical. If it works here, it works at scale.
