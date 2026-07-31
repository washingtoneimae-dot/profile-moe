# Profile-MoE

**Swappable Experts with Profile-Based Routing**

Traditional MoE routers are learned — they're co-trained with experts and can't swap experts without retraining. Profile-MoE replaces the learned router with **profile-vector similarity matching**. Each expert carries a benchmark-calibrated profile vector. The router computes cosine similarity between the input's profile and each expert's profile — pure math, zero learned parameters.

## Why

- **Swappable experts**: Replace any expert, recalibrate its profile (seconds), done. No router retraining.
- **Observable routing**: Every decision is traceable — which expert was picked, why, with what weight.
- **Infrastructure, not architecture**: Profiles are the API between experts and router.

## Theory → [THEORY.md](THEORY.md)

Formal proofs for routing isolation, swap guarantees, and scalability analysis (O(n·d) routing, d=50 dimensions can discriminate millions of experts).

## MVP → [mvp.py](mvp.py)

Minimal prediction calculator proving the concept on synthetic 2D regression:

```bash
python mvp.py
```

Outputs:
- Per-prediction verbose routing (which expert, why, timing)
- Full evaluation: routing accuracy, per-cluster MSE
- Swap test: before/after metrics showing 38.4x isolation
- Temperature analysis: how τ affects routing sharpness
- Expert scaling: performance at 2, 3, 4 experts

## Results

| Metric | Value |
|--------|-------|
| Routing accuracy | 99.88% (random: 25%) |
| Swap isolation | 38.4x |
| Avg prediction time | 0.53ms |
| Global MSE | 0.074 |

## Architecture

```
INPUT (x,y) → Profiler → input_profile [0.92, 0.03, 0.02, 0.03]
                                ↓
              Router: cos_sim(input_profile, expert_profiles)
                                ↓
              top-2 experts → weighted combination → OUTPUT

Expert profile = calibrated benchmark scores (run expert on test sets)
Router = pure similarity math (no learned parameters)
Swap = replace expert + update its profile (seconds, no retraining)
```

## Plan → [PLAN.md](PLAN.md)

MVP scope, benchmarks, prior art comparison, and roadmap to production LLM integration.
