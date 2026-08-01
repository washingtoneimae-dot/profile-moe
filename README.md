# Profile-MoE

**Swappable Expert Infrastructure with Profile-Based Routing**

Traditional MoE routers are learned — co-trained with experts, can't swap without retraining.
Profile-MoE replaces the learned router with profile-vector similarity matching.
Each expert carries a calibrated benchmark profile. The router computes cosine similarity —
pure math, zero learned parameters. Same experts, same speed, swappable by design.

---

## Quick Verification (3 minutes)

One command. No setup. Clones and proves everything.

```bash
git clone <this-repo> -b hackathon
cd profile-moe
bash setup.sh     # creates venv, installs deps (first time only)
bash run_all.sh   # runs all 6 proofs, saves output/
```

That's it. After 3 minutes you have:

```
output/
├── 01_mvp_output.txt                    ← Core proof: 99.88% routing, 38.4x swap isolation
├── 02_versioning_output.txt             ← Adding 5th expert: 96% law routing
├── 03_comparison_output.txt             ← vs DeepSeek: swappable, zero router params
├── 04_transformer_arch_output.txt       ← Transformer architecture: identical speed
├── 05_transformer_training_output.txt   ← Full training: 9.3 PPL vs 10.1 learned
├── 06_boundary_solutions_output.txt      ← Adaptive τ: 32% boundary error fix
├── 07_graphs_output.txt                  ← Graph + FINDINGS.xlsx generation
├── results.json                         ← Raw MVP data
├── transformer_results.json             ← Raw transformer data
├── versioning_demo.xlsx                 ← 6-sheet versioning workbook
└── comparison_benchmark.xlsx            ← 2-sheet DeepSeek comparison
```

Plus `graphs/*.png` (5 charts) and `FINDINGS.xlsx` (7-sheet master workbook).

### Key Numbers at a Glance

| Metric | Value | Source |
|--------|-------|--------|
| Routing accuracy | 99.88% | 01_mvp_output.txt |
| Swap isolation | 38.4× | 01_mvp_output.txt |
| Law routing (new domain) | 96.0% | 02_versioning_output.txt |
| Transformer PPL (Profile-MoE) | 9.3 | 05_transformer_training_output.txt |
| Transformer PPL (Learned Router) | 10.1 | 05_transformer_training_output.txt |
| Speed ratio | 0.999× | 05_transformer_training_output.txt |
| Router learned params | 0 | 05_transformer_training_output.txt |

### Individual Scripts

If you want to run one proof at a time:

```bash
python mvp.py                      # Core proof: routing, swap, temperature
python versioning_demo.py          # Adding a 5th expert, profile versioning
python comparison_benchmark.py     # Profile-MoE vs DeepSeek-style learned router
python transformer_benchmark.py    # nanoGPT MoE architecture (pure numpy)
python transformer_training.py     # Full transformer training benchmark (PyTorch)
python generate_graphs.py          # Publication-quality charts
python export_findings.py          # Generate FINDINGS.xlsx
```

---

## What Each Script Proves

### 1. `mvp.py` — Core Proof (30 seconds)

The minimal functional unit. 4 regression experts, profile-based routing.

**Look for:**
```
FULL EVALUATION
  code        : MSE=0.055  routing_acc=100.00%
  math        : MSE=0.191  routing_acc=99.50%
  creative    : MSE=0.027  routing_acc=100.00%
  reasoning   : MSE=0.021  routing_acc=100.00%
  OVERALL ROUTING ACCURACY: 99.88%
```

**Then the swap test:**
```
SWAP TEST
  ✓ ISOLATION CONFIRMED: swap impact is 38.4x larger than max spillover
```

One expert is replaced with a different one. Only that expert's domain changes. All others stay flat. The router is NOT retrained — it adapts automatically to the new profile.

**Also outputs:** `results.json` with raw evaluation data.

---

### 2. `versioning_demo.py` — Adding a New Domain (30 seconds)

A company wants to add a "law" expert to their existing 4-expert pool. Demonstrates why profile versioning matters.

**Look for:**
```
Law expert calibrated on v1 (4-dim): Expert_law: [code=0.573, creative=0.129, ...]
⚠ PROBLEM: Law expert's law capability is UNMEASURED in v1 profile!

[After v2 upgrade — adding 'law' as 5th dimension:]
  Law routing accuracy: 96.0% (144/150 routed to Expert_law)
  Law MSE: 5.165 → 0.265 (-94.9%)
```

**Also outputs:** `versioning_demo.xlsx` — 6-sheet workbook with raw predictions, recalibration cascade, error distributions.

---

### 3. `comparison_benchmark.py` — Head-to-Head vs DeepSeek (30 seconds)

Same data, same experts. Only routing mechanism differs.

**Look for:**
```
  DeepSeek-MoE: SWAPPABLE? NO — needs router retraining
  Profile-MoE:  SWAPPABLE? YES — profile update only
  Profile-MoE Router Parameters: ZERO (pure math)
  DeepSeek Router Parameters: 512 (learned, per layer)
```

**Also outputs:** `comparison_benchmark.xlsx` — 2-sheet comparison workbook.

---

### 4. `transformer_training.py` — Full Transformer Benchmark (1-2 minutes)

A 177K-param nanoGPT-style transformer with MoE FFN layers. Trains both routing mechanisms on multi-domain text (code + math + stories + wiki). **Requires PyTorch.**

**Look for:**
```
FINAL RESULTS
                     Learned Router   Profile Router
  Overall PPL              10.1              9.3
  Speed (ms)               4.30             4.30
  Speed (tok/s)           59570            59492
  Speed ratio:             0.999x  ← identical
  Router params              512                0
  Swappable?                  NO              YES
  Train time (s)             8.6              8.3
```

Profile-MoE matches or beats learned routing on every metric while using zero learned routing parameters and supporting hot-swapping.

**Also outputs:** `transformer_results.json` with full per-domain PPL and speed data.

---

### 5. `generate_graphs.py` — Charts (5 seconds)

Generates 5 publication-quality PNGs from the data files:

| Graph | What It Shows |
|-------|--------------|
| `graphs/swap_isolation.png` | Before/after MSE bars. 38.4× isolation annotated. |
| `graphs/ppl_comparison.png` | Profile-MoE vs Learned Router per domain. −8% to −17% improvements. |
| `graphs/profile_heatmap.png` | Expert × Domain capability matrix. Each expert >0.95 on its domain. |
| `graphs/routing_accuracy.png` | 99.9% routing vs 25% random baseline. |
| `graphs/speed_comparison.png` | Identical speed (0.999×) + ZERO router params vs 512. |

---

## Master Findings

**[FINDINGS.xlsx](FINDINGS.xlsx)** — 7-sheet Excel workbook consolidating every result:

| Sheet | Contents |
|-------|----------|
| Executive Summary | All 4 tickboxes verified, key numbers, architectural advantages |
| 1-Regression MVP | 99.88% routing, swap isolation, expert profiles |
| 2-Transformer Training | PPL comparison (9.3 vs 10.1), speed, swap test |
| 3-Versioning Demo | Law expert addition, recalibration cascade, agentic swarm speedup |
| 4-DeepSeek Comparison | 13-dim architectural comparison table |
| 5-Theory | Formal theorems, failure modes, mitigations |
| 6-Raw Data | Full JSON dumps for programmatic analysis |

---

## What This Is

**Not a better expert.** The expert FFN (`W₂·GELU(W₁·x)`) is identical to traditional MoE. You can take trained weights from any MoE expert, run them through benchmarks, and attach a calibrated profile. The expert's computation doesn't change.

**A better way to CONNECT experts.** The learned router (`W_r·x`) is replaced with `cos_sim(φ(x), expert_profiles)`. Zero learned routing parameters. Pure similarity math. The profile is the API between expert and router.

**This means:**
- Swap experts without retraining (seconds, not days)
- Add new domains without retraining the router (minutes, not weeks)
- Source experts from third parties (they submit model + benchmark scores)
- Every routing decision is traceable (which expert, why, what weight)
- Router parameter count stays flat as experts and layers scale (384× fewer at DeepSeek-V4 scale)
- **Adaptive temperature**: softens routing weights at decision boundaries, letting both experts contribute on ambiguous inputs

---

## Theory

**[THEORY.md](THEORY.md)** — Formal proofs:
- **Routing Isolation Theorem**: swapping expert i does not affect routing for experts j≠i
- **Localized Impact Theorem**: only inputs originally routed to i are affected
- **Scalability analysis**: O(n·d) routing, d=50 discriminates 10^15 experts, router parameter count comparison
- **Failure modes**: profile collapse, profiler drift, dimension collapse with detection and mitigation

---

## Prior Art

**[DEEPSEEK_REFERENCE.md](DEEPSEEK_REFERENCE.md)** — DeepSeekMoE/V2/V3 architecture reference: fine-grained experts, shared expert isolation, auxiliary-loss-free load balancing, Multi-Head Latent Attention.

**[PLAN.md](PLAN.md)** — MVP scope, benchmarks, prior art comparison against Symbolic-MoE, ModuleFormer, ICL-Router.

---

## Repository Structure

```
profile-moe/
├── README.md                    ← You are here
├── THEORY.md                    ← Formal proofs
├── PLAN.md                      ← Architecture spec + prior art
├── DEEPSEEK_REFERENCE.md        ← DeepSeek paper reference
│
├── mvp.py                       ← Core proof (99.88% routing, 38.4× swap isolation)
├── versioning_demo.py           ← Adding 5th expert + profile versioning
├── comparison_benchmark.py      ← Profile-MoE vs DeepSeek-style comparison
├── transformer_benchmark.py     ← nanoGPT MoE (pure numpy, no training)
├── transformer_training.py      ← Full transformer training (PyTorch)
├── generate_graphs.py           ← Publication-quality charts
├── export_findings.py           ← Generate FINDINGS.xlsx
│
├── graphs/                      ← 5 PNG charts
│   ├── swap_isolation.png
│   ├── ppl_comparison.png
│   ├── profile_heatmap.png
│   ├── routing_accuracy.png
│   └── speed_comparison.png
│
├── FINDINGS.xlsx                ← Master 7-sheet workbook
├── results.json                 ← MVP raw data
├── transformer_results.json     ← Transformer training raw data
├── versioning_demo.xlsx         ← Versioning demo raw data
└── comparison_benchmark.xlsx    ← DeepSeek comparison raw data
```

---

## Dependencies

```
numpy scikit-learn matplotlib openpyxl torch
```

All scripts print their findings to stdout and export structured data (JSON/XLSX) for analysis.
