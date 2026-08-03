# Profile-Routed Modular MoE — Technical Documentation

Status: research prototype. This document separates **proven** (verified by test, reproducible), **bounded/conditional** (proven under stated conditions, with known limits), **designed** (specified but not yet empirically tested), and **open** (hypothesis, not yet validated). Nothing below is stated more strongly than its evidence supports.

---

## 1. Introduction

Standard Mixture-of-Experts architectures (Switch Transformer, GShard, Mixtral, DeepSeek-MoE) route each input through a **learned gate**: a small neural network trained end-to-end, jointly with the experts it routes to. This works well, but it creates a specific coupling: the gate's decision boundaries are a function of the exact expert set it was trained with. Changing that expert set — adding a capability, removing one, swapping one for an improved version — generally requires retraining the gate, with the usual risks of retraining anything jointly optimized: cost, and the possibility of disturbing behavior that already worked (catastrophic forgetting, §6.1).

This project asks a narrower, testable question: **if routing is decoupled from training entirely** — replaced with similarity between a calibrated profile vector (what does this expert measure as good at) and an input's own profile — **what do you gain, what do you lose, and exactly where does it break?**

The short answer, argued and tested in this document: you gain a real, provable isolation guarantee for the common case (replacing an expert's function), you gain full interpretability of every routing decision, and you lose nothing on either of those counts compared to a learned gate. What you do *not* get for free is isolation when *adding* a genuinely new capability — that operation inherits a well-known failure mode from classifier theory (catastrophic forgetting / decision-boundary shift), and this document spends most of its length characterizing exactly how badly, and how much of it can be fixed.

---

## 2. Overview

### 2.1 Core components

- **Anchor**: a small, fixed base model (parameters frozen for the purposes of this document; the routing and expert-composition layer sits on top of it).
- **Experts**: lightweight, swappable prediction units (in production, LoRA-style adapters; in this document's test suite, small regressors/classifiers standing in for them). Each expert has a **calibrated profile vector** — a measured-competence score per known domain — rather than a learned routing embedding.
- **Profiler**: a lightweight classifier that maps an input to a profile vector in the same space as expert profiles (i.e., "which domain does this input look like it belongs to").
- **Router**: computes cosine similarity between the input's profile and every expert's profile, selects the top-k experts, and blends their outputs by softmax-weighted similarity.

### 2.2 Deployment shape (designed, not tested in this document)

The architecture is intended to scale across two hardware tiers by varying only *how many* experts are resident in fast memory at once, not the routing mechanism itself:

- **Small-scale**: consumer hardware, experts streamed from system RAM/disk into GPU memory on demand.
- **Large-scale**: server-class hardware, many experts resident simultaneously, serving concurrent requests with different expert selections in the same batch.

This tiering, and the batched multi-adapter serving problem it implies, is **not original engineering validated in this document** — it is the same problem efficient multi-LoRA serving systems (Punica, S-LoRA) already solve well, and this project should build on that prior art rather than re-derive it (§13). What *is* original here is everything downstream of "which experts are selected," not the memory/batching mechanics of serving them.

---

## 3. Comparison with Traditional (Learned-Gate) MoE

| Property | Traditional MoE (Switch/GShard/Mixtral/DeepSeek-style) | Profile-Routed MoE (this project) |
|---|---|---|
| Routing decision | Learned linear gate, trained jointly with experts via backprop | Cosine similarity between calibrated profile vectors — no gradient through the base model |
| Load-balancing loss (z-loss / auxiliary loss) | Required — unconstrained logits collapse onto a few experts without it | **Not required** — similarity is bounded in [-1, 1] by construction; see §4.2 (structural, not empirical) |
| Adding a new expert to an existing pool | Requires retraining the gate (fully or partially) across old + new experts jointly | No retraining of existing experts or the frozen base profiler required, **but** naively retraining the profiler jointly reproduces the same catastrophic-forgetting risk (§6) — the correct procedure is the gated approach in §6.3, not joint retraining |
| Swapping an expert's function (same slot) | Not naturally isolated — the gate's boundary for that slot was co-optimized with the old expert's actual behavior | **Provably isolated** (§5) — untouched experts show 0.00% collateral change in every test run |
| Interpretability of a routing decision | Opaque — a decision is "whatever the trained gate's forward pass produced," not decomposable into a reason | Every decision traces to a specific calibration score: "expert X was chosen because it scored Y on domain benchmark Z" |
| Parameter growth | Gate parameters scale with expert count × hidden dimension | Profiler size is independent of expert count — same classifier regardless of how many experts exist |
| Genuinely novel/subtle routing signal at large scale | **Advantage: traditional MoE** — a jointly-trained gate can, given enough data, discover routing signal humans wouldn't think to calibrate for | **Disadvantage here** — routing quality is bounded by what the calibration benchmark actually measures; this is an honest, unresolved trade-off, not a settled win (§11, limitation 5) |
| Isolation guarantee under expert *addition* | None claimed or typically characterized in the literature reviewed | **Bounded/conditional guarantee** (§6.3, Theorem 3) — provably safe below a calibrated threshold, provably unsafe (and provably *unsafeable*) above it for genuinely ambiguous inputs |
| Multi-adapter batched serving at scale | Solved in prior art (Punica's SGMV kernels, S-LoRA's unified paging) | Not re-solved here; intended to build on the same prior art (§2.2) |

The honest summary: traditional MoE trades flexibility for the ability to discover routing signal end-to-end at scale. This architecture trades that discovery ability for a **provable, calibratable, inspectable isolation guarantee** that traditional MoE does not have an equivalent of. Neither is strictly better; they are different points on a flexibility/guarantee trade-off, and this document's job is to state exactly where this project's side of that trade-off currently sits.

---

## 4. Formal Mechanism Definitions

Notation: domain set $D = \{d_1, ..., d_n\}$. Expert $e_i$ has model $f_i$ and profile $p_i \in \Delta^{|D|}$ (the probability simplex). Profiler $\pi: X \to \Delta^{|D|}$ maps an input to a profile vector $\phi(x) = \pi(x)$.

**4.1 Profile calibration.** For expert $e_i$ and domain benchmark set $B_d$ for each $d \in D$:

$$\text{MSE}_i(d) = \frac{1}{|B_d|}\sum_{(x,y)\in B_d} (y - f_i(x))^2$$

$$p_i[d] = \frac{1/(\text{MSE}_i(d) + \epsilon)}{\sum_{d' \in D} 1/(\text{MSE}_i(d') + \epsilon)}$$

(Implementation: `Expert.calibrate()` in `scripts/shared_data.py`.)

**4.2 Routing.** Cosine similarity between input profile and each expert's profile:

$$\text{sim}(x, e_i) = \frac{\phi(x) \cdot p_i}{\|\phi(x)\| \, \|p_i\|}$$

Top-k selection, softmax-weighted blend at temperature $\tau$:

$$w_i(x) = \frac{\exp(\text{sim}(x,e_i)/\tau)}{\sum_{j \in \text{top-}k(x)} \exp(\text{sim}(x,e_j)/\tau)}, \qquad y(x) = \sum_{i \in \text{top-}k(x)} w_i(x)\, f_i(x)$$

(Implementation: `cosine_top1()` in `scripts/shared_data.py`; adaptive-$\tau$ variant in `boundary_solutions.py`.)

**4.3 Gated domain addition (the §6.3 fix).** For a new domain $d_{new}$, train a one-vs-rest detector $g_{new}: X \to [0,1]$ independently of $\pi$. Calibrate threshold $\theta$ on held-out old-domain data $C$ so that the false-positive rate is bounded by target $\alpha$:

$$\theta = \text{Percentile}_{100(1-\alpha)}\big(\{g_{new}(x) : x \in C\}\big)$$

Gated profile injection:

$$\gamma(x) = \begin{cases} g_{new}(x) & \text{if } g_{new}(x) \ge \theta \\ 0 & \text{otherwise} \end{cases} \qquad \phi'(x) = \big[(1-\gamma(x))\cdot\phi(x),\ \gamma(x)\big]$$

**4.4 Multi-addition (Bonferroni) correction.** For $M$ simultaneous additions targeting an aggregate false-positive rate $\alpha_{\text{total}}$:

$$\alpha_{\text{individual}} = \alpha_{\text{total}} / M, \qquad \theta_j = \text{Percentile}_{100(1-\alpha_{\text{individual}})}\big(\{g_j(x) : x \in C\}\big) \ \ \forall j$$

**4.5 Theorem statements**

*Theorem 1 (Swap Isolation).* Let $E = \{e_1,...,e_n\}$ share a fixed profile dimensionality $D$. Replace $e_i$ with $e_i'$ (same domain slot; different model), holding all other experts, their calibration data, and $\pi$ fixed. Then for every $j \ne i$, $p_j$ is unchanged (by 4.1, $p_j$ depends only on $e_j$'s own MSE values), hence $\text{sim}(x,e_j)$ is unchanged for all $x$, hence relative routing among $\{e_j : j\ne i\}$ is unchanged. *Verified: §5.*

*Theorem 3 (Bounded Addition Isolation).* Let $\pi_{base}$ be frozen. For any $x$ with $g_{new}(x) < \theta$ (4.3), $\gamma(x)=0$, so $\phi'(x)$'s new-domain component is exactly zero and top-1 routing among the original experts is unchanged. *Verified: §6.3.*

*Corollary (Impossibility for ambiguous inputs).* No choice of $\theta$ in 4.3 can extend this guarantee to inputs $x$ where $P(d_{new}|x) \approx P(d_{true}|x)$ (genuine overlapping support) without reducing recall on genuine $d_{new}$ inputs at the same score level — a standard precision/recall trade-off of thresholded binary classification on non-separable data, not a defect in the construction. *Verified: §6.3, §6.4.*

---

## 5. Proven: Swap Isolation

Replacing one expert's underlying function, at fixed profile dimensionality, does not perturb routing or prediction quality for other experts (Theorem 1, §4.5).

- **The invariant property, not a fixed constant**: in every run tested, the swapped domain's own error changes substantially (magnitude varies by how different the replacement is — one canonical run showed +792%), while every untouched domain shows **exactly 0.00% change**. Reproduce: `scripts/addition_isolation_suite.py`, `section_2_swap_isolation()`.
- Also verified inside an already-provisioned 5-dimension space (not just the original 4-dimension case): swapping one expert produced 0.00% change in the other four domains' error, and 0/4 top-1 routing decisions changed.
- **Practical implication**: if a system's dimension set is provisioned comprehensively up front, routine expert improvement/upgrade work falls entirely inside this proven-safe operation. The riskier operation (below) is only triggered by genuinely new, unanticipated domains.

---

## 6. The Addition Problem

### 6.1 What breaks, and why

Adding a new domain by jointly retraining the profiler across old + new domains together causes two distinct failure modes:

1. **New-domain contamination**: existing inputs pick up spurious affinity for the new domain.
2. **Old-boundary shift**: retraining the classifier jointly can shift decision boundaries *among the old domains themselves*, with zero relation to the new domain. Confirmed directly: one flip case had a measured new-domain-detector score of 0.002 (effectively zero contamination) yet still flipped, because retraining moved an unrelated old boundary.

This is a specific instance of a well-documented phenomenon: **catastrophic forgetting in class-incremental learning** (McCloskey & Cohen, 1989 and the extensive literature since). It is not a novel discovery; the contribution here is identifying that profile-routed MoE inherits this failure mode at the *routing* level (misdirecting an entire query to the wrong specialist, not just misclassifying a label) and quantifying it for this architecture specifically.

### 6.2 Measured severity (original synthetic benchmark)

Adding one new domain (law) to a 4-domain pool, then recalibrating:

| domain | MSE before | MSE after | change |
|---|---|---|---|
| code | 0.125 | 0.215 | +71.8% |
| creative | 0.038 | 0.190 | +404.2% |
| math | 1.183 | 1.183 | +0.01% |
| reasoning | 0.016 | 0.031 | +89.2% |
| law (target) | 5.165 | 0.265 | −94.9% |

Of the non-law samples that lost their correct top-1 expert to the new domain, **7 of 8 got measurably worse**, with mean error roughly tripling (0.66 → 2.10). Source: pre-existing repository artifact `versioning_demo.xlsx` / `versioning_demo.py`, not authored in this review pass.

### 6.3 Bounded fix (Theorem 3, conditional)

Standard fix pattern from the **open-set recognition** literature (OpenMax family, Bendale & Boult ~2016), applied to this routing context: freeze the base profiler permanently; add each new domain via an independently-trained one-vs-rest detector, gated by a calibrated threshold (formulas: §4.3).

**Guarantee**: for any input whose new-domain detector score falls below the calibrated threshold, the addition provably does not change its top-1 routing decision.

Canonical run (reproduce: `scripts/addition_isolation_suite.py`, `section_3_addition_and_gating()`): **3 of 4** contamination-driven flips fixed; genuine new-domain recall preserved at 88.7%. The one unfixed case sits well inside the honest detector's own high-confidence range for the new domain — a well-calibrated detector genuinely cannot separate it, at any threshold, without sacrificing recall elsewhere. (An earlier development-session run found 2/3 fixed at 96.0% recall on a differently-realized dataset — see §12: the pattern, not the exact ratio, is the reproducible claim.)

**Corollary (provable limit, not an engineering gap)**: stated formally in §4.5.

### 6.4 Is the limit a modeling-capacity problem? Tested — no.

Hypothesis: maybe the "irreducible" cases aren't irreducible, just under-modeled by a small gate.

Tested directly: a 33-parameter gate vs. a 10,753-parameter gate (326×) on identical data. Reproduce: `scripts/capacity_ablation.py` — this one reproduces exactly, to the decimal, every run (see §12).

| | small (33 params) | large (10,753 params) |
|---|---|---|
| overall AUC | 0.99604 | 0.99614 |
| accuracy on genuinely-ambiguous subset (2.53% of test set) | 63.6% | 62.5% |
| overall accuracy | 97.610% | 97.570% |

Cases where the large model fixed a small-model error (32) were almost exactly balanced by cases where the small model was right and the large one wasn't (36) — no systematic advantage. **Conclusion: the ambiguity is a Bayes-error property of the data (genuine overlapping support), not a capacity bottleneck.** More parameters, without co-training, do not resolve it. This independently confirms an existing finding in this project (`possibility.md`'s profiler-depth ablation found the same null result in a transformer setting).

*Caveat: tested only in low-dimensional synthetic space; unconfirmed whether high-dimensional semantic embeddings behave identically.*

### 6.5 Mechanism, further characterized

Multi-seed test (5 independent training runs, same data distributions). Reproduce: `scripts/addition_isolation_suite.py`, `section_3_5_multi_seed_stability()`.

- Canonical run: **5 of 9** total observed flip points recurred in **5/5 seeds** — a stable core tied to specific, consistently poorly-resolved regions. The remainder recurred in only 1–2/5 seeds — a shifting penumbra, more sensitive to individual training runs. (An earlier run found a smaller stable core, 2 of 6 points — the *existence* of a stable-core/shifting-penumbra split is the reproducible finding; its exact size is not.)

Margin analysis on two stable-core points from one documented run (bug caught and fixed before reporting — an initial pass had a profile-vector ordering mismatch that produced internally inconsistent numbers; not yet re-run against the canonical script's own stable-core points, flagged here rather than silently left out):

| case | winner | winner sim. | correct expert sim. | margin |
|---|---|---|---|---|
| math #108 | law | 0.883 | 0.476 | 0.408 |
| creative #134 | law | 0.865 | 0.487 | 0.379 |

**The failure mode is "confidently wrong," not "narrowly displacing a strong match."** Margins of ~0.38–0.41 rule out a near-tie explanation. This matters for remediation: a narrow-margin problem would be addressed by widening top-k or tie-breaking; a confident-wrong problem needs the gating mechanism in §6.3.

### 6.6 Mitigation scope check: adaptive temperature

`boundary_solutions.py`'s adaptive-$\tau$ blending (formula: §4.2, adaptive variant) reduces error ~32% on flip cases but — confirmed by direct instrumentation — **never changes which expert wins top-1**, in any tested case. It dampens the symptom, not the cause (softmax argmax is temperature-invariant by construction).

Quantified how localized the cost is, across 750 test points:
- 74.5% of requests keep $\tau$ within 2× of baseline (essentially undisturbed).
- Only 2.7% get substantially softened ($\tau > 0.5$).
- Correlation between $\tau$ and the top1–top2 similarity gap: **r = −1.0** (deterministic, by formula construction).

So the mitigation's cost is genuinely boundary-localized, not a blanket tax on all traffic.

---

## 7. Multiple Simultaneous Additions

Adding M new domains at once, each with an independently-calibrated 1%-false-positive-rate gate, does **not** preserve a 1% aggregate rate — it compounds, a direct instance of the multiple-comparisons problem in statistics (formula: §4.4).

Tested with 3 simultaneous additions (law, medicine, finance). Reproduce: `scripts/multi_dimension_compounding.py`.

- Each gate alone: 1.00% FPR on old-domain data (as designed).
- **Aggregate** (≥1 gate fires spuriously): **3.00%**, matching the independence prediction (1−(1−0.01)³ ≈ 2.97%) almost exactly. This part of the result reproduces to the decimal, run after run — it follows directly from the calibration percentile, not from a longer stochastic pipeline.
- Bonferroni correction (each gate tightened to target/M ≈ 99.67th percentile) restores aggregate to 1.01%.
- Cost: new-domain recall drops meaningfully — canonical run: finance 70.0%→50.7%, law 88.0%→80.0%, medicine 61.3%→44.7%. (Recall figures vary a few points run to run per §12; the compounding and correction percentages above do not.)

**Design rule following directly from this: budget an aggregate false-capture tolerance across the full roadmap of planned additions, not per-addition in isolation, or the compounding will be discovered in production rather than planned for.**

---

## 8. Validation on Real Text (Partial)

No network access was available to test with a trained semantic embedding model (e.g. a sentence-transformer). Tested instead with TF-IDF + SVD (lexical/co-occurrence structure — a real step up from synthetic 2D coordinates, but not full semantic representation). Reproduce: `scripts/text_validation.py`, `part1_real_text_flip_test()`.

- 131 base-domain test prompts, including 14 deliberately cross-domain "boundary" prompts (e.g. *"calculate the statute of limitations deadline..."* — genuinely math-and-law at once).
- Canonical run: **6 flips, 100% concentrated on the boundary prompts, 0% on the ordinary prompts.** (An earlier run found 4 flips on the same design; count varies slightly per §12, concentration on boundary prompts has been 100% in every run.)
- Important nuance: unlike the synthetic case, there is no downstream task-quality metric for text. Inspecting the flips individually, most look like defensible reclassification of genuinely dual-domain content rather than clear degradation (one flip even corrected a pre-existing profiler error). This is a real and different finding from the synthetic case, not a weaker replication of the same one.
- The gating fix (§6.3) was **not successfully validated on this text corpus** — calibration on 300 hand-written sentences produced a degenerate threshold (0.01, fires on nearly everything) because the calibration set lacked genuine difficulty. This is a corpus-size problem, not a fix failure.

**A follow-up test using systematically-generated (not hand-picked) boundary examples for calibration fixed this**: crossing domain vocabularies mechanically to produce ~600 boundary examples (vs. 14 hand-picked) raised the threshold from a degenerate 0.0031 to a meaningful 0.9943, dropped false-capture on fresh unambiguous data from 1.60% to 0.00%, at a cost of law recall dropping from 100% to 92%. This is the first working prototype of a **systematic calibration-data generation procedure** (see §9). Reproduce: `scripts/text_validation.py`, `part2_printer_prototype()` — this section's numbers reproduce exactly, to the decimal, every run.

---

## 9. Proposed: Modular Dimension Composition (not yet validated at scale)

Since each new domain's gate is trained fully independently (no joint retraining, by design — §6.3), domains behave as pluggable, standardized units. This suggests a natural extension: maintain a standardized calibration-data bank per domain, and assemble custom profilers on demand by selecting a subset of available dimensions — a "profiler compiled from parts" rather than one trained per deployment.

**What's logically well-grounded:**
- Since gates are independent by construction, assembling any subset requires no retraining of the gates themselves.
- Adding multiple dimensions back into a composition is governed by the same, already-quantified Bonferroni compounding math in §7 — no new theory needed there.

**What is *not* yet proven, and needs to be stated precisely:** it's tempting to claim "dropping a dimension from a composed profiler is always at least as safe." Checked this directly before writing it down: it only holds if calibration is done as a **per-domain worst-case** (threshold = max over each individual domain's own quantile), not as a **pooled aggregate** over a mixture of domains (which is what every experiment in this document actually used). Under pooled calibration, it's possible for one "easy" domain to pull the aggregate threshold down while a "harder" domain sits above the nominal target FPR — in which case keeping only the harder domain in a subset could exceed the intended bound. Tested this on the current 4-domain data and it happened to hold (individual FPRs ranged 0.60–0.85%, all under the 1% target) — but that's because these domains are roughly balanced in difficulty, not because the calibration protocol guarantees it in general. **Recommendation: if this is built, calibrate per-domain worst-case from the start, not pooled.**

**Practical constraint, stated honestly**: building a genuinely standardized, well-populated dataset bank across many domains — with the boundary-example density §8 shows is required for meaningful calibration — is a substantial undertaking, not a weekend project, especially pre-team and pre-traction. The lower-risk path is validating the composition mechanism on 2–3 domains first (proving the mechanism, not the coverage), rather than attempting comprehensive breadth solo before the architecture-level claim is even confirmed at small scale.

---

## 10. Achievements Mapped to Architecture (vs. Traditional MoE)

| Achievement | Which component produces it | Why traditional MoE doesn't have this |
|---|---|---|
| No load-balancing (z-)loss needed | Router (§4.2) — similarity is bounded in [-1,1] by construction | Traditional gates produce unbounded logits; without an auxiliary loss, routing collapses onto a few experts |
| Every routing decision is traceable to a reason | Profile calibration (§4.1) — a decision reduces to "which benchmark score was highest" | A learned gate's decision is a forward pass through jointly-optimized weights; not decomposable into a stated reason |
| Provable isolation under expert *replacement* | Profile calibration formula's independence across experts (Theorem 1, §4.5) — $p_j$ mathematically cannot depend on $e_i$'s parameters for $j \ne i$ | The gate is co-optimized with every expert simultaneously; nothing in a jointly-trained gate guarantees one expert's change doesn't move the boundary for another |
| Conditional, calibratable isolation under expert *addition* | Frozen base profiler + independent one-vs-rest gate (§6.3) — decouples the new decision boundary from the old one entirely | Traditional MoE has no equivalent decoupling mechanism; adding a class to a jointly-trained gate is exactly the class-incremental-learning problem this project's fix borrows its solution pattern from |
| Profiler parameter count doesn't scale with expert count | Profiler is a fixed-size classifier over the domain set, independent of how many experts implement each domain (§2.1) | Gate parameter count scales with (hidden dim) × (expert count) in standard MoE |
| Multi-addition risk is quantifiable and budgetable | Bonferroni correction applied to independent one-vs-rest gates (§4.4, §7) | Not a concept that applies to a single jointly-trained gate — there's no equivalent of "adding several experts at once" as a separably-analyzable operation |

**What is explicitly *not* claimed as an achievement**: superior routing accuracy at scale, superior handling of subtle/emergent domain signal, or a solved multi-adapter serving story (§3, row 7 — flexibility/discovery trade-off is real and unresolved in traditional MoE's favor; §3, row 9 — batched serving is prior art, not this project's contribution).

---

## 11. Known Limitations (consolidated)

1. Full (unconditional) addition-isolation is not achievable — proven impossible for genuinely ambiguous inputs, not merely unsolved.
2. Multi-addition compounding requires explicit threshold budgeting or the aggregate risk grows silently.
3. Gating fix validated on synthetic data and (with proper calibration-set construction) partially on lexical text; not yet tested against true semantic embeddings, and not yet tested in a live serving environment (calibration latency, concurrent-request interference — untested outside offline evaluation).
4. Real production calibration data (for expert profiles, as opposed to gate thresholds) does not yet exist for any real domain in this project — everything validated so far uses synthetic or template-generated stand-ins.
5. Accuracy comparison against a learned router (DeepSeek/Mixtral-style) used an undertrained baseline (~8s training); not a fair comparison yet, and no claim of superiority should rest on it until rerun at proper scale. Relatedly (§3, row 7): traditional MoE's ability to discover subtle routing signal from data at scale is a genuine, unmatched advantage that this architecture has no answer to yet.
6. Modular composition (§9) is logically motivated and partially tested at the mechanism level, not validated at any realistic scale or domain count.
7. `d_profile` and `k` (top-k) knob effects in `knob_boundaries.png` are not yet reliably characterized — the existing figure's own data does not clearly support its captioned conclusions for either knob; treat both as using-convention defaults, not verified findings, until re-tested.
8. Exact reproducibility is run-dependent for multi-stage experiments (§12) — treat percentages and sample-level detail as illustrative of a real, repeatedly-observed pattern, not as fixed constants. The two-decimal-place-stable results (§6.4, the multi-dimension percentages in §7, the printer-prototype numbers in §8) are the ones safe to cite as fixed.
9. Margin/confidence analysis (§6.5) has been run and verified on one documented dataset realization, not yet re-verified against the canonical script's own specific stable-core points — a known, stated gap rather than a silently-dropped one.
10. The two-tier deployment story and multi-adapter batched serving (§2.2) are design intent, not tested engineering, in this document — they rest on citing prior art (Punica, S-LoRA), not original validation here.

---

## 12. Reproducibility

Every numbered finding above has a corresponding runnable script in `scripts/` (`shared_data.py` + four standalone scripts). Run any of them directly; no external data or network access required.

**Checked directly rather than assumed**: re-running these experiments in clean, standalone scripts (as opposed to the original incremental development session) produces the same *qualitative* findings every time, but not bit-identical numbers. Same nominal random seeds do not guarantee identical output once code structure changes how the random-number stream gets consumed — a common and under-discussed reproducibility gap in ML experimentation generally, not specific to this project.

- **Short, self-contained experiments reproduce exactly.** `capacity_ablation.py` and the calibration half of `text_validation.py` reproduce their numbers to the last decimal, run after run.
- **Longer multi-stage pipelines (several sequential model fits) reproduce the *pattern*, not the exact figures.** The addition-isolation and multi-seed-stability scripts consistently show: real flips occur, concentrated in the same qualitative way, gating fixes most but not all of them, and a stable-core-plus-shifting-penumbra split always appears in multi-seed testing — but the specific sample indices, exact percentages, and swap-isolation magnitude vary run to run.

The numbers in this document are this repository's canonical run (the actual output of the scripts in `scripts/`, verified immediately before writing this document). Anyone re-running the scripts should expect the same *conclusions*, not necessarily the same *digits* — treat the scripts, not the frozen numbers in this file, as the actual source of truth.

### Appendix: Companion Scripts

All in `scripts/`, each independently runnable (`python3 <script>.py`), CPU-only, no network access required:

| script | reproduces |
|---|---|
| `shared_data.py` | canonical data generator + Expert/profiler/routing infrastructure (implements the formulas in §4) used by all other scripts |
| `addition_isolation_suite.py` | §5 (swap isolation, both dimension counts), §6.2–6.3 (addition flips + gated fix), §6.5 (multi-seed stability) |
| `capacity_ablation.py` | §6.4 (parameter-scaling test) |
| `multi_dimension_compounding.py` | §7 (Bonferroni compounding across simultaneous additions) |
| `text_validation.py` | §8 (real-text flip replication + systematic calibration-generation prototype) |

The pre-existing repository files `boundary_solutions.py` and `versioning_demo.py`/`versioning_demo.xlsx` (§6.2, §6.6) were not authored in this review pass and are referenced, not reproduced, here.

---

## 13. Prior Art Acknowledgment

This project sits at the intersection of two established lines of work, and should be positioned as such rather than as a fully novel technique:

- **Efficient multi-adapter serving / training-free routing**: Punica (SGMV kernels), S-LoRA (unified paging), LoraRetriever, PHATGOOSE, MoLE — none of these, as far as reviewed, characterize isolation guarantees under *expert addition* specifically, which is this project's most distinctive contribution.
- **Catastrophic forgetting / class-incremental learning / open-set recognition**: the addition-failure mechanism (§6.1) and its mitigation (§6.3) are applications of decades-established technique (McCloskey & Cohen 1989; OpenMax, ~2016) to a routing context that hadn't previously had this connection made explicit, as far as this review has found.

The defensible claim is: *correct, quantified, empirically-verified application of established techniques to close a specific, previously uncharacterized gap in training-free MoE routing* — not invention from first principles.
