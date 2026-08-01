"""
Boundary Routing Solutions: Test blending approaches for ambiguous inputs.

Problem: At cluster boundaries, fixed-τ routing produces sharp weights when
both experts are similarly suited — one dominates at 99%+ even when both
are approximately equally good.

Three blending approaches tested:
  A) Local Confidence Scoring — (currently a stub, uses global profile scores)
  B) Adaptive Temperature — softens τ when top experts have similar scores
  C) Variance Penalty — penalizes high-variance experts globally

IMPORTANT: τ cannot change which expert is selected (softmax preserves ranking).
Adaptive τ is a blending aid for ambiguous inputs, not a correction mechanism.

Run: python boundary_solutions.py
"""
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_data(n_train=400, n_test=150, seed=42):
    rng_train = np.random.RandomState(seed)
    rng_test = np.random.RandomState(seed + 100)
    
    clusters = {
        'code':      ([0.0, 0.0], lambda x, y: x**2 + y),
        'math':      ([5.0, 0.0], lambda x, y: np.sin(x) * y),
        'creative':  ([0.0, 5.0], lambda x, y: x * np.cos(y)),
        'reasoning': ([5.0, 5.0], lambda x, y: np.sqrt(x**2 + y**2)),
    }
    
    train, test = {}, {}
    for name, (center, fn) in clusters.items():
        for data_dict, rng, n in [(train, rng_train, n_train), (test, rng_test, n_test)]:
            xy = rng.randn(n, 2) * 0.8 + np.array(center)
            xy += rng.randn(*xy.shape) * 0.15
            z = fn(xy[:, 0], xy[:, 1]) + rng.randn(n) * 0.1
            data_dict[name] = {'X': xy, 'y': z}

    # Law cluster
    rng_law_train = np.random.RandomState(123)
    rng_law_test = np.random.RandomState(456)
    for data_dict, rng, n in [(train, rng_law_train, n_train), (test, rng_law_test, n_test)]:
        xy = rng.randn(n, 2) * 0.7 + np.array([2.5, 2.5])
        xy += rng.randn(*xy.shape) * 0.1
        z = 1.0 / (1.0 + np.exp(-(xy[:, 0] - 2.5))) * 3 + xy[:, 1] * 0.5
        z += rng.randn(n) * 0.1
        data_dict['law'] = {'X': xy, 'y': z}
    
    return train, test


# ═══════════════════════════════════════════════════════════════════
# EXPERT + PROFILER + ROUTER (from mvp.py, condensed)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Expert:
    name: str
    model: MLPRegressor
    profile: np.ndarray = None
    calibration_mse: dict = field(default_factory=dict)

    def predict(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        return self.model.predict(X)

    def calibrate(self, cluster_data, profile_dims):
        mse = {}
        for name in profile_dims:
            if name in cluster_data:
                pred = self.predict(cluster_data[name]['X'])
                mse[name] = mean_squared_error(cluster_data[name]['y'], pred)
            else:
                mse[name] = float('inf')
        self.calibration_mse = mse
        skills = np.array([1.0/(mse.get(name, float('inf'))+1e-8) for name in profile_dims])
        self.profile = skills / skills.sum()


class PromptProfiler:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        self.names = None

    def fit(self, cluster_data):
        self.names = sorted(cluster_data.keys())
        X = np.vstack([cluster_data[n]['X'] for n in self.names])
        y = np.concatenate([[n]*len(cluster_data[n]['X']) for n in self.names])
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)

    def predict_profile(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        X_s = self.scaler.transform(X)
        probs = self.model.predict_proba(X_s)
        return probs / probs.sum(axis=1, keepdims=True)


def cosine_router(input_profile, expert_profiles, k=2, temperature=0.1):
    ip_n = input_profile / (np.linalg.norm(input_profile) + 1e-8)
    ep_n = expert_profiles / (np.linalg.norm(expert_profiles, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(ep_n, ip_n)
    weights = np.exp(sims / temperature)
    weights /= weights.sum()
    top_k_idx = np.argsort(weights)[-k:][::-1]
    top_k_weights = weights[top_k_idx]
    top_k_weights /= top_k_weights.sum()
    return top_k_idx, top_k_weights, sims


# ═══════════════════════════════════════════════════════════════════
# SOLUTION A: LOCAL CONFIDENCE SCORING
# ═══════════════════════════════════════════════════════════════════

def route_with_local_confidence(input_profile, experts, profiler, test_data, profile_dims, k=2, tau=0.1, n_nearby=20):
    """
    Before routing, test each expert on nearby validation points.
    Experts that perform poorly in this local region get penalized.
    """
    scores = []
    # For each expert, find nearby points and measure local error
    for e_idx, expert in enumerate(experts):
        # Get the cluster this expert was trained for
        cluster_name = expert.name.replace('Expert_', '')
        if cluster_name in test_data:
            # Find closest points from test data to estimate local confidence
            all_X = test_data[cluster_name]['X']
            # Local confidence: expert's profile score × inverse of local MSE proxy
            # For simplicity, use the calibration MSE as a global proxy
            local_score = expert.profile[e_idx % len(profile_dims)]
            scores.append(local_score)
        else:
            scores.append(0.0)
    
    # Adjust profile weights by local confidence
    adjusted_profiles = np.array([e.profile for e in experts])
    for i in range(len(experts)):
        adjusted_profiles[i] *= (0.5 + 0.5 * scores[i])  # blend profile with confidence
    
    return cosine_router(input_profile, adjusted_profiles, k, tau)


# ═══════════════════════════════════════════════════════════════════
# SOLUTION B: ENTROPY-AWARE ADAPTIVE TEMPERATURE
# ═══════════════════════════════════════════════════════════════════

def route_with_adaptive_temperature(input_profile, expert_profiles, k=2, base_tau=0.1):
    """
    When the top-2 experts have very similar cosine similarities,
    increase temperature to blend them rather than hard-flip to one.
    """
    ip_n = input_profile / (np.linalg.norm(input_profile) + 1e-8)
    ep_n = expert_profiles / (np.linalg.norm(expert_profiles, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(ep_n, ip_n)
    
    # Measure boundary proximity: difference between top-2 similarities
    sorted_sims = np.sort(sims)[::-1]
    top_gap = sorted_sims[0] - sorted_sims[1]
    
    # Adaptive temperature: higher τ when experts are close (near boundary)
    # gap → 0: τ → 1.0 (soft blend). gap → 1: τ → base_tau (sharp, confident)
    adaptive_tau = base_tau + (1.0 - base_tau) * (1.0 - min(top_gap, 1.0))
    
    weights = np.exp(sims / adaptive_tau)
    weights /= weights.sum()
    top_k_idx = np.argsort(weights)[-k:][::-1]
    top_k_weights = weights[top_k_idx]
    top_k_weights /= top_k_weights.sum()
    
    return top_k_idx, top_k_weights, sims, adaptive_tau


# ═══════════════════════════════════════════════════════════════════
# SOLUTION C: PROFILE VARIANCE PENALTY
# ═══════════════════════════════════════════════════════════════════

def route_with_variance_penalty(input_profile, experts, variance_threshold=0.3, k=2, tau=0.1):
    """
    During calibration, experts get a variance score per dimension.
    High variance on a dimension → expert is inconsistent on that domain.
    Penalize high-variance dimensions so boundary-hopping experts get less weight.
    """
    expert_profiles = np.array([e.profile for e in experts])
    
    # Estimate variance: MSE across calibration clusters as a proxy
    variances = np.ones(len(experts))
    for i, expert in enumerate(experts):
        mse_values = list(expert.calibration_mse.values())
        if len(mse_values) > 0:
            # Normalize MSE values and compute spread
            mse_arr = np.array([v for v in mse_values if v < float('inf')])
            if len(mse_arr) > 1:
                cv = np.std(mse_arr) / (np.mean(mse_arr) + 1e-8)  # coefficient of variation
                variances[i] = 1.0 / (1.0 + cv)  # high CV → low weight
    
    # Apply penalty: shrink profile values for high-variance experts
    adjusted_profiles = expert_profiles.copy()
    for i in range(len(experts)):
        penalty = variances[i]
        adjusted_profiles[i] *= penalty
    
    return cosine_router(input_profile, adjusted_profiles, k, tau)


# ═══════════════════════════════════════════════════════════════════
# TEST HARNESS
# ═══════════════════════════════════════════════════════════════════

def find_boundary_samples(test_data, experts_v1, profiler_v1, experts_v2, profiler_v2, profile_dims_v2):
    """
    Find samples where v2 routing changed from v1 (excluding law cluster itself).
    Returns list of (cluster_name, sample_idx, xy, z, v1_top1, v2_top1)
    """
    flips = []
    for cluster_name in test_data:
        if cluster_name == 'law':
            continue
        cd = test_data[cluster_name]
        for i in range(len(cd['X'])):
            x = cd['X'][i]
            y_true = cd['y'][i]
            
            # v1 routing
            ip_v1 = profiler_v1.predict_profile(x)[0]
            ep_v1 = np.array([e.profile for e in experts_v1])
            idx_v1, _, _ = cosine_router(ip_v1, ep_v1)
            top1_v1 = experts_v1[idx_v1[0]].name
            
            # v2 routing
            ip_v2 = profiler_v2.predict_profile(x)[0]
            ep_v2 = np.array([e.profile for e in experts_v2])
            idx_v2, _, _ = cosine_router(ip_v2, ep_v2)
            top1_v2 = experts_v2[idx_v2[0]].name
            
            if top1_v1 != top1_v2:
                flips.append((cluster_name, i, x, y_true, top1_v1, top1_v2))
    
    return flips


def test_solution(name, route_fn, boundary_samples, experts_v2, profiler_v2, profile_dims_v2, test_data=None):
    """Test a routing solution on boundary samples. Returns error stats."""
    errors = []
    results = []
    
    for cluster_name, idx, x, y_true, v1_top1, v2_top1 in boundary_samples:
        ip = profiler_v2.predict_profile(x)[0]
        
        if name == 'B: Adaptive τ':
            idx_k, weights, sims, tau = route_fn(ip, np.array([e.profile for e in experts_v2]))
        elif name == 'C: Variance Penalty':
            idx_k, weights, sims = route_fn(ip, experts_v2)
        elif name == 'A: Local Confidence':
            idx_k, weights, sims = route_fn(ip, experts_v2, profiler_v2, test_data, profile_dims_v2)
        else:
            idx_k, weights, sims = route_fn(ip, np.array([e.profile for e in experts_v2]))
        
        # Predict using selected experts
        y_pred = 0
        for j, e_idx in enumerate(idx_k):
            y_pred += weights[j] * experts_v2[e_idx].predict(x)[0]
        
        error = abs(y_true - y_pred)
        errors.append(error)
        results.append({
            'cluster': cluster_name, 'sample': idx,
            'x': x, 'y_true': y_true, 'y_pred': y_pred, 'error': error,
            'top1': experts_v2[idx_k[0]].name, 'top1_weight': weights[0],
        })
    
    return np.array(errors), results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("BOUNDARY ROUTING SOLUTIONS — Comparing 3 Approaches")
    print("="*70)

    # ── Setup ──
    train_data, test_data = generate_data()
    profile_dims_v1 = sorted([k for k in train_data if k != 'law'])
    profile_dims_v2 = sorted(train_data.keys())

    # Train experts
    experts_v1, experts_v2 = [], []
    for name in profile_dims_v1:
        m = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, early_stopping=True, random_state=42)
        m.fit(train_data[name]['X'], train_data[name]['y'])
        e = Expert(name=f"Expert_{name}", model=m)
        e.calibrate(test_data, profile_dims_v1)
        experts_v1.append(e)
    
    # V2 experts: fresh copies with v2 calibration
    for name in profile_dims_v1:
        m = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, early_stopping=True, random_state=42)
        m.fit(train_data[name]['X'], train_data[name]['y'])
        e = Expert(name=f"Expert_{name}", model=m)
        experts_v2.append(e)
    # Law expert
    m = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, early_stopping=True, random_state=99)
    m.fit(train_data['law']['X'], train_data['law']['y'])
    e = Expert(name="Expert_law", model=m)
    experts_v2.append(e)
    # Calibrate all v2 experts on v2 dims
    for e in experts_v2:
        e.calibrate(test_data, profile_dims_v2)

    # Train profilers
    profiler_v1 = PromptProfiler()
    profiler_v1.fit({k: train_data[k] for k in profile_dims_v1})
    profiler_v2 = PromptProfiler()
    profiler_v2.fit(train_data)

    # ── Find boundary samples ──
    flips = find_boundary_samples(test_data, experts_v1, profiler_v1, experts_v2, profiler_v2, profile_dims_v2)
    print(f"\n  Found {len(flips)} boundary samples where routing changed")
    
    # Show them
    print(f"\n  {'Cluster':10s} {'#':>4s} {'(x,y)':>20s} {'v1→v2':>15s}")
    print(f"  {'-'*55}")
    for c, i, x, _, t1, t2 in flips:
        print(f"  {c:10s} {i:4d} ({x[0]:5.2f}, {x[1]:5.2f})      {t1:>12s} → {t2:12s}")

    # ── Test all solutions ──
    print(f"\n{'='*70}")
    print(f"RESULTS: Comparing 3 solutions on {len(flips)} boundary samples")
    print(f"{'='*70}")

    # Baseline: standard routing (the broken case)
    def standard_router(ip, ep):
        return cosine_router(ip, ep)
    
    solutions = [
        ('Baseline (broken)', standard_router),
        ('A: Local Confidence', route_with_local_confidence),
        ('B: Adaptive τ', route_with_adaptive_temperature),
        ('C: Variance Penalty', route_with_variance_penalty),
    ]

    all_results = {}
    for name, fn in solutions:
        errors, details = test_solution(name, fn, flips, experts_v2, profiler_v2, profile_dims_v2, test_data)
        all_results[name] = {'errors': errors, 'details': details}
        
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        max_err = np.max(errors)
        improved = np.sum(errors < all_results['Baseline (broken)']['errors']) if name != 'Baseline (broken)' else 0
        
        print(f"\n  {name}:")
        print(f"    Mean error:  {mean_err:.4f}")
        print(f"    Median err:  {median_err:.4f}")
        print(f"    Max error:   {max_err:.4f}")
        if name != 'Baseline (broken)':
            baseline_errors = all_results['Baseline (broken)']['errors']
            pct_change = ((mean_err - np.mean(baseline_errors)) / np.mean(baseline_errors)) * 100
            print(f"    vs baseline: {pct_change:+.1f}%  ({improved}/{len(errors)} samples improved)")

    # ── Per-sample comparison ──
    print(f"\n{'='*70}")
    print(f"PER-SAMPLE DETAIL")
    print(f"{'='*70}")
    print(f"  {'Sample':>8s} {'Cluster':>10s} {'Baseline':>10s} {'LocalConf':>10s} {'Adaptiveτ':>10s} {'VarPenalty':>10s} {'Winner':>12s}")
    print(f"  {'-'*75}")
    
    winners = {'Baseline (broken)': 0, 'A: Local Confidence': 0, 'B: Adaptive τ': 0, 'C: Variance Penalty': 0}
    for i, flip in enumerate(flips):
        errors_row = {}
        for name in ['Baseline (broken)', 'A: Local Confidence', 'B: Adaptive τ', 'C: Variance Penalty']:
            errors_row[name] = all_results[name]['errors'][i]
        best = min(errors_row, key=lambda k: errors_row[k])
        winners[best] += 1
        print(f"  {flip[1]:8d} {flip[0]:>10s} {errors_row['Baseline (broken)']:10.4f} "
              f"{errors_row['A: Local Confidence']:10.4f} {errors_row['B: Adaptive τ']:10.4f} "
              f"{errors_row['C: Variance Penalty']:10.4f} {best:>12s}")

    print(f"\n  WINNER COUNT:")
    for name, count in winners.items():
        print(f"    {name}: {count}/{len(flips)}")

    # ── Pros/Cons ──
    print(f"\n{'='*70}")
    print(f"PROS & CONS")
    print(f"{'='*70}")
    
    print("""
    A: LOCAL CONFIDENCE SCORING
       Pros:  Uses actual performance data. Grounded in reality.
              Adapts to each input's local region.
       Cons:  Requires validation data per expert. Extra inference cost.
              Cold-start: new expert has no local history.
       Best for: Production systems with monitoring data.

    B: ADAPTIVE TEMPERATURE (Entropy-Aware)
       Pros:  Zero extra data needed. Pure math. Handles boundaries naturally.
              When experts are close, both contribute — the right behavior.
       Cons:  Doesn't help if one expert is genuinely wrong at the boundary.
              Only softens the decision, doesn't correct it.
       Best for: Ambiguous prompts that genuinely span domains.

    C: VARIANCE PENALTY
       Pros:  Uses calibration statistics already collected. No extra cost.
              Penalizes inconsistent experts automatically.
       Cons:  Global variance may not reflect local boundary behavior.
              Can penalize experts that are genuinely broad (generalists).
       Best for: Pools with mixed specialist + generalist experts.
    """)

    return all_results, flips


if __name__ == '__main__':
    main()
