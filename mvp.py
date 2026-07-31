"""
Profile-MoE MVP: Swappable Experts with Profile-Based Routing

Minimal prediction calculator proving:
1. Profile-based routing works (picks right expert for right input)
2. Experts are swappable without retraining the router
3. Every decision is observable (which expert, why, how fast)

Architecture:
    INPUT (x,y) → Profiler → input_profile
                             ↓
                Router: cos_sim(input_profile, expert_profiles)
                             ↓
                top-k experts → weighted combination → OUTPUT

Run:  python mvp.py
"""
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass, field
import time
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_cluster_data(n_samples=500, noise=0.1, seed=42):
    """Generate 4 clusters in 2D, each with a different underlying function."""
    rng = np.random.RandomState(seed)
    clusters = {
        'code': {
            'center': np.array([0.0, 0.0]),
            'spread': 0.8,
            'fn': lambda x, y: x**2 + y,
            'color': 'blue',
        },
        'math': {
            'center': np.array([5.0, 0.0]),
            'spread': 0.8,
            'fn': lambda x, y: np.sin(x) * y,
            'color': 'red',
        },
        'creative': {
            'center': np.array([0.0, 5.0]),
            'spread': 0.8,
            'fn': lambda x, y: x * np.cos(y),
            'color': 'green',
        },
        'reasoning': {
            'center': np.array([5.0, 5.0]),
            'spread': 0.8,
            'fn': lambda x, y: np.sqrt(x**2 + y**2),
            'color': 'orange',
        },
    }

    data = {}
    for name, cfg in clusters.items():
        xy = rng.randn(n_samples, 2) * cfg['spread'] + cfg['center']
        # Add jitter so clusters overlap slightly at edges
        xy += rng.randn(*xy.shape) * 0.15
        z = cfg['fn'](xy[:, 0], xy[:, 1]) + rng.randn(n_samples) * noise
        data[name] = {'X': xy, 'y': z, 'color': cfg['color']}

    return data


# ═══════════════════════════════════════════════════════════════════
# EXPERT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Expert:
    """A tiny neural network + its capability profile."""
    name: str
    model: MLPRegressor
    profile: np.ndarray = None          # d_profile dims, normalized
    calibration_mse: dict = field(default_factory=dict)  # raw scores per cluster

    def predict(self, X):
        return self.model.predict(X)

    def calibrate(self, cluster_data: dict):
        """Build profile by testing on each cluster. Lower MSE → higher score.
        Profile dimensions are in SORTED cluster name order (alphabetical).
        This must match PromptProfiler's ordering.
        """
        sorted_names = sorted(cluster_data.keys())
        mse_per_cluster = {}
        for cluster_name in sorted_names:
            cd = cluster_data[cluster_name]
            pred = self.predict(cd['X'])
            mse_per_cluster[cluster_name] = mean_squared_error(cd['y'], pred)

        self.calibration_mse = mse_per_cluster

        # Invert MSE → skill score (add epsilon to avoid div by zero)
        skills = np.array([1.0 / (mse_per_cluster[name] + 1e-8)
                          for name in sorted_names])
        self.profile = skills / skills.sum()       # normalize to sum=1

    def describe(self):
        """Human-readable profile description."""
        if self.profile is None:
            return f"{self.name}: [uncalibrated]"
        dims = PROFILE_DIMS
        parts = [f"{d}={self.profile[i]:.3f}" for i, d in enumerate(dims)]
        return f"{self.name}: [{', '.join(parts)}]"


# Profile dimension names (alphabetical — must match profiler + calibrate ordering)
PROFILE_DIMS = ['code', 'creative', 'math', 'reasoning']


# ═══════════════════════════════════════════════════════════════════
# PROMPT PROFILER
# ═══════════════════════════════════════════════════════════════════

class PromptProfiler:
    """Maps input (x,y) → profile vector predicting which expert is best."""

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.cluster_names = None          # in the order profiles are returned
        self._name_to_idx = None           # mapping from cluster name → profile index

    def fit(self, cluster_data: dict):
        """Train to classify which cluster an input belongs to."""
        X_all = []
        y_all = []
        self.cluster_names = sorted(cluster_data.keys())  # alphabetical, matches sklearn

        for name in self.cluster_names:
            cd = cluster_data[name]
            X_all.append(cd['X'])
            y_all.extend([name] * len(cd['X']))

        X_all = np.vstack(X_all)
        X_all = self.scaler.fit_transform(X_all)
        self.model.fit(X_all, y_all)

        # Build name → index mapping (sklearn returns classes_ in sorted order)
        self._name_to_idx = {name: i for i, name in enumerate(self.model.classes_)}

    def predict_profile(self, X):
        """Return probability distribution over clusters (the input profile).
        Profile order: alphabetical by cluster name ['code','creative','math','reasoning'].
        """
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        probs = self.model.predict_proba(X_scaled)[0]
        return probs / probs.sum()   # ensure sum=1

    def profile_index_for(self, cluster_name):
        """Return the profile vector index for a given cluster name."""
        return self._name_to_idx.get(cluster_name, -1)


# ═══════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════

class ProfileRouter:
    """Routes inputs to experts based on profile similarity. Zero learned params."""

    def __init__(self, temperature=0.1):
        self.temperature = temperature
        self.stats = {'total_calls': 0, 'route_history': []}

    def route(self, input_profile, experts, k=2):
        """
        Match input profile to expert profiles via cosine similarity.

        Args:
            input_profile: shape (d_profile,)
            experts: list of Expert objects
            k: top-k experts to select

        Returns:
            selected_indices: list of k expert indices
            weights: normalized weights for selected experts
            similarities: all similarity scores (for analysis)
        """
        expert_profiles = np.array([e.profile for e in experts])

        # Cosine similarity
        input_norm = input_profile / (np.linalg.norm(input_profile) + 1e-8)
        expert_norms = expert_profiles / (np.linalg.norm(expert_profiles, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(expert_norms, input_norm)

        # Softmax with temperature
        weights = np.exp(similarities / self.temperature)
        weights /= weights.sum()

        # Top-k
        top_k_idx = np.argsort(weights)[-k:][::-1]
        top_k_weights = weights[top_k_idx]
        top_k_weights /= top_k_weights.sum()

        self.stats['total_calls'] += 1
        self.stats['route_history'].append({
            'input_profile': input_profile.tolist(),
            'similarities': similarities.tolist(),
            'weights': weights.tolist(),
            'selected': top_k_idx.tolist(),
            'selected_weights': top_k_weights.tolist(),
        })

        return top_k_idx, top_k_weights, similarities


# ═══════════════════════════════════════════════════════════════════
# PROFILE-MoE (combines everything)
# ═══════════════════════════════════════════════════════════════════

class ProfileMoE:
    """Full Profile-MoE system: profiler + router + experts."""

    def __init__(self, experts, profiler, router, k=2):
        self.experts = experts
        self.profiler = profiler
        self.router = router
        self.k = k

    def predict(self, X, verbose=False):
        """Predict for a single input, with optional verbose output."""
        t0 = time.perf_counter()

        # Step 1: Profile the input
        t1 = time.perf_counter()
        input_profile = self.profiler.predict_profile(X)
        t_profiler = time.perf_counter() - t1

        # Step 2: Route to experts
        t2 = time.perf_counter()
        selected_idx, weights, similarities = self.router.route(
            input_profile, self.experts, self.k
        )
        t_router = time.perf_counter() - t2

        # Step 3: Selected experts predict
        t3 = time.perf_counter()
        expert_outputs = []
        for idx in selected_idx:
            pred = self.experts[idx].predict(X.reshape(1, -1))[0]
            expert_outputs.append(pred)
        t_experts = time.perf_counter() - t3

        # Step 4: Weighted combination
        output = np.dot(weights, expert_outputs)
        t_total = time.perf_counter() - t0

        if verbose:
            self._print_verbose(X, input_profile, similarities, selected_idx,
                               weights, expert_outputs, output,
                               t_profiler, t_router, t_experts, t_total)

        return output, {
            'input_profile': input_profile,
            'selected_experts': selected_idx,
            'weights': weights,
            'expert_outputs': expert_outputs,
            'similarities': similarities,
            'timing': {
                'profiler_ms': t_profiler * 1000,
                'router_ms': t_router * 1000,
                'experts_ms': t_experts * 1000,
                'total_ms': t_total * 1000,
            }
        }

    def _print_verbose(self, X, input_profile, similarities, selected_idx,
                       weights, expert_outputs, output, t_p, t_r, t_e, t_t):
        dims = PROFILE_DIMS
        print(f"\n{'═'*65}")
        print(f"INPUT:        ({X[0]:.2f}, {X[1]:.2f})")
        print(f"INPUT PROFILE: {dict(zip(dims, [f'{v:.4f}' for v in input_profile]))}")
        print(f"\nEXPERT PROFILES:")
        for i, e in enumerate(self.experts):
            marker = ' ← SELECTED' if i in selected_idx else ''
            print(f"  {e.describe()}{marker}")
        print(f"\nCOSINE SIMILARITIES:")
        for i, sim in enumerate(similarities):
            print(f"  {self.experts[i].name}: {sim:.4f}")
        print(f"\nROUTER: top-{self.k} experts")
        for j, (idx, w, eout) in enumerate(zip(selected_idx, weights, expert_outputs)):
            print(f"  {self.experts[idx].name}: weight={w:.4f}, output={eout:.4f}")
        print(f"FINAL OUTPUT: {output:.4f}")
        print(f"\nTIMING: profiler={t_p*1e6:.0f}μs router={t_r*1e6:.0f}μs "
              f"experts={t_e*1e6:.0f}μs total={t_t*1e6:.0f}μs")
        print(f"{'═'*65}")


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_experts(cluster_data, hidden_sizes=(16,), seed=42):
    """Train one expert per cluster. Returns experts sorted by cluster name
    (alphabetical) so expert[i].profile matches profiler profile[i]."""
    experts = []
    for name in sorted(cluster_data.keys()):   # sorted = alphabetical
        cd = cluster_data[name]
        model = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            activation='relu',
            max_iter=1000,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        model.fit(cd['X'], cd['y'])
        expert = Expert(name=f"Expert_{name}", model=model)
        experts.append(expert)
    return experts


def calibrate_experts(experts, test_data):
    """Calibrate all experts on test data — build their profile vectors."""
    for expert in experts:
        expert.calibrate(test_data)


def train_profiler(cluster_data):
    """Train the prompt profiler on labeled cluster data."""
    profiler = PromptProfiler()
    profiler.fit(cluster_data)
    return profiler


# ═══════════════════════════════════════════════════════════════════
# EVALUATION & ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def evaluate(moe, test_data, verbose_samples=3):
    """Full evaluation: routing accuracy, per-cluster MSE, timing."""
    results = {
        'per_cluster': {},
        'routing_stats': {},
        'timing': [],
        'global_mse': 0,
    }

    all_y_true = []
    all_y_pred = []

    for cluster_name, cd in test_data.items():
        cluster_errors = []
        cluster_routes = []

        # The "correct" expert for this cluster = expert whose name matches the cluster
        correct_expert_idx = None
        for i, e in enumerate(moe.experts):
            if e.name == f"Expert_{cluster_name}":
                correct_expert_idx = i
                break

        for i in range(len(cd['X'])):
            x = cd['X'][i]
            y_true = cd['y'][i]
            y_pred, meta = moe.predict(x, verbose=(i < verbose_samples))

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            cluster_errors.append((y_true - y_pred) ** 2)
            cluster_routes.append(meta['selected_experts'][0])  # top-1 expert
            results['timing'].append(meta['timing']['total_ms'])

        # Routing accuracy: does top-1 expert match the "correct" expert?
        if correct_expert_idx is not None:
            cluster_route_acc = sum(1 for r in cluster_routes
                                   if r == correct_expert_idx) / len(cluster_routes)
        else:
            cluster_route_acc = 0.0

        results['per_cluster'][cluster_name] = {
            'mse': float(np.mean(cluster_errors)),
            'routing_accuracy': float(cluster_route_acc),
            'n_samples': len(cluster_errors),
        }

    results['global_mse'] = float(mean_squared_error(all_y_true, all_y_pred))

    # Overall routing accuracy
    total_correct = 0
    total_samples = 0
    for cluster_name in test_data:
        correct_expert_idx = None
        for i, e in enumerate(moe.experts):
            if e.name == f"Expert_{cluster_name}":
                correct_expert_idx = i
                break
        if correct_expert_idx is not None:
            total_correct += results['per_cluster'][cluster_name]['routing_accuracy'] * \
                            results['per_cluster'][cluster_name]['n_samples']
            total_samples += results['per_cluster'][cluster_name]['n_samples']

    results['routing_stats']['overall_accuracy'] = float(total_correct / total_samples) if total_samples > 0 else 0
    results['routing_stats']['avg_time_ms'] = float(np.mean(results['timing']))
    results['routing_stats']['total_samples'] = total_samples

    return results


def expert_utilization(moe, test_data):
    """Calculate how often each expert is selected."""
    counts = np.zeros(len(moe.experts))
    for cd in test_data.values():
        for i in range(len(cd['X'])):
            _, meta = moe.predict(cd['X'][i], verbose=False)
            for idx in meta['selected_experts']:
                counts[idx] += 1
    return counts / counts.sum()


# ═══════════════════════════════════════════════════════════════════
# SWAP TEST
# ═══════════════════════════════════════════════════════════════════

def create_swapped_expert(original_expert, new_fn, cluster_data, test_data):
    """Create a new expert trained on a different function, then calibrate it.
    Keeps the same NAME as the original (same slot, different occupant)."""
    name = original_expert.name  # Keep same slot name

    # Train on modified data (different function for the same cluster)
    cluster_name = name.replace('Expert_', '')
    X = cluster_data[cluster_name]['X']
    y_new = new_fn(X[:, 0], X[:, 1])

    model = MLPRegressor(
        hidden_layer_sizes=(16,),
        activation='relu',
        max_iter=1000,
        random_state=99,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X, y_new)
    new_expert = Expert(name=name, model=model)
    new_expert.calibrate(test_data)
    return new_expert


def swap_test(moe, test_data, cluster_data):
    """
    Full swap test:
    1. Record baseline performance
    2. Swap Expert_code with a differently-trained version
    3. Measure: only the swapped cluster should change
    """
    print("\n" + "="*70)
    print("SWAP TEST")
    print("="*70)

    # --- BASELINE ---
    print("\n[1] BASELINE (before swap)")
    baseline = evaluate(moe, test_data, verbose_samples=0)
    for name, stats in baseline['per_cluster'].items():
        print(f"  {name:12s}: MSE={stats['mse']:.6f}  routing_acc={stats['routing_accuracy']:.2%}")

    # --- SWAP ---
    print("\n[2] SWAPPING Expert_code → new Expert_code (different function)")
    old_expert = moe.experts[0]
    old_profile = old_expert.profile.copy()

    # New function: different from original x²+y
    new_fn = lambda x, y: x**3 - y**2 + x*y
    new_expert = create_swapped_expert(old_expert, new_fn, cluster_data, test_data)

    # Before replacing, let's verify the new expert's profile differs
    print(f"  Old profile: {old_expert.describe()}")
    print(f"  New profile: {new_expert.describe()}")
    print(f"  Profile delta: {np.linalg.norm(new_expert.profile - old_profile):.4f}")

    # Perform the swap
    moe.experts[0] = new_expert
    # NOTE: router is NOT retrained. Only the expert and its profile changed.

    # --- AFTER SWAP ---
    print("\n[3] AFTER SWAP (no router retraining)")
    after = evaluate(moe, test_data, verbose_samples=0)
    for name, stats in after['per_cluster'].items():
        delta = stats['mse'] - baseline['per_cluster'][name]['mse']
        direction = '↑' if delta > 0 else '↓'
        print(f"  {name:12s}: MSE={stats['mse']:.6f}  Δ={delta:+.6f} {direction}  "
              f"routing_acc={stats['routing_accuracy']:.2%}")

    # --- ISOLATION CHECK ---
    print("\n[4] ISOLATION CHECK: Only the swapped cluster should change significantly")
    deltas = {}
    for name in test_data.keys():
        deltas[name] = abs(after['per_cluster'][name]['mse'] -
                           baseline['per_cluster'][name]['mse'])

    swapped_cluster = list(test_data.keys())[0]
    other_max_delta = max(deltas[n] for n in test_data if n != swapped_cluster)
    swap_delta = deltas[swapped_cluster]

    print(f"  Swapped cluster delta: {swap_delta:.6f}")
    print(f"  Max other cluster delta: {other_max_delta:.6f}")

    if swap_delta > 2 * other_max_delta:
        print(f"  ✓ ISOLATION CONFIRMED: swap impact is {swap_delta/other_max_delta:.1f}x larger "
              f"than max spillover")
    else:
        print(f"  ⚠ Isolation weaker than expected. Ratio: {swap_delta/other_max_delta:.1f}x")

    # Restore for subsequent tests
    moe.experts[0] = old_expert

    return {
        'baseline': baseline,
        'after_swap': after,
        'deltas': deltas,
        'isolation_ratio': float(swap_delta / other_max_delta) if other_max_delta > 0 else float('inf'),
    }


# ═══════════════════════════════════════════════════════════════════
# TEMPERATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_temperature(moe, test_data, temperatures=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]):
    """How does temperature τ affect routing accuracy and MSE?"""
    print("\n" + "="*70)
    print("TEMPERATURE ANALYSIS")
    print("="*70)
    print(f"{'τ':>8s}  {'Route Acc':>10s}  {'Global MSE':>12s}  {'Avg Experts Used':>16s}")
    print("-"*55)

    results = []
    for tau in temperatures:
        moe.router.temperature = tau
        eval_r = evaluate(moe, test_data, verbose_samples=0)

        # Average number of experts with non-negligible weight
        total_route_samples = len(moe.router.stats['route_history'])
        if total_route_samples > 0:
            recent = moe.router.stats['route_history'][-1000:]
            n_active = np.mean([sum(1 for sw in r['selected_weights'] if sw > 0.01)
                               for r in recent])
        else:
            n_active = moe.k

        print(f"{tau:8.3f}  {eval_r['routing_stats']['overall_accuracy']:10.2%}  "
              f"{eval_r['global_mse']:12.6f}  {n_active:16.2f}")

        results.append({
            'temperature': tau,
            'routing_accuracy': eval_r['routing_stats']['overall_accuracy'],
            'global_mse': eval_r['global_mse'],
            'avg_experts_used': float(n_active),
        })

    # Reset to default
    moe.router.temperature = 0.1
    return results


# ═══════════════════════════════════════════════════════════════════
# EXPERT COUNT SCALING
# ═══════════════════════════════════════════════════════════════════

def analyze_expert_scaling(cluster_data, test_data, n_experts_list=[2, 3, 4]):
    """How does performance change as we reduce expert count?"""
    print("\n" + "="*70)
    print("EXPERT SCALING ANALYSIS")
    print("="*70)

    all_names = sorted(cluster_data.keys())
    results = []

    for n in n_experts_list:
        names_subset = all_names[:n]

        # Train experts only on the subset of clusters
        subset_data = {k: cluster_data[k] for k in names_subset}
        subset_test = {k: test_data[k] for k in names_subset}

        subset_experts = train_experts(subset_data)
        # Calibrate on ALL test data (so profiles are full d_profile-dim)
        calibrate_experts(subset_experts, test_data)

        # Profiler MUST be trained on ALL clusters for profile dimensions to match
        profiler = train_profiler(cluster_data)
        router = ProfileRouter(temperature=0.1)
        moe_small = ProfileMoE(subset_experts, profiler, router, k=min(2, n))

        eval_r = evaluate(moe_small, subset_test, verbose_samples=0)

        print(f"  n_experts={n}: MSE={eval_r['global_mse']:.6f}  "
              f"route_acc={eval_r['routing_stats']['overall_accuracy']:.2%}")

        results.append({'n_experts': n, 'mse': eval_r['global_mse'],
                       'routing_accuracy': eval_r['routing_stats']['overall_accuracy']})

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("Profile-MoE MVP: Profile-Based Expert Routing Analysis")
    print("="*70)

    # --- Generate data ---
    print("\n[GENERATING DATA]")
    train_data = generate_cluster_data(n_samples=500, noise=0.1, seed=42)
    test_data = generate_cluster_data(n_samples=200, noise=0.1, seed=99)

    for name, cd in train_data.items():
        print(f"  {name}: {cd['X'].shape[0]} train, {test_data[name]['X'].shape[0]} test samples")

    # --- Train experts ---
    print("\n[TRAINING EXPERTS]")
    experts = train_experts(train_data)
    calibrate_experts(experts, test_data)
    for e in experts:
        print(f"  {e.describe()}")
        print(f"    Raw calibration MSE: { {k: f'{v:.4f}' for k,v in e.calibration_mse.items()} }")

    # --- Train profiler ---
    print("\n[TRAINING PROMPT PROFILER]")
    profiler = train_profiler(train_data)
    # Quick accuracy check
    X_test_all = np.vstack([test_data[n]['X'] for n in test_data])
    y_test_all = np.concatenate([[n]*len(test_data[n]['X']) for n in test_data])
    X_test_scaled = profiler.scaler.transform(X_test_all)
    profiler_acc = profiler.model.score(X_test_scaled, y_test_all)
    print(f"  Profiler accuracy: {profiler_acc:.2%}")

    # --- Build MoE ---
    router = ProfileRouter(temperature=0.1)
    moe = ProfileMoE(experts, profiler, router, k=2)

    # --- Verbose predictions ---
    print("\n[VERBOSE PREDICTIONS (3 samples per cluster)]")
    for name in test_data:
        for i in range(3):
            x = test_data[name]['X'][i]
            y_true = test_data[name]['y'][i]
            y_pred, meta = moe.predict(x, verbose=True)
            print(f"  True={y_true:.4f} Pred={y_pred:.4f}  Error={abs(y_true-y_pred):.4f}")

    # --- Full evaluation ---
    print("\n" + "="*70)
    print("FULL EVALUATION")
    print("="*70)
    eval_results = evaluate(moe, test_data, verbose_samples=0)
    for name, stats in eval_results['per_cluster'].items():
        print(f"  {name:12s}: MSE={stats['mse']:.6f}  "
              f"routing_acc={stats['routing_accuracy']:.2%}  "
              f"n={stats['n_samples']}")
    print(f"\n  GLOBAL MSE: {eval_results['global_mse']:.6f}")
    print(f"  OVERALL ROUTING ACCURACY: {eval_results['routing_stats']['overall_accuracy']:.2%}")
    print(f"  AVG TIME PER PREDICTION: {eval_results['routing_stats']['avg_time_ms']:.3f}ms")

    # --- Expert utilization ---
    util = expert_utilization(moe, test_data)
    print(f"\n  EXPERT UTILIZATION:")
    for i, e in enumerate(experts):
        bar = '█' * int(util[i] * 40)
        print(f"  {e.name}: {util[i]:.1%} {bar}")

    # --- Swap test ---
    swap_results = swap_test(moe, test_data, train_data)

    # --- Temperature analysis ---
    temp_results = analyze_temperature(moe, test_data)

    # --- Expert scaling ---
    scale_results = analyze_expert_scaling(train_data, test_data)

    # --- Summary ---
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
    Architecture verified:
    ├── Routing accuracy: {eval_results['routing_stats']['overall_accuracy']:.1%}
    │   (random baseline: {1/len(experts):.1%})
    ├── Swap isolation ratio: {swap_results['isolation_ratio']:.1f}x
    ├── Expert utilization: { {e.name: f'{u:.1%}' for e,u in zip(experts, util)} }
    ├── Avg prediction time: {eval_results['routing_stats']['avg_time_ms']:.2f}ms
    └── Profiler accuracy: {profiler_acc:.1%}
    """)

    # Export full results as JSON for further analysis
    export = {
        'evaluation': eval_results,
        'swap_test': {
            'baseline_mse': {k: v['mse'] for k, v in swap_results['baseline']['per_cluster'].items()},
            'after_swap_mse': {k: v['mse'] for k, v in swap_results['after_swap']['per_cluster'].items()},
            'deltas': swap_results['deltas'],
            'isolation_ratio': swap_results['isolation_ratio'],
        },
        'temperature_analysis': temp_results,
        'expert_scaling': scale_results,
        'expert_profiles': {e.name: e.profile.tolist() for e in experts},
        'expert_calibration': {e.name: e.calibration_mse for e in experts},
    }

    with open('/home/someone/profile-moe/results.json', 'w') as f:
        json.dump(export, f, indent=2, default=float)
    print(f"\nFull results exported to: /home/someone/profile-moe/results.json")

    return moe, eval_results, swap_results


if __name__ == '__main__':
    main()
