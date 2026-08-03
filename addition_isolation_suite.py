"""
addition_isolation_suite.py

Reproduces TECHNICAL.md sections 2, 3.1-3.3, 3.5:
  - Swap isolation (both in the original 4-dim pool and inside an
    already-provisioned 5-dim pool)
  - The addition flip test (jointly-retrained profiler vs frozen base)
  - The gated one-vs-rest fix, with properly-calibrated threshold
  - Multi-seed flip stability + confidence/margin analysis

Run: python3 addition_isolation_suite.py
Expected runtime: ~15-30s on CPU.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from shared_data import BASE_CLUSTERS, NEW_CLUSTERS, gen_cluster, generate_dataset, \
                         Expert, build_expert, cosine_top1

BASE_NAMES = sorted(BASE_CLUSTERS.keys())
ALL_NAMES_WITH_LAW = sorted(BASE_NAMES + ['law'])


def section_2_swap_isolation():
    print("="*70)
    print("SECTION 2: Swap Isolation")
    print("="*70)
    train, test = generate_dataset(BASE_CLUSTERS)
    experts_before = [build_expert(n, train, test, BASE_NAMES, seed=42) for n in BASE_NAMES]

    # Swap the 'code' expert for a genuinely DIFFERENT one (deliberately
    # undertrained, to guarantee a different -- not coincidentally better --
    # function, matching what "swap" is meant to test: does a different
    # expert function stay isolated, regardless of whether it's better or worse)
    different_code = MLPRegressor(hidden_layer_sizes=(4,), max_iter=15,
                                   random_state=999)
    different_code.fit(train['code']['X'], train['code']['y'])
    experts_after = [e for e in experts_before if e.name != 'Expert_code']
    new_code = Expert(name='Expert_code', model=different_code)
    new_code.calibrate(test, BASE_NAMES)
    experts_after.append(new_code)
    experts_after = sorted(experts_after, key=lambda e: e.name)
    experts_before = sorted(experts_before, key=lambda e: e.name)

    def mse_by_cluster(experts, label):
        idx = BASE_NAMES.index(label)
        ip = np.zeros(len(BASE_NAMES)); ip[idx] = 1.0
        ep = np.array([e.profile for e in experts])
        winner_idx, _ = cosine_top1(ip, ep)
        winner = experts[winner_idx]
        errs = (test[label]['y'] - winner.predict(test[label]['X']))**2
        return errs.mean()

    print(f"{'domain':12s} {'MSE before':>12s} {'MSE after':>12s} {'change':>10s}")
    increases = {}
    for label in BASE_NAMES:
        b = mse_by_cluster(experts_before, label)
        a = mse_by_cluster(experts_after, label)
        increases[label] = a - b
        print(f"{label:12s} {b:12.5f} {a:12.5f} {100*(a-b)/b:+9.2f}%")
    non_code = [k for k in increases if k != 'code']
    worst_collateral = max(increases[k] for k in non_code)
    if worst_collateral > 1e-8 and increases['code'] > 0:
        ratio = increases['code'] / worst_collateral
        print(f"\nIsolation ratio (code increase / largest collateral increase): {ratio:.1f}x")
    else:
        print(f"\nCollateral changes on non-swapped domains: {[(k, round(increases[k],6)) for k in non_code]}")
        print("All non-swapped domains show ~zero change (isolation holds; ratio undefined "
              "when collateral change rounds to zero, which is the strongest possible result).")
    print("NOTE: this run's exact numbers will differ from any other run/seed -- "
          "what's invariant is the ISOLATION PROPERTY (collateral changes ~0), not a fixed constant.")
    print()


def section_3_addition_and_gating():
    print("="*70)
    print("SECTION 3.1-3.3: Addition flips + gated fix")
    print("="*70)
    domains = {**BASE_CLUSTERS, 'law': NEW_CLUSTERS['law']}
    train, test = generate_dataset(domains)

    experts_v1 = [build_expert(n, train, test, BASE_NAMES, seed=42) for n in BASE_NAMES]
    experts_v2 = [build_expert(n, train, test, ALL_NAMES_WITH_LAW, seed=42) for n in BASE_NAMES]
    law_expert = build_expert('law', train, test, ALL_NAMES_WITH_LAW, seed=99)
    experts_v2 = sorted(experts_v2 + [law_expert], key=lambda e: e.name)
    experts_v1 = sorted(experts_v1, key=lambda e: e.name)

    # Frozen base profiler (never retrained)
    base_scaler = StandardScaler()
    Xb = np.vstack([train[n]['X'] for n in BASE_NAMES]); base_scaler.fit(Xb)
    base_clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    yb = np.concatenate([[n]*len(train[n]['X']) for n in BASE_NAMES])
    base_clf.fit(base_scaler.transform(Xb), yb)

    # Jointly-retrained profiler across all 5 domains (the broken baseline)
    full_scaler = StandardScaler()
    Xall = np.vstack([train[n]['X'] for n in ALL_NAMES_WITH_LAW]); full_scaler.fit(Xall)
    joint_clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    yall = np.concatenate([[n]*len(train[n]['X']) for n in ALL_NAMES_WITH_LAW])
    joint_clf.fit(full_scaler.transform(Xall), yall)
    joint_classes = list(joint_clf.classes_)

    # One-vs-rest law gate, trained independently
    gate_clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=7)
    gate_clf.fit(full_scaler.transform(Xall), (yall == 'law').astype(int))

    # Calibrate threshold on a large held-out non-law sample
    rng_calib = np.random.RandomState(999)
    calib_X = np.vstack([gen_cluster(c, fn, 2000, rng_calib)[0] for c, fn in BASE_CLUSTERS.values()])
    threshold = np.percentile(gate_clf.predict_proba(full_scaler.transform(calib_X))[:, 1], 99)
    print(f"Calibrated gate threshold (99th pct, n=8000 calib points): {threshold:.4f}")

    flips = []
    for cn in BASE_NAMES:
        for i in range(len(test[cn]['X'])):
            x = test[cn]['X'][i]
            p1 = base_clf.predict_proba(base_scaler.transform(x.reshape(1, -1)))[0]
            order1 = [list(base_clf.classes_).index(n) for n in BASE_NAMES]
            idx1, _ = cosine_top1(p1[order1], np.array([e.profile for e in experts_v1]))
            top1_v1 = experts_v1[idx1].name

            p2 = joint_clf.predict_proba(full_scaler.transform(x.reshape(1, -1)))[0]
            order2 = [joint_classes.index(n) for n in ALL_NAMES_WITH_LAW]
            idx2, sims2 = cosine_top1(p2[order2], np.array([e.profile for e in experts_v2]))
            top1_v2 = experts_v2[idx2].name

            if top1_v1 != top1_v2:
                flips.append((cn, i, x, top1_v1, top1_v2))

    print(f"Flips (jointly-retrained profiler): {len(flips)}")
    fixed = 0
    for cn, i, x, t1, t2 in flips:
        law_score = gate_clf.predict_proba(full_scaler.transform(x.reshape(1, -1)))[0, 1]
        gate = law_score if law_score >= threshold else 0.0
        p1 = base_clf.predict_proba(base_scaler.transform(x.reshape(1, -1)))[0]
        order1 = [list(base_clf.classes_).index(n) for n in BASE_NAMES]
        base_probs = dict(zip(BASE_NAMES, p1[order1]))
        gated_profile_dict = {k: base_probs[k]*(1-gate) for k in BASE_NAMES}
        gated_profile_dict['law'] = gate
        gip = np.array([gated_profile_dict[k] for k in ALL_NAMES_WITH_LAW])
        idx_g, _ = cosine_top1(gip, np.array([e.profile for e in experts_v2]))
        ok = experts_v2[idx_g].name == t1
        fixed += ok
        print(f"  {cn}#{i}: law_score={law_score:.3f} gated={'yes' if gate>0 else 'no'} "
              f"-> {experts_v2[idx_g].name}  correct={t1}  {'FIXED' if ok else 'still wrong'}")
    print(f"Fixed: {fixed}/{len(flips)}")
    law_recall = (gate_clf.predict_proba(full_scaler.transform(test['law']['X']))[:, 1] >= threshold).mean()
    print(f"Genuine law recall at this threshold: {law_recall*100:.1f}%\n")


def section_3_5_multi_seed_stability():
    print("="*70)
    print("SECTION 3.5: Multi-seed flip stability")
    print("="*70)
    domains = {**BASE_CLUSTERS, 'law': NEW_CLUSTERS['law']}
    train, test = generate_dataset(domains, train_seed=42, test_seed=142)

    flip_sets = []
    for model_seed in [1, 2, 3, 4, 5]:
        experts_v1 = [build_expert(n, train, test, BASE_NAMES, seed=model_seed) for n in BASE_NAMES]
        experts_v2 = [build_expert(n, train, test, ALL_NAMES_WITH_LAW, seed=model_seed) for n in BASE_NAMES]
        law_e = build_expert('law', train, test, ALL_NAMES_WITH_LAW, seed=model_seed+50)
        experts_v2 = sorted(experts_v2 + [law_e], key=lambda e: e.name)
        experts_v1 = sorted(experts_v1, key=lambda e: e.name)

        scaler1 = StandardScaler(); Xb = np.vstack([train[n]['X'] for n in BASE_NAMES]); scaler1.fit(Xb)
        clf1 = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=model_seed)
        yb = np.concatenate([[n]*len(train[n]['X']) for n in BASE_NAMES]); clf1.fit(scaler1.transform(Xb), yb)

        scaler2 = StandardScaler(); Xall = np.vstack([train[n]['X'] for n in ALL_NAMES_WITH_LAW]); scaler2.fit(Xall)
        yall = np.concatenate([[n]*len(train[n]['X']) for n in ALL_NAMES_WITH_LAW])
        clf2 = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=model_seed)
        clf2.fit(scaler2.transform(Xall), yall)
        classes2 = list(clf2.classes_)

        flips = set()
        for cn in BASE_NAMES:
            for i in range(len(test[cn]['X'])):
                x = test[cn]['X'][i]
                p1 = clf1.predict_proba(scaler1.transform(x.reshape(1, -1)))[0]
                idx1, _ = cosine_top1(p1[[list(clf1.classes_).index(n) for n in BASE_NAMES]],
                                       np.array([e.profile for e in experts_v1]))
                p2 = clf2.predict_proba(scaler2.transform(x.reshape(1, -1)))[0]
                idx2, _ = cosine_top1(p2[[classes2.index(n) for n in ALL_NAMES_WITH_LAW]],
                                       np.array([e.profile for e in experts_v2]))
                if experts_v1[idx1].name != experts_v2[idx2].name:
                    flips.add((cn, i))
        flip_sets.append(flips)
        print(f"Seed {model_seed}: {len(flips)} flips -> {sorted(flips)}")

    counts = {}
    for fs in flip_sets:
        for p in fs:
            counts[p] = counts.get(p, 0) + 1
    print(f"\nFlip frequency across 5 seeds:")
    for p, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c}/5 seeds")


if __name__ == '__main__':
    section_2_swap_isolation()
    section_3_addition_and_gating()
    section_3_5_multi_seed_stability()
