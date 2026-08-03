"""
multi_dimension_compounding.py

Reproduces TECHNICAL.md section 4: adding M new domains simultaneously,
each with an independently-calibrated 1%-FPR gate, does not preserve a 1%
AGGREGATE false-capture rate on old-domain data -- it compounds. Also checks
whether Bonferroni-style threshold correction restores the target rate, and
what it costs in recall.

Run: python3 multi_dimension_compounding.py
Expected runtime: ~30-60s on CPU.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from shared_data import BASE_CLUSTERS, NEW_CLUSTERS, gen_cluster, generate_dataset, ALL_CLUSTERS

BASE_NAMES = sorted(BASE_CLUSTERS.keys())
NEW_NAMES = sorted(NEW_CLUSTERS.keys())  # finance, law, medicine
ALL_NAMES = sorted(BASE_NAMES + NEW_NAMES)


def main():
    train, test = generate_dataset(ALL_CLUSTERS, train_seed=42, test_seed=142)

    full_scaler = StandardScaler()
    Xall = np.vstack([train[n]['X'] for n in ALL_NAMES]); full_scaler.fit(Xall)
    yall = np.concatenate([[n]*len(train[n]['X']) for n in ALL_NAMES])

    gates = {}
    for nf in NEW_NAMES:
        clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=7)
        clf.fit(full_scaler.transform(Xall), (yall == nf).astype(int))
        gates[nf] = clf

    rng_calib = np.random.RandomState(999)
    calib_X = np.vstack([gen_cluster(c, fn, 2000, rng_calib)[0] for c, fn in BASE_CLUSTERS.values()])

    print(f"Adding {len(NEW_NAMES)} new fields simultaneously: {NEW_NAMES}\n")

    # Uncorrected: each gate independently at 99th pct
    thr_unc = {nf: np.percentile(gates[nf].predict_proba(full_scaler.transform(calib_X))[:, 1], 99)
               for nf in NEW_NAMES}
    fires = np.zeros(len(calib_X), dtype=bool)
    for nf in NEW_NAMES:
        s = gates[nf].predict_proba(full_scaler.transform(calib_X))[:, 1]
        fired = s >= thr_unc[nf]
        print(f"  {nf} gate alone: {fired.mean()*100:.2f}% false-capture on old-domain data")
        fires |= fired
    print(f"  AGGREGATE (at least one gate fires): {fires.mean()*100:.2f}%")
    naive = 100*(1 - np.prod([1-(gates[nf].predict_proba(full_scaler.transform(calib_X))[:,1]>=thr_unc[nf]).mean() for nf in NEW_NAMES]))
    print(f"  Naive independence prediction: ~{naive:.2f}%\n")

    # Bonferroni-corrected
    target_alpha = 0.01
    per_gate_alpha = target_alpha / len(NEW_NAMES)
    thr_cor = {nf: np.percentile(gates[nf].predict_proba(full_scaler.transform(calib_X))[:, 1],
                                   100*(1-per_gate_alpha)) for nf in NEW_NAMES}
    fires_cor = np.zeros(len(calib_X), dtype=bool)
    for nf in NEW_NAMES:
        s = gates[nf].predict_proba(full_scaler.transform(calib_X))[:, 1]
        fires_cor |= (s >= thr_cor[nf])
    print(f"Bonferroni-corrected (each gate at {100*(1-per_gate_alpha):.2f}th pct, target={target_alpha*100:.0f}%):")
    print(f"  AGGREGATE: {fires_cor.mean()*100:.2f}%\n")

    print("Recall on genuine new-field test data (uncorrected vs corrected):")
    for nf in NEW_NAMES:
        s = gates[nf].predict_proba(full_scaler.transform(test[nf]['X']))[:, 1]
        r_unc = (s >= thr_unc[nf]).mean()
        r_cor = (s >= thr_cor[nf]).mean()
        print(f"  {nf}: uncorrected={r_unc*100:.1f}%  corrected={r_cor*100:.1f}%")


if __name__ == '__main__':
    main()
