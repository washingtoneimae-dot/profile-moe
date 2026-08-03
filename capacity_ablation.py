"""
capacity_ablation.py

Reproduces TECHNICAL.md section 3.4: does giving the gate detector far more
parameters (without co-training) resolve the genuinely-ambiguous boundary
cases, or is that ambiguity a property of the data (Bayes error)?

Run: python3 capacity_ablation.py
Expected runtime: ~20-40s on CPU.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from shared_data import BASE_CLUSTERS, NEW_CLUSTERS, gen_cluster

LAW_CENTER_FN = NEW_CLUSTERS['law']


def main():
    rng_train = np.random.RandomState(42)
    rng_test = np.random.RandomState(142)

    train_X, train_y = [], []
    for name, (c, fn) in BASE_CLUSTERS.items():
        xy, _ = gen_cluster(c, fn, 400, rng_train); train_X.append(xy); train_y += [0]*400
    xy, _ = gen_cluster(*LAW_CENTER_FN, 400, rng_train); train_X.append(xy); train_y += [1]*400
    train_X = np.vstack(train_X); train_y = np.array(train_y)

    test_X, test_y = [], []
    for name, (c, fn) in BASE_CLUSTERS.items():
        xy, _ = gen_cluster(c, fn, 2000, rng_test); test_X.append(xy); test_y += [0]*2000
    xy, _ = gen_cluster(*LAW_CENTER_FN, 2000, rng_test); test_X.append(xy); test_y += [1]*2000
    test_X = np.vstack(test_X); test_y = np.array(test_y)

    scaler = StandardScaler().fit(train_X)

    small = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=1)
    small.fit(scaler.transform(train_X), train_y)
    n_small = sum(w.size for w in small.coefs_) + sum(b.size for b in small.intercepts_)

    large = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=1)
    large.fit(scaler.transform(train_X), train_y)
    n_large = sum(w.size for w in large.coefs_) + sum(b.size for b in large.intercepts_)

    print(f"Small gate: {n_small} params   Large gate: {n_large} params ({n_large/n_small:.0f}x)")

    small_scores = small.predict_proba(scaler.transform(test_X))[:, 1]
    large_scores = large.predict_proba(scaler.transform(test_X))[:, 1]

    print(f"Overall AUC -- small: {roc_auc_score(test_y, small_scores):.5f}  "
          f"large: {roc_auc_score(test_y, large_scores):.5f}")

    ambiguous = (large_scores > 0.3) & (large_scores < 0.7)
    print(f"Genuinely ambiguous points (large gate's own score in 0.3-0.7): "
          f"{ambiguous.sum()}/{len(test_X)} ({100*ambiguous.mean():.2f}%)")

    small_acc = ((small_scores[ambiguous] > 0.5).astype(int) == test_y[ambiguous]).mean()
    large_acc = ((large_scores[ambiguous] > 0.5).astype(int) == test_y[ambiguous]).mean()
    print(f"Accuracy on ambiguous subset -- small: {small_acc*100:.1f}%  large: {large_acc*100:.1f}%")

    small_overall = ((small_scores > 0.5).astype(int) == test_y).mean()
    large_overall = ((large_scores > 0.5).astype(int) == test_y).mean()
    print(f"Overall accuracy -- small: {small_overall*100:.3f}%  large: {large_overall*100:.3f}%")

    large_fixes_small = (((small_scores>0.5).astype(int)!=test_y) & ((large_scores>0.5).astype(int)==test_y)).sum()
    small_fixes_large = (((large_scores>0.5).astype(int)!=test_y) & ((small_scores>0.5).astype(int)==test_y)).sum()
    print(f"Large fixes what small got wrong: {large_fixes_small}   "
          f"Small fixes what large got wrong: {small_fixes_large}")
    print("(if these two numbers are close, there is no systematic capacity advantage --"
          " the ambiguity is data-driven, not model-driven)")


if __name__ == '__main__':
    main()
