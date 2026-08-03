"""
shared_data.py -- canonical synthetic data generator and shared infrastructure
used by every script in this reproduction suite. Every finding in TECHNICAL.md
that uses synthetic data uses THIS generator, with the seeds specified at each
call site, so results are exactly reproducible.

Domains and their (center, generating function):
  code:      (0,0)     f(x,y) = x^2 + y
  math:      (5,0)     f(x,y) = sin(x) * y
  creative:  (0,5)     f(x,y) = x * cos(y)
  reasoning: (5,5)     f(x,y) = sqrt(x^2 + y^2)
  law:       (2.5,2.5) f(x,y) = sigmoid(x-2.5)*3 + y*0.5
  medicine:  (3.5,-1.0) f(x,y) = cos(x)*2 + y
  finance:   (-1.0,3.0) f(x,y) = x*1.5 + sin(y)

law/medicine/finance are the "addable" domains used in addition-isolation
experiments; code/math/creative/reasoning are the fixed base pool.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass, field

BASE_CLUSTERS = {
    'code':      ([0.0, 0.0], lambda x, y: x**2 + y),
    'math':      ([5.0, 0.0], lambda x, y: np.sin(x) * y),
    'creative':  ([0.0, 5.0], lambda x, y: x * np.cos(y)),
    'reasoning': ([5.0, 5.0], lambda x, y: np.sqrt(x**2 + y**2)),
}
NEW_CLUSTERS = {
    'law':      ([2.5, 2.5],  lambda x, y: 1/(1+np.exp(-(x-2.5)))*3 + y*0.5),
    'medicine': ([3.5, -1.0], lambda x, y: np.cos(x)*2 + y),
    'finance':  ([-1.0, 3.0], lambda x, y: x*1.5 + np.sin(y)),
}
ALL_CLUSTERS = {**BASE_CLUSTERS, **NEW_CLUSTERS}


def gen_cluster(center, fn, n, rng, noise=0.15, ynoise=0.1):
    """Draw n points from one domain's generating distribution."""
    xy = rng.randn(n, 2) * 0.8 + np.array(center)
    xy += rng.randn(*xy.shape) * noise
    z = fn(xy[:, 0], xy[:, 1]) + rng.randn(n) * ynoise
    return xy, z


def generate_dataset(domains, n_train=400, n_test=150, train_seed=42, test_seed=142):
    """Generate train/test data for a given dict of {name: (center, fn)}.
    Use ALL_CLUSTERS (or a subset) as `domains`."""
    rng_train = np.random.RandomState(train_seed)
    rng_test = np.random.RandomState(test_seed)
    train, test = {}, {}
    for name, (center, fn) in domains.items():
        xy, z = gen_cluster(center, fn, n_train, rng_train)
        train[name] = {'X': xy, 'y': z}
        xy, z = gen_cluster(center, fn, n_test, rng_test)
        test[name] = {'X': xy, 'y': z}
    return train, test


@dataclass
class Expert:
    """One expert: a regressor plus a calibrated profile vector."""
    name: str
    model: object
    profile: np.ndarray = None
    calibration_mse: dict = field(default_factory=dict)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def calibrate(self, calibration_data, profile_dims):
        """Profile[i] = normalized inverse-MSE on domain i's calibration set.
        This is THE profile-vector formula used throughout the project."""
        mse = {d: mean_squared_error(calibration_data[d]['y'], self.predict(calibration_data[d]['X']))
               for d in profile_dims}
        self.calibration_mse = mse
        skills = np.array([1.0 / (mse[d] + 1e-8) for d in profile_dims])
        self.profile = skills / skills.sum()


def build_expert(name, train_data, calib_data, profile_dims, seed,
                  hidden_layer_sizes=(16,)):
    """Train one expert's regressor and calibrate its profile."""
    m = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000,
                      early_stopping=True, random_state=seed)
    m.fit(train_data[name]['X'], train_data[name]['y'])
    e = Expert(name=f"Expert_{name}", model=m)
    e.calibrate(calib_data, profile_dims)
    return e


def cosine_top1(input_profile, expert_profile_matrix):
    """THE routing formula: cosine similarity, top-1 by argmax."""
    ipn = input_profile / (np.linalg.norm(input_profile) + 1e-8)
    epn = expert_profile_matrix / (np.linalg.norm(expert_profile_matrix, axis=1, keepdims=True) + 1e-8)
    sims = epn @ ipn
    return np.argmax(sims), sims


def build_profiler(train_data, domains, hidden_layer_sizes=(16, 8), seed=42):
    """A profiler = a classifier over `domains`, predicting a probability
    distribution used as the input's profile vector. This is used both as
    the FROZEN base profiler and as the JOINTLY-RETRAINED profiler --
    the distinction is which set of `domains` and which data you fit it on."""
    scaler = StandardScaler()
    X = np.vstack([train_data[d]['X'] for d in domains])
    scaler.fit(X)
    y = np.concatenate([[d] * len(train_data[d]['X']) for d in domains])
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, random_state=seed)
    clf.fit(scaler.transform(X), y)
    classes = list(clf.classes_)

    def profile_fn(X_query, order=None):
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)
        p = clf.predict_proba(scaler.transform(X_query))
        if order is None:
            order = sorted(domains)
        idx = [classes.index(n) for n in order]
        return p[:, idx]

    return profile_fn, clf, scaler
