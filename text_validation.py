"""
text_validation.py

Reproduces TECHNICAL.md section 5: does the addition-flip mechanism replicate
on real text vocabulary (not just synthetic 2D coordinates), and does
systematically-generated (vs. hand-picked) boundary calibration data fix the
degenerate-threshold problem found with a small hand-written corpus?

Uses TF-IDF + SVD (lexical/co-occurrence structure) -- NOT a trained semantic
embedding model, since no network access was available to fetch one. This is
a real limitation, stated explicitly in TECHNICAL.md section 7.

Run: python3 text_validation.py
Expected runtime: ~10-20s on CPU.
"""
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DOMAINS = {
    'code': {'subjects': ['the function', 'this algorithm', 'the recursive method', 'the API endpoint', 'this class'],
              'actions': ['throws an exception when', 'needs to be refactored because', 'has a bug where', 'should be optimized to reduce'],
              'objects': ['the array index is out of bounds', 'memory usage grows unbounded', 'the syntax is invalid', 'the recursion never terminates']},
    'math': {'subjects': ['the theorem', 'this integral', 'the matrix', 'the probability distribution', 'the derivative'],
              'actions': ['can be solved by applying', 'converges when', 'requires proving that', 'simplifies to'],
              'objects': ['the fundamental theorem of calculus', 'the denominator approaches zero', 'the eigenvalues are real', 'integration by parts']},
    'creative': {'subjects': ['the protagonist', 'this poem', 'the narrative arc', 'the dialogue', 'the character'],
                  'actions': ['reveals their true motivation when', 'uses vivid imagery to show', 'builds tension through'],
                  'objects': ['a sense of loss and longing', 'the changing seasons', 'an unreliable narrator']},
    'reasoning': {'subjects': ['the argument', 'this syllogism', 'the premise', 'the logical fallacy'],
                   'actions': ['fails because it assumes', 'is valid only if', 'commits the error of'],
                   'objects': ['correlation implies causation', 'circular reasoning', 'a false dichotomy']},
    'law': {'subjects': ['the contract', 'this statute', 'the plaintiff', 'the defendant', 'the court'],
             'actions': ['is enforceable only if', 'establishes liability when', 'requires compliance with'],
             'objects': ['both parties provided consideration', 'the statute of limitations has expired', 'a material breach occurred']},
}

BOUNDARY_PROMPTS = {
    'code': ["review the open source license agreement attached to this repository for compliance issues",
             "the software patent covers the specific algorithm implementation used in this function",
             "draft terms of service clauses governing API usage limits for this application",
             "assess whether this code snippet infringes on a competitor's registered copyright"],
    'math': ["calculate the statute of limitations deadline given the filing date and applicable tolling rules",
             "prove that the settlement formula satisfies the contractual interest rate cap",
             "derive the amortization schedule required under the loan agreement's payment terms"],
    'creative': ["write a closing argument in the voice of a defense attorney pleading for leniency",
                 "compose a courtroom drama scene where the plaintiff's testimony contradicts the evidence"],
    'reasoning': ["evaluate whether the prosecutor's argument commits a logical fallacy in establishing intent",
                  "assess if the contract's ambiguous clause creates a valid basis for rescission"],
}

BASE_NAMES = ['code', 'math', 'creative', 'reasoning']


def make_sentences(d, n, rng):
    v = DOMAINS[d]
    return [f"{rng.choice(v['subjects'])} {rng.choice(v['actions'])} {rng.choice(v['objects'])}." for _ in range(n)]


def make_boundary_systematic(domain_a, domain_b, n, rng):
    va, vb = DOMAINS[domain_a], DOMAINS[domain_b]
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            s = f"{rng.choice(va['subjects'])} {rng.choice(vb['actions'])} {rng.choice(vb['objects'])}."
        else:
            s = f"{rng.choice(vb['subjects'])} {rng.choice(va['actions'])} {rng.choice(va['objects'])}."
        out.append(s)
    return out


def part1_real_text_flip_test():
    print("="*70)
    print("PART 1: does the flip mechanism replicate on real text?")
    print("="*70)
    random.seed(42)
    rng = random.Random(42)
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for d in BASE_NAMES:
        tr = make_sentences(d, 60, rng)
        te = make_sentences(d, 30, rng) + BOUNDARY_PROMPTS.get(d, [])
        train_texts += tr; train_labels += [d]*len(tr)
        test_texts += te; test_labels += [d]*len(te)
    law_train = make_sentences('law', 60, rng)
    law_test = make_sentences('law', 30, rng)
    train_texts += law_train; train_labels += ['law']*len(law_train)
    test_texts += law_test; test_labels += ['law']*len(law_test)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X_train_tfidf = tfidf.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_train = svd.fit_transform(X_train_tfidf)
    X_test = svd.transform(tfidf.transform(test_texts))
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train); X_test_s = scaler.transform(X_test)

    base_mask = np.array([l in BASE_NAMES for l in train_labels])
    base_clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    base_clf.fit(X_train_s[base_mask], np.array(train_labels)[base_mask])

    joint_clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    joint_clf.fit(X_train_s, np.array(train_labels))

    flips, n_base_test = [], 0
    for i in range(len(test_texts)):
        if test_labels[i] not in BASE_NAMES:
            continue
        n_base_test += 1
        x = X_test_s[i].reshape(1, -1)
        base_pred = base_clf.predict(x)[0]
        joint_pred = joint_clf.predict(x)[0]
        if base_pred != joint_pred:
            flips.append((test_labels[i], test_texts[i], base_pred, joint_pred))

    print(f"Base-domain test prompts: {n_base_test}   Flips after adding law: {len(flips)}")
    for true_d, text, b, j in flips:
        print(f"  true={true_d:10s} base={b:10s} joint={j:10s}  \"{text[:70]}\"")


def part2_printer_prototype():
    print("\n" + "="*70)
    print("PART 2: systematic boundary generation ('printer' prototype) fixes calibration")
    print("="*70)
    rng = random.Random(7)
    calib_clean, calib_boundary = [], []
    for d in BASE_NAMES:
        calib_clean += make_sentences(d, 300, rng)
        calib_boundary += make_boundary_systematic(d, 'law', 150, rng)

    rng2 = random.Random(99)
    train_texts, train_labels = [], []
    for d in BASE_NAMES:
        tr = make_sentences(d, 300, rng2)
        train_texts += tr; train_labels += [d]*len(tr)
    law_train = make_sentences('law', 300, rng2)
    train_texts += law_train; train_labels += ['law']*len(law_train)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X_train_tfidf = tfidf.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    scaler = StandardScaler().fit(X_train_svd)
    X_train = scaler.transform(X_train_svd)

    gate = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=7)
    gate.fit(X_train, (np.array(train_labels) == 'law').astype(int))

    def embed(texts):
        return scaler.transform(svd.transform(tfidf.transform(texts)))

    thr_clean = np.percentile(gate.predict_proba(embed(calib_clean))[:, 1], 99)
    thr_printed = np.percentile(gate.predict_proba(embed(calib_clean + calib_boundary))[:, 1], 99)
    print(f"Threshold, clean-only calibration:            {thr_clean:.4f}")
    print(f"Threshold, printed (clean+systematic boundary): {thr_printed:.4f}")

    law_test = make_sentences('law', 100, random.Random(555))
    law_scores = gate.predict_proba(embed(law_test))[:, 1]
    print(f"Genuine law recall @ clean-only threshold: {(law_scores>=thr_clean).mean()*100:.1f}%")
    print(f"Genuine law recall @ printed threshold:    {(law_scores>=thr_printed).mean()*100:.1f}%")

    fresh_base = []
    for d in BASE_NAMES:
        fresh_base += make_sentences(d, 500, random.Random(321))
    fresh_scores = gate.predict_proba(embed(fresh_base))[:, 1]
    print(f"False-capture on fresh unambiguous data @ clean-only: {(fresh_scores>=thr_clean).mean()*100:.2f}%")
    print(f"False-capture on fresh unambiguous data @ printed:    {(fresh_scores>=thr_printed).mean()*100:.2f}%")


if __name__ == '__main__':
    part1_real_text_flip_test()
    part2_printer_prototype()
