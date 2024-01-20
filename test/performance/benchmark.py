"""
This file is modelled off 
sklearn (version 1.5)/benchmarks/bench_hist_gradient_boosting
"""

import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from realkd.rule_boosting import RuleBoostingEstimator, XGBRuleEstimator

TEST_SIZE = 0.5
N_SAMPLES_MAX = 1e5
N_FEATURES_MAX = 10

def get_data():
    # We're making a binary classifier
    n_classes = 2
    X, y = make_classification(
        n_samples=int(N_SAMPLES_MAX / TEST_SIZE),
        n_features=N_FEATURES_MAX,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_informative=n_classes,
        random_state=0,
    )
    return X, y


X, y = get_data()

X_train_, X_test_, y_train_, y_test_ = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=0
)

def one_run(n_samples, num_rules):
    X_train = X_train_[:n_samples]
    X_test = X_test_[:n_samples]
    y_train = y_train_[:n_samples]
    y_test = y_test_[:n_samples]
    assert X_train.shape[0] == n_samples
    assert X_test.shape[0] == n_samples

    print("Data size: %d samples train, %d samples test." % (n_samples, n_samples))
    print("Fitting a sklearn model...")
    tic = time()
    est = RuleBoostingEstimator(
        base_learner=XGBRuleEstimator(
            loss="logistic",
            search="exhaustive",
            # reg=10,
            # search="greedy",
            # search_params={
            #     "order": "bestboundfirst",
            #     "apx": [1.0, 1.0, 0.8],
            #     "max_depth": None,
            # },
        ),
        num_rules=num_rules,
        verbose=0,
    )
    est.fit(X_train, y_train)
    fit_duration = time() - tic
    tic = time()
    predictions = est.predict_proba(X_test)
    predict_duration = time() - tic
    roc_auc = roc_auc_score(y_test, predictions[:, 1])
    print("ROC AUC: {:.4f}".format(roc_auc))
    print("fit duration: {:.3f}s,".format(fit_duration))
    print("predict duration: {:.3f}s,".format(predict_duration))

    return (
        roc_auc,
        fit_duration,
        predict_duration
    )


n_samples_list = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
n_samples_list = [
    n_samples for n_samples in n_samples_list if n_samples <= N_SAMPLES_MAX
]

roc_auc_scores = []
fit_durations = []
predict_durations = []

for n_samples in n_samples_list:
    (
        roc_auc,
        fit_duration,
        predict_duration,
    ) = one_run(n_samples, num_rules=5)

    for scores, score in (
        (roc_auc_scores, roc_auc),
        (fit_durations, fit_duration),
        (predict_durations, predict_duration),
    ):
        scores.append(score)

fig, axs = plt.subplots(3, sharex=True)

axs[0].plot(n_samples_list, scores, label="sklearn")
axs[1].plot(n_samples_list, fit_durations, label="sklearn")
axs[2].plot(n_samples_list, predict_durations, label="sklearn")

for ax in axs:
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.set_xlabel("n_samples")

axs[0].set_title("scores")
axs[1].set_title("fit duration (s)")
axs[2].set_title("score duration (s)")

title = "Classification"
fig.suptitle(title)


plt.tight_layout()
plt.show()


# (env) alastair@DESKTOP-AVFRJTF:/mnt/c/Users/locke/Dropbox/Programming/Python/RA/realkd.py$ python3 test/performance/benchmark.py
# Data size: 1000 samples train, 1000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9881
# fit duration: 1.095s,
# predict duration: 0.000s,
# Data size: 10000 samples train, 10000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9823
# fit duration: 7.886s,
# predict duration: 0.001s,
# Data size: 100000 samples train, 100000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9828
# fit duration: 61.370s,
# predict duration: 0.014s,
# Data size: 500000 samples train, 500000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9818
# fit duration: 479.359s,
# predict duration: 0.517s,
# Data size: 1000000 samples train, 1000000 samples test.
# Fitting a sklearn model...
#   Killed
# (env) alastair@DESKTOP-AVFRJTF:/mnt/c/Users/locke/Dropbox/Programming/Python/RA/realkd.py$ python3 test/performance/benchmark.py
# Data size: 1000 samples train, 1000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9909
# fit duration: 11.157s,
# predict duration: 0.000s,
# Data size: 5000 samples train, 5000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9886
# fit duration: 31.810s,
# predict duration: 0.001s,
# Data size: 10000 samples train, 10000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9873
# fit duration: 57.984s,
# predict duration: 0.001s,
# Data size: 50000 samples train, 50000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9876
# fit duration: 593.533s,
# predict duration: 0.152s,
# Data size: 100000 samples train, 100000 samples test.
# Fitting a sklearn model...
# ROC AUC: 0.9879
# fit duration: 1457.894s,
# predict duration: 0.595s