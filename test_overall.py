
from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator, logistic_loss
from realkd.search import search_methods
from numba import njit, parallel_chunksize
import pandas as pd
import sortednp as snp
import doctest
from realkd.search import GreedySearch
import matplotlib.pyplot as plt

from collections import defaultdict, deque
from sortedcontainers import SortedSet
from math import inf
from heapq import heappop, heappush
from numpy import array
from bitarray import bitarray
import collections.abc
from numba.typed import List
from math import inf
from numpy import arange, argsort, array, cumsum, exp, full_like, log2, stack, zeros, zeros_like
from pandas import qcut, Series
import time
from sklearn.base import BaseEstimator, clone

from realkd.search import Conjunction, Context, KeyValueProposition, Constraint
from bitarray.util import subset
import numpy as np
import pandas as pd
import warnings
from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator, logistic_loss


from realkd.logic import Conjunction, Constraint, KeyValueProposition, TabulatedProposition

RNG = np.random.default_rng(seed=0)

def rand_array(size, alpha=0.2):
    n, k = size
    d = np.arange(n*k)
    RNG.shuffle(d)
    d = (d < alpha*len(d)).astype(int)
    return d.reshape(n, k)

# n = number of columns
# m = number of rows
ns = np.arange(500, 501, 50)
ms = np.arange(15000, 15051, 1000)

alpha = 0.5

d = {}
for m in ms:
    for n in ns:
        X = rand_array((m, n), alpha=alpha)
        true_weights = RNG.random(n) * 10
        y = X @ true_weights + RNG.random(m)
        y = np.sign(y - y.mean())
        dfX = pd.DataFrame(data=X, index=None, columns=[f'x{i}' for i in range(X.shape[1])])
        dfy = pd.Series(data=y)
        d[(m, n)] = (dfX, dfy)


re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss, search='greedy'))

def re_fit(data):
  re.fit(data[0], data[1])


re_fit(d[(ms[0], ns[0])])