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
warnings.filterwarnings("ignore")


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

@njit
def intersect_sorted_arrays(A, B):
  """
  Returns the sorted intersection of A and B
  - Assumes A and B are sorted
  - Assumes A and B each have no duplicates
  """
  i = 0
  j = 0
  intersection = List()

  while i < len(A) and j < len(B):
      if A[i] == B[j]:
          intersection.append(A[i])
          i += 1
          j += 1
      elif A[i] < B[j]:
          i += 1
      else:
          j += 1
  return np.asarray(intersection)

def get_greedy_search(objective_function):
    @njit
    def run_greedy_search(initial_extent, n, extents):
        """
        Runs the configured search.

        :return: :class:`~realkd.logic.Conjunction` that (approximately) maximizes objective
        """
        intent = List([-1])
        extent = initial_extent
        value = objective_function(extent)
        while True:
            best_i, best_ext = None, None
            for i in range(n):
                for index in intent:
                    if index == i:
                        continue
                _extent = intersect_sorted_arrays(extent, extents[i])
                _value = objective_function(_extent)
                if _value > value:
                    value = _value
                    best_ext = _extent
                    best_i = i
            if best_i is not None:
                # Found a good addition, update intent and try again
                intent.append(best_i)
                extent = best_ext
            else:
                # Intent can't get any better
                break
        return intent[1:]
    return run_greedy_search

@njit
def run_greedy_search(initial_extent, n, extents, objective_function):
    """
    Runs the configured search.

    :return: :class:`~realkd.logic.Conjunction` that (approximately) maximizes objective
    """
    intent = [-1]
    extent = initial_extent
    value = objective_function(extent)
    while True:
        best_i, best_ext = None, None
        for i in range(n):
            for index in intent:
                if index == i:
                    continue
            _extent = intersect_sorted_arrays(extent, extents[i])
            _value = objective_function(_extent)
            if _value > value:
                value = _value
                best_ext = _extent
                best_i = i
        if best_i is not None:
            # Found a good addition, update intent and try again
            intent.append(best_i)
            extent = best_ext
        else:
            # Intent can't get any better
            break
    return intent[1:]

class NumbaGreedySearch:
    def __init__(self, ctx, obj, bdn, verbose=False, **kwargs):
        """

        :param Context ctx: the context defining the search space
        :param callable obj: objective function
        :param callable bnd: bounding function satisfying that ``bnd(q) >= max{obj(r) for r in successors(q)}`` (for signature compatibility only, not currently used)
        :param int verbose: level of verbosity

        """
        self.ctx = ctx
        self.f = obj
        self.verbose = verbose

    def run(self):
        """
        Runs the configured search.

        :return: :class:`~realkd.logic.Conjunction` that (approximately) maximizes objective
        """
        initial_extent = np.array(self.ctx.extension([]))
        n = self.ctx.n
        extents = List(self.ctx.extents)

        intent = run_greedy_search(initial_extent, self.ctx.n, extents, objective_function=self.f)

        return Conjunction(map(lambda i: self.ctx.attributes[i], intent))

def build_numba_obj_function(X, y):
    """
    This would be part of the objective e.g GradientBoostingObjective
    """
    # The following is only ONE example of this class of objective function, it would be different for
    # different X, y, losses, regs and predictions.
    loss = logistic_loss
    reg = 1.0
    predictions = zeros_like(y)
    g = array(loss.g(y, predictions))
    h = array(loss.h(y, predictions))
    r = g / h
    order = argsort(r)[::-1]
    g = g[order]
    h = h[order]
    n = y.shape[0]

    @njit
    def objective_function(ext):
        if len(ext) == 0:
            return -inf
        g_q = g[ext]
        h_q = h[ext]
        return g_q.sum() ** 2 / (2 * n * (reg + h_q.sum()))

    return objective_function

# run_greedy_search = get_greedy_search(get_greedy_search)

def run_search_numba(data, context):
    ctx, obj_fn = context
    search = NumbaGreedySearch(ctx=ctx, obj=obj_fn, bdn=None)
    search.run()


def run_search_base(data, context):
    ctx, obj_fn = context
    search = GreedySearch(ctx=ctx, obj=obj_fn, bdn=None)
    search.run()




t_numba = {}
t_base = {}

ms_to_plot = ms
ns_to_plot = ns

pre_made_ctx = {}

for m in ms_to_plot:
    data = d[(m, ns[0])]
    ctx = Context.from_df(data[0])
    obj_fn = build_numba_obj_function(data[0], data[1])
    pre_made_ctx[(m, ns[0])] = ctx, obj_fn

for n in ns_to_plot:
    data = d[(ms[0], n)]
    ctx = Context.from_df(data[0])
    obj_fn = build_numba_obj_function(data[0], data[1])
    pre_made_ctx[(ms[0], n)] = ctx, obj_fn


# Dry runs to compile numba code
run_search_numba(d[(ms[0], ns[0])], pre_made_ctx[(ms[0], ns[0])])
run_search_base(d[(ms[0], ns[0])], pre_made_ctx[(ms[0], ns[0])])

for m in ms_to_plot:
    run_search_numba(d[(m, ns[0])], pre_made_ctx[(m, ns[0])])
    # for i in range(700):
        # run_search_base(d[(m, ns[0])], pre_made_ctx[(m, ns[0])])

for n in ns_to_plot:
    run_search_numba(d[(ms[0], n)], pre_made_ctx[(ms[0], n)])
    # for i in range(700):
        # t_base[(ms[0], n)] = run_search_base(d[(ms[0], n)], pre_made_ctx[(ms[0], n)])