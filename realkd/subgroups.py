"""
Early experimental interface to subgroup discovery methods.
"""

from math import inf
from numpy import arange, argsort, cumsum
import numpy as np

from sklearn.base import BaseEstimator

from realkd.search import (
    Conjunction,
    Context,
    IndexValueProposition,
    Constraint,
    search_methods,
)
from realkd.rules import Rule
from realkd.utils import validate_data


class Impact:
    """
    Impact objective function for conjunctive queries with respect to a specific
    dataset D and target variable y. Formally:

    impact(q) = |ext(q)|/|D| (mean(y; ext(q)) - mean(y; D)) .

    Accepts list-like, dict-like, and Pandas dataframe objects. For example:
    >>> import pandas as pd
    >>> titanic = pd.read_csv("./datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> old_male = Conjunction([IndexValueProposition(3, 'Age', Constraint.greater_equals(60)),
    ...                         IndexValueProposition(2, 'Sex', Constraint.equals('male'))])
    >>> imp_survival = Impact(titanic, 1)
    >>> imp_survival(old_male)
    -0.006110487591969073
    >>> imp_survival.bound(old_male)
    0.002074618236234398
    >>> imp_survival.search()
    Sex==female
    """

    def __init__(self, data, target):
        # target = target.to_numpy() # TODO: accept numpy
        data = validate_data(data)
        self.m = len(data)
        # print(data, target)
        # self.data = data.sort_values(target, ascending=False)  # data
        self.data = data[argsort(data[:, target])]  # data
        # self.data.reset_index(drop=True, inplace=True)
        self.target = target
        self.mean = self.data[:, self.target].mean()

    def __call__(self, q):
        extent = q(self.data)
        local_mean = extent[self.target].mean()
        return len(extent) / self.m * (local_mean - self.mean)

    def bound(self, q):
        extent = q(self.data)
        data = extent[self.target]
        n = len(extent)
        if n == 0:
            return -inf
        s = cumsum(data)
        return (s - arange(1, n + 1) * self.mean).max() / self.m

    def search(self, search="exhaustive", verbose=False):
        ctx = Context.from_array(self.data, without=[self.target], max_col_attr=10)
        return search_methods[search](ctx, self, self.bound, verbose=verbose).run()
        # return ctx.exhaustive(self, self.bound, order=order, verbose=verbose)


class ImpactRuleEstimator(BaseEstimator):
    """
    Fits rules with conjunctive query based on multiplicative combination
    of query coverage and effect of query satisfaction on target mean.
    Formally, for dataset D and target variable y:

    .. math::
        :nowrap:

        \begin{equation}
        \mathrm{imp}(q) = \left(\frac{|\mathrm{ext}(q)|}{|D|}\right)^\alpha (\mathrm{mean}(y; \mathrm{ext}(q)) - \mathrm{mean}(y; D)) .
        \end{equation}

    >>> import pandas as pd
    >>> titanic = pd.read_csv("./datasets/titanic/train.csv")
    >>> survived = titanic['Survived']
    >>> titanic.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> subgroup = ImpactRuleEstimator(search='exhaustive', verbose=False)
    >>> subgroup.fit(titanic, survived).rule_
       +0.7420 if Sex==female
    >>> subgroup.score(titanic, survived)
    0.12623428448344273
    >>> subgroup2 = ImpactRuleEstimator(alpha=0.5, search='exhaustive', verbose=False)
    >>> subgroup2.fit(titanic, survived).rule_
       +0.9471 if Pclass<=2 & Sex==female
    >>> subgroup2.score(titanic, survived)
    0.24601637556150627
    """

    def __init__(self, alpha=1.0, search="greedy", search_params={}, verbose=False):
        """

        :param alpha: (exponential) weight of coverage term
        :param str|type search: search method either specified via string identifier (e.g., ``'greedy'`` or ``'exhaustive'``) or directly as search type (see :func:`realkd.search.search_methods`)
        :param dict search_params: parameters to apply to discretization (when creating binary search context from
                              dataframe via :func:`~realkd.search.Context.from_df`) as well as to actual search method
                              (specified by ``method``). See :mod:`~realkd.search`.
        :param verbose:
        """
        self.alpha = alpha
        self.search = search
        self.search_params = search_params
        self.verbose = verbose
        self.rule_ = None

    def score(self, data, target):
        data = validate_data(data)
        target = target.to_numpy()

        ext = self.rule_.q(data).nonzero
        global_mean = target.mean()
        local_mean = target[ext].mean()
        return (len(ext) / len(data)) ** self.alpha * (local_mean - global_mean)

    def fit(self, data, target, labels=None):
        data = validate_data(data, labels)

        m = len(data)

        order = argsort(target)[::-1]
        data = data[order]
        target = target[order]

        global_mean = target.mean()

        def obj(extent):
            local_mean = target[extent].mean()
            return (len(extent) / m) ** self.alpha * (local_mean - global_mean)

        def bnd(extent):
            _target = target[extent]
            n = len(extent)
            if n == 0:
                return -inf
            s = cumsum(_target)
            i = arange(1, n + 1)
            covs = i / m
            means = s / i
            vals = covs**self.alpha * means
            return vals.max()

        ctx = Context.from_array(data, max_col_attr=10)
        q = search_methods[self.search](
            ctx, obj, bnd, verbose=self.verbose, **self.search_params
        ).run()
        ext = q(data).nonzero()
        y = target[ext].mean()
        self.rule_ = Rule(q, y)
        return self


if __name__ == "__main__":
    import doctest

    doctest.testmod()
