"""
Early experimental interface to subgroup discovery methods.
"""

from math import inf
from numpy import arange, argsort, cumsum

from sklearn.base import BaseEstimator

from realkd.search import Conjunction, Context, KeyValueProposition, Constraint, search_methods
from realkd.rules import Rule


class Impact:
    """
    Impact objective function for conjunctive queries with respect to a specific
    dataset D and target variable y. Formally:

    impact(q) = |ext(q)|/|D| (mean(y; ext(q)) - mean(y; D)) .

    Accepts list-like, dict-like, and Pandas dataframe objects. For example:
    >>> import pandas as pd
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> old_male = Conjunction([KeyValueProposition('Age', Constraint.greater_equals(60)),
    ...                         KeyValueProposition('Sex', Constraint.equals('male'))])
    >>> imp_survival = Impact(titanic, 'Survived')
    >>> imp_survival(old_male)
    -0.006110487591969073
    >>> imp_survival.bound(old_male)
    0.002074618236234398
    >>> imp_survival.search()
    Sex==female
    """

    def __init__(self, data, target):
        self.m = len(data)
        self.data = data.sort_values(target, ascending=False)  # data
        self.data.reset_index(drop=True, inplace=True)
        self.target = target
        self.mean = self.data[self.target].mean()

    def __call__(self, q):
        extent = self.data.loc[q]
        local_mean = extent[self.target].mean()
        return len(extent)/self.m * (local_mean - self.mean)

    def bound(self, q):
        extent = self.data.loc[q]
        data = extent[self.target]
        n = len(extent)
        if n == 0:
            return -inf
        s = cumsum(data)
        return (s - arange(1, n + 1)*self.mean).max() / self.m

    def search(self, search='exhaustive', verbose=False):
        ctx = Context.from_df(self.data, without=[self.target], max_col_attr=10)
        return search_methods[search](ctx, self, self.bound, verbose=verbose).run()
        # return ctx.exhaustive(self, self.bound, order=order, verbose=verbose)


class ImpactRuleEstimator(BaseEstimator):
    """
    Fits rules with conjunctive query based on multiplicative combination
    of query coverage and effect of query satisfaction on target mean.
    Formally, for dataset D and target variable y:

    .. math::

        \mathrm{imp}(q) = (|\mathrm{ext}(q)|/|D|) (\mathrm{mean}(y; \mathrm{ext}(q)) - \mathrm{mean}(y; D)) .


    >>> import pandas as pd
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> survived = titanic['Survived']
    >>> titanic.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> subgroup = ImpactRuleEstimator(search='exhaustive', verbose=False)
    >>> subgroup.fit(titanic, survived)
    ImpactRuleEstimator(search='exhaustive')
    >>> subgroup.rule_
       +0.7420 if Sex==female
    >>> subgroup.score(titanic, survived)
    0.1262342844834427
    """

    def __init__(self, gamma=1.0, search='greedy', search_params={}, verbose=False):
        # TODO: use gamma in score and optimisation
        self.gamma = gamma
        self.search = search
        self.search_params = search_params
        self.verbose = verbose
        self.rule_ = None

    def score(self, data, target):
        ext = data.loc[self.rule_.q].index
        global_mean = target.mean()
        local_mean = target[ext].mean()
        return len(ext)*(local_mean-global_mean)/len(data)

    def fit(self, data, target):
        m = len(data)

        order = argsort(target)[::-1]
        data = data.iloc[order].reset_index(drop=True)
        target = target.iloc[order].reset_index(drop=True)

        global_mean = target.mean()

        def obj(extent):
            local_mean = target[extent].mean()
            return len(extent) / m * (local_mean - global_mean)

        def bnd(extent):
            _target = target[extent]
            n = len(extent)
            if n == 0:
                return -inf
            s = cumsum(_target)
            return (s - arange(1, n + 1) * global_mean).max() / m

        ctx = Context.from_df(data, max_col_attr=10)
        q = search_methods[self.search](ctx, obj, bnd, verbose=self.verbose, **self.search_params).run()
        ext = data.loc[q].index
        y = target[ext].mean()
        self.rule_ = Rule(q, y)
        return self


if __name__ == '__main__':
    import doctest
    doctest.testmod()
