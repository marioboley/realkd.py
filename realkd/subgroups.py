from math import inf
from numpy import arange, cumsum
from realkd.search import Conjunction, Context, KeyValueProposition, Constraint


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

    def search(self, order='breadthfirst', verbose=False):
        ctx = Context.from_df(self.data, without=[self.target], max_col_attr=10)
        return ctx.search(self, self.bound, order=order, verbose=verbose)


if __name__=='__main__':

    from timeit import timeit

    setup1 = \
"""import pandas as pd
from realkd.subgroups import Impact
titanic = pd.read_csv("../datasets/titanic/train.csv")
titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
imp_survival = Impact(titanic, 'Survived')"""

    setup2 = \
"""import pandas as pd
from realkd.search import Impact
titanic = pd.read_csv("../datasets/titanic/train.csv")
titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
imp_survival = Impact(titanic, 'Survived')"""

    print(timeit('imp_survival.search()', setup1, number=5))
    print(timeit('imp_survival.search()', setup2, number=5))
