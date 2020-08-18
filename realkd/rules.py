from math import inf
from numpy import arange, cumsum
from realkd.search import Conjunction, Context, KeyValueProposition, Constraint


class SquaredLossObjective:
    """
    Rule boosting objective function for squared loss.

    >>> import pandas as pd
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> obj = SquaredLossObjective(titanic.drop(columns=['Survived']), titanic['Survived'])
    >>> female = Conjunction([KeyValueProposition('Sex', Constraint.equals('female'))])
    >>> first_class = Conjunction([KeyValueProposition('Pclass', Constraint.less_equals(1))])
    >>> obj(female)
    0.1940459084832758
    >>> obj(first_class)
    0.09610508375940474
    >>> obj.bound(first_class)
    0.1526374859708193

    >>> reg_obj = SquaredLossObjective(titanic.drop(columns=['Survived']), titanic['Survived'], reg=2)
    >>> reg_obj(female)
    0.19342988972618602
    >>> reg_obj(first_class)
    0.09566220318908492

    # >>> reg_obj._mean(female)
    # 0.7420382165605095
    # >>> reg_obj._mean(first_class)
    # 0.6296296296296297
    >>> q = reg_obj.search(verbose=True)
    >>> q
    Sex==female
    >>> reg_obj.opt_value(q)
    0.7396825396825397
    """

    def __init__(self, data, target, reg=0):
        """
        :param data:
        :param target: _series_ of target values of matching dimension
        :param reg:
        """
        self.m = len(data)
        self.data = data #.copy()
        self.target = target #target.sort_values()
        #self.data.index = self.target.index
        #self.target.reset_index(drop=True, inplace=True)
        #self.data.reset_index(drop=True, inplace=True)
        #self.data = data.sort_values(key=lambda i: target[i])
        self.reg = reg

    def __call__(self, q):
        ext = self.data.loc[q] # check if already index
        y_q = self.target[ext.index]
        return y_q.sum()**2 / (self.m * (self.reg / 2 + len(ext)))

    def bound(self, q):
        ext = self.data.loc[q]
        n = len(ext)
        if n == 0:
            return -inf
        y_q = self.target[ext.index].sort_values()
        ss_neg = cumsum(y_q)**2
        ss_pos = cumsum(y_q[::-1])**2
        i = self.reg / 2 + arange(1, n + 1) #bug?
        neg_bound = (ss_neg / i).max() / self.m
        pos_bound = (ss_pos / i).max() / self.m
        return max(neg_bound, pos_bound)

    def search(self, order='bestboundfirst', verbose=False):
        ctx = Context.from_df(self.data, max_col_attr=10)
        return ctx.search(self, self.bound, order=order, verbose=verbose)

    def opt_value(self, q):
        ext = self.data.loc[q]
        res_q = self.target[ext.index]
        return res_q.sum() / (self.reg/2 + len(ext))
        # s, c = 0.0, 0
        # for i in rows:
        #     s += self.target[i]
        #     c += 1
        #
        # return s / (self.reg/2 + c) if (c > 0 or self.reg > 0) else 0.0


if __name__=='__main__':

    from timeit import timeit

    setup1 = \
"""import pandas as pd
from realkd.rules import SquaredLossObjective
titanic = pd.read_csv("../datasets/titanic/train.csv")
sql_survival = SquaredLossObjective(titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']), titanic['Survived'], reg=2)"""

    setup2 = \
"""import pandas as pd
from realkd.search import SquaredLossObjective
titanic = pd.read_csv("../datasets/titanic/train.csv")
sql_survival = SquaredLossObjective(titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']), titanic['Survived'], reg=2)"""

    print(timeit('sql_survival.search()', setup1, number=5))
    print(timeit('sql_survival.search()', setup2, number=5))
