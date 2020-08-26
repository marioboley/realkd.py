from math import inf
from numpy import arange, array, cumsum, full_like, zeros_like
from pandas import Series
from realkd.search import Conjunction, Context, KeyValueProposition, Constraint


class SquaredLoss:
    """
    >>> squared_loss
    squared_loss
    >>> y = array([-2, 0, 3])
    >>> s = array([0, 1, 2])
    >>> squared_loss(s, y)
    array([4, 1, 1])
    >>> squared_loss.g(s, y)
    array([-4, -2,  2])
    >>> squared_loss.h(s, y)
    array([2, 2, 2])
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SquaredLoss, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def __call__(y, s):
        return (y - s)**2

    @staticmethod
    def g(y, s):
        return -2*(y - s)

    @staticmethod
    def h(y, s):
        return Series(full_like(s, 2))

    @staticmethod
    def __repr__():
        return 'squared_loss'

    @staticmethod
    def __str__():
        return 'squared'


squared_loss = SquaredLoss()

loss_functions = {
    SquaredLoss.__repr__(): squared_loss,
    SquaredLoss.__str__(): squared_loss
}


def loss_function(identifier):
    return loss_function[identifier]


class GradientBoostingObjective:
    """
    >>> import pandas as pd
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> obj = GradientBoostingObjective(titanic.drop(columns=['Survived']), titanic['Survived'], reg=0.0)
    >>> female = Conjunction([KeyValueProposition('Sex', Constraint.equals('female'))])
    >>> first_class = Conjunction([KeyValueProposition('Pclass', Constraint.less_equals(1))])
    >>> obj(female)
    0.1940459084832758
    >>> obj(first_class)
    0.09610508375940474
    >>> obj.bound(first_class)
    0.1526374859708193

    >>> reg_obj = GradientBoostingObjective(titanic.drop(columns=['Survived']), titanic['Survived'], reg=2)
    >>> reg_obj(female)
    0.19342988972618602
    >>> reg_obj(first_class)
    0.09566220318908492

    >>> q = reg_obj.search(verbose=True)
    <BLANKLINE>
    Found optimum after inspecting 100 nodes
    >>> q
    Sex==female
    >>> reg_obj.opt_weight(q)
    0.7396825396825397
    """

    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0):
        self.data = data
        self.target = target
        self.predictions = predictions or Series(zeros_like(target))
        self.loss = loss
        self.reg = reg
        self.g = self.loss.g(self.target, self.predictions)
        self.h = self.loss.h(self.target, self.predictions)
        self.n = len(target)

    def ext(self, q):
        return self.data.loc[q]  # check if already index

    def __call__(self, q):
        ext = self.ext(q)
        g_q = self.g[ext.index]
        h_q = self.h[ext.index]
        return g_q.sum() ** 2 / (2 * self.n * (self.reg + h_q.sum()))

    def bound(self, q):
        ext = self.ext(q)
        m = len(ext)
        if m == 0:
            return -inf

        g_q = self.g[ext.index]
        h_q = self.h[ext.index]
        r_q = (g_q / h_q).sort_values(ascending=False)
        g_q = g_q[r_q.index]
        h_q = h_q[r_q.index]

        num_pre = cumsum(g_q)**2
        num_suf = cumsum(g_q[::-1])**2
        den_pre = cumsum(h_q) + self.reg
        den_suf = cumsum(h_q[::-1]) + self.reg
        neg_bound = (num_suf / den_suf).max() / (2 * self.n)
        pos_bound = (num_pre / den_pre).max() / (2 * self.n)
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        ext = self.ext(q)
        g_q = self.g[ext.index]
        h_q = self.h[ext.index]
        return -g_q.sum() / (self.reg + h_q.sum())

    def search(self, order='bestboundfirst', verbose=False):
        ctx = Context.from_df(self.data, max_col_attr=10)
        return ctx.search(self, self.bound, order=order, verbose=verbose)


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


if __name__=='__main__':

    from timeit import timeit

    setup1 = \
"""import pandas as pd
from realkd.rules import GradientBoostingObjective
titanic = pd.read_csv("../datasets/titanic/train.csv")
sql_survival = GradientBoostingObjective(titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']), titanic['Survived'], reg=2)"""

    setup2 = \
"""import pandas as pd
from realkd.search import SquaredLossObjective
titanic = pd.read_csv("../datasets/titanic/train.csv")
sql_survival = SquaredLossObjective(titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']), titanic['Survived'], reg=2)"""

    print(timeit('sql_survival.search()', setup1, number=5))
    print(timeit('sql_survival.search()', setup2, number=5))
