"""
Loss functions and models for rule learning.
"""

import collections.abc

from math import inf
from numpy import arange, argsort, array, cumsum, exp, full_like, log2, stack, zeros, zeros_like
from pandas import qcut, Series

from realkd.search import Conjunction, Context, KeyValueProposition, Constraint


class SquaredLoss:
    """
    Squared loss function l(y, s) = (y-s)^2.

    >>> squared_loss
    squared_loss
    >>> y = array([-2, 0, 3])
    >>> s = array([0, 1, 2])
    >>> squared_loss(y, s)
    array([4, 1, 1])
    >>> squared_loss.g(y, s)
    array([ 4,  2, -2])
    >>> squared_loss.h(y, s)
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
    def predictions(s):
        return s

    @staticmethod
    def g(y, s):
        return -2*(y - s)

    @staticmethod
    def h(y, s):
        return full_like(s, 2)  # Series(full_like(s, 2))

    @staticmethod
    def __repr__():
        return 'squared_loss'

    @staticmethod
    def __str__():
        return 'squared'


class LogisticLoss:
    """
    Logistic loss function l(y, s) = log2(1 + exp(-ys)).

    Function assumes that positive and negative values are encoded as +1 and -1, respectively.

    >>> y = array([1, -1, 1, -1])
    >>> s = array([0, 0, 10, 10])
    >>> logistic_loss(y, s)
    array([1.00000000e+00, 1.00000000e+00, 6.54967668e-05, 1.44270159e+01])
    >>> logistic_loss.g(y, s)
    array([-5.00000000e-01,  5.00000000e-01, -4.53978687e-05,  9.99954602e-01])
    >>> logistic_loss.h(y, s)
    array([2.50000000e-01, 2.50000000e-01, 4.53958077e-05, 4.53958077e-05])
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogisticLoss, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def __call__(y, s):
        return log2(1 + exp(-y*s))

    @staticmethod
    def sigmoid(a):
        return 1 / (1 + exp(-a))

    @staticmethod
    def predictions(s):
        preds = zeros_like(s)
        preds[s >= 0] = 1
        preds[s < 0] = -1
        return preds  # this case now returns np array

    @staticmethod
    def probabilities(s):
        pos = LogisticLoss.sigmoid(s)
        return stack((1-pos, pos), axis=1)

    @staticmethod
    def g(y, s):
        return -y*LogisticLoss.sigmoid(-y*s)

    @staticmethod
    def h(y, s):
        sig = LogisticLoss.sigmoid(-y*s)
        return sig*(1.0-sig)

    @staticmethod
    def __repr__():
        return 'logistic_loss'

    @staticmethod
    def __str__():
        return 'logistic'


logistic_loss = LogisticLoss()
squared_loss = SquaredLoss()


loss_functions = {
    LogisticLoss.__repr__(): logistic_loss,
    SquaredLoss.__repr__(): squared_loss,
    LogisticLoss.__str__(): logistic_loss,
    SquaredLoss.__str__(): squared_loss
}


def loss_function(loss):
    if callable(loss):
        return loss
    else:
        return loss_functions[loss]


class GradientBoostingObjective:
    """
    >>> import pandas as pd
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> survived = titanic['Survived']
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> obj = GradientBoostingObjective(titanic, survived, reg=0.0)
    >>> female = Conjunction([KeyValueProposition('Sex', Constraint.equals('female'))])
    >>> first_class = Conjunction([KeyValueProposition('Pclass', Constraint.less_equals(1))])
    >>> obj(obj.data[female].index)
    0.1940459084832758
    >>> obj(obj.data[first_class].index)
    0.09610508375940474
    >>> obj.bound(obj.data[first_class].index)
    0.1526374859708193

    >>> reg_obj = GradientBoostingObjective(titanic, survived, reg=2)
    >>> reg_obj(reg_obj.data[female].index)
    0.19342988972618602
    >>> reg_obj(reg_obj.data[first_class].index)
    0.09566220318908492

    >>> q = reg_obj.search(verbose=True)
    <BLANKLINE>
    Found optimum after inspecting 103 nodes: [16]
    Greedy simplification: [16]
    >>> q
    Sex==female
    >>> reg_obj.opt_weight(q)
    0.7396825396825397

    >>> obj = GradientBoostingObjective(titanic, survived.replace(0, -1), loss='logistic')
    >>> obj(obj.data[female].index)
    0.04077109318199465
    >>> obj.opt_weight(female)
    0.9559748427672956
    >>> best = obj.search(order='bestvaluefirst', verbose=True)
    <BLANKLINE>
    Found optimum after inspecting 446 nodes: [27, 29]
    Greedy simplification: [27, 29]
    >>> best
    Pclass>=2 & Sex==male
    >>> obj(obj.data[best].index)
    0.13072995752734315
    >>> obj.opt_weight(best)
    -1.4248366013071896
    """

    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0):
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = array(self.loss.h(target, predictions))
        r = g / h
        order = argsort(r)[::-1]
        self.g = g[order]
        self.h = h[order]
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        self.n = len(target)

    def __call__(self, ext):
        if len(ext) == 0:
            return -inf
        g_q = self.g[ext]
        h_q = self.h[ext]
        return g_q.sum() ** 2 / (2 * self.n * (self.reg + h_q.sum()))

    def bound(self, ext):
        m = len(ext)
        if m == 0:
            return -inf

        g_q = self.g[ext]
        h_q = self.h[ext]

        num_pre = cumsum(g_q)**2
        num_suf = cumsum(g_q[::-1])**2
        den_pre = cumsum(h_q) + self.reg
        den_suf = cumsum(h_q[::-1]) + self.reg
        neg_bound = (num_suf / den_suf).max() / (2 * self.n)
        pos_bound = (num_pre / den_pre).max() / (2 * self.n)
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        # ext = self.ext(q)
        ext = self.data.loc[q].index
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg + h_q.sum())

    def search(self, order='bestboundfirst', max_col_attr=10, discretization=qcut, apx=1.0, max_depth=None, verbose=False):
        ctx = Context.from_df(self.data, max_col_attr=max_col_attr, discretization=discretization)
        if verbose >= 2:
            print(f'Created search context with {len(ctx.attributes)} attributes')  #:\n {ctx.attributes}')
        if order == 'greedy':
            return ctx.greedy_search(self, verbose=verbose)
        else:
            return ctx.search(self, self.bound, order=order, apx=apx, max_depth=max_depth, verbose=verbose)


class Rule:
    """
    Represents a rule of the form "r(x) = y if q(x) else z"
    for some binary query function q.

    >>> import pandas as pd
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> titanic[['Name', 'Sex', 'Survived']].iloc[0]
    Name        Braund, Mr. Owen Harris
    Sex                            male
    Survived                          0
    Name: 0, dtype: object
    >>> titanic[['Name', 'Sex', 'Survived']].iloc[1]
    Name        Cumings, Mrs. John Bradley (Florence Briggs Th...
    Sex                                                    female
    Survived                                                    1
    Name: 1, dtype: object

    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> r = Rule(female, 1.0, 0.0)
    >>> r(titanic.iloc[0]), r(titanic.iloc[1])
    (0.0, 1.0)
    >>> target = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> opt = Rule(reg=0.0)
    >>> opt.fit(titanic, target)
       +0.7420 if Sex==female

    >>> best_logistic = Rule(loss='logistic')
    >>> best_logistic.fit(titanic, target.replace(0, -1))
       -1.4248 if Pclass>=2 & Sex==male

    >>> best_logistic.predict(titanic) # doctest: +ELLIPSIS
    array([-1.,  1.,  1.,  1., ...,  1.,  1., -1.])

    >>> greedy = Rule(loss='logistic', reg=1.0, method='greedy')
    >>> greedy.fit(titanic, target.replace(0, -1))
       -1.4248 if Pclass>=2 & Sex==male

    >>> empty = Rule()
    >>> empty
       +0.0000 if True
    """

    # max_col attribute to change number of propositions
    def __init__(self, q=Conjunction([]), y=0.0, z=0.0, loss=SquaredLoss, reg=1.0, max_col_attr=10,
                 discretization=qcut, method='bestboundfirst', apx=1.0, max_depth=None):
        """

        :param q:
        :param y:
        :param z:
        :param loss:
        :param reg:
        :param max_col_attr:
        :param discretization:
        :param method:
        :param apx: approximation ratio (ignored when method 'greedy')
        """
        self.q = q
        self.y = y
        self.z = z
        self.reg = reg
        self.max_col_attr = max_col_attr
        self.discretization = discretization
        self.loss = loss
        self.method = method
        self.apx = apx
        self.max_depth = max_depth

    def __call__(self, x):
        """ Predicts score for input data based on loss function.

        For instance for logistic loss will return log odds of the positive class.

        :param x:
        :return:
        """
        sat = self.q(x)
        return sat*self.y + (1-sat)*self.z

    def __repr__(self):
        # TODO: if existing also print else part
        return f'{self.y:+10.4f} if {self.q}'

    def fit(self, data, target, scores=None, verbose=False):
        """
        Fits rule to provide best loss reduction on given data
        (where the baseline prediction scores are either given
        explicitly through the scores parameter or are assumed
        to be 0.

        :param data: pandas DataFrame containing only the feature columns
        :param target: pandas Series containing the target values
        :param scores: prior prediction scores according to which the reduction in prediction loss is optimised
        :param verbose: whether to print status update and summary of query search
        :return: self

        """
        obj = GradientBoostingObjective(data, target, predictions=scores, loss=self.loss, reg=self.reg)
        self.q = obj.search(order=self.method, max_col_attr=self.max_col_attr, discretization=self.discretization,
                            apx=self.apx, max_depth=self.max_depth, verbose=verbose)
        self.y = obj.opt_weight(self.q)
        return self

    def predict(self, data):
        loss = loss_function(self.loss)
        return loss.predictions(self(data))

    def predict_proba(self, data):
        """Generates probability predictions for

        :param data: pandas dataframe with data to predict probabilities for
        :return: two-dimensional array of probabilities
        """
        loss = loss_function(self.loss)
        return loss.probabilities(self(data))


class GradientBoostingRuleEnsemble:
    """Additive rule ensemble fitted by gradient boosting.

    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> r1 = Rule(Conjunction([]), -0.5, 0.0)
    >>> r2 = Rule(female, 1.0, 0.0)
    >>> r3 = Rule(female, 0.3, 0.0)
    >>> r4 = Rule(Conjunction([]), -0.2, 0.0)
    >>> ensemble = GradientBoostingRuleEnsemble(members=[r1, r2, r3, r4])
    >>> len(ensemble)
    4
    >>> ensemble[2]
       +0.3000 if Sex==female

    >>> ensemble[:2]
       -0.5000 if True
       +1.0000 if Sex==female

    >>> import pandas as pd
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> survived = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> re = GradientBoostingRuleEnsemble(loss=logistic_loss)
    >>> re.fit(titanic, survived.replace(0, -1), verbose=0) # doctest: +SKIP
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       +2.5598 if Age<=19.0 & Fare>=7.8542 & Parch>=1.0 & Sex==male & SibSp<=1.0

    # performance with bestboundfirst:
    # <BLANKLINE>
    # Found optimum after inspecting 443 nodes
    #    -1.4248 if Pclass>=2 & Sex==male
    # <BLANKLINE>
    # Found optimum after inspecting 786 nodes
    #    +1.7471 if Pclass<=2 & Sex==female
    # ******
    # Found optimum after inspecting 6564 nodes
    #

    >>> re_with_offset = GradientBoostingRuleEnsemble(max_rules=2, loss='logistic', offset_rule=True)
    >>> re_with_offset.fit(titanic, survived.replace(0, -1))
       -0.4626 if True
       +2.3076 if Pclass<=2 & Sex==female

    >>> greedy = GradientBoostingRuleEnsemble(max_rules=3, loss='logistic', method='greedy')
    >>> greedy.fit(titanic, survived.replace(0, -1)) # doctest: +SKIP
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       -0.4225 if Parch<=1.0 & Sex==male
    """

    def __init__(self, max_rules=3, loss=SquaredLoss, members=[], reg=1.0, max_col_attr=10, discretization=qcut,
                 offset_rule=False, method='bestboundfirst', apx=1.0, max_depth=None):
        self.reg = reg
        self.members = members[:]
        self.max_col_attr = max_col_attr
        self.max_rules = max_rules
        self.discretization = discretization
        self.loss = loss
        self.offset_rule = offset_rule
        self.method = method
        if callable(apx):
            self.apx = apx
        elif isinstance(apx, collections.abc.Sequence):
            self.apx = lambda i: apx[min(i, len(apx)-1)]
        else:
            self.apx = lambda _: apx
        self.max_depth = max_depth

    def __call__(self, x):  # look into swapping to Series and numpy
        """Computes combined prediction scores using all ensemble members.

        :param x: dataframe to make predictions for

        :return: vector of prediction scores for all rows in x
        """
        res = zeros(len(x))  # TODO: a simple reduce should do if we can rule out empty ensemble
        for r in self.members:
            res += r(x)
        return res
        #return sum(r(x) for r in self.members)

    def __repr__(self):
        return str.join('\n', (str(r) for r in self.members))

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        """Index access to the individual members of the ensemble.

        Also supports slicing, resulting in a new ensemble.

        :param item: index
        :return: rule of index
        """
        if isinstance(item, slice):
            _members = self.members[item]
            return GradientBoostingRuleEnsemble(self.max_rules, self.loss, _members, self.reg, self.max_col_attr,
                                                self.discretization, self.offset_rule, self.method, self.apx,
                                                self.max_depth)
        else:
            return self.members[item]

    def fit(self, data, target, verbose=False):
        if len(self.members) < self.max_rules and self.offset_rule:
            obj = GradientBoostingObjective(data, target, loss=self.loss, reg=self.reg)
            q = Conjunction([])
            y = obj.opt_weight(q)
            r = Rule(q=q, y=y, loss=self.loss, reg=self.reg, method=self.method)
            if verbose:
                print(r)
            self.members.append(r)

        while len(self.members) < self.max_rules:
            scores = self(data)
            apx = self.apx(len(self.members))
            r = Rule(loss=self.loss, reg=self.reg, max_col_attr=self.max_col_attr, discretization=self.discretization,
                     method=self.method, apx=apx, max_depth=self.max_depth)
            r.fit(data, target, scores, verbose)
            if verbose:
                print(r)
            self.members.append(r)
        return self

    def predict(self, data):
        loss = loss_function(self.loss)
        return loss.predictions(self(data))

    def predict_proba(self, data):
        loss = loss_function(self.loss)
        return loss.probabilities(self(data))

    def size(self):
        return sum(len(r.q) for r in self.members) + len(self.members)

    def consolidated(self, inplace=False):
        """ Consolidates rules with equivalent queries into one.

        :param inplace: whether to update self or to create new ensemble
        :return: reference to consolidated ensemble (self if inplace=True)

        For example:

        >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
        >>> r1 = Rule(Conjunction([]), -0.5, 0.0)
        >>> r2 = Rule(female, 1.0, 0.0)
        >>> r3 = Rule(female, 0.3, 0.0)
        >>> r4 = Rule(Conjunction([]), -0.2, 0.0)
        >>> ensemble = GradientBoostingRuleEnsemble(members=[r1, r2, r3, r4])
        >>> ensemble.consolidated(inplace=True) # doctest: +NORMALIZE_WHITESPACE
        -0.7000 if True
        +1.3000 if Sex==female
        """

        _members = self.members[:]
        for i, r1 in enumerate(_members):
            q = r1.q
            y = r1.y
            z = r1.z
            for j in range(len(_members)-1, i, -1):
                r2 = _members[j]
                if q == r2.q:
                    y += r2.y
                    z += r2.z
                    _members.pop(j)
            _members[i] = Rule(q, y, z)

        if inplace:
            self.members = _members
            return self
        else:
            return GradientBoostingRuleEnsemble(self.max_rules, self.loss, _members, self.reg, self.max_col_attr,
                                                self.discretization, self.offset_rule, self.method, self.apx,
                                                self.max_depth)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
