"""
Loss functions and models for rule learning.
"""

import collections.abc

from math import inf
from numpy import arange, argsort, array, cumsum, exp, full_like, log2, stack, zeros, zeros_like
from pandas import qcut, Series
from sklearn.base import BaseEstimator, clone

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


class PoissonLoss:
    """
    Poisson Loss function l(y, s) = exp(s) - s * y +log(y) * y - y

    s is the log value of the actual predicted value
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PoissonLoss, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def __call__(y, s):
        return np.array(
            [exp(s[i]) if y[i] == 0 else exp(s[i]) - s[i] * y[i] + log(y[i]) * y[i] - y[i] for i in range(len(y))])

    @staticmethod
    def predictions(s):
        return exp(s)

    @staticmethod
    def g(y, s):
        res = exp(s) - y
        return res

    @staticmethod
    def h(y, s):
        res = exp(s)
        return res

    @staticmethod
    def __repr__():
        return 'poisson_loss'

    @staticmethod
    def __str__():
        return 'poisson'

    @staticmethod
    def pw(y, s, q):
        return q * (exp(s) - exp(y))


logistic_loss = LogisticLoss()
squared_loss = SquaredLoss()
poisson_loss = PoissonLoss()

#: Dictionary of available loss functions with keys corresponding to their string representations.
loss_functions = {
    LogisticLoss.__repr__(): logistic_loss,
    SquaredLoss.__repr__(): squared_loss,
    LogisticLoss.__str__(): logistic_loss,
    SquaredLoss.__str__(): squared_loss,
    PoissonLoss.__repr__(): poisson_loss,
    PoissonLoss.__str__(): poisson_loss
}


def loss_function(loss):
    """Provides loss functions from string representation.

    :param loss: string identifier of loss function loss function
    :return: loss function matching corresponding to input string (or unchanged input if was already loss function)
    """
    if callable(loss):
        return loss
    else:
        return loss_functions[loss]


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

    >>> empty = Rule()
    >>> empty
       +0.0000 if True
    """

    def __init__(self, q=Conjunction([]), y=0.0, z=0.0):
        """
        :param `~realkd.logic.Conjunction` q: rule query (antecedent/condition)
        :param float y: prediction value if query satisfied
        :param float z: prediction value if query not satisfied
        """
        self.q = q
        self.y = y
        self.z = z

    def __call__(self, x):
        """ Predicts score for input data based on loss function.

        For instance for logistic loss will return log odds of the positive class.

        :param ~pandas.DataFrame x: input data
        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)
        """
        sat = self.q(x)
        return sat*self.y + (1-sat)*self.z

    def __repr__(self):
        # TODO: if existing also print else part
        return f'{self.y:+10.4f} if {self.q}'


class AdditiveRuleEnsemble:
    """Rules ensemble that combines scores of its member rules additively to form predictions.

    While order of rules does not influence predictions, it is important for indexing and
    slicing, which provides convenient access to individual ensemble members and modified
    ensembles.

    For example:

    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> r1 = Rule(Conjunction([]), -0.5, 0.0)
    >>> r2 = Rule(female, 1.0, 0.0)
    >>> r3 = Rule(female, 0.3, 0.0)
    >>> r4 = Rule(Conjunction([]), -0.2, 0.0)
    >>> ensemble = AdditiveRuleEnsemble(members=[r1, r2, r3, r4])
    >>> len(ensemble)
    4
    >>> ensemble[2]
       +0.3000 if Sex==female
    >>> ensemble[:2]
       -0.5000 if True
       +1.0000 if Sex==female
    """

    def __init__(self, members=[]):
        """

        :param List[Rule] members: the individual rules that make up the ensemble
        """
        self.members = members[:]

    def __repr__(self):
        return str.join('\n', (str(r) for r in self.members))

    def __len__(self):
        """Length of the ensemble.

        :return: number of contained rules
        """
        return len(self.members)

    def __getitem__(self, item):
        """Index access to the individual members of the ensemble.

        Also supports slicing, resulting in a new ensemble.

        :param int item: index
        :return: rule of index
        """
        if isinstance(item, slice):
            _members = self.members[item]
            return AdditiveRuleEnsemble(_members)
        else:
            return self.members[item]

    def __call__(self, x):  # look into swapping to Series and numpy
        """Computes combined prediction scores using all ensemble members.

        :param ~pandas.DataFrame x: input data
        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)
        """
        res = zeros(len(x))  # TODO: a simple reduce should do if we can rule out empty ensemble
        for r in self.members:
            res += r(x)
        return res

    def append(self, rule):
        """Adds a rule to the ensemble.

        :param Rule rule: the rule to be added
        :return: self
        """
        self.members.append(rule)
        return self

    def size(self):
        """ Computes the total size of the ensemble.

        Currently, this is defined as the number of rules (length of the ensemble)
        plus the the number of elementary conditions in all rule queries.

        In the future this is subject to change to a more general notion of size (taking into account
        the possible greater number of parameters of more complex rules).

        :return: size of ensemble as defined above
        """
        return sum(len(r.q) for r in self.members) + len(self.members)

    def consolidated(self, inplace=False):
        """ Consolidates rules with equivalent queries into one.

        :param bool inplace: whether to update self or to create new ensemble
        :return: reference to consolidated ensemble (self if inplace=True)

        For example:

        >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
        >>> r1 = Rule(Conjunction([]), -0.5, 0.0)
        >>> r2 = Rule(female, 1.0, 0.0)
        >>> r3 = Rule(female, 0.3, 0.0)
        >>> r4 = Rule(Conjunction([]), -0.2, 0.0)
        >>> ensemble = AdditiveRuleEnsemble([r1, r2, r3, r4])
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
            return AdditiveRuleEnsemble(_members)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
