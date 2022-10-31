"""
Loss functions and models for rule learning.
"""

from math import inf, sqrt
import numpy as np
import scipy
from numpy import arange, argsort, array, cumsum, exp, full_like, log2, stack, zeros, zeros_like, log
from pandas import qcut, Series
from sklearn.base import BaseEstimator, clone

from realkd.search import Conjunction, Context, KeyValueProposition, Constraint

### WEIGHT CORRECTION METHODS

def norm(xs):
    return sqrt(sum([x * x for x in xs]))

def golden_ratio_search(func, left, right, dir, origin):
    """
    Use golden ratio search to search for an optimal distance along a direction
    to make the function minimized
    :param func: function to be minimized
    :param left: left bound of the search interval
    :param right: right bound of the search interval
    :param direction: search direction
    :param origin: origin point
    :param epsilon: the precision of the search
    """
    ratio = (sqrt(5) - 1) / 2
    while right - left > 1e-3:
        lam = left + (1 - ratio) * (right - left)
        mu = left + ratio * (right - left)
        r_lam = func(origin + lam * dir)
        r_mu = func(origin + mu * dir)
        if r_lam <= r_mu:
            right = mu
        else:
            left = lam
    return (left + right) / 2

def gradient_descent(weights_to_calc, gradient, sum_loss, hessian):
    old_w = zeros_like(weights_to_calc) * 1.0
    w = weights_to_calc
    i = 0
    while norm(old_w - w) > 1e-3 and i < 20:
        old_w = np.array(w)
        if norm(gradient(w)) == 0:
            break
        p = -gradient(w) / norm(gradient(w))
        left = 0
        right = norm(w)
        w += golden_ratio_search(sum_loss, left, right, p, old_w) * p
    
    return w

def line_descent(weights_to_calc, gradient, sum_loss, hessian):
    w = weights_to_calc

    if norm(gradient(w)) != 0:
        p = -gradient(w) / norm(gradient(w))
        left = 0
        right = norm(w)
        distance = golden_ratio_search(sum_loss, left, right, p, w)
        w += distance * p

    return w

CUSTOM_CORRECTION_METHODS = {
    'GD': gradient_descent,
    'line': line_descent
}

def get_correction_method(correction_method='Newton-CG'):
    """Provides correction methods from string representation.

    :param correction_method: string identifier of correction method
    :return: correction method matching corresponding to input string (or unchanged input if was already correction method)
    """
    if callable(correction_method):
        return correction_method
    elif correction_method in CUSTOM_CORRECTION_METHODS:
        return CUSTOM_CORRECTION_METHODS[correction_method]
    else:
        def scipy_correction(weights_to_calc, gradient, sum_loss, hessian):
            w = weights_to_calc
            w = scipy.optimize.minimize(sum_loss, w, method=correction_method, jac=gradient, hess=hessian,
                                        options={'disp': False}).x
            return w
        return scipy_correction

### WEIGHT UPDATE METHODS

def get_gradient(g, y, q_mat, weights, reg):
    def gradient(weight):
        all_weights = np.append(weights, weight)
        grad_vec = g(y, q_mat.dot(all_weights))
        return np.array([(q_mat.T.dot(grad_vec) + reg * all_weights)[-1]])

    return gradient

def get_risk(loss, y, q_mat, weights, reg):
    def sum_loss(weight):
        all_weights = np.append(weights, weight)
        return sum(loss(y, q_mat.dot(all_weights))) + reg * sum(all_weights * all_weights) / 2

    return sum_loss

def get_hessian(h, y, q_mat, reg):
    def hessian(weights):
        h_vec = h(y, q_mat.dot(weights))
        return q_mat.T.dot(np.diag(h_vec)).dot(q_mat) + np.diag([reg] * len(weights))

    return hessian

def fully_corrective(rules, loss, weight_update_method_params, data, target, reg):
    '''
        FullyCorrective updates all weights
    '''
    if weight_update_method_params is None:
        weight_update_method_params = {'correction_method': 'Newton-CG'}

    w = np.array([3.5 if r.y > 100 and loss == 'poisson' else r.y for r in rules])

    q_mat = np.column_stack([rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])

    gradient = get_gradient(loss.g, target, q_mat, np.array([]), reg)
    sum_loss = get_risk(loss, target, q_mat, np.array([]), reg)
    hessian = get_hessian(loss.h, target, q_mat, reg)

    w = get_correction_method(weight_update_method_params['correction_method'])(w, gradient, sum_loss, hessian)

    return w

def line_search(rules, loss, weight_update_method_params, data, target, reg):
    '''
        Line search only updates the most recent weight
    '''
    if weight_update_method_params is None:
        weight_update_method_params = {'correction_method': 'Newton-CG'}

    if weight_update_method_params['correction_method'] != 'GD':
        # TODO: Not sure if this is correct
        raise NotImplementedError()
    
    all_weights = np.array([rule.y for rule in rules][:-1])
    w = np.array([3.5 if rules[-1].y > 100 and loss == 'poisson' else rules[-1].y])

    q_mat = np.column_stack([rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])

    gradient = get_gradient(loss.g, target, q_mat, all_weights, reg)
    sum_loss = get_risk(loss, target, q_mat, all_weights, reg)
    hessian = get_hessian(loss.h, target, q_mat, reg)

    w = get_correction_method(weight_update_method_params['correction_method'])(w, gradient, sum_loss, hessian)
    
    all_weights = np.append(all_weights, w)
    return all_weights

def no_update(rules, loss, weight_update_method_params, data, target, reg):
    '''
        Return existing weights
    '''
    return np.array([rule.y for rule in rules])

WEIGHT_UPDATE_METHODS = {
    'line': line_search,
    'fully_corrective': fully_corrective,
    'no_update': no_update
}

def get_weight_update_method(weight_update_method='fully_corrective'):
    """Provides weight update methods from string representation.

    :param weight_update_method: string identifier of weight update method
    :return: weight update method matching corresponding to input string (or unchanged input if was already weight update method)
    """
    if callable(weight_update_method):
        return weight_update_method
    else:
        return WEIGHT_UPDATE_METHODS[weight_update_method]



### LOSS FUNCTIONS
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

    >>> q = reg_obj.search(method='exhaustive', verbose=True)
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
    >>> best = obj.search(method='exhaustive', order='bestvaluefirst', verbose=True)
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
        # TODO: this should probably just be defined for ext (saving the q evaluation)
        # ext = self.ext(q)
        ext = self.data.loc[q].index
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg + h_q.sum())

    def search(self, method='greedy', verbose=False, **search_params):
        from realkd.search import search_methods
        ctx = Context.from_df(self.data, **search_params)
        if verbose >= 2:
            print(f'Created search context with {len(ctx.attributes)} attributes')
        # return getattr(ctx, method)(self, self.bound, verbose=verbose, **search_params)
        return search_methods[method](ctx, self, self.bound, verbose=verbose, **search_params).run()

    #def search(self, order='bestboundfirst', max_col_attr=10, discretization=qcut, apx=1.0, max_depth=None, verbose=False):
        # ctx = Context.from_df(self.data, max_col_attr=max_col_attr, discretization=discretization)
        # if verbose >= 2:
        #     print(f'Created search context with {len(ctx.attributes)} attributes')
        # if order == 'greedy':
        #     return ctx.greedy_search(self, verbose=verbose)
        # else:
        #     return ctx.search(self, self.bound, order=order, apx=apx, max_depth=max_depth, verbose=verbose)


class XGBRuleEstimator(BaseEstimator):
    r"""
    Fits a rule based on first and second loss derivatives of some prior prediction values.

    In more detail, given some prior prediction values :math:`f(x)` and a twice differentiable loss function
    :math:`l(y,f(x))`, a rule :math:`r(x)=wq(x)` is fitted by finding a binary query :math:`q` via maximizing the objective function

    .. math::

        \mathrm{obj}(q) = \frac{\left( \sum_{i \in I(q)} g_i \right )^2}{2n \left(\lambda + \sum_{i \in I(q)} h_i \right)}


    and finding the optimal weight as

    .. math::

        w = -\frac{\sum_{i \in I(q)} g_i}{\lambda + \sum_{i \in I(q)} h_i} \enspace .

    Here, :math:`I(q)` denotes the indices of training examples selected by :math:`q` and

    .. math::

        g_i=\frac{\mathrm{d} l(y_i, y)}{\mathrm{d}y}\Bigr|_{\substack{y=f(x_i)}} \enspace ,
        \quad
        h_i=\frac{\mathrm{d}^2 l(y_i, y)}{\mathrm{d}y^2}\Bigr|_{\substack{y=f(x_i)}}

    refer to the first and second order gradient statistics of the prior prediction values.


    >>> import pandas as pd
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> target = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> opt = XGBRuleEstimator(reg=0.0)
    >>> opt.fit(titanic, target).rule_
       +0.7420 if Sex==female

    >>> best_logistic = XGBRuleEstimator(loss='logistic')
    >>> best_logistic.fit(titanic, target.replace(0, -1)).rule_
       -1.4248 if Pclass>=2 & Sex==male

    >>> best_logistic.predict(titanic) # doctest: +ELLIPSIS
    array([-1.,  1.,  1.,  1., ...,  1.,  1., -1.])

    >>> greedy = XGBRuleEstimator(loss='logistic', reg=1.0, search='greedy')
    >>> greedy.fit(titanic, target.replace(0, -1)).rule_
       -1.4248 if Pclass>=2 & Sex==male
    """

    # max_col attribute to change number of propositions
    def __init__(self, loss='squared', reg=1.0, search='exhaustive',
                 search_params={'order': 'bestboundfirst', 'apx': 1.0, 'max_depth': None, 'discretization': qcut, 'max_col_attr': 10},
                 query=None):
        """
        :param str|callable loss: loss function either specified via string identifier (e.g., ``'squared'`` for regression or ``'logistic'`` for classification) or directly has callable loss function with defined first and second derivative (see :data:`~realkd.rules.loss_functions`)
        :param float reg: the regularization parameter :math:`\\lambda`
        :param str|type search: search method either specified via string identifier (e.g., ``'greedy'`` or ``'exhaustive'``) or directly as search type (see :func:`realkd.search.search_methods`)
        :param dict search_params: parameters to apply to discretization (when creating binary search context from
                              dataframe via :func:`~realkd.search.Context.from_df`) as well as to actual search method
                              (specified by ``method``). See :mod:`~realkd.search`.
        """
        self.reg = reg
        self.loss = loss
        self.search = search
        self.search_params = search_params
        self.query = query
        self.rule_ = None

    def decision_function(self, x):
        """ Predicts score for input data based on loss function.

        For instance for logistic loss will return log odds of the positive class.

        :param ~pandas.DataFrame x: input data
        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)

        """
        return self.rule_(x)

    def __repr__(self):
        return f'{type(self).__name__}(reg={self.reg}, loss={self.loss})'

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
        q = obj.search(method=self.search, verbose=verbose, **self.search_params) if self.query is None else self.query
        y = obj.opt_weight(q)
        self.rule_ = Rule(q, y)
        return self

    def predict(self, data):
        """Generates predictions for input data.

        :param data: pandas dataframe with co-variates for which to make predictions
        :return: array of predictions
        """
        loss = loss_function(self.loss)
        return loss.predictions(self.rule_(data))

    def predict_proba(self, data):
        """Generates probability predictions for input data.

        This method is only supported for suitable loss functions.

        :param data: pandas dataframe with data to predict probabilities for
        :return: array of probabilities (shape according to number of classes)
        """
        loss = loss_function(self.loss)
        return loss.probabilities(self.rule_(data))

SINGLE_RULE_ESTIMATORS = {
    'XGBRuleEstimator': XGBRuleEstimator
}

class RuleBoostingEstimator(BaseEstimator):
    """Additive rule ensemble fitted by boosting.

    That is, rules are fitted iteratively by one or more base learners until a desired number of rules has been
    learned. In each iteration, the base learner fits the training data taking into account the prediction scores
    of the already fixed part of the ensemble.

    Therefore, base learners need to provide a fit method that can take into account prior predictions
    (see :func:`XGBRuleEstimator.fit`).

    >>> import pandas as pd
    >>> from sklearn.metrics import roc_auc_score
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> survived = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss))
    >>> re.fit(titanic, survived.replace(0, -1), verbose=0) # doctest: +SKIP
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       +2.5598 if Age<=19.0 & Fare>=7.8542 & Parch>=1.0 & Sex==male & SibSp<=1.0

    Multiple base learners can be specified and are used sequentially. The last based learner is used as many times
    as necessary to learn the desired number of rules. This mechanism can, e.g., be used to fit an "offset rule":

    >>> re_with_offset = RuleBoostingEstimator(num_rules=2, base_learner=[XGBRuleEstimator(loss='logistic', query = Conjunction([])), XGBRuleEstimator(loss='logistic')])
    >>> re_with_offset.fit(titanic, survived.replace(0, -1)).rules_
       -0.4626 if True
       +2.3076 if Pclass<=2 & Sex==female

    >>> greedy = RuleBoostingEstimator(num_rules=3, base_learner=XGBRuleEstimator(loss='logistic', search='greedy'))
    >>> greedy.fit(titanic, survived.replace(0, -1)).rules_ # doctest: -SKIP
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       -0.4225 if Parch<=1.0 & Sex==male
    >>> roc_auc_score(survived, greedy.rules_(titanic))
    0.8321136782454011
    >>> opt = RuleBoostingEstimator(num_rules=3, base_learner=XGBRuleEstimator(loss='logistic', search='exhaustive'))
    >>> opt.fit(titanic, survived.replace(0, -1)).rules_ # doctest: -SKIP
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       +2.5598 if Age<=19.0 & Fare>=7.8542 & Parch>=1.0 & Sex==male & SibSp<=1.0
    >>> roc_auc_score(survived, opt.rules_(titanic)) # doctest: -SKIP
    0.8490530363553084
    """

    def __init__(self, num_rules=3, base_learner='XGBRuleEstimator', base_learner_params=None,
                 verbose=False, weight_update_method='no_update', weight_update_method_params=None):
        """

        :param int num_rules: the desired number of ensemble members
        :param Estimator|Sequence[Estimator] base_learner: the base learner(s) to be used in each iteration (last base
                                    learner is used as many times as necessary to fit desired number of rules)
        :param bool|int verbose: Level of verbosity, theoretically "number of levels deep of printing"
        :weight_update_method: the method to do the fully-correction
        :correction_obj_fn: the method to do the fully-correction
        """
        if base_learner_params == None:
            base_learner_params = {'loss':'squared', 'reg':1.0, 'search':'greedy'}

        self.num_rules = num_rules
        self.base_learner = SINGLE_RULE_ESTIMATORS[base_learner](**base_learner_params)
        self.rules_ = AdditiveRuleEnsemble([])
        self.weight_update_method = weight_update_method
        self.weight_update_method_params = weight_update_method_params
        self.verbose = verbose

    def decision_function(self, x):
        """Computes combined prediction scores using all ensemble members.

        :param ~pandas.DataFrame x: input data

        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)
        """
        return self.rules_(x)

    def __repr__(self):
        return f'{type(self).__name__}(max_rules={self.num_rules}, base_learner={self.base_learner}, weight_update_method={self.weight_update_method})'

    def fit(self, data, target):
        self.history = []
        while len(self.rules_) < self.num_rules:
            scores = self.rules_(data)
            # Estimate
            estimator = self.base_learner
            estimator.fit(data, target, scores, max(self.verbose - 1, 0))
            if self.verbose:
                print(estimator.rule_)
            self.rules_.append(estimator.rule_)

            # Correct weights
            loss = loss_function(self.base_learner.loss)
            reg = self.base_learner.reg
            update_method = get_weight_update_method(self.weight_update_method)
            new_weights = update_method(self.rules_, loss, self.weight_update_method_params, data=data, target=target, reg=reg)
            self.rules_ = AdditiveRuleEnsemble([Rule(q=rule.q, y=new_weights[i]) for i, rule in enumerate(self.rules_.members)])
            self.history.append(self.rules_)
        return self

    def predict(self, data):
        loss = loss_function(self.base_learner.loss)
        return loss.predictions(self.rules_(data))

    def predict_proba(self, data):
        loss = loss_function(self.base_learner.loss)
        return loss.probabilities(self.rules_(data))


def get_titanic():
    import pandas as pd
    titanic = pd.read_csv('./datasets/titanic/train.csv')
    survived = titanic.Survived
    titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    return survived.replace(0, -1), titanic

if __name__ == '__main__':
    survived, titanic = get_titanic()
    re1 = RuleBoostingEstimator(weight_update_method='fully_corrective', weight_update_method_params={'correction_method': 'GD'})
    re1.fit(titanic, survived)
    re2 = RuleBoostingEstimator()
    re2.fit(titanic, survived)
    re3 = RuleBoostingEstimator(weight_update_method='line', weight_update_method_params={'correction_method': 'GD'})
    re3.fit(titanic, survived)
    re4 = RuleBoostingEstimator(weight_update_method='fully_corrective', weight_update_method_params={'correction_method': 'line'})
    re4.fit(titanic, survived)
    # re2 = RuleBoostingEstimator()
    # re2.fit(titanic, survived)

    # No correction:
    #    -0.7179 if Pclass>=2 & Sex==male
    #    +0.8915 if Pclass<=2 & Sex==female
    #    -0.2864 if Age>=41.0 & Fare>=10.5 & SibSp<=1.0
    # Fully corrective, correction method=GD:
    #    -0.7183 if Pclass>=2 & Sex==male
    #    +0.8906 if Pclass<=2 & Sex==female
    #    -0.2867 if Age>=41.0 & Fare>=10.5 & SibSp<=1.0
    # Line search, correction method=GD:
    #    -0.7175 if Pclass>=2 & Sex==male
    #    +0.8912 if Pclass<=2 & Sex==female
    #    -0.2869 if Age>=41.0 & Fare>=10.5 & SibSp<=1.0
    # Fully corrective, correction method=line:
    #    -0.7183 if Pclass>=2 & Sex==male
    #    +0.8906 if Pclass<=2 & Sex==female
    #    -0.2867 if Age>=41.0 & Fare>=10.5 & SibSp<=1.0

    print('No correction:')
    print(re2.rules_)
    print('Fully corrective, correction method=GD:')
    print(re1.rules_)
    print('Line search, correction method=GD:')
    print(re3.rules_)
    print('Fully corrective, correction method=line:')
    print(re4.rules_)