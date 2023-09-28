import collections.abc

from math import inf
import numpy as np
from pandas import qcut
from numbers import Integral
from sklearn.base import BaseEstimator, clone, _fit_context
from sklearn.utils._param_validation import HasMethods

# Imported for doctests. #TODO: Fix
from realkd.logic import Conjunction, IndexValueProposition  # noqa: F401
from realkd.datasets import titanic_data  # noqa: F401

from realkd.loss import SquaredLoss, loss_function
from realkd.rules import Rule, AdditiveRuleEnsemble
from realkd.search import SearchContext, search_methods


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
        predictions = np.zeros_like(target) if predictions is None else predictions
        g = np.array(self.loss.g(target, predictions))
        h = np.array(self.loss.h(target, predictions))
        r = g / h
        order = np.argsort(r)[::-1]
        self.g = g[order]
        self.h = h[order]
        self.data = data[order]
        self.target = target[order]
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

        num_pre = np.cumsum(g_q) ** 2
        num_suf = np.cumsum(g_q[::-1]) ** 2
        den_pre = np.cumsum(h_q) + self.reg
        den_suf = np.cumsum(h_q[::-1]) + self.reg
        neg_bound = (num_suf / den_suf).max() / (2 * self.n)
        pos_bound = (num_pre / den_pre).max() / (2 * self.n)
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        # TODO: this should probably just be defined for ext (saving the q evaluation)
        # ext = self.ext(q)
        ext = q(self.data).nonzero()[0]
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg + h_q.sum())

    def search(self, method="greedy", verbose=False, **search_params):
        ctx = SearchContext.from_array(self.data, **search_params)
        if verbose >= 2:
            print(f"Created search context with {len(ctx.attributes)} attributes")
        return search_methods[method](
            ctx, self, self.bound, verbose=verbose, **search_params
        ).run()


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

    _parameter_constraints = {
        "loss": [str, callable],
        "reg": "no-validation",
        # "reg": [Interval(Real, 0, None, closed="neither")], #TODO: Understand this
    }

    # max_col attribute to change number of propositions
    def __init__(
        self,
        loss="squared",
        reg=1.0,
        search="exhaustive",
        search_params={
            "order": "bestboundfirst",
            "apx": 1.0,
            "max_depth": None,
            "discretization": qcut,
            "max_col_attr": 10,
        },
        query=None,
    ):
        """
        :param str|callable loss: loss function either specified via string identifier (e.g., ``'squared'`` for regression or ``'logistic'`` for classification) or directly has callable loss function with defined first and second derivative (see :data:`~realkd.rules.loss_functions`)
        :param float reg: the regularization parameter :math:`\\lambda`
        :param str|type search: search method either specified via string identifier (e.g., ``'greedy'`` or ``'exhaustive'``) or directly as search type (see :func:`realkd.search.search_methods`)
        :param dict search_params: parameters to apply to discretization (when creating binary search context from
                              dataframe via :func:`~realkd.search.SearchContext.from_df`) as well as to actual search method
                              (specified by ``method``). See :mod:`~realkd.search`.
        """
        self.reg = reg
        self.loss = loss
        self.search = search
        self.search_params = search_params
        self.query = query
        self.rule_ = None

    def decision_function(self, x):
        """Predicts score for input data based on loss function.

        For instance for logistic loss will return log odds of the positive class.

        :param ~pandas.DataFrame x: input data
        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)

        """
        return self.rule_(x)

    def __repr__(self):
        return f"{type(self).__name__}(reg={self.reg}, loss={self.loss})"

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
        obj = GradientBoostingObjective(
            data, target, predictions=scores, loss=self.loss, reg=self.reg
        )
        q = (
            obj.search(method=self.search, verbose=verbose, **self.search_params)
            if self.query is None
            else self.query
        )
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
    >>> re.fit(titanic, survived.replace(0, -1)).rules_
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

    The fitted model can be used to predict for new Xs.

    >>> columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    >>> new_passengers = \
        [[2, 'male', 32, 1, 0, 10, 'Q'], \
         [2, 'female', 62, 0, 0, 7, 'S']]
    >>> new_data = pd.DataFrame(new_passengers, columns=columns)
    >>> re.predict(new_data)
    array([-1.,  1.])
    >>> re.predict_proba(new_data)
    array([[0.80609552, 0.19390448],
           [0.14841001, 0.85158999]])
    """

    _parameter_constraints = {
        "num_rules": [Integral],
        "base_learner": [HasMethods(["fit"])],
    }

    def __init__(
        self,
        num_rules=3,
        base_learner=XGBRuleEstimator(loss="squared", reg=1.0, search="greedy"),
        verbose=False,
    ):
        """

        :param int num_rules: the desired number of ensemble members
        :param Estimator|Sequence[Estimator] base_learner: the base learner(s) to be used in each iteration (last base learner is used as many time as necessary to fit desired number of rules)

        """
        self.num_rules = num_rules
        self.base_learner = base_learner
        self.rules_ = AdditiveRuleEnsemble([])
        self.verbose = verbose

    def _next_base_learner(self):
        if isinstance(self.base_learner, collections.abc.Sequence):
            return self.base_learner[min(len(self.rules_), len(self.base_learner) - 1)]
        else:
            return clone(self.base_learner)

    def decision_function(self, x):
        """Computes combined prediction scores using all ensemble members.

        :param ~pandas.DataFrame x: input data

        :return: :class:`~numpy.array` of prediction scores (one for each rows in x)
        """
        return self.rules_(x)

    def __repr__(self):
        return f"{type(self).__name__}(max_rules={self.num_rules}, base_learner={self.base_learner})"

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, data, target, feature_names=None):
        data, target = self._validate_data(
            data, target, multi_output=True, force_all_finite=False
        )
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        while len(self.rules_) < self.num_rules:
            scores = self.rules_(data)
            estimator = self._next_base_learner()
            estimator.fit(data, target, scores, max(self.verbose - 1, 0))
            if self.verbose:
                print(estimator.rule_)
            self.rules_.append(estimator.rule_)

        return self

    def predict(self, data):
        loss = loss_function(self._next_base_learner().loss)
        return loss.predictions(self.rules_(data))

    def predict_proba(self, data):
        loss = loss_function(self._next_base_learner().loss)
        return loss.probabilities(self.rules_(data))

    def _more_tags(self):
        return {"allow_nan": True}


if __name__ == "__main__":
    import doctest

    doctest.testmod()
