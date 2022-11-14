from math import sqrt, inf

import numpy as np
import scipy
import scipy.optimize
from numpy import zeros_like, array, argsort, cumsum
from numpy.linalg import inv
from pandas import qcut
from sklearn.base import BaseEstimator

from realkd.rules import AdditiveRuleEnsemble, loss_function, Rule, SquaredLoss
from realkd.search import Context, GoldenRatioSearch


class ObjectFunction:
    def __init__(self, data, target, predictions, loss, reg, rules=None):
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
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
        raise NotImplementedError()

    def bound(self, ext):
        raise NotImplementedError()

    def search(self, method='greedy', verbose=False, **search_params):
        from realkd.search import search_methods
        ctx = Context.from_df(self.data, **search_params)
        if verbose >= 2:
            print(
                f'Created search context with {len(ctx.attributes)} attributes')
        return search_methods[method](ctx, self, self.bound, verbose=verbose, **search_params).run()


class GradientBoostingObjectiveMWG(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None):
        super().__init__(data, target, predictions, loss, reg, rules)
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = np.ones_like(g)
        r = g
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
        return abs(g_q.sum())

    def bound(self, ext):
        m = len(ext)
        if m == 0:
            return -inf
        g_q = self.g[ext]
        num_pre = abs(cumsum(g_q))
        num_suf = abs(cumsum(g_q[::-1]))
        neg_bound = num_suf.max()
        pos_bound = num_pre.max()
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        ext = self.data.loc[q].index
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg + h_q.sum())


class OrthogonalBoostingObjective(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None):
        super().__init__(data, target, predictions, loss, reg, rules)
        self.rules = [] if rules is None else rules
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        r = g
        order = argsort(r)[::-1]
        self.g = g[order]
        self.h = np.ones_like(g)
        self.data = data.iloc[order].reset_index(drop=True)
        self.target = target.iloc[order].reset_index(drop=True)
        self.n = len(target)
        if len(rules) != 0:
            self.q_mat = np.column_stack(
                [rules[i].q(self.data) + np.zeros(len(self.data)) for i in range(len(rules))])

    def __call__(self, ext):
        if len(ext) == 0:
            return -inf
        if len(self.rules) == 0:
            g_q = self.g[ext]
            h_q = self.h[ext]
            return abs(g_q.sum()) / (self.reg + np.sqrt(h_q.sum()))
        q = zeros_like(self.g)
        q[ext] = 1
        q_mat = self.q_mat
        q_proj = q_mat.dot(inv(q_mat.T.dot(q_mat))).dot(q_mat.T).dot(q)
        q_orthogonal = q - q_proj
        length = norm(q_orthogonal)
        obj = abs(self.g.dot(q_orthogonal / length)) if length > 1e-6 else 0
        return obj

    def bound(self, ext):
        """
        Temporately use this bounding function, need to change to a better one
        :param ext: the extent of a query
        :return: currently it is the maximum cumulate sum of the gradient, need to be
                 changed
        """
        m = len(ext)
        if m == 0:
            return -inf
        g_q = self.g[ext]
        h_q = self.h[ext]
        num_pre = np.abs(cumsum(g_q))
        num_suf = np.abs(cumsum(g_q[::-1]))
        den_pre = np.sqrt(cumsum(h_q)) + self.reg
        den_suf = np.sqrt(cumsum(h_q[::-1])) + self.reg
        neg_bound = (num_suf / den_suf).max()
        pos_bound = (num_pre / den_pre).max()
        return max(neg_bound, pos_bound)


class GradientBoostingObjectiveGPE(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0, rules=None):
        super().__init__(data, target, predictions, loss, reg, rules)
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(
            target) if predictions is None else predictions
        g = array(self.loss.g(target, predictions))
        h = np.ones_like(g)
        r = g
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
        return abs(g_q.sum()) / (self.reg + np.sqrt(h_q.sum())) / (2 * self.n)

    def bound(self, ext):
        m = len(ext)
        if m == 0:
            return -inf
        g_q = self.g[ext]
        h_q = self.h[ext]
        num_pre = np.abs(cumsum(g_q))
        num_suf = np.abs(cumsum(g_q[::-1]))
        den_pre = np.sqrt(cumsum(h_q)) + self.reg
        den_suf = np.sqrt(cumsum(h_q[::-1])) + self.reg
        neg_bound = (num_suf / den_suf).max() / (2 * self.n)
        pos_bound = (num_pre / den_pre).max() / (2 * self.n)
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
        ext = self.data.loc[q].index
        g_q = self.g[ext]
        h_q = self.h[ext]
        return -g_q.sum() / (self.reg + h_q.sum())


def norm(x):
    """
    Calculate the L-2 norm of a vector

    :param x: the vector whose L-2 norm is to be calculated
    :return: the L-2 norm of the vector
    """
    return (x * x).sum() ** 0.5


class WeightUpdateMethod:
    def __init__(self, loss, reg=1.0):
        self.loss = loss
        self.reg = reg

    def calc_weight(self, data, target, rules):
        raise NotImplementedError()


class FullyCorrective(WeightUpdateMethod):
    def __init__(self, loss='squared', reg=1.0, solver='Newton-CG'):
        super().__init__(loss, reg)
        self.solver = solver

    @staticmethod
    def get_risk(loss, y, q_mat, reg):
        def sum_loss(weights):
            return sum(loss(y, q_mat.dot(weights))) + reg * sum(weights * weights) / 2

        return sum_loss

    @staticmethod
    def get_gradient(g, y, q_mat, reg):
        def gradient(weights):
            grad_vec = g(y, q_mat.dot(weights))
            return q_mat.T.dot(grad_vec) + reg * weights

        return gradient

    @staticmethod
    def get_hessian(h, y, q_mat, reg):
        def hessian(weights):
            h_vec = h(y, q_mat.dot(weights))
            return q_mat.T.dot(np.diag(h_vec)).dot(q_mat) + np.diag([reg] * len(weights))

        return hessian

    def calc_weight(self, data, target, rules):
        g = loss_function(self.loss).g
        h = loss_function(self.loss).h
        loss = loss_function(self.loss)
        y = np.array(target)
        q_mat = np.column_stack(
            [rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])
        sum_loss = self.get_risk(loss, y, q_mat, self.reg)
        gradient = self.get_gradient(g, y, q_mat, self.reg)
        hessian = self.get_hessian(h, y, q_mat, self.reg)

        if self.solver == 'GD':  # Gradient descent
            w = np.array([r.y for r in rules])
            old_w = np.ones_like(w) * (1.0 if len(w) - sum(w) > 1e-5 else 2.0)
            i = 0
            while norm(old_w - w) > 1e-3 and i < 100:
                old_w = np.array(w)
                if norm(gradient(w)) == 0:
                    break
                p = -gradient(w) / norm(gradient(w))
                w += GoldenRatioSearch(sum_loss, old_w, p, gradient).run() * p
                i += 1
        elif self.solver == 'Line':
            w = np.array([r.y for r in rules])
            if norm(gradient(w)) != 0:
                p = -gradient(w) / norm(gradient(w))
                distance = GoldenRatioSearch(sum_loss, w, p, gradient).run()
                w += distance * p
        else:
            w = np.array([r.y for r in rules])
            w = scipy.optimize.minimize(sum_loss, w, method=self.solver, jac=gradient, hess=hessian,
                                        options={'disp': False}).x
        return w


class LineSearch(WeightUpdateMethod):
    def __init__(self, loss='squared', reg=1.0):
        super().__init__(loss, reg)

    @staticmethod
    def get_risk(loss, y, q_mat, weights: np.array, reg):
        def sum_loss(weight):
            all_weights = np.append(weights, weight)
            return sum(loss(y, q_mat.dot(all_weights))) + reg * sum(all_weights * all_weights) / 2

        return sum_loss

    @staticmethod
    def get_gradient(g, y, q_mat, weights: np.array, reg):
        def gradient(weight):
            all_weights = np.append(weights, weight)
            grad_vec = g(y, q_mat.dot(all_weights))
            return np.array([(q_mat.T.dot(grad_vec) + reg * all_weights)[-1]])

        return gradient

    def calc_weight(self, data, target, rules):
        w = np.array([rules[-1].y])
        all_weights = np.array([rule.y for rule in rules][:-1])
        loss = loss_function(self.loss)
        g = loss_function(self.loss).g
        y = np.array(target)
        q_mat = np.column_stack(
            [rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])
        sum_loss = self.get_risk(loss, y, q_mat, all_weights, self.reg)
        gradient = self.get_gradient(g, y, q_mat, all_weights, self.reg)
        if norm(gradient(w)) != 0:
            p = -gradient(w) / norm(gradient(w))
            distance = GoldenRatioSearch(sum_loss, w, p, gradient).run()
            w += distance * p
        all_weights = np.append(all_weights, w)
        return all_weights


class KeepWeight(WeightUpdateMethod):
    def __init__(self, loss='squared', reg=1.0):
        super().__init__(loss, reg)

    def calc_weight(self, data, target, rules):
        all_weights = np.array([rule.y for rule in rules])
        return all_weights


class GeneralRuleBoostingEstimator(BaseEstimator):
    def __init__(self, num_rules, objective_function, weight_update_method, loss='squared', reg=1.0,
                 search='exhaustive', max_col_attr=10,
                 search_params=None, verbose=False):
        if search_params is None:
            search_params = {'order': 'bestboundfirst', 'apx': 1.0, 'max_depth': None, 'discretization': qcut,
                             'max_col_attr': max_col_attr}
        self.num_rules = num_rules
        self.objective = objective_function
        self.weight_update_method = weight_update_method
        self.loss = loss_function(loss)
        self.reg = reg
        self.weight_update_method.loss = loss
        self.weight_update_method.reg = reg
        self.verbose = verbose
        self.search = search
        self.rules_ = AdditiveRuleEnsemble([])
        self.search_params = search_params
        self.history = []

    def set_reg(self, reg):
        self.reg = reg
        self.objective.reg = reg
        self.weight_update_method.reg = reg

    def fit(self, data, target, verbose=False):
        self.history = []
        self.rules_.members = []
        while len(self.rules_) < self.num_rules:
            # Search for a rule
            scores = self.rules_(data)
            obj = self.objective(data, target, predictions=scores,
                                 loss=self.loss, reg=self.reg, rules=self.rules_)
            q = obj.search(method=self.search, verbose=verbose,
                           **self.search_params)
            if hasattr(self.objective, 'opt_weight') and callable(getattr(self.objective, 'opt_weight')):
                y = obj.opt_weight(q)
            else:
                y = 1.0
            rule = Rule(q, y)
            if self.verbose:
                print(rule)
            self.rules_.append(rule)
            # Calculate weights
            weights = self.weight_update_method.calc_weight(
                data, target, self.rules_)
            for i in range(len(self.rules_)):
                self.rules_[i].y = weights[i]
            self.history.append(AdditiveRuleEnsemble(
                [Rule(q=rule.q, y=rule.y) for rule in self.rules_.members]))
        return self

    def predict(self, data):
        loss = loss_function(self.loss)
        return loss.preidictions(self.rules_(data))

    def decision_function(self, data):
        return self.rules_(data)
