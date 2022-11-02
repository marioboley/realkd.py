import collections.abc
import numpy as np
from numpy import zeros_like, array, argsort, cumsum
from sklearn.base import BaseEstimator, clone
from realkd.rules import AdditiveRuleEnsemble, loss_function, Rule, SquaredLoss
from realkd.search import Context
import scipy
import scipy.optimize
from math import sqrt, inf


class ObjectFunction:
    def __init__(self, data, target, predictions, loss, reg):
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
        raise NotImplementedError()

    def bound(self, ext):
        raise NotImplementedError()

    def search(self, method='greedy', verbose=False, **search_params):
        from realkd.search import search_methods
        ctx = Context.from_df(self.data, **search_params)
        if verbose >= 2:
            print(f'Created search context with {len(ctx.attributes)} attributes')
        return search_methods[method](ctx, self, self.bound, verbose=verbose, **search_params).run()


class GradientBoostingObjectiveMWG(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0):
        super().__init__(data, target, predictions, loss, reg)
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(target) if predictions is None else predictions
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
        neg_bound = (num_suf).max()
        pos_bound = (num_pre).max()
        return max(neg_bound, pos_bound)

    def opt_weight(self, q):
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


class GradientBoostingObjectiveGPE(ObjectFunction):
    def __init__(self, data, target, predictions=None, loss=SquaredLoss, reg=1.0):
        super().__init__(data, target, predictions, loss, reg)
        self.loss = loss_function(loss)
        self.reg = reg
        predictions = zeros_like(target) if predictions is None else predictions
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

    def search(self, method='greedy', verbose=False, **search_params):
        from realkd.search import search_methods
        ctx = Context.from_df(self.data, **search_params)
        if verbose >= 2:
            print(f'Created search context with {len(ctx.attributes)} attributes')
        # return getattr(ctx, method)(self, self.bound, verbose=verbose, **search_params)
        return search_methods[method](ctx, self, self.bound, verbose=verbose, **search_params).run()


class BaseLearner(BaseEstimator):
    def __init__(self, loss, reg, object_function, search='exhaustive', search_params={}, query=None):
        self.rule_ = None
        self.loss = loss
        self.search = search
        self.reg = reg
        self.object_function = object_function
        self.query = query
        self.search_params = search_params

    def fit(self, data, target):
        obj = self.object_function(self.loss, self.reg)
        q = obj.search(data, target, method=self.search, search_params=self.search_params)
        y = 0
        self.rule_ = Rule(q, y)
        return self

    def predict(self, data):
        loss = loss_function(self.loss)
        return loss.predictions(self.rule_(data))


class WeightUpdateMethod:
    def __init__(self, loss, reg=1.0):
        self.loss = loss
        self.reg = reg

    def calc_weight(self, data, target, rules):
        raise NotImplementedError()

    @staticmethod
    def golden_ratio_search(func, left, right, dir, origin):
        ratio = (sqrt(5) - 1) / 2
        while right - left > max(1e-5 * left, 1e-5):
            lam = left + (1 - ratio) * (right - left)
            mu = left + ratio * (right - left)
            r_lam = func(origin + lam * dir)
            r_mu = func(origin + mu * dir)
            if r_lam <= r_mu:
                right = mu
            else:
                left = lam
        return (left + right) / 2


class FullyCorrective(WeightUpdateMethod):
    def __init__(self, loss='squared', reg=1.0, correction_method='Newton-CG'):
        super().__init__(loss, reg)
        self.correction_method = correction_method

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
        q_mat = np.column_stack([rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])
        sum_loss = self.get_risk(loss, y, q_mat, self.reg)
        gradient = self.get_gradient(g, y, q_mat, self.reg)
        hessian = self.get_hessian(h, y, q_mat, self.reg)

        def norm(xs):
            return sqrt(sum([x * x for x in xs]))

        if self.correction_method == 'GD':  # Gradient descent
            w = np.array([10 if r.y > 40 and self.loss == 'poisson' else r.y for r in rules])
            old_w = zeros_like(w) * 1.0
            i = 0
            while norm(old_w - w) > 1e-3 and i < 20:
                old_w = np.array(w)
                if norm(gradient(w)) == 0:
                    break
                p = -gradient(w) / norm(gradient(w))
                left = 0
                right = norm(w) * 5
                w += self.golden_ratio_search(sum_loss, left, right, p, old_w) * p
                i += 1
        elif self.correction_method == 'Line':
            w = np.array([10 if r.y > 40 and self.loss == 'poisson' else r.y for r in rules])
            if norm(gradient(w)) != 0:
                p = -gradient(w) / norm(gradient(w))
                left = 0
                right = norm(w) * 5
                distance = self.golden_ratio_search(sum_loss, left, right, p, w)
                w += distance * p
        else:
            w = np.array([rules[i].y for i in range(int(len(rules)))])
            w = scipy.optimize.minimize(sum_loss, w, method=self.correction_method, jac=gradient, hess=hessian,
                                        options={'disp': False}).x
        return w


class LineSearch(WeightUpdateMethod):
    def __init__(self, loss='squared', reg=1.0):
        super().__init__(loss, reg)

    @staticmethod
    def norm(xs):
        return sqrt(sum([x * x for x in xs]))

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
        w = np.array([3.5 if rules[-1].y > 100 and self.loss == 'poisson' else rules[-1].y])
        all_weights = np.array([rule.y for rule in rules][:-1])
        loss = loss_function(self.loss)
        g = loss_function(self.loss).g
        y = np.array(target)
        q_mat = np.column_stack([rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])
        sum_loss = self.get_risk(loss, y, q_mat, all_weights, self.reg)
        gradient = self.get_gradient(g, y, q_mat, all_weights, self.reg)
        if self.norm(gradient(w)) != 0:
            p = -gradient(w) / self.norm(gradient(w))
            left = 0
            right = self.norm(w)
            distance = self.golden_ratio_search(sum_loss, left, right, p, w)
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
    def __init__(self, num_rules, base_learner, objective_function, weight_update_method, loss='squared', reg=1.0,
                 verbose=False):
        self.num_rules = num_rules
        self.base_learner = base_learner
        self.objective = objective_function
        self.weight_update_method = weight_update_method
        self.loss = loss_function(loss)
        self.reg = reg
        self.base_learner.loss = loss
        self.base_learner.reg = reg
        self.weight_update_method.loss = loss
        self.weight_update_method.reg = reg
        self.base_learner.object_func = self.objective
        self.verbose = verbose
        self.rules_ = AdditiveRuleEnsemble([])
        self.history = []

    def set_reg(self, reg):
        self.reg = reg
        self.base_learner.reg = reg
        self.objective.reg = reg
        self.weight_update_method.reg = reg

    def _next_base_learner(self):
        if isinstance(self.base_learner, collections.abc.Sequence):
            return self.base_learner[min(len(self.rules_), len(self.base_learner) - 1)]
        else:
            return clone(self.base_learner)

    def fit(self, data, target):
        self.history = []
        self.rules_.members = []
        while len(self.rules_) < self.num_rules:
            scores = self.rules_(data)
            self.base_learner.fit(data, target, scores, verbose=max(self.verbose - 1, 0))
            if self.verbose:
                print(self.base_learner.rule_)
            rule = self.base_learner.rule_
            self.rules_.append(rule)
            weights = self.weight_update_method.calc_weight(data, target, self.rules_)
            for i in range(len(self.rules_)):
                self.rules_[i].y = weights[i]
            self.history.append(AdditiveRuleEnsemble([Rule(q=rule.q, y=rule.y) for rule in self.rules_.members]))
        return self

    def predict(self, data):
        loss = loss_function(self.loss)
        return loss.preidictions(self.rules_(data))

    def decision_function(self, data):
        return self.rules_(data)
