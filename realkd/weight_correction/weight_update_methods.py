from math import inf, sqrt

import numpy as np
from numpy import zeros_like
import scipy

from realkd.rules import AdditiveRuleEnsemble, Rule
from correction_methods import get_correction_method

class WeightUpdateMethod:
    def __init__(self, loss, reg=1.0):
        self.loss = loss
        self.reg = reg

    def calc_weight(self, data, target, rules):
        raise NotImplementedError()

    @staticmethod
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

class FullyCorrective(WeightUpdateMethod):
    '''
        FullyCorrective updates all weights
    '''

    def get_corrected_weights(self, data, target, rules, loss, reg, correction_method='Newton-CG'):
        w = np.array([3.5 if r.y > 100 and self.loss == 'poisson' else r.y for r in rules])

        # w = get_correction_method(correction_method)(w, other_params)

        return w

class LineSearch(WeightUpdateMethod):
    '''
        Line search only updates the most recent weight
    '''
    def get_corrected_weights(self, data, target, rules, correction_method='Newton-CG'):
        all_weights = np.array([rule.y for rule in rules][:-1])

        w = np.array([3.5 if rules[-1].y > 100 and self.loss == 'poisson' else rules[-1].y])

        # w = get_correction_method(correction_method)(w, other_params)
        
        all_weights = np.append(all_weights, w)
        return all_weights

class NoUpdate(WeightUpdateMethod):
    '''
        Line search only updates the most recent weight
    '''
    def get_corrected_weights(self, data, target, rules, correction_method='Newton-CG'):
        return np.array([rule.y for rule in rules])

WEIGHT_UPDATE_METHODS = {
    'line': LineSearch,
    'fully_corrective': FullyCorrective,
    'no_update': NoUpdate
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