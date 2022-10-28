from typing import Any, Callable, Dict, Optional, Union
import numpy as np
from numpy import floating
from numpy.typing import NDArray

from realkd.rules import AdditiveRuleEnsemble, Rule
from realkd.correction_methods import get_correction_method

WeightUpdateMethod = Callable[[AdditiveRuleEnsemble, Any, Optional[Union[str, Callable]]], NDArray[floating]]

def get_gradient(g, y, q_mat, weights: NDArray[floating], reg):
    def gradient(weight):
        all_weights = np.append(weights, weight)
        grad_vec = g(y, q_mat.dot(all_weights))
        return np.array([(q_mat.T.dot(grad_vec) + reg * all_weights)[-1]])

    return gradient

def get_risk(loss, y, q_mat, weights: NDArray[floating], reg):
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

WEIGHT_UPDATE_METHODS: Dict[str, WeightUpdateMethod] = {
    'line': line_search,
    'fully_corrective': fully_corrective,
    'no_update': no_update
}

def get_weight_update_method(weight_update_method='fully_corrective') -> WeightUpdateMethod:
    """Provides weight update methods from string representation.

    :param weight_update_method: string identifier of weight update method
    :return: weight update method matching corresponding to input string (or unchanged input if was already weight update method)
    """
    if callable(weight_update_method):
        return weight_update_method
    else:
        return WEIGHT_UPDATE_METHODS[weight_update_method]
