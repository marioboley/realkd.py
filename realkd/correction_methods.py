from math import inf, sqrt
from typing import Callable, Type
import numpy as np

from numpy.typing import NDArray
from numpy import gradient, ndarray, zeros_like, floating
import scipy

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

def get_gradient(g, y, q_mat, weights: NDArray[floating], reg):
    def gradient(weight):
        all_weights = np.append(weights, weight)
        grad_vec = g(y, q_mat.dot(all_weights))
        return np.array([(q_mat.T.dot(grad_vec) + reg * all_weights)[-1]])

    return gradient

# def get_gradient(g, y, q_mat, reg):
#     def gradient(weights):
#         grad_vec = g(y, q_mat.dot(weights))
#         return q_mat.T.dot(grad_vec) + reg * weights

#     return gradient

def get_risk(loss, y, q_mat, weights: NDArray[floating], reg):
    def sum_loss(weight):
        all_weights = np.append(weights, weight)
        return sum(loss(y, q_mat.dot(all_weights))) + reg * sum(all_weights * all_weights) / 2

    return sum_loss

# :param weights: a 1d or 0d array of weights to correct, in the form:

# Rule # (could be 0d),   Weight
# 1                          w1
# 2                          w2
# 3                          w3

# Should return an array of corrected weights of the same dimensions.

def gradient_descent(weights_to_calc, other_weights, rules, loss, data, target: NDArray[floating], reg):
    q_mat = np.column_stack([rules[i].q(data) + np.zeros(len(data)) for i in range(len(rules))])

    gradient = get_gradient(loss.g, target, q_mat, other_weights, reg)
    sum_loss = get_risk(loss, target, q_mat, other_weights, reg)

    old_w = zeros_like(weights_to_calc) * 1.0
    i = 0
    w = weights_to_calc
    while norm(old_w - w) > 1e-3 and i < 20:
        old_w = np.array(w)
        if norm(gradient(w)) == 0:
            break
        p = -gradient(w) / norm(gradient(w))
        left = 0
        right = norm(w)
        w += golden_ratio_search(sum_loss, left, right, p, old_w) * p
    
    return w

# TODO: less conflicting name?
def line_descent(initial_weights, rules, loss, data, target: NDArray[floating], reg):
    pass


def newton_CG(initial_weights, rules, loss, data, target: NDArray[floating], reg):
    pass

CORRECTION_METHODS = {
    'Newton-CG': newton_CG,
    'GD': gradient_descent,
    'line': line_descent
}

def get_correction_method(correction_method='Newton-CG') -> Callable[..., NDArray[floating]]:
    """Provides correction methods from string representation.

    :param correction_method: string identifier of correction method
    :return: correction method matching corresponding to input string (or unchanged input if was already correction method)
    """
    if callable(correction_method):
        return correction_method
    else:
        return CORRECTION_METHODS[correction_method]