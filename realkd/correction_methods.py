from math import inf, sqrt
from typing import Callable, Type
import numpy as np

from numpy.typing import NDArray
from numpy import zeros_like, floating
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

def get_correction_method(correction_method='Newton-CG') -> Callable[..., NDArray[floating]]:
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