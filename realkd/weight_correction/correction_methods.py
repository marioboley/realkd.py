import numpy as np

# :param weights: a 1d or 0d array of weights to correct, in the form:

# Rule # (could be 0d),   Weight
# 1                          w1
# 2                          w2
# 3                          w3

# Should return an array of corrected weights of the same dimensions.

def gradient_descent(initial_weights, data, target: np.array, loss, rules, reg):
    pass

# TODO: less conflicting name?
def line_descent(initial_weights, data, target: np.array, loss, rules, reg):
    pass


def newton_CG(initial_weights, data, target: np.array, loss, rules, reg):
    pass

CORRECTION_METHODS = {
    'Newton-CG': newton_CG,
    'GD': gradient_descent,
    'line': line_descent
}

def get_correction_method(correction_method='Newton-CG') -> callable:
    """Provides correction methods from string representation.

    :param correction_method: string identifier of correction method
    :return: correction method matching corresponding to input string (or unchanged input if was already correction method)
    """
    if callable(correction_method):
        return correction_method
    else:
        return CORRECTION_METHODS[correction_method]