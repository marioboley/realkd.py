import numpy as np

class CorrectionMethod:
    def __init__():
        pass

    def run(weights):
        '''
            :param weights: a 1d or 0d array of weights to correct, in the form:

            Rule # (could be 0d),   Weight
            1                          w1
            2                          w2
            3                          w3

            Should return an array of corrected weights of the same dimensions.
        '''
        pass


class GradientDescent(CorrectionMethod):
    def run():
        pass

# TODO: less conflicting name?
class LineDescent(CorrectionMethod):
    def run():
        pass


class NewtonCG(CorrectionMethod):
    def run():
        pass

CORRECTION_METHODS = {
    'Newton-CG': NewtonCG,
    'GD': GradientDescent,
    'line': LineDescent
}

def get_correction_method(correction_method='Newton-CG'):
    """Provides correction methods from string representation.

    :param correction_method: string identifier of correction method
    :return: correction method matching corresponding to input string (or unchanged input if was already correction method)
    """
    if callable(correction_method):
        return correction_method
    else:
        return CORRECTION_METHODS[correction_method]