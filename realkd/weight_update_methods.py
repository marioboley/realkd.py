
from typing import Any, Callable, Dict, Optional, Type, Union
import numpy as np
from numpy import zeros_like, floating
from numpy.typing import NDArray
import scipy

from realkd.rules import AdditiveRuleEnsemble, Rule
from realkd.correction_methods import get_correction_method

WeightUpdateMethod = Callable[[AdditiveRuleEnsemble, Any, Optional[Union[str, Callable]]], NDArray[floating]]

def fully_corrective(rules, loss, correction_method='Newton-CG', **kwargs):
    '''
        FullyCorrective updates all weights
    '''
    w = np.array([3.5 if r.y > 100 and loss == 'poisson' else r.y for r in rules])

    w = get_correction_method(correction_method)(w, **kwargs)

    return w

def line_search(rules, loss, correction_method='Newton-CG', **kwargs):
    '''
        Line search only updates the most recent weight
    '''
    all_weights = np.array([rule.y for rule in rules][:-1])

    w = np.array([3.5 if rules[-1].y > 100 and loss == 'poisson' else rules[-1].y])

    w = get_correction_method(correction_method)(w, **kwargs)
    
    all_weights = np.append(all_weights, w)
    return all_weights

def no_update(rules, loss, correction_method='Newton-CG', **kwargs):
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