Rules
=====

.. automodule:: realkd.rules

Overview
--------

.. autosummary::

    realkd.rules.AdditiveRuleEnsemble
    realkd.rules.logistic_loss
    realkd.rules.Rule
    realkd.rules.RuleBoostingEstimator
    realkd.rules.squared_loss
    realkd.rules.XGBRuleEstimator

.. _loss_functions:

Details
-------

.. autodata:: logistic_loss
.. autodata:: loss_functions
.. autodata:: squared_loss

.. autoclass:: realkd.rules.AdditiveRuleEnsemble
    :special-members: __call__
    :members:
.. autoclass:: realkd.rules.Rule
    :special-members: __call__
    :members:
.. autoclass:: realkd.rules.RuleBoostingEstimator
    :members:
.. autoclass:: XGBRuleEstimator
    :members:
