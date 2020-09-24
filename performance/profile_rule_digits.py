import cProfile
import pstats
from pstats import SortKey

from os.path import isfile

FILENAME = 'profile_rule_digits.stats'
RERUN = False
NUM_RULES = 2

if RERUN or not isfile(FILENAME):
    from pandas import DataFrame, Series
    from realkd.rules import GradientBoostingRuleEnsemble
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    bunch = datasets.load_digits()
    data = DataFrame(bunch.data, columns=bunch.feature_names)
    target = Series(bunch.target)

    target[target != 3] = -1
    target.replace(3, 1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)

    rules = GradientBoostingRuleEnsemble(max_rules=NUM_RULES, loss='logistic', reg=10)
    cProfile.run('rules.fit(x_train, y_train, verbose=3)', FILENAME)

p = pstats.Stats(FILENAME)
p.sort_stats(SortKey.CUMULATIVE).print_stats(50)
