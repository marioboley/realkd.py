import cProfile
import pstats
from pstats import SortKey

from os.path import isfile

RERUN = True
NUM_RULES = 3
FILENAME = f'profile_rule_digits_{NUM_RULES}.stats'

if RERUN or not isfile(FILENAME):
    from pandas import DataFrame, Series
    from realkd.rules import RuleBoostingEstimator
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    bunch = datasets.load_digits()
    data = DataFrame(bunch.data, columns=bunch.feature_names)
    target = Series(bunch.target)

    target[target != 3] = -1
    target.replace(3, 1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)

    rules = RuleBoostingEstimator(num_rules=NUM_RULES, loss='logistic', reg=10, method='bestboundfirst', apx=[1.0, 1.0, 0.8], max_depth=None)
    cProfile.run('rules.fit(x_train, y_train, verbose=3)', FILENAME)

p = pstats.Stats(FILENAME)
p.sort_stats(SortKey.CUMULATIVE).print_stats(50)
