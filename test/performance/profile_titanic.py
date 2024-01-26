import cProfile
import pstats
from pstats import SortKey


from os.path import isfile

NUM_RULES = 3
FILENAME = f"rule_profiling_titanic_{NUM_RULES}.stats"
RERUN = True

if RERUN or not isfile(FILENAME):
    import numpy as np
    from realkd.rule_boosting import RuleBoostingEstimator
    from realkd.datasets import titanic_column_trans, titanic_data

    titanic = titanic_data()
    survived = np.where(titanic.Survived == 0, -1, titanic.Survived)
    titanic = titanic_column_trans.fit_transform(titanic)
    re = RuleBoostingEstimator(num_rules=NUM_RULES)
    cProfile.run("re.fit(titanic, survived)", FILENAME)

p = pstats.Stats(FILENAME)
p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
