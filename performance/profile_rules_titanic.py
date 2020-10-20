import cProfile
import pstats
from pstats import SortKey

from os.path import isfile

NUM_RULES = 10
FILENAME = f'rule_profiling_titanic_{NUM_RULES}.stats'
RERUN = True

if RERUN or not isfile(FILENAME):
    import pandas as pd
    from realkd.rules import GradientBoostingRuleEnsemble
    titanic = pd.read_csv('../datasets/titanic/train.csv')
    survived = titanic.Survived
    titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    re = GradientBoostingRuleEnsemble(max_rules=NUM_RULES, loss='logistic')
    cProfile.run('re.fit(titanic, survived.replace(0, -1), verbose=3)', FILENAME)

p = pstats.Stats(FILENAME)
p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
