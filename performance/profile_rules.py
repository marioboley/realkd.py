import cProfile
import pstats
from pstats import SortKey

from os.path import isfile

FILENAME = 'rule_profiling.stats'
RERUN = True

if RERUN or not isfile(FILENAME):
    import pandas as pd
    from realkd.rules import GradientBoostingRuleEnsemble
    titanic = pd.read_csv('../datasets/titanic/train.csv')
    survived = titanic.Survived
    titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    re = GradientBoostingRuleEnsemble(loss='logistic')
    cProfile.run('re.fit(titanic, survived.replace(0, -1), verbose=True)', FILENAME)

p = pstats.Stats(FILENAME)
p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
