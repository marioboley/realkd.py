from timeit import timeit

setup1 = \
"""import pandas as pd
from realkd.rules import GradientBoostingObjective
titanic = pd.read_csv("../datasets/titanic/train.csv")
sql_survival = GradientBoostingObjective(titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']), titanic['Survived'], reg=2)"""

setup2 = \
"""import pandas as pd
from realkd.legacy import SquaredLossObjective
titanic = pd.read_csv("../datasets/titanic/train.csv")
sql_survival = SquaredLossObjective(titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']), titanic['Survived'], reg=2)"""

print(timeit('sql_survival.search()', setup1, number=5))
print(timeit('sql_survival.search()', setup2, number=5))
