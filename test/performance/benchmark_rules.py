# from timeit import timeit

# setup1 = """import pandas as pd
# from realkd.rules import GradientBoostingObjective
# titanic = pd.read_csv("../datasets/titanic/train.csv")
# sql_survival = GradientBoostingObjective(titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']), titanic['Survived'], reg=2)"""

# setup2 = """import pandas as pd
# from realkd.legacy import SquaredLossObjective
# titanic = pd.read_csv("../datasets/titanic/train.csv")
# sql_survival = SquaredLossObjective(titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']), titanic['Survived'], reg=2)"""

# print(timeit("sql_survival.search()", setup1, number=5))
# print(timeit("sql_survival.search()", setup2, number=5))
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
# print(X, y)
print(y, clf.predict_proba(X)[:, 1])
print(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
# roc_auc_score(y, clf.decision_function(X))