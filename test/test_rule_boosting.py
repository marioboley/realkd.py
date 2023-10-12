
import numpy as np
import pandas as pd
import importlib
from sklearn.compose import make_column_transformer

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from realkd.datasets import concat_with_unique
from realkd.loss import LogisticLoss
importlib.reload(realkd.rule_boosting)
import realkd.rule_boosting
importlib.reload(realkd.search.context)
import realkd.search.context
importlib.reload(realkd.search)
import realkd.search
import realkd.rules
# import realkd.loss
# import realkd.loss

titanic = pd.read_csv('../datasets/titanic/train.csv')
survived = titanic.Survived
column_trans = make_column_transformer(
    (OneHotEncoder(feature_name_combiner=concat_with_unique, drop=None), ['Sex', 'Embarked']),
    ('passthrough', ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']), verbose_feature_names_out=False)
X = column_trans.fit_transform(titanic)


titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
titanic.drop(columns=['Sex', 'Embarked'],  inplace=True)
feature_names = column_trans.get_feature_names_out()

re = realkd.rule_boosting.RuleBoostingEstimator(num_rules=3, base_learner=realkd.rule_boosting.XGBRuleEstimator(loss=LogisticLoss))
print(re.fit(X, survived.replace(0, -1)).rules_)
# re.fit(X, survived.replace(0, -1)).get_rules_(labels)
print(feature_names)  # (no feature names in)

X[pd.isnull(X[:, 5])][0]
re.get_params()
