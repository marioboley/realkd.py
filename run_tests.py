import numpy as np
from realkd.search import Context
from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator, logistic_loss

import pandas as pd
from sklearn.metrics import roc_auc_score
titanic = pd.read_csv('./datasets/titanic/train.csv')
# titanic = titanic.iloc[888:890]
survived = titanic.Survived
titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss))
re.fit(titanic, survived.replace(0, -1))
# 
# OLD: [Pclass==1, Pclass==2, Pclass==3, Sex==female, Sex==male, Age<=19.0, Age>=19.0, Age<=25.0, Age>=25.0, Age<=31.800000000000068, Age>=31.800000000000068, Age<=41.0, Age>=41.0, Age<=80.0, Age>=80.0, SibSp<=1.0, SibSp>=1.0, SibSp<=8.0, SibSp>=8.0, Parch<=1.0, Parch>=1.0, Parch<=6.0, Parch>=6.0, Fare<=7.8542, Fare>=7.8542, Fare<=10.5, Fare>=10.5, Fare<=21.67920000000004, Fare>=21.67920000000004, Fare<=39.6875, Fare>=39.6875, Fare<=512.3292, Fare>=512.3292, Embarked==C, Embarked==Q, Embarked==S, Embarked==nan]
# DEV: [Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==female, Sex==male, Age<=19.0, Age>=19.0, Age<=25.0, Age>=25.0, Age<=31.800000000000068, Age>=31.800000000000068, Age<=41.0, Age>=41.0, Age<=80.0, Age>=80.0, SibSp<=1.0, SibSp>=1.0, SibSp<=8.0, SibSp>=8.0, Parch<=1.0, Parch>=1.0, Parch<=6.0, Parch>=6.0, Fare<=7.8542, Fare>=7.8542, Fare<=10.5, Fare>=10.5, Fare<=21.67920000000004, Fare>=21.67920000000004, Fare<=39.6875, Fare>=39.6875, Fare<=512.3292, Fare>=512.3292, Embarked==S, Embarked==C, Embarked==Q, Embarked==nan]
# NEW: [Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==female, Sex==male, Age<=19.0, Age>=19.0, Age<=25.0, Age>=25.0, Age<=31.800000000000068, Age>=31.800000000000068, Age<=41.0, Age>=41.0, Age<=80.0, Age>=80.0, SibSp<=1.0, SibSp>=1.0, SibSp<=8.0, SibSp>=8.0, Parch<=1.0, Parch>=1.0, Parch<=6.0, Parch>=6.0, Fare<=7.8542, Fare>=7.8542, Fare<=10.5, Fare>=10.5, Fare<=21.67920000000004, Fare>=21.67920000000004, Fare<=39.6875, Fare>=39.6875, Fare<=512.3292, Fare>=512.3292, Embarked==C, Embarked==Q, Embarked==S, Embarked==nan]
print(re.rules_)
# Context.from_array(
#     np.array([
#         ['as',1,3],
#         ['asdf', 4,5],
#         ['asdf', 5,15],
#         ['asdf', 6,70],
#         ['asdf', 2,30],
#         ['as', 1,100],
# ]), ['abc', 'a2', 'a3'], max_col_attr=5)
