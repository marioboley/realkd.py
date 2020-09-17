# realkd.py

Methods for knowledge discovery from data and interpretable machine learning.
Currently, package contains primarily rule ensembles learners.

```
 >>> import pandas as pd
 >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
 >>> survived = titanic.Survived
 >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
 >>> re = GradientBoostingRuleEnsemble(loss=logistic_loss)
 >>> re.fit(titanic, survived.replace(0, -1), verbose=0) 
    -1.4248 if Pclass>=2 & Sex==male
    +1.7471 if Pclass<=2 & Sex==female
    +2.5598 if Age<=19.0 & Fare>=7.8542 & Parch>=1.0 & Sex==male & SibSp<=1.0
```
