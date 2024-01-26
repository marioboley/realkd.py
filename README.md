# realkd.py

Methods for knowledge discovery from data and interpretable machine learning.
Currently, package contains primarily rule ensembles learners.

```
    >>> import pandas as pd
    >>> from sklearn.metrics import roc_auc_score
    >>> from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> survived = titanic.Survived
    >>> column_trans = make_column_transformer(
            (OneHotEncoder(drop=None), ['Sex', 'Embarked']),
            ('passthrough', ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']), verbose_feature_names_out=False)
    >>> X = column_trans.fit_transform(titanic)
    >>> re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss))
    >>> fitted = re.fit(X, survived.replace(0, -1), verbose=0)
    >>> 
       -1.4248 if Pclass>=2 & Sex==male
       +1.7471 if Pclass<=2 & Sex==female
       +2.5598 if Age<=19.0 & Fare>=7.8542 & Parch>=1.0 & Sex==male & SibSp<=1.0
```

See the full [documentation](https://realkdpy.readthedocs.io/en/latest/index.html).