import io
import sys
import pandas as pd
import numpy as np

import unittest
import realkd.rule_boosting

from realkd.logic import Conjunction, IndexValueProposition  # noqa: F401
from realkd.datasets import titanic_column_trans

from doctest import DocTestSuite


def get_extents(filter, data):
    return filter(data).nonzero()[0]


class GradientBoostingObjectiveTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        titanic = pd.read_csv("./datasets/titanic/train.csv")
        cls.X = titanic_column_trans.fit_transform(titanic)
        cls.y = np.array(titanic["Survived"])
        cls.female = Conjunction([IndexValueProposition.equals(0, 1)])
        cls.first_class = Conjunction([IndexValueProposition.less_equals(10, 1)])
        
        cls.maxDiff = None

    def test_no_reg(self):
        obj = realkd.rule_boosting.GradientBoostingObjective(self.X, self.y, reg=0.0)

        self.assertEqual(obj(get_extents(self.female, obj.data)), 0.1940459084832758)

        first_class_extents = get_extents(self.first_class, obj.data)
        self.assertEqual(obj(first_class_extents), 0.09610508375940474)
        self.assertEqual(obj.bound(first_class_extents), 0.1526374859708193)

    def test_with_reg(self):
        obj = realkd.rule_boosting.GradientBoostingObjective(self.X, self.y, reg=2)
        self.assertEqual(obj(get_extents(self.female, obj.data)), 0.19342988972618603)
        self.assertEqual(
            obj(get_extents(self.first_class, obj.data)), 0.09566220318908492
        )

    # @unittest.skip("Currently Failing")
    def test_logistic_loss(self):
        obj = realkd.rule_boosting.GradientBoostingObjective(
            self.X, np.where(self.y == 0, -1, self.y), loss="logistic"
        )

        print(obj.data)
        print(obj.target)
        print(self.female(obj.data))
        print(self.female(obj.data).nonzero()[0])
        print(obj(self.female(obj.data).nonzero()[0]))

        self.assertEqual(obj(get_extents(self.female, obj.data)), 0.04077109318199465)
        self.assertEqual(obj.opt_weight(self.female), 0.9559748427672956)

        # TODO: Unskip after I'm confident about context creation 
        
        # capturedOutput = io.StringIO()  # Create StringIO object
        # sys.stdout = capturedOutput  # and redirect stdout.
        # best = obj.search(method="exhaustive", order="bestvaluefirst", verbose=True, max_col_attr=6)
        # sys.stdout = sys.__stdout__  # Reset redirect.
        # self.assertEqual(
        #     capturedOutput.getvalue(),
        #     """
        
        # Found optimum after inspecting 446 nodes: [27, 29]
        # Greedy simplification: [27, 29]
        # """,
        # )

        # self.assertEqual(str(best), "Pclass>=2 & Sex==male")
        # self.assertEqual(obj(self.best(obj.data).nonzero()[0]), 0.13072995752734315)
        # self.assertEqual(
        #     obj.opt_weight(self.best(obj.data).nonzero()[0]), -1.4248366013071896
        # )


def load_tests(loader, tests, ignore):
    # tests.addTests(DocTestSuite(module=realkd.rule_boosting))
    return tests


if __name__ == "__main__":
    unittest.main(verbosity=2)


# import numpy as np
# import pandas as pd
# import importlib
# from sklearn.compose import make_column_transformer

# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import OneHotEncoder
# from realkd.datasets import concat_with_unique
# from realkd.loss import LogisticLoss
# importlib.reload(realkd.rule_boosting)
# import realkd.rule_boosting
# importlib.reload(realkd.search.context)
# import realkd.search.context
# importlib.reload(realkd.search)
# import realkd.search
# import realkd.rules
# # import realkd.loss
# import realkd.loss

#     >>> import pandas as pd
#     >>> from sklearn.metrics import roc_auc_score
#     >>> from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator
#     >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
#     >>> survived = titanic.Survived
#     >>> column_trans = make_column_transformer(
#             (OneHotEncoder(drop=None), ['Sex', 'Embarked']),
#             ('passthrough', ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']), verbose_feature_names_out=False)
#     >>> X = column_trans.fit_transform(titanic)
#     >>> re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss))
#     >>> fitted = re.fit(X, survived.replace(0, -1), verbose=0)
#     >>>
#        -1.4248 if Pclass>=2 & Sex==male
#        +1.7471 if Pclass<=2 & Sex==female
#        +2.5598 if Age<=19.0 & Fare>=7.8542 & Parch>=1.0 & Sex==male & SibSp<=1.0
# titanic = pd.read_csv('../datasets/titanic/train.csv')
# survived = titanic.Survived
# column_trans = make_column_transformer(
#     (OneHotEncoder(feature_name_combiner=concat_with_unique, drop=None), ['Sex', 'Embarked']),
#     ('passthrough', ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']), verbose_feature_names_out=False)
# X = column_trans.fit_transform(titanic)


# titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
# titanic.drop(columns=['Sex', 'Embarked'],  inplace=True)
# feature_names = column_trans.get_feature_names_out()

# re = realkd.rule_boosting.RuleBoostingEstimator(num_rules=3, base_learner=realkd.rule_boosting.XGBRuleEstimator(loss=LogisticLoss))
# print(re.fit(X, survived.replace(0, -1)).rules_)
# # re.fit(X, survived.replace(0, -1)).get_rules_(labels)
# print(feature_names)  # (no feature names in)

# X[pd.isnull(X[:, 5])][0]
# re.get_params()


[x0==1,
 x1==1,
 x2==1,
 x3==1,
 x4==1,
 x5==1,
 x6<=7.8542,
 x6>=7.8542,
 x6<=10.5,
 x6>=10.5,
 x6<=21.67920000000004,
 x6>=21.67920000000004,
 x6<=39.6875,
 x6>=39.6875,
 x6<=512.3292,
 x6>=512.3292,
 x7<=1.0,
 x7>=1.0,
 x7<=8.0,
 x7>=8.0,
 x8<=1.0,
 x8>=1.0,
 x8<=6.0,
 x8>=6.0,
 x9==nan,
 x9<=19.0,
 x9>=19.0,
 x9<=25.0,
 x9>=25.0,
 x9<=31.800000000000068,
 x9>=31.800000000000068,
 x9<=41.0,
 x9>=41.0,
 x9<=80.0,
 x9>=80.0,
 x10<=1.0,
 x10<=2.0,
 x10>=2.0,
 x10>=3.0]