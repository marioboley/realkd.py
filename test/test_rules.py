from tabnanny import verbose
import unittest
from realkd.logic import IndexValueProposition
import realkd.rules
from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator, logistic_loss, GradientBoostingObjective, Conjunction, Constraint
import pandas as pd

from doctest import DocTestSuite


def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.rules))
    return tests


if __name__ == '__main__':
    # unittest.main(verbosity=2)
    
    import pandas as pd
    titanic = pd.read_csv('../datasets/titanic/train.csv')
    survived = titanic.Survived
    titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived', 'Pclass'], inplace=True)
    print(titanic)
    re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss), verbose=100)
    re.fit(titanic, survived.replace(0, -1))
    
    print(len(survived))
    print(re.rules_)

    # titanic = pd.read_csv("../datasets/titanic/train.csv")
    # survived = titanic['Survived']
    # titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    # obj = GradientBoostingObjective(titanic, survived, reg=0.0)
    # female = Conjunction([IndexValueProposition(3,'Sex', Constraint.equals('female'))])
    # first_class = Conjunction([IndexValueProposition(4, 'Pclass', Constraint.less_equals(1))])
    # obj(obj.data[female].index)
    # obj(obj.data[first_class].index)
    # obj.bound(obj.data[first_class].index)
    # reg_obj = GradientBoostingObjective(titanic, survived, reg=2)
    # reg_obj(reg_obj.data[female].index)
    # reg_obj(reg_obj.data[first_class].index)

