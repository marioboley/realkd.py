# import io
# import sys
# import pandas as pd
# import numpy as np
# from timeit import timeit

# import unittest
# import realkd.rule_boosting

# from realkd.logic import Conjunction, IndexValueProposition  # noqa: F401
# from realkd.datasets import titanic_column_trans, titanic_data

# from doctest import DocTestSuite


# class GradientBoostingObjectiveTestCase(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         titanic = pd.read_csv("./datasets/titanic/train.csv")
#         cls.X = titanic_column_trans.fit_transform(titanic)
#         cls.y = np.array(titanic["Survived"])
#         cls.female = Conjunction([IndexValueProposition.equals(0, 1)])
#         cls.first_class = Conjunction([IndexValueProposition.less_equals(10, 1)])

#         cls.maxDiff = None

#     def test_search_titanic(self):
#         setup1 = """
# from realkd.datasets import titanic_column_trans, titanic_data
# from realkd.rule_boosting import GradientBoostingObjective
# titanic = titanic_data()
# survived = titanic['Survived']
# titanic = titanic_column_trans.fit_transform(titanic)
# sql_survival = GradientBoostingObjective(titanic, survived, reg=2)"""
#         print(timeit("sql_survival.search()", setup1, number=5))


# def load_tests(loader, tests, ignore):
#     # tests.addTests(DocTestSuite(module=realkd.rule_boosting))
#     return tests


# if __name__ == "__main__":
#     unittest.main(verbosity=2)
