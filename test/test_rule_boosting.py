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

    def test_logistic_loss(self):
        obj = realkd.rule_boosting.GradientBoostingObjective(
            self.X, np.where(self.y == 0, -1, self.y), loss="logistic"
        )

        self.assertEqual(obj(get_extents(self.female, obj.data)), 0.04077109318199465)
        self.assertEqual(obj.opt_weight(self.female), 0.9559748427672956)

        capturedOutput = io.StringIO()  # Create StringIO object
        sys.stdout = capturedOutput  # and redirect stdout.
        best = obj.search(method="exhaustive", order="bestvaluefirst", verbose=True)
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertEqual(
            capturedOutput.getvalue().strip(),
            "Found optimum after inspecting 446 nodes: [27, 29]\nGreedy simplification: [27, 29]",
        )

        self.assertEqual(str(best), "x10>=2.0 & x1==1")
        predicted_indexes = best(obj.data).nonzero()[0]
        self.assertEqual(obj(predicted_indexes), 0.13072995752734315)
        self.assertEqual(obj.opt_weight(best), -1.4248366013071896)


def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.rule_boosting))
    return tests


if __name__ == "__main__":
    unittest.main(verbosity=2)
