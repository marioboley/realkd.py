import unittest
import numpy as np
import pandas as pd

from realkd.propositions import *


class TestPropositions(unittest.TestCase):
    def test_constraint(self):
        c = Constraint.less_equals(21)
        
        self.assertEqual(repr(c), 'Constraint(x<=21)')
        self.assertEqual(c.to_string(), 'x<=21')
        self.assertEqual(format(c, 'age'), 'age<=21')

        self.assertTrue(c(18))
        self.assertFalse(c(63))

        values = np.array([True,  True,  True,  True, False, False, False])
        self.assertTrue(np.array_equal(c(np.arange(18, 25)), values))
        
    def test_proposition(self):
        prop1 = Proposition()
        prop2 = Proposition()
        
        self.assertEqual(repr(prop1), 'Proposition()')
        self.assertEqual(prop1.to_string(), '')
        self.assertEqual(str(prop1), '')
        
        self.assertEqual(prop1, prop2)
        self.assertFalse(prop1 < prop2)
        self.assertFalse(prop1 > prop2)
        
    def test_key_proposition(self):
        prop1 = KeyProposition('x')
        prop2 = KeyProposition('y')
        prop3 = KeyProposition('x')
        
        self.assertIsInstance(prop1, Proposition)
        self.assertEqual(repr(prop1), 'KeyProposition(x)')
        
        self.assertEqual(prop1.to_string(), 'x')
        self.assertEqual(str(prop1), 'x')
        self.assertEqual(len(prop1), 1)
        
        self.assertTrue(prop1 < prop2)
        self.assertTrue(prop1 == prop3)

        self.assertTrue(prop1.extension({'x': 1}))
        self.assertFalse(prop1.extension({'y': 1}))

        values = prop1.extension({'x': np.array([1, 2, 3])})
        self.assertIsInstance(values, np.ndarray)
        self.assertIsInstance(prop1.extension({'x': pd.Series([1, 2, 3])}),
                              np.ndarray)
        self.assertTrue(values.sum() == 3)
        
    def test_key_value_proposition(self):
        prop1 = KeyValueProposition('x', Constraint.less_equals(5))
        prop2 = KeyValueProposition('x', Constraint.equals(5))
        prop3 = KeyValueProposition('x', Constraint.greater_equals(5))
        
        self.assertIsInstance(prop1, Proposition)
        self.assertIsInstance(prop1.constraint, Constraint)
        self.assertEqual(repr(prop1), 'KeyValueProposition(x<=5)')
        
        self.assertEqual(prop1.to_string(), 'x<=5')
        self.assertEqual(str(prop1), 'x<=5')
        self.assertEqual(len(prop1), 4)
        
        self.assertTrue(prop1.extension({'x': 4}))
        self.assertTrue(np.array_equal(prop1.extension({'x': np.array([4, 5, 6])}),
                                       np.array([True, True, False])))
        
        self.assertTrue(prop1 < prop2 < prop3)
        
    def test_tabulated_proposition(self):
        table = [[0, 1, 0, 1], [1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]]
        prop1 = TabulatedProposition(1)
        
        self.assertIsInstance(prop1, Proposition)
        self.assertEqual(prop1.index, 1)
        
        self.assertEqual(repr(prop1), 'TabulatedProposition(column=:,row=1)')
        self.assertTrue(np.array_equal(prop1.extension(table),
                                       np.array(table).T[1]))
        
    def test_conjunction(self):
        old = KeyValueProposition('age', Constraint.greater_equals(60))
        male = KeyValueProposition('sex', Constraint.equals('male'))
        female = KeyValueProposition('sex', Constraint.equals('female'))

        stephanie = {'age': 30, 'sex': 'female'}
        erika = {'age': 72, 'sex': 'female'}
        ron = {'age': 67, 'sex': 'male'}

        high_risk = Conjunction([male, old])
        self.assertEqual(str(high_risk), "age>=60 & sex=='male'")
        self.assertEqual(high_risk.to_string(), "age>=60 & sex=='male'")
        self.assertEqual(repr(high_risk), "Conjunction(age>=60 & sex=='male')")

        self.assertTrue(len(high_risk) == 2)
        self.assertTrue(high_risk[0] is old)
        self.assertTrue(high_risk.get(1) is male)

        male2 = KeyValueProposition('sex', Constraint.equals('male'))
        self.assertTrue(male2 in high_risk)
        self.assertFalse(high_risk.has(female))

        for person, result in zip([stephanie, erika, ron], [False, False, True]):
            self.assertTrue(high_risk(person) == result)
            self.assertTrue(high_risk.extension(person) == result)
        
