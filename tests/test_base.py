import random
import unittest
import numpy as np
import pandas as pd

from realkd.base import *


class TestContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.read_csv("datasets/titanic/train.csv", low_memory=False)
        data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'],
                  inplace=True)

        # Provide default data for testing
        cls.data = data
        cls.table = [[0, 1, 0, 1], [1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]]

        # Default context
        cls.context = Context.from_dataframe(data, nbins=5,
                                             sort_attributes=False)

    # Fixture for generating a valid conjunction
    def create_conjunction(self, context, selection_frequency=0.9,
                           return_indices=True):
        size = 0
        while not size:
            props = []
            for key in context.data:
                attributes = [attr for attr in context.attributes
                              if attr.key == key]

                value = np.random.random_sample()
                if value > selection_frequency or len(attributes) < 3:
                    continue

                # Ignore attributes at the bounds
                props.append(random.choice(attributes[1:-1]))

            # Update conjunction
            conjunction = Conjunction(props)
            size = np.sum(context.extension(conjunction))

        result = conjunction
        if return_indices:
            indices = [i for i, attr in enumerate(context.attributes)
                       if attr in props]
            result = (conjunction, indices)
        return result

    def test_context(self):
        conjunction, indices = self.create_conjunction(self.context)

        # Check extension
        extension = self.context.extension(conjunction)
        self.assertEqual(extension.dtype.kind, 'b')
        self.assertTrue(np.sum(extension) > 0)

        self.assertTrue(np.array_equal(self.context.extension(indices),
                                       extension))

        # Check query
        expression = str(conjunction)
        query = self.context.query(expression)
        extension = self.context.extension(conjunction, return_indices=True)

        self.assertEqual(extension.dtype.kind, 'i')
        self.assertTrue(np.array_equal(extension, query.index))

        # Check iterators
        for key, value in zip(self.context.keys(), self.context.values()):
            a, b = self.data[key], value
            mask = pd.isnull(a) & pd.isnull(b)
            self.assertTrue(np.array_equal(a[~mask], b[~mask]))
            
    def test_from_table(self):
        ctx = Context.from_table(self.table)

        extension = ctx.extension([0, 2], return_indices=True)
        self.assertTrue(np.array_equal(extension, [1, 2]))

        extension = ctx.extension([0, 2], return_indices=False)
        self.assertTrue(np.array_equal(extension, [False, True, True, False]))

    def test_from_dataframe(self):
        self.assertTrue(len(self.context) == self.data.shape[1])
        self.assertTrue(self.context.n == self.data.shape[0])
        self.assertTrue(self.context.m == len(self.context.attributes))

        self.assertIsInstance(self.context.data, pd.DataFrame)
    
