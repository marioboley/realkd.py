"""
Elements of propositional logic: constraints, propositions, and
conjunctions.
"""

import numpy as np

from realkd.datasets import titanic_data
from numpy import logical_and, ones

class IndexValueProposition:
    # Type: ">=" or "<=" or "=="
    def __init__(self, comparison, index, value):
        self.comparison = comparison
        self.index = index
        self.value = value

    def __call__(self, rows):
        right_column = np.array(rows).take(self.index, -1)

        if self.comparison == ">=":
            return right_column >= self.value
        if self.comparison == "<=":
            return right_column <= self.value
        if self.comparison == "==":
            return right_column == self.value

    def __repr__(self):
        return f'x{self.index}{self.comparison}{self.value}'
    def __eq__(self, other):
        return self.index == other.index and self.comparison == other.comparison and self.value == other.value 
    def __le__(self, other):
        return str(self) <= str(other)
    
    @staticmethod
    def greater_equals(*args, **kwargs):
        return IndexValueProposition(">=", *args, **kwargs)
    @staticmethod
    def less_equals(*args, **kwargs):
        return IndexValueProposition("<=", *args, **kwargs)
    @staticmethod
    def equals(*args, **kwargs):
        return IndexValueProposition("==", *args, **kwargs)

class TabulatedProposition:

    def __init__(self, table, col_idx):
        self.table = table
        self.col_idx = col_idx
        self.repr = 'c'+str(col_idx)

    def __call__(self, row_idx):
        return self.table[row_idx][self.col_idx]

    def __repr__(self):
        return self.repr


class Conjunction:
    def __init__(self, props):
        self.props = sorted(props, key=str)

    def __call__(self, x):
        # TODO: check performance of the logical_and.reduce implementation (with list materialization)
        if not self.props:
            return ones(len(x), dtype='bool')  # TODO: check if this is correct handling for scalar x
        return logical_and.reduce([p(x) for p in self.props])

    def __repr__(self):
        return str.join(" & ", map(str, self.props)) if self.props else 'True'

    def __len__(self):
        return len(self.props)

    def __eq__(self, other):
        """
        Checks equality of conjunctions based on string representation.
        """
        return str(self) == str(other)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
