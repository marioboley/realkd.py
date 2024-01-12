"""
Elements of propositional logic: propositions and conjunctions
"""

import numpy as np

from numpy import logical_and, ones


class IndexValueProposition:
    """
    Parameters
    ----------
    comparison : {">=", "==", "<="}
    index : int
    value : int or float

    Examples
    --------
    >>> p = IndexValueProposition.greater_equals(1, 10)
    >>> p == IndexValueProposition(">=", 1, 10)
    True
    >>> p
    x1>=10
    >>> p([[1, 9, 105], [1, 10, 99]])
    array([False,  True])

    A 1D array is treated as a single sample
    >>> p([[1, 10, 99]])
    array([ True])
    >>> p([1, 10, 99])
    True
    """

    def __init__(self, comparison, index, value):
        self.comparison = comparison
        self.index = index
        self.value = value

    def __call__(self, rows):
        """
        Args:
            rows : array-like of shape (n_samples, n_features) or (n_features)

        Returns:
            bool array-like of shape (n_samples), or scalar boolean.
        """
        right_column = np.array(rows).take(self.index, -1)

        if self.comparison == ">=":
            return right_column >= self.value
        if self.comparison == "<=":
            return right_column <= self.value
        if self.comparison == "==":
            return right_column == self.value

    def __repr__(self):
        return f"x{self.index}{self.comparison}{self.value}"

    def __eq__(self, other):
        return (
            self.index == other.index
            and self.comparison == other.comparison
            and self.value == other.value
        )

    def __le__(self, other):
        if self.index != other.index:
            return self.index <= other.index
        if self.comparison != other.comparison:
            return self.comparison <= other.comparison
        return self.value <= other.value

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
        self.repr = "c" + str(col_idx)

    def __call__(self, row_idx):
        return self.table[row_idx][self.col_idx]

    def __repr__(self):
        return self.repr


class Conjunction:
    """
    Conjunctive aggregation of propositions.
    Parameters
    ----------
    props : enumerable of callable propositions.

    Examples
    --------
    >>> prop1 = IndexValueProposition.greater_equals(1, 10)
    >>> prop2 = IndexValueProposition.greater_equals(2, 100)
    >>> c = Conjunction([prop1, prop2])
    >>> c([[1, 9, 105], [1, 10, 99], [1, 10, 105]])
    array([False, False,  True])

    Elements are sorted by index, then by operation.
    >>> c
    x1>=10 & x2>=100

    A 1D array is treated as a single sample
    >>> c([1, 10, 105])
    True
    """

    def __init__(self, props):
        self.props = sorted(props, key=str)

    def __call__(self, x):
        # TODO: check performance of the logical_and.reduce implementation (with list materialization)
        if not self.props:
            # TODO: check if this is correct handling for scalar x
            return ones(len(x), dtype="bool")
        return logical_and.reduce([p(x) for p in self.props])

    def __repr__(self):
        return str.join(" & ", map(str, self.props)) if self.props else "True"

    def __len__(self):
        return len(self.props)

    def __eq__(self, other):
        """
        Checks equality of conjunctions based on string representation.
        """
        return str(self) == str(other)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
