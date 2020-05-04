# -*- coding: utf-8 -*-
"""
@package    realkd.propositions

@copyright  Copyright (c) 2020+ RealKD-Team,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
            Mario Boley <mario.boley@monash.edu>
@license    See LICENSE file for details.

Licensed under the MIT License (the "License").
You may not use this file except in compliance with the License.
"""

import functools

import numpy as np
import pandas as pd

from typing import Union, Optional, Any, Dict, Sequence, Iterable, Callable 


class Constraint:
    """Boolean condition on a single value with string representation.

    Usage:
    >>> c = Constraint.less_equals(21)
    >>> c
    Constraint(x<=21)
    >>> (c.to_string(), format(c, 'age'))
    ('x<=21', 'age<=21')
    >>> c(18)
    True
    >>> c(63)
    False
    >>> c(np.arange(18, 25))
    array([ True,  True,  True,  True, False, False, False])
    """

    def __init__(self, condition: Callable[[Any], bool],
                 formatter: Optional[Callable[[str], str]] = None):
        """Constructor.
        """
        self.condition = condition
        self.formatter = formatter or (lambda name: str(condition) + "(" + name + ")")

    def __call__(self, value: Any) -> bool:
        """Called when the constraint is 'called' as a function.
        """
        return self.condition(value)

    def __format__(self, name: str) -> str:
        """Format constraint when used in new-style string formatting.
        """
        return self.formatter(name)

    def __repr__(self) -> str:
        """String representation of the constraint.
        """
        return "{:s}({:s})".format(self.__class__.__name__, self.to_string())

    def to_string(self) -> str:
        """String representation.
        """
        return format(self, "x")

    @classmethod
    def less_equals(cls, value: Any) -> 'Constraint':
        """Constraint with a less-than operator.
        """
        return cls(lambda v: v <= value, lambda n: "{}<={!r}".format(n, value))

    @classmethod
    def greater_equals(cls, value: Any) -> 'Constraint':
        """Constraint with a greater-than operator.
        """
        return cls(lambda v: v >= value, lambda n: "{}>={!r}".format(n, value))

    @classmethod
    def equals(cls, value: Any) -> 'Constraint':
        """Constraint with an equality operator.
        """
        return cls(lambda v: v == value, lambda n: "{}=={!r}".format(n, value))


@functools.total_ordering
class Proposition:
    """Callable proposition base class.

    Usage:
    >>> p1 = Proposition()
    >>> p1
    Proposition()
    >>> p1.to_string()
    ''
    >>> p1.extension(False)
    False
    >>> p2 = Proposition()
    >>> p1 < p2
    False
    >>> p1 == p2
    True
    """

    def __repr__(self) -> str:
        """String representation of the proposition.
        """
        return "{:s}({:s})".format(self.__class__.__name__, self.to_string())

    def __str__(self) -> str:
        """String representation of the proposition.
        """
        return self.to_string()

    def __call__(self, *args: Any) -> Any:
        """Called when the proposition is 'called' as a function.
        """
        return self.extension(*args)

    def __len__(self) -> int:
        """Returns the length of the proposition.
        """
        return len(str(self))

    def __eq__(self, other: "Proposition") -> bool:
        """Comparison method for the equality operator.
        """
        return str(self) == str(other)

    def __lt__(self, other: "Proposition") -> bool:
        """Comparison method for the less-than operator.
        """
        return str(self) < str(other)

    def extension(self, data: Any, return_indices: Optional[bool] = False) -> Any:
        """Query proposition.
        """
        mask = np.ones(len(data), dtype=np.bool_)
        if return_indices:
            mask = np.nonzero(mask)[0]
        return mask

    def to_string(self) -> str:
        """String representation.
        """
        return ""


class KeyProposition(Proposition):
    """Callable proposition for a key in a dict-like object such as Pandas
    row series.

    Usage:
    >>> prop1 = KeyProposition('x')
    >>> prop1
    KeyProposition(x)
    >>> prop1.to_string()
    'x'
    >>> str(prop1)
    'x'
    >>> prop2 = KeyProposition('y')
    >>> prop1 < prop2
    True
    >>> prop3 = KeyProposition('x')
    >>> prop1 == prop3
    True

    Example:
    >>> male = KeyProposition('male')
    >>> female = KeyProposition('female')
    >>> male <= female
    False
    """

    def __init__(self, key: str):
        """Constructor.
        """
        Proposition.__init__(self)
        self.key = key

    def extension(self, data: Any, return_indices: Optional[bool] = False) -> Any:
        """Query proposition.
        """
        value = (self.key in data)
        if value and isinstance(data[self.key], (np.ndarray, pd.Series)):
            size = len(data[self.key])
            value = np.ones(size, dtype=np.bool_)

            if return_indices:
                value = np.nonzero(mask)[0]
        return value

    def to_string(self) -> str:
        """String representation.
        """
        return self.key


class KeyValueProposition(Proposition):
    """Callable proposition that represents a constraint on value for some
    fixed key in a dict-like object such as Pandas row series.

    Usage:
    >>> prop1 = KeyValueProposition('x', Constraint.less_equals(5))
    >>> prop1
    KeyValueProposition(x<=5)
    >>> str(prop1)
    'x<=5'
    >>> prop1.extension({'x': 4})
    True
    >>> prop1.extension({'x': np.array([4, 5, 6])})
    array([ True,  True, False])
    >>> prop2 = KeyValueProposition('x', Constraint.equals(5))
    >>> prop3 = KeyValueProposition('x', Constraint.greater_equals(5))
    >>> prop1 < prop2 < prop3
    True

    Example:
    >>> male = KeyValueProposition('Sex', Constraint.equals('male'))
    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> infant = KeyValueProposition('Age', Constraint.less_equals(4))
    >>> male == male2, male == infant
    (True, False)
    >>> male <= female, male >= female, infant <= female
    (False, True, True)
    """

    def __init__(self, key: str, constraint):
        """Constructor.
        """
        Proposition.__init__(self)

        self.key = key
        self.constraint = constraint

    def extension(self, data: Any, return_indices: Optional[bool] = False) -> Any:
        """Query proposition.
        """
        mask = self.constraint(data[self.key])
        if return_indices:
            mask = np.nonzero(mask)[0]
        return mask

    def to_string(self) -> str:
        """String representation.
        """
        return format(self.constraint, self.key)


class TabulatedProposition(Proposition):
    """Callable proposition for tabulated data.

    Usage:
    >>> table = [[0, 1, 0, 1], [1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]]
    >>> p = TabulatedProposition(1)
    >>> p
    TabulatedProposition(c1)
    >>> p.extension(0)
    1
    >>> p.extension(np.array([0, 1, 2]))
    array([1, 1, 0])

    Example:
    >>> p1 = TabulatedProposition(1)
    >>> p2 = TabulatedProposition(2)
    >>> p1
    TabulatedProposition(c1)
    >>> p2
    TabulatedProposition(c2)
    >>> p1 < p2
    """

    def __init__(self, index: int):
        """Constructor.
        """
        Proposition.__init__(self)
        self.key = index

    def extension(self, data: Any, return_indices: Optional[bool] = False) -> Any:
        """Query proposition.
        """
        mask = np.array([column[self.key] for column in data],
                        dtype=np.bool_)
        if return_indices:
            mask = np.nonzero(mask)[0]
        return mask

    def to_string(self) -> str:
        """String representation.
        """
        return "c{:d}".format(self.key)


class Conjunction:
    """Conjunctive aggregation of propositions.

    Usage:
    >>> old = KeyValueProposition('age', Constraint.greater_equals(60))
    >>> male = KeyValueProposition('sex', Constraint.equals('male'))
    >>> high_risk = Conjunction([male, old])
    >>> high_risk
    Conjunction(age>=60 & sex==male)
    >>> stephanie = {'age': 30, 'sex': 'female'}
    >>> erika = {'age': 72, 'sex': 'female'}
    >>> ron = {'age': 67, 'sex': 'male'}
    >>> high_risk(stephanie), high_risk(erika), high_risk(ron)
    (False, False, True)

    Elements can be accessed via index and are sorted lexicographically.
    >>> str(high_risk)
    'age>=60 & sex==male'
    >>> len(high_risk)
    2
    >>> [p for p in high_risk]
    [KeyValueProposition(age>=60), KeyValueProposition(sex==male)]
    >>> high_risk[0]
    KeyValueProposition(age>=60)
    >>> high_risk[1]
    KeyValueProposition(sex==male)
    >>> female = KeyValueProposition('sex', Constraint.equals('female'))
    >>> (male in high_risk, female in high_risk)
    (True, False)
    """

    def __init__(self, props: Optional[Iterable[Proposition]] = []):
        """Constructor.
        """
        self.propositions = sorted(props, key=str)

    def __repr__(self) -> str:
        """String representation of the conjunction.
        """
        return "{:s}({:s})".format(self.__class__.__name__, self.to_string())

    def __str__(self) -> str:
        """String representation of the conjunction.
        """
        return self.to_string()

    def __call__(self, *args: Any) -> Any:
        """Called when the proposition is 'called' as a function.
        """
        return self.extension(*args)

    def __len__(self) -> int:
        """Returns the length of the proposition.
        """
        return len(self.propositions)

    def __getitem__(self, item) -> Any:
        """Called to implement evaluation of self[key].
        """
        return self.get(item)

    def __contains__(self, item) -> bool:
        """Called to implement membership test operators.
        """
        return self.has(item)

    def get(self, index: int) -> Proposition:
        """Get proposition via index.
        """
        return self.propositions[index]

    def has(self, prop: Proposition) -> bool:
        """Check if a proposition exists in conjunction.
        """
        return prop in self.propositions

    def extension(self, data: Any, return_indices: Optional[bool] = False) -> Any:
        """Query proposition.
        """
        mask = np.logical_and.reduce([p(data) for p in self.propositions])
        if not self.propositions:
            mask = np.full(len(data), mask, dtype=np.bool)
        if return_indices:
            mask = np.nonzero(mask)[0]
        return mask

    def to_string(self) -> str:
        """String representation.
        """
        return str.join(" & ", map(str, self.propositions))
