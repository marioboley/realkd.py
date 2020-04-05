# -*- coding: utf-8 -*-
"""
@package    realkd.propositions

@copyright  Copyright (c) 2020+ RealKD-Team,
            Mario Boley <mario.boley@monash.edu>
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the MIT License (the "License").
You may not use this file except in compliance with the License.
"""

import functools

import numpy as np
import pandas as pd

from typing import Union, Sequence, Dict, Iterable, Callable, Any, Optional


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

    def __init__(self, cond: Callable[[Any], bool], str_repr: Optional[Callable[[str], str]] = None,):
        """Constructor.
        """
        self.cond = cond
        self.str_repr = str_repr or (lambda vn: str(cond) + "(" + vn + ")")

    def __call__(self, value: Any) -> bool:
        """Called when the constraint is 'called' as a function.
        """
        return self.cond(value)

    def __format__(self, varname: str) -> str:
        """Format constraint when used in new-style string formatting.
        """
        return self.str_repr(varname)

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
        return cls(lambda v: v <= value, lambda n: "{}<={}".format(n, value))

    @classmethod
    def greater_equals(cls, value: Any) -> 'Constraint':
        """Constraint with a greater-than operator.
        """
        return cls(lambda v: v >= value, lambda n: "{}>={}".format(n, value))

    @classmethod
    def equals(cls, value: Any) -> 'Constraint':
        """Constraint with an equality operator.
        """
        return cls(lambda v: v == value, lambda n: "{}=={}".format(n, value))


@functools.total_ordering
class Proposition:
    """Callable proposition base class.

    Usage:
    >>> p1 = Proposition()
    >>> p1
    Proposition()
    >>> p1.to_string()
    ''
    >>> p1.query(False)
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
        return self.query(*args)

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

    def query(self, data: Any) -> Any:
        """Query proposition.
        """
        return data

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

    def query(self, data: Any) -> Any:
        """Query proposition.
        """
        value = (self.key in data)
        if value and isinstance(data[self.key], (np.ndarray, pd.Series)):
            size = len(data[self.key])
            value = np.ones(size, dtype=np.bool_)
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
    >>> prop1.query({'x': 4})
    True
    >>> prop1.query({'x': np.array([4, 5, 6])})
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

    def query(self, data: Any) -> Any:
        """Query proposition.
        """
        return self.constraint(data[self.key])

    def to_string(self) -> str:
        """String representation.
        """
        return format(self.constraint, self.key)


class TabulatedProposition(Proposition):
    """Callable proposition for tabulated data.

    Usage:
    >>> table = [[0, 1, 0, 1], [1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]]
    >>> prop1 = TabulatedProposition(table, 1)
    >>> prop1
    TabulatedProposition(1: [1, 1, 0, 1])
    >>> prop1.query(0)
    1
    >>> prop1.query(np.array([0, 1, 2]))
    array([1, 1, 0])

    Example:
    >>> prop1 = TabulatedProposition(table, 1)
    >>> prop2 = TabulatedProposition(table, 2)
    >>> prop1
    TabulatedProposition(1: [1, 1, 0, 1])
    >>> prop2
    TabulatedProposition(2: [0, 1, 1, 0])
    >>> prop1 < prop2
    """

    def __init__(self, table: Union[Sequence[Sequence], np.ndarray], index: int):
        """Constructor.
        """
        Proposition.__init__(self)

        self.table = np.atleast_2d(table).T
        self.index = index

    def query(self, data: Union[int, np.ndarray]) -> Any:
        """Query proposition.
        """
        return self.table[self.index, data]

    def to_string(self) -> str:
        """String representation.
        """
        return "{:d}: {!r}".format(self.index, self.table[self.index].tolist())


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

    def __init__(self, props: Iterable[Proposition]):
        """Constructor.
        """
        self.props = sorted(props, key=str)

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
        return self.query(*args)

    def __len__(self) -> int:
        """Returns the length of the proposition.
        """
        return len(self.props)

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
        return self.props[index]

    def has(self, prop: Proposition) -> bool:
        """Check if a proposition exists in conjunction.
        """
        return prop in self.props

    def query(self, data: Any) -> Any:
        """Query conjunction.
        """
        return all(p(data) for p in self.props)

    def to_string(self) -> str:
        """String representation.
        """
        return str.join(" & ", map(str, self.props))
