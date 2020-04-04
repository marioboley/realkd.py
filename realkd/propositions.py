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

class Constraint:
    """
    Boolean condition on a single value with string representation. For example:
    >>> t = 21
    >>> c = Constraint.less_equals(21)
    >>> c
    Constraint(x<=21)
    >>> format(c, 'age')
    'age<=21'
    >>> c(18)
    True
    >>> c(63)
    False
    """

    def __init__(self, cond, str_repr=None):
        self.cond = cond
        self.str_repr = str_repr or (lambda vn: str(cond)+'('+vn+')')

    def __call__(self, value):
        return self.cond(value)

    def __format__(self, varname):
        return self.str_repr(varname)

    def __repr__(self):
        return 'Constraint('+format(self, 'x')+')'

    @staticmethod
    def less_equals(value):
        return Constraint(lambda v: v <= value, lambda n: str(n)+'<='+str(value))

    @staticmethod
    def greater_equals(value):
        return Constraint(lambda v: v >= value, lambda n: str(n)+'>='+str(value))

    @staticmethod
    def equals(value):
        return Constraint(lambda v: v == value, lambda n: str(n)+'=='+str(value))


class KeyValueProposition:
    """
    Callable proposition that represents constraint on value for some fixed key in a dict-like object
    such as Pandas row series.

    For example:
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> male = KeyValueProposition('Sex', Constraint.equals('male'))
    >>> male
    Sex==male

    ---> WARNING: string values need probably be quoted in representation to work as pandas query as intended

    >>> titanic.iloc[10]
    Survived         1
    Pclass           3
    Sex         female
    Age              4
    SibSp            1
    Parch            1
    Fare          16.7
    Embarked         S
    Name: 10, dtype: object
    >>> male(titanic.iloc[10])
    False

    >>> male2 = KeyValueProposition('Sex', Constraint.equals('male'))
    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> infant = KeyValueProposition('Age', Constraint.less_equals(4))
    >>> male == male2, male == infant
    (True, False)
    >>> male <= female, male >= female, age <= female
    (False, True, True)
    """

    def __init__(self, key, constraint):
        self.key = key
        self.constraint = constraint
        self.repr = format(constraint, key)

    def __call__(self, row):
        return self.constraint(row[self.key])

    def __repr__(self):
        return self.repr

    def __eq__(self, other):
        return str(self) == str(other)


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
    """
    Conjunctive aggregation of propositions.

    For example:
    >>> old = KeyValueProposition('age', Constraint.greater_equals(60))
    >>> male = KeyValueProposition('sex', Constraint.equals('male'))
    >>> high_risk = Conjunction([male, old])
    >>> stephanie = {'age': 30, 'sex': 'female'}
    >>> erika = {'age': 72, 'sex': 'female'}
    >>> ron = {'age': 67, 'sex': 'male'}
    >>> high_risk(stephanie), high_risk(erika), high_risk(ron)
    (False, False, True)

    Elements can be accessed via index and are sorted lexicographically.
    >>> high_risk
    age>=60 & sex==male
    >>> high_risk[0]
    age>=60
    >>> len(high_risk)
    2
    """

    def __init__(self, props):
        self.props = sorted(props, key=str)
        self.repr = str.join(" & ", map(str, self.props))

    def __call__(self, x):
        return all(map(lambda p: p(x), self.props))

    def __repr__(self):
        return self.repr

    def __getitem__(self, item):
        return self.props[item]

    def __len__(self):
        return len(self.props)
