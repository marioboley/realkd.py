# -*- coding: utf-8 -*-
"""
@package    realkd.base

@copyright  Copyright (c) 2020+ RealKD-Team,
            Mario Boley <mario.boley@monash.edu>
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the MIT License (the "License").
You may not use this file except in compliance with the License.
"""

import importlib
import numpy as np
import pandas as pd

from .propositions import Constraint, Proposition, KeyProposition, \
                          KeyValueProposition, TabulatedProposition, Conjunction

from typing import Optional, Union, Sequence, Iterable, List, Tuple, Dict, Any
        

class Context:
    """A formal search context of the data set.
    
    A formal context, i.e., a binary relation between the data set and a set
    of attributes, provides a search context (search space) over conjunctions
    that can be formed from the individual attributes.
    """

    # Data type constants
    ORDINAL = 'ordinal'
    NOMINAL = 'nominal'
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'
    
    def __init__(self, data: Union[dict, pd.DataFrame], attributes: Sequence[Proposition],
                 sort_attributes: bool = True):
        """Constructor.
        """
        self.attributes = attributes
        self.data = pd.DataFrame(data)
        
        self.n = len(self.data)
        self.m = len(attributes)
        
        # Rerpresent binary relations as numpy masks
        self.extents = [attribute(data) for attribute in attributes]

        # Sort attribute in ascending order of extent size
        if sort_attributes:
            attribute_order = sorted(range(self.m), key=lambda i: np.sum(self.extents[i]))
            self.attributes = [self.attributes[i] for i in attribute_order]
            self.extents = [self.extents[i] for i in attribute_order]

    def __len__(self):
        return self.data.shape[-1]

    def keys(self) -> Iterable:
        """Return the keys of the data set.
        """
        return self.data.keys()

    def values(self) -> Iterable:
        """Return the values of the data set.
        """
        return (self.data[k].values for k in self.keys())

    def items(self) -> Iterable:
        """Iterates over the data set, returning a tuple with the data name and the values.
        """
        return self.data.iteritems()

    def dtypes(self, raw: bool = False) -> Iterable[str]:
        """Return the data types of the data set.
        """
        return (self.dtype(k, raw=raw) for k in self.keys())

    def dtype(self, column: str, raw: bool = False) -> str:
        """Get data type of column.
        """
        values = self.data[column]
        return values.dtype.kind if raw else self._guess_datatype(values)

    def query(self, expression: str, **kwargs) -> pd.DataFrame:
        """Query context with a boolean expression.
        """
        return self.data.query(expression, inplace=False, **kwargs)

    def extension(self, intent: Optional[Union[Iterable[int], Conjunction]] = None,
                  return_indices: Optional[bool] = False) -> np.ndarray:
        """Query context.
        """
        # Determine data mask
        if isinstance(intent, Conjunction):
            mask = intent.extension(self.data)
        elif intent is not None:
            mask = np.logical_and.reduce([self.extents[i] for i in intent])
            if not len(intent):
                mask = np.full(self.n, mask, dtype=np.bool)
        else:
            mask = np.ones(self.n, dtype=np.bool)

        if return_indices:
            mask = np.nonzero(mask)[0]
        return mask

    @staticmethod
    def from_table(table, sort_attributes=False):
        """Converts an input table where each row represents an object into
        a formal context (which uses column-based representation).

        Uses Boolean interpretation of table values to determine attribute
        presence for an object.

        For instance:

        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_table(table)
        >>> ctx.extension([0, 2], return_indices=True)
        array([1, 2])

        :param table: table that explicitly encodes object/attribute relation
        :return: context over table row indices as objects and one tabulated feature per column index
        """
        table = np.atleast_2d(table)
        m, n = table.shape
        
        attributes = [TabulatedProposition(j) for j in range(n)]
        return Context(table, attributes, sort_attributes)

    @staticmethod
    def from_data(data: Union[dict, pd.DataFrame], nbins=None, attributes_only=False, ignore=None,
                  precision=2, sort_attributes=False):
        """
        Generates a formal context from the data by applying inter-ordinal scaling to numerical data columns
        and for object columns creating one attribute per value.
        """
        
        ignore = ignore or []
        attributes = []
        
        for key in data:
            # Exclude keys
            if key in ignore:
                continue

            # Create single propositions
            if attributes_only:
                attributes.append(KeyProposition(key))
                continue                
                
            values = pd.unique(data[key])
            values = np.compress(values == values, values)
            values.sort(kind='mergesort')
            
            dtype = values.dtype.kind
            size = len(values)

            # Categorical and nominal propositions
            if dtype in 'bOSU' or (dtype in 'iu' and size == 2) or \
                   pd.api.types.is_categorical(data[key]):
                attributes.extend(KeyValueProposition(key, Constraint.equals(v))
                                  for v in values)

            # Numeric propositions
            elif dtype in 'iuf':                    
                # Reduce number of distinct values per column
                discretized = False
                if nbins and size > 2 * nbins:
                    _, values = pd.qcut(data[key], q=nbins - 1, retbins=True, duplicates='drop')

                    # Round quantiles to nearest float value with precision
                    prec = (0 if dtype in 'iu' else precision)
                    values = np.hstack((np.round(values[0] - 0.5 * 10**(-prec), prec),
                                        np.around(values[1:-1], decimals=prec),
                                        np.round(values[-1] + 0.5 * 10**(-prec), prec)))

                    # Format negative zero always as positive zero
                    values = values.astype(data[key].dtype) + 0
                    discretized = True
                    size = len(values)
                        
                # Generate attributes
                for i, v in enumerate(values, 1):
                    propositions = [
                        KeyValueProposition(key, Constraint.less_equals(v)),
                        KeyValueProposition(key, Constraint.greater_equals(v))
                    ]

                    # Add equality attribute
                    if not discretized and dtype in 'iu':
                        attributes.append(KeyValueProposition(key, Constraint.equals(v)))
                        if i == 1 or i == size:
                            continue

                    # Reduce number of attributes at value edges
                    if i == 1:
                        propositions.pop()
                    elif i == size:
                        propositions.pop(0)
                        
                    attributes.extend(propositions)
                    
            # Anything else
            else:
                TypeError("Data type {!r} not understood.".format(values.dtype))

        return Context(data, attributes, sort_attributes)

    @staticmethod
    def from_dict(data: dict, nbins=None, attributes_only=False, ignore=None,
                  precision=2, sort_attributes=False):
        """
        Generates a formal context from a dictionary by applying inter-ordinal scaling to numerical data columns
        and for object columns creating one attribute per value.
        """
        return Context.from_data(data, nbins, attributes_only, ignore,
                                 precision, sort_attributes)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, nbins=None, attributes_only=False,
                       ignore=None, precision=2, sort_attributes=False):
        """
        Generates formal context from pandas dataframe by applying inter-ordinal scaling to numerical data columns
        and for object columns creating one attribute per value.

        For inter-ordinal scaling a maximum number of attributes per column can be specified. If required, threshold
        values are then selected quantile-based.

        The restriction should also be implemented for object columns in the future (by merging small categories
        into disjunctive propositions).

        The generated attributes correspond to pandas-compatible query strings. For example:

        >>> titanic_df = pd.read_csv("../datasets/titanic/train.csv")
        >>> titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        >>> titanic_ctx = Context.from_df(titanic_df, max_col_attr=6, sort_attributes=False)
        >>> titanic_ctx.m
        891
        >>> titanic_ctx.attributes
        [Survived<=0, Survived>=1, Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==male, Sex==female,
        Age<=23.0, Age>=23.0, Age<=34.0, Age>=34.0, Age<=80.0, Age>=80.0, SibSp<=8.0, SibSp>=8.0,
        Parch<=6.0, Parch>=6.0, Fare<=8.6625, Fare>=8.6625, Fare<=26.0, Fare>=26.0, Fare<=512.3292,
        Fare>=512.3292, Embarked==S, Embarked==C, Embarked==Q, Embarked==nan]
        >>> titanic_ctx.n
        28
        >>> titanic_df.query('Survived==1 & Pclass>=3 & Sex=="male" & Age>=34')
             Survived  Pclass   Sex   Age  SibSp  Parch   Fare Embarked
        338         1       3  male  45.0      0      0  8.050        S
        400         1       3  male  39.0      0      0  7.925        S
        414         1       3  male  44.0      0      0  7.925        S
        >>> titanic_ctx.extension([1, 5, 6, 11])
        SortedSet([338, 400, 414])

        :param df: pandas dataframe to be converted to formal context
        :param max_col_attr: maximum number of attributes generated per column
        :param without: columns to ommit
        :return: context representing dataframe [38, 45, 49, 29]
        """
        return Context.from_data(data, nbins, attributes_only, ignore,
                                 precision, sort_attributes)

    def _guess_datatype(self, data: Union[np.ndarray, pd.Series]) -> str:
        """Guess data types from inut data.
        """
        # Normalize data
        values = pd.unique(data)
        values = np.compress(values == values, values)

        size = len(values)    
        dtype = values.dtype.kind

        # Nominal (categorical) propositions
        if dtype in 'OSU':
            datatype = self.NOMINAL

        # Ordinal (categorical) and discrete (numerical) propositions
        if dtype in 'biu':
            if size == 2 or pd.api.types.is_categorical(data):
                datatype = self.ORDINAL
            else:
                datatype = self.DISCRETE

        # Continuous (numerical) propositions
        elif dtype in 'f':
            datatype = self.CONTINUOUS

        # Anything else
        else:
            TypeError('Data type {!r} not understood.'.format(values.dtype))

        return datatype
