import pandas as pd
import numpy as np
import sortednp as snp

from collections import defaultdict
from sortedcontainers import SortedSet
from numpy import array
from bitarray import bitarray
from bitarray.util import subset

from realkd.datasets import titanic_data, titanic_column_trans
from realkd.logic import IndexValueProposition, TabulatedProposition


class SearchContext:
    """
    Formal context, i.e., a binary relation between a set of objects and a set of attributes,
    i.e., Boolean functions that are defined on the objects.

    A formal context provides a search context (search space) over conjunctions that can be
    formed from the individual attributes.
    """

    @staticmethod
    def from_tab(table, sort_attributes=False):
        """
        Converts an input table where each row represents an object into
        a formal context (which uses column-based representation).

        Uses Boolean interpretation of table values to determine attribute
        presence for an object.

        For instance:

        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = SearchContext.from_tab(table)
        >>> list(ctx.extension([0, 2]))
        [1, 2]

        :param table: table that explicitly encodes object/attribute relation
        :return: context over table row indices as objects and one tabulated feature per column index
        """

        m = len(table)
        n = len(table[0])
        attributes = [TabulatedProposition(table, j) for j in range(n)]
        return SearchContext(attributes, list(range(m)), sort_attributes)

    @staticmethod
    def from_array(
        data,
        without=None,
        max_col_attr=10,
        sort_attributes=True,
        discretization=pd.qcut,
        **kwargs,
    ):
        without = without or []

        attributes = []
        for col_index, column in enumerate(data.T):
            # if c in without:
            #     continue

            # column = data[:, col_index]

            # TODO: Handle nulls
            # No matter what, if there's a null value, add it as an option
            if np.any(pd.isnull(column)):
                attributes.append(IndexValueProposition("==", col_index, np.nan))

            # If the column is already binary, we don't have to do anything fancy
            if ((column == 0) | (column == 1)).all():
                attributes.append(IndexValueProposition("==", col_index, 1))
            else:
                vals = sorted(np.unique(column[~pd.isnull(column)]))
                if max_col_attr and len(vals) * 2 > max_col_attr:
                    _, vals = discretization(
                        np.asfarray(column),
                        max_col_attr // 2,
                        retbins=True,
                        duplicates="drop",
                    )
                    # Drop the first bin because it's redundant information
                    vals = vals[1:]
                    for v in vals:
                        attributes.append(IndexValueProposition("<=", col_index, v))
                        attributes.append(IndexValueProposition(">=", col_index, v))
                else:
                    for i, v in enumerate(vals):
                        if i < len(vals) - 1:
                            attributes.append(IndexValueProposition("<=", col_index, v))
                        if i > 0:
                            attributes.append(IndexValueProposition(">=", col_index, v))
        print(attributes)
        return SearchContext(attributes, data, sort_attributes)

    @staticmethod
    def get_bit_array_from_indexes(indexes, length):
        result = bitarray(length)
        result.setall(0)
        for index in indexes:
            result[index] = 1
        return result

    def __init__(self, attributes, objects, sort_attributes=True):
        self.attributes = attributes
        self.objects = objects
        self.n = len(attributes)
        self.m = len(objects)
        # for now we materialise the whole binary relation; in the future can be on demand
        # self.extents = [SortedSet([i for i in range(self.m) if attributes[j](objects[i])]) for j in range(self.n)]
        self.extents = [
            array(
                [i for i in range(self.m) if attributes[j](objects[i])], dtype="int64"
            )
            for j in range(self.n)
        ]
        self.bit_extents = [
            SearchContext.get_bit_array_from_indexes(self.extents[j], self.m)
            for j in range(self.n)
        ]

        # sort attribute in ascending order of extent size
        if sort_attributes:
            attribute_order = list(
                sorted(range(self.n), key=lambda i: len(self.extents[i]))
            )
            self.attributes = [self.attributes[i] for i in attribute_order]
            self.extents = [self.extents[i] for i in attribute_order]
            self.bit_extents = [self.bit_extents[i] for i in attribute_order]

        # switches
        self.crit_propagation = True

        # stats
        self.popped = 0
        self.created = 0
        self.avg_created_length = 0
        self.rec_crit_hits = 0
        self.crit_hits = 0
        self.del_bnd_hits = 0
        self.clo_hits = 0
        self.non_lexmin_hits = 0
        self.bnd_post_children_hits = 0
        self.bnd_immediate_hits = 0

    def greedy_simplification(self, intent, extent):
        to_cover = SortedSet(range(self.m)).difference(SortedSet(extent))
        available = list(range(len(intent)))
        covering = [
            SortedSet(range(self.m)).difference(SortedSet(self.extents[j]))
            for j in intent
        ]
        result = []
        while to_cover:
            j = max(available, key=lambda i: len(covering[i]))
            result += [intent[j]]
            available.remove(j)
            to_cover -= covering[j]
            for l in available:
                covering[l] -= covering[j]

        return result

    def extension(self, intent):
        """
        :param intent: attributes describing a set of objects
        :return: indices of objects that have all attributes in intent in common
        """
        if not intent:
            return array(range(len(self.objects)))

        result = self.extents[intent[0]]
        for i in range(1, len(intent)):
            result = snp.intersect(result, self.extents[intent[i]])

        return result

    def find_small_crit_index(self, gen_idx, bit_extension, part_closure):
        """
        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = SearchContext.from_tab(table)
        >>> root = Node([], bitarray('0000'), array([0,1,2,3]), bitarray('1111'), -1, -4, 1, inf)
        >>> ctx.find_small_crit_index(0, bitarray('0110'), bitarray('1000'))
        4
        >>> ctx.find_small_crit_index(1, bitarray('1101'), bitarray('0100'))
        4
        >>> ctx.find_small_crit_index(2, bitarray('0110'), bitarray('0010'))
        0
        >>> ctx.find_small_crit_index(3, bitarray('1001'), bitarray('0001'))
        1
        >>> ctx.find_small_crit_index(3, bitarray('1001'), bitarray('0101'))
        4
        """
        for j in range(0, gen_idx):
            # and len(extension) <= len(self.extents[j])
            if not part_closure[j] and subset(bit_extension, self.bit_extents[j]):
                return j
        return len(part_closure)

    def complete_closure(self, gen_idx, bit_extension, part_closure):
        """
        :param gen_idx:
        :param bit_extension:
        :param closure_prefix:
        :return:

        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = SearchContext.from_tab(table)
        >>> clo = bitarray('1000')
        >>> ctx.complete_closure(0, bitarray('0110'), clo)
        2
        >>> clo
        bitarray('1010')
        >>> clo = bitarray('1110')
        >>> ctx.complete_closure(1, bitarray('0100'), clo)
        4
        >>> clo
        bitarray('1110')
        """
        n = len(part_closure)
        crit_idx = n
        for j in range(gen_idx + 1, n):
            # TODO: for the moment put guard out because it seems faster, but this could change with
            #       numba and/or be different for different datasets
            # and len(extension) <= len(self.extents[j])
            if not part_closure[j] and subset(bit_extension, self.bit_extents[j]):
                crit_idx = min(crit_idx, j)
                part_closure[j] = True

        return crit_idx


if __name__ == "__main__":
    import doctest

    doctest.testmod()
