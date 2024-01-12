import pandas as pd
import numpy as np
import sortednp as snp

from sortedcontainers import SortedSet
from numpy import array
from bitarray import bitarray
from bitarray.util import subset

from realkd.logic import IndexValueProposition, TabulatedProposition

# Imported for doctests
from realkd.datasets import titanic_column_trans, titanic_data  # noqa: F401


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
        """
        Generates formal context from an array by applying inter-ordinal scaling to numerical data columns.

        For inter-ordinal scaling a maximum number of attributes per column can be specified. If required, threshold
        values are then selected by the provided discretization function (per default quantile-based).

        >>> titanic = titanic_data()
        >>> X = titanic_column_trans.fit_transform(titanic)
        >>> y = np.array(titanic["Survived"])
        >>> titanic_ctx = SearchContext.from_array(X, max_col_attr=6, sort_attributes=False)
        >>> titanic_ctx.m
        891
        >>> titanic_ctx.attributes # doctest: +NORMALIZE_WHITESPACE
        [x0==1, x1==1, x2==1, x3==1, x4==1, x5==1,
        x6<=8.6625, x6>=8.6625, x6<=26.0, x6>=26.0, x6<=512.3292, x6>=512.3292,
        x7<=8.0, x7>=8.0, x8<=6.0, x8>=6.0, x9<=23.0, x9>=23.0, x9<=34.0, x9>=34.0, x9<=80.0, x9>=80.0,
        x10<=1.0, x10<=2.0, x10>=2.0, x10>=3.0]
        >>> titanic_ctx.n
        26
        >>> titanic_ctx.extension([25, 1, 19]) # doctest: +NORMALIZE_WHITESPACE
        array([  4,  13,  94, 104, 108, 116, 129, 152, 153, 160, 179, 188, 189,
            197, 202, 203, 222, 280, 326, 338, 349, 360, 363, 400, 406, 414,
            461, 465, 471, 482, 525, 528, 561, 590, 592, 595, 597, 603, 605,
            614, 616, 631, 661, 663, 668, 696, 699, 758, 761, 771, 811, 818,
            843, 845, 847, 851, 860, 873])


        :param array: array to be converted to formal context
        :param int max_col_attr: maximum number of attributes generated per column;
                             or None if an arbitrary number of attributes is permitted;
                             or dict (usually defaultdict) with keys being columns ids of df and values
                             being the maximum number of attributes for the corresponding column (again using
                             None if no bound for a specific column);
                             Note: use defaultdict(lambda: None) instead of defaultdict(None) to specify no maximum
                             per default
        :param callable discretization: the discretization function to be used when number of thresholds has to be reduced to
                               a specificed maximum (function has to have identical signature to pandas.qcut, which
                               is the default)
        :param Iterable[str] without: columns to ommit
        :return: :class:`Context` representing dataframe
        """
        without = without or []

        attributes = []
        for col_index, column in enumerate(data.T):
            # if c in without:
            #     continue

            # column = data[:, col_index]

            # TODO: Handle nulls
            # No matter what, if there's a null value, add it as an option
            if np.any(pd.isnull(column)) and col_index not in [9]:
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
        self.extents = [attributes[j](objects).nonzero()[0] for j in range(self.n)]
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
            for L in available:
                covering[L] -= covering[j]

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
