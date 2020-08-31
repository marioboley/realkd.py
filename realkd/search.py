import pandas as pd

from pandas import Index
from collections import deque
from sortedcontainers import SortedSet
from math import inf
from heapq import heappop, heappush

from realkd.logic import Conjunction, Constraint, KeyValueProposition, TabulatedProposition


class Node:
    """
    Represents a potential node (and incoming edge) for searches in the concept graph
    with edges representing the direct prefix-preserving successor relation (dpps).
    """

    def __init__(self, gen, clo, ext, idx, crit_idx, val, bnd):
        self.generator = gen
        self.closure = clo
        self.extension = ext
        self.gen_index = idx
        self.crit_idx = crit_idx
        self.val = val
        self.val_bound = bnd
        self.valid = self.crit_idx > self.gen_index

    def __repr__(self):
        return f'N({list(self.generator)}, {list(self.closure)}, {self.val:.5g}, {self.val_bound:.5g}, {list(self.extension)})'

    def value(self):
        return self.val

    def __le__(self, other):
        return self.val <= other.val

    def __eq__(self, other):
        return self.val == other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __lt__(self, other):
        return self.val < other.val

    def __gt__(self, other):
        return self.val > other.val


class BreadthFirstBoundary:

    def __init__(self):
        self.deq = deque()

    def __bool__(self):
        return bool(self.deq)

    def push(self, augmented_node):
        self.deq.append(augmented_node)

    def pop(self):
        return self.deq.popleft()


class DepthFirstBoundary:

    def __init__(self):
        self.stack = []

    def __bool__(self):
        return bool(self.stack)

    def push(self, augmented_node):
        self.stack.append(augmented_node)

    def pop(self):
        return self.stack.pop()


class BestBoundFirstBoundary:

    def __init__(self):
        self.heap = []

    def __bool__(self):
        return bool(self.heap)

    def push(self, augmented_node):
        _, node = augmented_node
        heappush(self.heap, (-node.val_bound, -node.val, augmented_node))

    def pop(self):
        _, _, augmented_node = heappop(self.heap)
        return augmented_node


class BestValueFirstBoundary:

    def __init__(self):
        self.heap = []

    def __bool__(self):
        return bool(self.heap)

    def push(self, augmented_node):
        _, node = augmented_node
        heappush(self.heap, (-node.val, -node.val_bound, augmented_node))

    def pop(self):
        _, _, augmented_node = heappop(self.heap)
        return augmented_node


class Context:
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
        >>> ctx = Context.from_tab(table)
        >>> list(ctx.extension([0, 2]))
        [1, 2]

        :param table: table that explicitly encodes object/attribute relation
        :return: context over table row indices as objects and one tabulated feature per column index
        """

        m = len(table)
        n = len(table[0])
        attributes = [TabulatedProposition(table, j) for j in range(n)]
        return Context(attributes, list(range(m)), sort_attributes)

    @staticmethod
    def from_df(df, without=None, max_col_attr=None, sort_attributes=True, discretization=pd.qcut):
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
        [Survived<=0, Survived>=1, Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==male, Sex==female, Age<=23.0, Age>=23.0, Age<=34.0, Age>=34.0, Age<=80.0, Age>=80.0, SibSp<=8.0, SibSp>=8.0, Parch<=6.0, Parch>=6.0, Fare<=8.6625, Fare>=8.6625, Fare<=26.0, Fare>=26.0, Fare<=512.3292, Fare>=512.3292, Embarked==S, Embarked==C, Embarked==Q, Embarked==nan]
        >>> titanic_ctx.n
        28
        >>> titanic_df.query('Survived>=1 & Pclass>=3 & Sex=="male" & Age>=34')
             Survived  Pclass   Sex   Age  SibSp  Parch   Fare Embarked
        338         1       3  male  45.0      0      0  8.050        S
        400         1       3  male  39.0      0      0  7.925        S
        414         1       3  male  44.0      0      0  7.925        S
        >>> titanic_ctx.extension([1, 5, 6, 11])
        Int64Index([338, 400, 414], dtype='int64')

        (prev was SortedSet([338, 400, 414]))

        :param df: pandas dataframe to be converted to formal context
        :param max_col_attr: maximum number of attributes generated per column
        :param without: columns to ommit
        :return: context representing dataframe
        """

        without = without or []
        attributes = []
        for c in df:
            if c in without:
                continue
            if df[c].dtype.kind in 'uif':
                vals = df[c].unique()
                reduced = False
                if max_col_attr and len(vals)*2 > max_col_attr:
                    _, vals = discretization(df[c], max_col_attr // 2, retbins=True, duplicates='drop')
                    vals = vals[1:]
                    reduced = True
                vals = sorted(vals)
                for i, v in enumerate(vals):
                    if reduced or i < len(vals) - 1:
                        attributes += [KeyValueProposition(c, Constraint.less_equals(v))]
                    if reduced or i > 0:
                        attributes += [KeyValueProposition(c, Constraint.greater_equals(v))]
            if df[c].dtype.kind in 'O':
                attributes += [KeyValueProposition(c, Constraint.equals(v)) for v in df[c].unique()]

        return Context(attributes, [df.iloc[i] for i in range(len(df.axes[0]))], sort_attributes)

    def __init__(self, attributes, objects, sort_attributes=True):
        self.attributes = attributes
        self.objects = objects
        self.n = len(attributes)
        self.m = len(objects)
        # for now we materialise the whole binary relation; in the future can be on demand
        # self.extents = [SortedSet([i for i in range(self.m) if attributes[j](objects[i])]) for j in range(self.n)]
        self.extents = [Index([i for i in range(self.m) if attributes[j](objects[i])]) for j in range(self.n)]

        # sort attribute in ascending order of extent size
        if sort_attributes:
            attribute_order = list(sorted(range(self.n), key=lambda i: len(self.extents[i])))
            self.attributes = [self.attributes[i] for i in attribute_order]
            self.extents = [self.extents[i] for i in attribute_order]

    def greedy_simplification(self, intent, extent):
        to_cover = SortedSet([i for i in range(self.m) if i not in extent])
        available = list(range(len(intent)))
        covering = [SortedSet([i for i in range(self.m) if i not in self.extents[j]]) for j in intent]
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
            return Index(range(len(self.objects)))
            # return SortedSet(range(len(self.objects)))

        # result = SortedSet.intersection(*map(lambda i: self.extents[i], intent))
        result = self.extents[intent[0]]
        for i in range(1, len(intent)):
            result = result.intersection(self.extents[intent[i]])
        #result = SortedSet.intersection(*map(lambda i: self.extents[i], intent))

        return result

    def refinement(self, node, i, f, g, opt_val):
        """
        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> f, g = lambda e: -len(e), lambda e: 1
        >>> root = Node(SortedSet([]),SortedSet([]),Index([0,1,2,3]), -1, -4, 1, inf)
        >>> ref = ctx.refinement(root, 0, f, g, -4)
        >>> list(ref.closure)
        [0, 2]
        """
        if i in node.closure:
            print(f"WARNING: redundant augmentation {self.attributes[i]}")
            return None

#        if node.extension <= self.extents[i]:
#           print(f"WARNING: redundant augmentation {self.attributes[i]}")

        #generator = node.generator + [i]
        generator = node.generator.copy()
        generator.add(i)
        # extension = node.extension & self.extents[i]
        extension = node.extension.intersection(self.extents[i])

        val = f(extension)
        bound = g(extension)

        if bound < opt_val:
            return None

        closure = []
        for j in range(0, i):
            if j in node.closure:
                closure.append(j)
            # elif extension <= self.extents[j]:
            elif extension.isin(self.extents[j]).all():
                return Node(generator, closure, extension, i, j, val, bound)

        closure.append(i)

        crit_idx = self.n
        for j in range(i + 1, self.n):
            if j in node.closure:
                closure.append(j)
            # elif extension <= self.extents[j]:
            elif extension.isin(self.extents[j]).all():
                crit_idx = min(crit_idx, self.n)
                closure.append(j)

        return Node(generator, SortedSet(closure), extension, i, crit_idx, val, bound)

    traversal_orders = {
        'breadthfirst': BreadthFirstBoundary,
        'bestboundfirst': BestBoundFirstBoundary,
        'bestvaluefirst': BestValueFirstBoundary,
        'depthfirst': DepthFirstBoundary
    }

    def traversal(self, f, g, order='breadthfirst'):
        """
        A first example with trivial objective and bounding function is as follows. In this example
        the optimal extension is the empty extension, which is generated via the
        the lexicographically smallest and shortest generator [0, 1, 3].
        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> search = ctx.traversal(lambda e: -len(e), lambda e: 1)
        >>> for n in search:
        ...     print(n)
        N([], [], -4, inf, [0, 1, 2, 3])
        N([0], [0, 2], -2, 1, [1, 2])
        N([1], [1], -3, 1, [0, 1, 3])
        N([1, 3], [1, 3], -2, 1, [0, 3])
        N([0, 1], [0, 1, 2], -1, 1, [1])
        N([0, 1, 3], [0, 1, 2, 3], 0, 1, [])

        Let's use more realistic objective and bounding functions based on values associated with each
        object (row in the table).
        >>> values = [-1, 1, 1, -1]
        >>> f = lambda e: sum((values[i] for i in e))/4
        >>> g = lambda e: sum((values[i] for i in e if values[i]==1))/4
        >>> search = ctx.traversal(f, g)
        >>> for n in search:
        ...     print(n)
        N([], [], 0, inf, [0, 1, 2, 3])
        N([0], [0, 2], 0.5, 0.5, [1, 2])

        Finally, here is a complex example taken from the UdS seminar on subgroup discovery.
        >>> table = [[1, 1, 1, 1, 0],
        ...          [1, 1, 0, 0, 0],
        ...          [1, 0, 1, 0, 0],
        ...          [0, 1, 1, 1, 1],
        ...          [0, 0, 1, 1, 1],
        ...          [1, 1, 0, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> labels = [1, 0, 1, 0, 0, 0]
        >>> from realkd.legacy import impact, cov_incr_mean_bound, impact_count_mean
        >>> f = impact(labels)
        >>> g = cov_incr_mean_bound(labels, impact_count_mean(labels))
        >>> search = ctx.traversal(f, g)
        >>> for n in search:
        ...     print(n)
        N([], [], 0, inf, [0, 1, 2, 3, 4, 5])
        N([0], [0], 0.11111, 0.22222, [0, 1, 2, 5])
        N([1], [1], -0.055556, 0.11111, [0, 1, 3, 5])
        N([2], [2], 0.11111, 0.22222, [0, 2, 3, 4])
        N([0, 2], [0, 2], 0.22222, 0.22222, [0, 2])

        >>> ctx.search(f, g)
        c0 & c2

        >>> labels = [1, 0, 0, 1, 1, 0]
        >>> f = impact(labels)
        >>> g = cov_incr_mean_bound(labels, impact_count_mean(labels))
        >>> search = ctx.traversal(f, g)
        >>> for n in search:
        ...     print(n)
        N([], [], 0, inf, [0, 1, 2, 3, 4, 5])
        N([0], [0], -0.16667, 0.083333, [0, 1, 2, 5])
        N([1], [1], 0, 0.16667, [0, 1, 3, 5])
        N([2], [2], 0.16667, 0.25, [0, 2, 3, 4])
        N([4], [4], 0.083333, 0.16667, [3, 4, 5])
        N([2, 3], [2, 3], 0.25, 0.25, [0, 3, 4])

        :param f: objective function
        :param g: bounding function satisfying that g(I) >= max {f(J): J >= I}
        """
        boundary = self.traversal_orders[order]()
        full = self.extension([])
        root = Node(SortedSet([]), SortedSet([]), full, -1, self.n, f(full), inf)
        opt = root
        yield root
        boundary.push((range(self.n), root))

        while boundary:
            ops, current = boundary.pop()
            children = []
            for a in ops:
                child = self.refinement(current, a, f, g, opt.val)
                if child:
                    if child.valid:
                        opt = max(opt, child, key=Node.value)
                        yield child
                    children += [child]
            filtered = list(filter(lambda c: c.val_bound > opt.val, children))
            ops = []
            for child in reversed(filtered):
                if child.valid:
                    boundary.push(([i for i in ops if i not in child.closure], child))
                # [i for i in ops if i not in child.closure]
                #ops = [child.gen_index] + ops
                ops = [child.gen_index] + ops

    def search(self, f, g, order='breadthfirst', verbose=False):
        opt = None
        opt_value = -inf
        k = 0
        for node in self.traversal(f, g, order):
            if opt_value < node.val:
                opt = node
                opt_value = node.val
            k += 1
            if verbose and k % 1000 == 0:
                print('*', end='', flush=True)
        if verbose:
            print('')
            print(f'Found optimum after inspecting {k} nodes')
        min_generator = self.greedy_simplification(opt.closure, opt.extension)
        return Conjunction(map(lambda i: self.attributes[i], min_generator))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
