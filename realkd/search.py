import pandas as pd
import sortednp as snp

from collections import defaultdict, deque
from sortedcontainers import SortedSet
from math import inf
from heapq import heappop, heappush
from numpy import array, arange
from bitarray import bitarray

from realkd.logic import Conjunction, Constraint, KeyValueProposition, TabulatedProposition


class Node:
    """
    Represents a potential node (and incoming edge) for searches in the concept graph
    with edges representing the direct prefix-preserving successor relation (dpps).
    """

    def __init__(self, gen, clo, ext, bit_ext, idx, crit_idx, val, bnd):
        self.generator = gen
        self.closure = clo
        self.extension = ext
        self.bit_extension = bit_ext
        self.gen_index = idx
        self.crit_idx = crit_idx
        self.val = val
        self.val_bound = bnd
        self.valid = self.crit_idx > self.gen_index

    def __repr__(self):
        return f'N({list(self.generator)}, {array([i for i in range(len(self.closure)) if self.closure[i]])}, {self.val:.5g}, {self.val_bound:.5g}, {list(self.extension)})'

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
        values are then selected by the provided discretization function (per default quantile-based).

        The restriction should also be implemented for object columns in the future (by merging small categories
        into disjunctive propositions).

        The generated attributes correspond to pandas-compatible query strings. For example:

        >>> titanic_df = pd.read_csv("../datasets/titanic/train.csv")
        >>> titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        >>> titanic_ctx = Context.from_df(titanic_df, max_col_attr=6, sort_attributes=False)
        >>> titanic_ctx.m
        891
        >>> titanic_ctx.attributes # doctest: +NORMALIZE_WHITESPACE
        [Survived<=0, Survived>=1, Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==male, Sex==female, Age<=23.0,
        Age>=23.0, Age<=34.0, Age>=34.0, Age<=80.0, Age>=80.0, SibSp<=8.0, SibSp>=8.0, Parch<=6.0, Parch>=6.0,
        Fare<=8.6625, Fare>=8.6625, Fare<=26.0, Fare>=26.0, Fare<=512.3292, Fare>=512.3292, Embarked==S, Embarked==C,
        Embarked==Q, Embarked==nan]
        >>> titanic_ctx.n
        28
        >>> titanic_df.query('Survived>=1 & Pclass>=3 & Sex=="male" & Age>=34')
             Survived  Pclass   Sex   Age  SibSp  Parch   Fare Embarked
        338         1       3  male  45.0      0      0  8.050        S
        400         1       3  male  39.0      0      0  7.925        S
        414         1       3  male  44.0      0      0  7.925        S
        >>> titanic_ctx.extension([1, 5, 6, 11])
        array([338, 400, 414])

        (prev was SortedSet([338, 400, 414]))

        >>> titanic_ctx = Context.from_df(titanic_df, max_col_attr=defaultdict(lambda: None, Age=6, Fare=6),
        ...                               sort_attributes=False)
        >>> titanic_ctx.attributes # doctest: +NORMALIZE_WHITESPACE
        [Survived<=0, Survived>=1, Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==male, Sex==female, Age<=23.0,
        Age>=23.0, Age<=34.0, Age>=34.0, Age<=80.0, Age>=80.0, SibSp<=0, SibSp<=1, SibSp>=1, SibSp<=2, SibSp>=2,
        SibSp<=3, SibSp>=3, SibSp<=4, SibSp>=4, SibSp<=5, SibSp>=5, SibSp>=8, Parch<=0, Parch<=1, Parch>=1, Parch<=2,
        Parch>=2, Parch<=3, Parch>=3, Parch<=4, Parch>=4, Parch<=5, Parch>=5, Parch>=6, Fare<=8.6625, Fare>=8.6625,
        Fare<=26.0, Fare>=26.0, Fare<=512.3292, Fare>=512.3292, Embarked==S, Embarked==C, Embarked==Q, Embarked==nan]


        :param df: pandas dataframe to be converted to formal context
        :param max_col_attr: maximum number of attributes generated per column;
                             or None if an arbitrary number of attributes is permitted;
                             or dict (usually defaultdict) with keys being columns ids of df and values
                             being the maximum number of attributes for the corresponding column (again using
                             None if no bound for a specific column);
                             Note: use defaultdict(lambda: None) instead of defaultdict(None) to specify no maximum
                             per default
        :param discretization: the discretization function to be used when number of thresholds has to be reduced to
                               a specificed maximum (function has to have identical signature to pandas.qcut, which
                               is the default)
        :param without: columns to ommit
        :return: context representing dataframe
        """

        without = without or []

        if not isinstance(max_col_attr, dict):
            const = max_col_attr
            max_col_attr = defaultdict(lambda: const)

        attributes = []
        for c in df:
            if c in without:
                continue
            if df[c].dtype.kind in 'uif':
                vals = df[c].unique()
                reduced = False
                max_cols = max_col_attr[str(c)]
                if max_cols and len(vals)*2 > max_cols:
                    _, vals = discretization(df[c], max_cols // 2, retbins=True, duplicates='drop')
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
        self.extents = [array([i for i in range(self.m) if attributes[j](objects[i])], dtype='int64') for j in range(self.n)]
        self.bit_extents = [bitarray([True if attributes[j](objects[i]) else False for i in range(self.m)]) for j in range(self.n)]

        # sort attribute in ascending order of extent size
        if sort_attributes:
            attribute_order = list(sorted(range(self.n), key=lambda i: len(self.extents[i])))
            self.attributes = [self.attributes[i] for i in attribute_order]
            self.extents = [self.extents[i] for i in attribute_order]
            self.bit_extents = [self.bit_extents[i] for i in attribute_order]

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
            return array(range(len(self.objects)))

        result = self.extents[intent[0]]
        for i in range(1, len(intent)):
            result = snp.intersect(result, self.extents[intent[i]])

        return result

    def refinement(self, node, i, f, g, opt_val, apx=1.0):
        """
        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> f, g = lambda e: -len(e), lambda e: 1
        >>> root = Node([], bitarray('0000'), array([0,1,2,3]), bitarray('1111'), -1, -4, 1, inf)
        >>> ref = ctx.refinement(root, 0, f, g, -4)
        >>> list(ref.closure)
        [True, False, True, False]
        """
        # if i in node.closure:
        #     print(f"WARNING: redundant augmentation {self.attributes[i]}")
        #     return None

        generator = node.generator[:]
        generator.append(i)
        extension = snp.intersect(node.extension, self.extents[i])
        bit_extension = node.bit_extension & self.bit_extents[i]

        val = f(extension)
        bound = g(extension)

        # TODO: this can apparently harm result quality: if val > opt it should still become the new
        #       opt even if the improvement (and bound) is less what is required for enqueuing
        if bound * apx < opt_val:
            return None

        closure = bitarray(node.closure)
        closure[i] = True
        for j in range(0, i):
            if not closure[j] and len(extension) <= len(self.extents[j]) and \
                    (bit_extension & self.bit_extents[j]).count() == len(extension):
                return Node(generator, closure, extension, bit_extension, i, j, val, bound)

        crit_idx = self.n
        for j in range(i + 1, self.n):
            if not closure[j] and len(extension) <= len(self.extents[j]) and \
                    (bit_extension & self.bit_extents[j]).count() == len(extension):
                crit_idx = min(crit_idx, self.n)
                closure[j] = True

        return Node(generator, closure, extension, bit_extension, i, crit_idx, val, bound)

    traversal_orders = {
        'breadthfirst': BreadthFirstBoundary,
        'bestboundfirst': BestBoundFirstBoundary,
        'bestvaluefirst': BestValueFirstBoundary,
        'depthfirst': DepthFirstBoundary
    }

    def traversal(self, f, g, order='breadthfirst', apx=1.0, verbose=False):
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
        N([0], [0 2], -2, 1, [1, 2])
        N([1], [1], -3, 1, [0, 1, 3])
        N([0, 1], [0 1 2], -1, 1, [1])
        N([1, 3], [1 3], -2, 1, [0, 3])
        N([0, 1, 3], [0 1 2 3], 0, 1, [])

        Let's use more realistic objective and bounding functions based on values associated with each
        object (row in the table).
        >>> values = [-1, 1, 1, -1]
        >>> f = lambda e: sum((values[i] for i in e))/4
        >>> g = lambda e: sum((values[i] for i in e if values[i]==1))/4
        >>> search = ctx.traversal(f, g)
        >>> for n in search:
        ...     print(n)
        N([], [], 0, inf, [0, 1, 2, 3])
        N([0], [0 2], 0.5, 0.5, [1, 2])

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
        N([0, 2], [0 2], 0.22222, 0.22222, [0, 2])

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
        N([1, 2], [1 2 3], 0.16667, 0.16667, [0, 3])
        N([2, 3], [2 3], 0.25, 0.25, [0, 3, 4])

        :param f: objective function
        :param g: bounding function satisfying that g(I) >= max {f(J): J >= I}
        :param order: traversal order ('breadthfirst', 'depthfirst', 'bestboundfirst' or 'bestvaluefirst')
        :param apx: approximation factor that determines guarantee of what fraction
                    of search space will be traversed, i.e., all nodes q are visited
                    with f(q) >= apx * opt (default 1.0, i.e., optimum will be visited;
                    smaller values means less traversal elements
        """
        boundary = self.traversal_orders[order]()
        full = self.extension([])
        full_bits = bitarray(len(full))
        full_bits.setall(1)
        root = Node(SortedSet([]), bitarray((0 for _ in range(self.n))), full, full_bits, -1, self.n, f(full), inf)
        opt = root
        yield root
        # boundary.push((range(self.n), root))
        boundary.push(([(i, self.n, inf) for i in range(self.n)], root))

        k = 0
        rec_crit_hits = 0
        crit_hits = 0
        del_bnd_hits = 0
        clo_hits = 0
        non_lexmin_hits = 0
        bnd_post_children_hits = 0
        bnd_immediate_hits = 0

        while boundary:
            ops, current = boundary.pop()

            k += 1
            if verbose >= 2 and k % 1000 == 0:
                print('*', end='', flush=True)
            if verbose >= 1 and k % 10000 == 0:
                print(f' (best/bound: {opt.val}, {current.val_bound})', flush=True)

            children = []
            # for a in ops:
            for aug, crit, bnd in ops:
                if aug <= current.gen_index:  # need to also check == case it seems
                    continue
                if crit < current.gen_index:
                    rec_crit_hits += 1
                    continue
                if bnd * apx <= opt.val:
                    del_bnd_hits += 1
                    continue
                if current.closure[aug]:
                    clo_hits += 1
                    continue

                # TODO: the following check guarantees that the augmentation will be invalid;
                #       however, it might still be needed as augmentation option for children;
                #       hence, it is incorrect to skip recursively but one could skip specific
                #       refinement operation and instead directly build invalid node
                if crit < aug and not current.closure[crit]:
                     crit_hits += 1
                #     continue

                child = self.refinement(current, aug, f, g, opt.val, apx)
                if child:
                    if child.valid:
                        # yield child
                        # TODO: this is a conservative implementation that means that an
                        #       invalid child does not contribute to raising the current opt value.
                        opt = max(opt, child, key=Node.value)
                        yield child
                    children += [child]
                else:
                    bnd_immediate_hits += 1

            # filtered = filter(lambda c: c.val_bound * apx > opt.val, children)
            # augs = [(child.gen_index, child.crit_idx, child.val_bound) for child in filtered]

            # augs = [(child.gen_index, child.crit_idx, child.val_bound) for child in children if child.val_bound * apx > opt.val]

            augs = []
            for child in children:
                if child.val_bound * apx > opt.val:
                    augs.append((child.gen_index, child.crit_idx, child.val_bound))
                else:
                    bnd_post_children_hits += 1

            for child in children:
                if child.valid:
                    boundary.push((augs, child))
                else:
                    non_lexmin_hits += 1

        if verbose >= 3:
            print()
            print('Pruning rule hits')
            print('-----------------')
            print('bound propagation   (rec):', del_bnd_hits)
            print('crit propagation    (rec):', rec_crit_hits)
            print('crit propagation    (sgl):', crit_hits)
            print('crit violation      (rec):', non_lexmin_hits)
            print('equivalence         (rec):', clo_hits)
            print('bnd immediate       (rec):', bnd_immediate_hits)
            print('bnd post children   (rec):', bnd_post_children_hits)

            # filtered = list(filter(lambda c: c.val_bound * apx > opt.val, children))

            # ops = []
            # for child in reversed(filtered):
            #     if child.valid:
            #         boundary.push(([i for i in ops if i not in child.closure], child))
            #     # [i for i in ops if i not in child.closure]
            #     #ops = [child.gen_index] + ops
            #     ops = [child.gen_index] + ops

    def greedy_search(self, f, verbose=False):
        """
        >>> table = [[1, 1, 1, 1, 0],
        ...          [1, 1, 0, 0, 0],
        ...          [1, 0, 1, 0, 0],
        ...          [0, 1, 1, 1, 1],
        ...          [0, 0, 1, 1, 1],
        ...          [1, 1, 0, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> labels = [1, 0, 1, 0, 0, 0]
        >>> from realkd.legacy import impact
        >>> f = impact(labels)
        >>> ctx.greedy_search(f)
        c0 & c2
        """
        intent = SortedSet([])
        extent = self.extension([])
        value = f(extent)
        while True:
            best_i, best_ext = None, None
            for i in range(self.n):
                if i in intent:
                    continue
                _extent = snp.intersect(extent, self.extents[i])
                _value = f(_extent)
                if _value > value:
                    value = _value
                    best_ext = _extent
                    best_i = i
            if best_i is not None:
                intent.add(best_i)
                extent = best_ext
            else:
                break
            if verbose:
                print('*', end='', flush=True)
        return Conjunction(map(lambda i: self.attributes[i], intent))

    def search(self, f, g, order='breadthfirst', apx=1.0, verbose=False):
        if verbose >= 2:
            print(f'Searching with apx factor {apx} in order {order}')
        opt = None
        opt_value = -inf
        k = 0
        for node in self.traversal(f, g, order, apx, verbose=verbose):
            k += 1
            if opt_value < node.val:
                opt = node
                opt_value = node.val
        if verbose:
            print('')
            print(f'Found optimum after inspecting {k} nodes')
        min_generator = self.greedy_simplification([i for i in range(len(opt.closure)) if opt.closure[i]],
                                                   opt.extension)
        return Conjunction(map(lambda i: self.attributes[i], min_generator))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
