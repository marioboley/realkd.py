import sortednp as snp

from collections import deque
from sortedcontainers import SortedSet
from math import inf
from heapq import heappop, heappush
from numpy import array
from bitarray import bitarray
from bitarray.util import subset

from realkd.datasets import titanic_data, titanic_column_trans
from realkd.logic import Conjunction


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
        return f"N({list(self.generator)}, {array([i for i in range(len(self.closure)) if self.closure[i]])}, {self.val:.5g}, {self.val_bound:.5g}, {self.extension})"

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

    def __len__(self):
        return len(self.deq)

    def push(self, augmented_node):
        self.deq.append(augmented_node)

    def pop(self):
        return self.deq.popleft()


class DepthFirstBoundary:
    def __init__(self):
        self.stack = []

    def __bool__(self):
        return bool(self.stack)

    def __len__(self):
        return len(self.stack)

    def push(self, augmented_node):
        self.stack.append(augmented_node)

    def pop(self):
        return self.stack.pop()


class BestBoundFirstBoundary:
    def __init__(self):
        self.heap = []

    def __bool__(self):
        return bool(self.heap)

    def __len__(self):
        return len(self.heap)

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

    def __len__(self):
        return len(self.heap)

    def push(self, augmented_node):
        _, node = augmented_node
        heappush(self.heap, (-node.val, -node.val_bound, augmented_node))

    def pop(self):
        _, _, augmented_node = heappop(self.heap)
        return augmented_node


class CoreQueryTreeSearch:
    r"""Searches the prefix-tree of core queries given a certain :class:`SearchContext`.

    The idea of core queries is that there is only one core query per potential query extensions. Thus using them
    as solution candidates results in a strongly condensed search space compared to naively searching all conjunctions.

    Core queries have been originally introduced for closed itemset mining :cite:`uno2004efficient`.
    The following is a simple recursive definition that does not require the notion of closures:

    The trivial query :math:`q = \top` is a core query. Moreover, the tail augmentation :math:`qp_i` of a core query
    :math:`q` is also a core query if :math:`q \not\rightarrow p_i` and

    .. math::
        :nowrap:

        \begin{equation}
        \text{for all } j < i, \text{ if } q' \rightarrow p_j \text{ then } q \rightarrow p_j \enspace .
        \end{equation}

    """

    #: dictionary of available traversal orders for core query search
    traversal_orders = {
        "breadthfirst": BreadthFirstBoundary,
        "bestboundfirst": BestBoundFirstBoundary,
        "bestvaluefirst": BestValueFirstBoundary,
        "depthfirst": DepthFirstBoundary,
    }

    def __init__(
        self,
        ctx,
        obj,
        bnd,
        order="bestboundfirst",
        apx=1.0,
        max_depth=10,
        verbose=False,
        **kwargs,
    ):
        """

        :param SearchContext ctx: the context defining the search space
        :param callable obj: objective function
        :param callable bnd: bounding function satisfying that ``bnd(q) >= max{obj(r) for r in successors(q)}``
        :param str order: traversal order (``'breadthfirst'``, ``'depthfirst'``, ``'bestboundfirst'``, or ``'bestvaluefirst'``; see :data:`traversal_orders`)
        :param float apx: approximation factor that determines guarantee of what fraction of search space will be traversed, i.e., all nodes q are visited with ``obj(q) >= apx * opt`` (default ``1.0``, i.e., optimum will be visited; smaller values means less traversal elements
        :param int max_depth: maximum depth of explored search nodes
        :param int verbose: level of verbosity

        """
        self.ctx = ctx
        self.f = obj
        self.g = bnd
        self.order = order
        self.apx = apx
        self.max_depth = max_depth
        self.verbose = verbose

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

    def traversal(self):
        """
        A first example with trivial objective and bounding function is as follows. In this example
        the optimal extension is the empty extension, which is generated via the
        the non-lexicographically smallest generator [0, 3].

        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = SearchContext.from_tab(table)
        >>> search = CoreQueryTreeSearch(ctx, lambda e: -len(e), lambda e: 0, order='breadthfirst')
        >>> for n in search.traversal():
        ...     print(n)
        N([], [], -4, inf, [0 1 2 3])
        N([0], [0 2], -2, 0, [1 2])
        N([1], [1], -3, 0, [0 1 3])
        N([2], [0 2], -2, 0, [1 2])
        N([3], [1 3], -2, 0, [0 3])
        N([0, 1], [0 1 2], -1, 0, [1])
        N([0, 3], [0 1 2 3], 0, 0, [])

        >>> search.verbose = True
        >>> search.run()
        <BLANKLINE>
        Found optimum after inspecting 7 nodes: [0, 3]
        Completing closure
        Greedy simplification: [0, 3]
        c0 & c3

        >>> CoreQueryTreeSearch(ctx, lambda e: 5-len(e), lambda e: 4+(len(e)>=2), apx=0.7).run()
        c0 & c1

        Let's use more realistic objective and bounding functions based on values associated with each
        object (row in the table).

        >>> values = [-1, 1, 1, -1]
        >>> f = lambda e: sum((values[i] for i in e))/4
        >>> g = lambda e: sum((values[i] for i in e if values[i]==1))/4
        >>> search = CoreQueryTreeSearch(ctx, f, g)
        >>> for n in search.traversal():
        ...     print(n)
        N([], [], 0, inf, [0 1 2 3])
        N([0], [0 2], 0.5, 0.5, [1 2])
        N([2], [0 2], 0.5, 0.5, [1 2])

        Finally, here is a complex example taken from the UdS seminar on subgroup discovery.

        >>> table = [[1, 1, 1, 1, 0],
        ...          [1, 1, 0, 0, 0],
        ...          [1, 0, 1, 0, 0],
        ...          [0, 1, 1, 1, 1],
        ...          [0, 0, 1, 1, 1],
        ...          [1, 1, 0, 0, 1]]
        >>> ctx = SearchContext.from_tab(table)
        >>> labels = [1, 0, 1, 0, 0, 0]
        >>> from realkd.legacy import impact, cov_incr_mean_bound, impact_count_mean
        >>> f = impact(labels)
        >>> g = cov_incr_mean_bound(labels, impact_count_mean(labels))
        >>> search = CoreQueryTreeSearch(ctx, f, g)
        >>> for n in search.traversal():
        ...     print(n)
        N([], [], 0, inf, [0 1 2 3 4 5])
        N([0], [0], 0.11111, 0.22222, [0 1 2 5])
        N([1], [1], -0.055556, 0.11111, [0 1 3 5])
        N([2], [2], 0.11111, 0.22222, [0 2 3 4])
        N([3], [2 3], 0, 0.11111, [0 3 4])
        N([0, 2], [0 2], 0.22222, 0.22222, [0 2])
        >>> search.run()
        c0 & c2

        >>> labels = [1, 0, 0, 1, 1, 0]
        >>> f = impact(labels)
        >>> g = cov_incr_mean_bound(labels, impact_count_mean(labels))
        >>> search = CoreQueryTreeSearch(ctx, f, g)
        >>> for n in search.traversal():
        ...     print(n)
        N([], [], 0, inf, [0 1 2 3 4 5])
        N([0], [0], -0.16667, 0.083333, [0 1 2 5])
        N([1], [1], 0, 0.16667, [0 1 3 5])
        N([2], [2], 0.16667, 0.25, [0 2 3 4])
        N([3], [2 3], 0.25, 0.25, [0 3 4])
        """
        boundary = CoreQueryTreeSearch.traversal_orders[self.order]()
        full = self.ctx.extension([])
        full_bits = bitarray(len(full))
        full_bits.setall(1)
        root = Node(
            SortedSet([]),
            bitarray((0 for _ in range(self.ctx.n))),
            full,
            full_bits,
            -1,
            self.ctx.n,
            self.f(full),
            inf,
        )
        opt = root
        yield root

        boundary.push(([(i, self.ctx.n, inf) for i in range(self.ctx.n)], root))
        self.created += 1

        while boundary:
            ops, current = boundary.pop()

            self.popped += 1

            if self.verbose >= 2 and self.popped % 1000 == 0:
                print("*", end="", flush=True)
            if self.verbose >= 1 and self.popped % 10000 == 0:
                print(
                    f" (lwr/upp/rat: {opt.val:.4f}/{current.val_bound:.4f}/{opt.val/current.val_bound:.4f},"
                    f" opt/avg depth: {len(opt.generator)}/{self.avg_created_length:.2f},"
                    f" bndry: {len(boundary)})",
                    flush=True,
                )

            children = []
            # for a in ops:
            for aug, crit, bnd in ops:
                if aug <= current.gen_index:  # need to also check == case it seems
                    continue
                if self.crit_propagation and crit < current.gen_index:
                    self.rec_crit_hits += 1
                    continue
                if (
                    bnd * self.apx <= opt.val
                ):  # checking old bound against potentially updated opt value
                    self.del_bnd_hits += 1
                    continue
                if current.closure[aug]:
                    self.clo_hits += 1
                    continue

                extension = snp.intersect(current.extension, self.ctx.extents[aug])
                val = self.f(extension)
                bound = self.g(extension)

                generator = current.generator[:]
                generator.append(aug)

                self.created += 1
                self.avg_created_length = (
                    self.avg_created_length * ((self.created - 1) / self.created)
                    + len(generator) / self.created
                )

                if bound * self.apx < opt.val and val <= opt.val:
                    self.bnd_immediate_hits += 1
                    continue

                bit_extension = current.bit_extension & self.ctx.bit_extents[aug]
                closure = bitarray(current.closure)
                closure[aug] = True
                if self.crit_propagation and crit < aug and not current.closure[crit]:
                    # aug still needed for descendants but for current is guaranteed
                    # to lead to not lexmin child; hence can recycle current crit index
                    # (as upper bound to real crit index)
                    self.crit_hits += 1
                    crit_idx = crit
                else:
                    crit_idx = self.ctx.find_small_crit_index(
                        aug, bit_extension, closure
                    )

                if crit_idx > aug:  # in this case crit_idx == n (sentinel)
                    crit_idx = self.ctx.complete_closure(aug, bit_extension, closure)
                else:
                    closure[crit_idx] = True

                child = Node(
                    generator,
                    closure,
                    extension,
                    bit_extension,
                    aug,
                    crit_idx,
                    val,
                    bound,
                )
                opt = max(opt, child, key=Node.value)
                yield child

                # early termination if opt value approximately exceeds best active upper bound
                if (
                    opt.val >= self.apx * current.val_bound
                    and self.order == "bestboundfirst"
                ):
                    if self.verbose:
                        print(
                            f"best value {opt.val:.4f} {self.apx}-apx. exceeds best active bound {current.val_bound:.4f}"
                        )
                        print(f"terminating traversal")
                    return

                children += [child]

            augs = []
            for child in children:
                if child.val_bound * self.apx > opt.val:
                    augs.append((child.gen_index, child.crit_idx, child.val_bound))
                else:
                    self.bnd_post_children_hits += 1

            for child in children:
                if child.valid and (
                    not self.max_depth or len(child.generator) < self.max_depth
                ):
                    boundary.push((augs, child))
                else:
                    self.non_lexmin_hits += 1

    def print_stats(self):
        print()
        print("Pruning rule hits")
        print("-----------------")
        print("bound propagation   (sgl):", self.del_bnd_hits)
        print("crit propagation    (rec):", self.rec_crit_hits)
        print("crit propagation    (sgl):", self.crit_hits)
        print("crit violation      (rec):", self.non_lexmin_hits)
        print("equivalence         (rec):", self.clo_hits)
        print("bnd immediate       (rec):", self.bnd_immediate_hits)
        print("bnd post children   (rec):", self.bnd_post_children_hits)

    def run(self):
        """
        Runs the configured search.

        :return: :class:`~realkd.logic.Conjunction` that (approximately) maximizes objective

        """
        if self.verbose >= 2:
            print(
                f"Searching with apx factor {self.apx} and depth limit {self.max_depth} in order {self.order}"
            )
        opt = None
        opt_value = -inf
        k = 0
        for node in self.traversal():
            k += 1
            if opt_value < node.val:
                opt = node
                opt_value = node.val
        if self.verbose:
            print("")
            print(f"Found optimum after inspecting {k} nodes: {opt.generator}")

        if self.verbose >= 3:
            self.print_stats()

        if not opt.valid:
            if self.verbose:
                print("Completing closure")
            self.ctx.complete_closure(opt.gen_index, opt.bit_extension, opt.closure)
        min_generator = self.ctx.greedy_simplification(
            [i for i in range(len(opt.closure)) if opt.closure[i]], opt.extension
        )
        if self.verbose:
            print("Greedy simplification:", min_generator)
        return Conjunction(map(lambda i: self.ctx.attributes[i], min_generator))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
