import sortednp as snp

from sortedcontainers import SortedSet

from realkd.logic import Conjunction


class GreedySearch:
    """
    Simple best-in greedy search for conjunctive query.

    Search starts from the trivial (empty) query and adds an objective-maximizing conditions until no
    further improvement with a single augmentation is possible.

    >>> table = [[1, 1, 1, 1, 0],
    ...          [1, 1, 0, 0, 0],
    ...          [1, 0, 1, 0, 0],
    ...          [0, 1, 1, 1, 1],
    ...          [0, 0, 1, 1, 1],
    ...          [1, 1, 0, 0, 1]]
    >>> ctx = SearchContext.from_tab(table)
    >>> labels = [1, 0, 1, 0, 0, 0]
    >>> from realkd.legacy import impact
    >>> f = impact(labels)
    >>> GreedySearch(ctx, f).run()
    c0 & c2

    """

    def __init__(self, ctx, obj, bdn=None, verbose=False, **kwargs):
        """

        :param SearchContext ctx: the context defining the search space
        :param callable obj: objective function
        :param callable bnd: bounding function satisfying that ``bnd(q) >= max{obj(r) for r in successors(q)}`` (for signature compatibility only, not currently used)
        :param int verbose: level of verbosity

        """
        self.ctx = ctx
        self.f = obj
        self.verbose = verbose

    def run(self):
        """
        Runs the configured search.

        :return: :class:`~realkd.logic.Conjunction` that (approximately) maximizes objective
        """
        intent = SortedSet([])
        extent = self.ctx.extension([])
        value = self.f(extent)
        while True:
            best_i, best_ext = None, None
            for i in range(self.ctx.n):
                if i in intent:
                    continue
                _extent = snp.intersect(extent, self.ctx.extents[i])
                _value = self.f(_extent)
                if _value > value:
                    value = _value
                    best_ext = _extent
                    best_i = i
            if best_i is not None:
                intent.add(best_i)
                extent = best_ext
            else:
                break
            if self.verbose:
                print("*", end="", flush=True)
        return Conjunction(map(lambda i: self.ctx.attributes[i], intent))
