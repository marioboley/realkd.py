import pandas as pd

from collections import deque
from sortedcontainers import SortedSet
from math import inf


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
        return f'N({self.generator}, {self.closure}, {self.val:.5g}, {self.val_bound:.5g}, {list(self.extension)})'

    def value(self):
        return self.val


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
    def from_df(df, without=None, max_col_attr=None, sort_attributes=True):
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
        SortedSet([338, 400, 414])

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
                    _, vals = pd.qcut(df[c], q=max_col_attr // 2, retbins=True, duplicates='drop')
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
        self.extents = [SortedSet([i for i in range(self.m) if attributes[j](objects[i])]) for j in range(self.n)]

        # sort attribute in ascending order of extent size
        if sort_attributes:
            attribute_order = list(sorted(range(self.n), key=lambda i: len(self.extents[i])))
            self.attributes = [self.attributes[i] for i in attribute_order]
            self.extents = [self.extents[i] for i in attribute_order]

    def search(self, f, g):
        opt = max(self.bfs(f, g), key=Node.value)
        min_generator = self.greedy_simplification(opt.closure, opt.extension)
        return Conjunction(map(lambda i: self.attributes[i], min_generator))

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
            return SortedSet(range(len(self.objects)))

        result = SortedSet.intersection(*map(lambda i: self.extents[i], intent))

        return result

    def refinement(self, node, i, f, g, opt_val):
        """
        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> f, g = lambda e: -len(e), lambda e: 1
        >>> root = Node([],[],SortedSet([0,1,2,3]), -1, -4, 1, inf)
        >>> ref = ctx.refinement(root, 0, f, g, -4)
        >>> ref.closure
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
        extension = node.extension & self.extents[i]

        val = f(extension)
        bound = g(extension)

        if bound < opt_val:
            return None

        closure = []
        for j in range(0, i):
            if j in node.closure:
                closure.append(j)
            elif extension <= self.extents[j]:
                return Node(generator, closure, extension, i, j, val, bound)

        closure.append(i)

        crit_idx = self.n
        for j in range(i + 1, self.n):
            if j in node.closure:
                closure.append(j)
            elif extension <= self.extents[j]:
                crit_idx = min(crit_idx, self.n)
                closure.append(j)

        return Node(generator, SortedSet(closure), extension, i, crit_idx, val, bound)

    def bfs(self, f, g):
        """
        A first example with trivial objective and bounding function is as follows. In this example
        the optimal extension is the empty extension, which is generated via the
        the lexicographically smallest and shortest generator [0, 3].
        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> search = ctx.bfs(lambda e: -len(e), lambda e: 1)
        >>> for n in search:
        ...     print(n)
        N([0, 3], [0, 1, 2, 3], 0, 1, [])

        Let's use more realistic objective and bounding functions based on values associated with each
        object (row in the table).
        >>> values = [-1, 1, 1, -1]
        >>> f = lambda e: sum((values[i] for i in e))/4
        >>> g = lambda e: sum((values[i] for i in e if values[i]==1))/4
        >>> search = ctx.bfs(f, g)
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
        >>> f = impact(labels)
        >>> g = cov_incr_mean_bound(labels, impact_count_mean(labels))
        >>> search = ctx.bfs(f, g)
        >>> for n in search:
        ...     print(n)
        N([], [], 0, inf, [0, 1, 2, 3, 4, 5])
        N([0], [0], 0.11111, 0.22222, [0, 1, 2, 5])
        N([1], [1], -0.055556, 0.11111, [0, 1, 3, 5])
        N([2], [2, 3], 0.11111, 0.22222, [0, 2, 3, 4])
        N([3], [3], 0, 0.11111, [0, 3, 4])
        N([0, 2], [0, 2], 0.22222, 0.22222, [0, 2])

        >>> ctx.search(f, g)
        c0 & c2

        :param f: objective function
        :param g: bounding function satisfying that g(I) >= max {f(J): J >= I}
        """
        boundary = deque()
        full = self.extension([])
        root = Node(SortedSet([]), SortedSet([]), full, -1, self.n, f(full), inf)
        opt = root
        yield root
        boundary.append((range(self.n), root))

        while boundary:
            ops, current = boundary.popleft()
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
                    boundary.append(([i for i in ops if i not in child.closure], child))
                # [i for i in ops if i not in child.closure]
                #ops = [child.gen_index] + ops
                ops = [child.gen_index] + ops


def cov_squared_dev(labels):
    n = len(labels)
    global_mean = sum(labels) / n

    def f(count, mean):
        return count/n * pow(mean - global_mean, 2)

    return f


def impact_count_mean(labels):
    """
    >>> labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> f = impact_count_mean(labels)
    >>> f(5, 0.4) # 1/2 * (2/5-1/5) = 1/6
    0.1
    """
    n = len(labels)
    m0 = sum(labels)/n

    def f(c, m):
        return c/n * (m - m0)

    return f


class DfWrapper:

    def __init__(self, df): self.df = df

    def __getitem__(self, item): return self.df.iloc[item]

    def __len__(self): return len(self.df)

    def __iter__(self):
        return (r for (_, r) in self.df.iterrows())


class Impact:
    """
    Impact objective function for conjunctive queries with respect to a specific
    dataset D and target variable y. Formally:

    impact(q) = |ext(q)|/|D| (mean(y; ext(q)) - mean(y; D)) .

    Accepts list-like, dict-like, and Pandas dataframe objects. For example:
    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> old_male = Conjunction([KeyValueProposition('Age', Constraint.greater_equals(60)),
    ...                         KeyValueProposition('Sex', Constraint.equals('male'))])
    >>> imp_survival = Impact(titanic, 'Survived')
    >>> imp_survival(old_male)
    -0.006110487591969073
    >>> imp_survival.search()
    Sex==female
    """

    def _mean(self, q):
        s, c = 0.0, 0.0
        for r in filter(q, self.data):
            s += r[self.target]
            c += 1
        return s/c

    def _coverage(self, q):
        return sum(1 for _ in filter(q, self.data))/self.m

    def __init__(self, data, target):
        self.m = len(data)
        self.data = DfWrapper(data) if isinstance(data, pd.DataFrame) else data
        self.target = target
        self.average = self._mean(lambda _: True)

    def __call__(self, q):
        return self._coverage(q) * (self._mean(q) - self.average)

    def search(self):
        ctx = Context.from_df(self.data.df, without=[self.target], max_col_attr=10)
        f = impact(self.data.df[self.target])
        g = cov_incr_mean_bound(self.data.df[self.target], impact_count_mean(self.data.df[self.target]))
        return ctx.search(f, g)


class SquaredLossObjective:
    """
    Rule boosting objective function for squared loss.

    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> obj = SquaredLossObjective(titanic, titanic['Survived'])
    >>> female = Conjunction([KeyValueProposition('Sex', Constraint.equals('female'))])
    >>> first_class = Conjunction([KeyValueProposition('Pclass', Constraint.less_equals(1))])
    >>> obj(female)
    0.19404590848327577
    >>> reg_obj = SquaredLossObjective(titanic.drop(columns=['Survived']), titanic['Survived'], reg=2)
    >>> reg_obj(female)
    0.19342988972618597
    >>> reg_obj(first_class)
    0.09566220318908493
    >>> reg_obj._mean(female)
    0.7420382165605095
    >>> reg_obj._mean(first_class)
    0.6296296296296297
    >>> reg_obj.search()
    Sex==female
    """

    def __init__(self, data, target, reg=0):
        """
        :param data:
        :param target: _series_ of target values of matching dimension
        :param reg:
        """
        self.m = len(data)
        self.data = DfWrapper(data) if isinstance(data, pd.DataFrame) else data
        self.target = target
        self.reg = reg

    def _f(self, count, mean):
        return self._reg_term(count)*count/self.m * pow(mean, 2)

    def _reg_term(self, c):
        return 1 / (1 + self.reg / (2 * c))

    def _count(self, q): #almost code duplication: Impact
        return sum(1 for _ in filter(q, self.data))

    def _mean(self, q): #code duplication: Impact
        s, c = 0.0, 0.0
        for i in range(self.m):
            if q(self.data[i]):
                s += self.target[i]
                c += 1
        return s/c

    def search(self, max_col_attr=10):
        # here we need the function in list of row indices; can we save some of these conversions?
        def f(rows):
            c = len(rows)
            if c == 0:
                return 0.0
            m = sum(self.target[i] for i in rows) / c
            return self._f(c, m)

        g = cov_mean_bound(self.target, lambda c, m: self._f(c, m))

        ctx = Context.from_df(self.data.df, max_col_attr=max_col_attr)
        return ctx.search(f, g)

    def opt_value(self, rows):
        s, c = 0.0, 0
        for i in rows:
            s += self.target[i]
            c += 1

        return s / (self.reg/2 + c) if (c > 0 or self.reg > 0) else 0.0

    def __call__(self, q):
        c = self._count(q)
        m = self._mean(q)
        return self._f(c, m)


def impact(labels):
    """
    Compiles objective function for extension I defined by
    f(I) = len(I)/n (mean_I(l)-mean(l)) for some set of labels l of size n.

    >>> labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> f = impact(labels)
    >>> f([0, 1, 2, 3, 4]) # 0.5 * (0.4 - 0.2)
    0.1
    >>> f(range(len(labels)))
    0.0
    """
    g = impact_count_mean(labels)

    def f(extension):
        if len(extension) == 0:
            return -inf
        m = sum((labels[i] for i in extension))/len(extension)
        return g(len(extension), m)

    return f


def squared_loss_obj(labels):
    """
    Builds objective function that maps index set to product
    of relative size of index set times squared difference
    of mean label value described by index set to overall
    mean label value. For instance:

    >>> labels = [-4, -2, -1, -1, 0, 1, 10, 21]
    >>> sum(labels)/len(labels)
    3.0
    >>> obj = squared_loss_obj(labels)
    >>> obj([4, 5, 6, 7])  # local avg 8, relative size 1/2
    12.5

    :param labels: y-values
    :return: f(I) = |I|/n * (mean(y)-mean_I(y))^2
    """

    f = cov_squared_dev(labels)

    def label(i): return labels[i]

    def obj(extent):
        k = len(extent)
        local_mean = sum(map(label, extent)) / k

        return f(k, local_mean)

    return obj


def cov_incr_mean_bound(labels, f):
    """
    >>> labels = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
    >>> f = impact_count_mean(labels)
    >>> g = cov_incr_mean_bound(labels, f)
    >>> g(range(len(labels)))
    0.25
    """

    def label(i): return labels[i]

    def bound(extent):
        ordered = sorted(extent, key=label)
        k = len(ordered)
        opt = -inf

        s = 0
        for i in range(k):
            s += labels[ordered[-i-1]]
            opt = max(opt, f(i+1, s/(i+1)))

        return opt

    return bound


def cov_mean_bound(labels, f):
    """
    >>> labels = [-13, -2, -1, -1, 0, 1, 19, 21]
    >>> f = cov_squared_dev(labels)
    >>> obj = squared_loss_obj(labels)
    >>> obj(range(6,8))
    72.25
    >>> f(2, 20.0)
    72.25
    >>> bound = cov_mean_bound(labels, f)
    >>> bound(range(len(labels)))  # local avg 8, relative size 1/2
    72.25

    :param labels:
    :param f: any function that can be re-written as the maximum f(c, m)=max(g(c,m), h(c,m)) over functions g and h
              where g is monotonically increasing in its first and second argument (count and mean)
              and h is monotonically increasing in its first argument and monotonically decreasing in its second
              argument
    :return: bounding function that returns for any set of indices I, the maximum f-value over subsets J <= I
             where f is evaluated as f(|J|, mean(labels; J))
    """

    def label(i): return labels[i]

    def bound(extent):
        ordered = sorted(extent, key=label)
        k = len(ordered)
        opt = -inf

        s = 0
        for i in range(k):
            s += labels[ordered[-i-1]]
            opt = max(opt, f(i+1, s/(i+1)))

        s = 0
        for i in range(k):
            s += labels[ordered[i]]
            opt = max(opt, f(i+1, s/(i+1)))

        return opt

    return bound

