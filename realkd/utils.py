import numpy as np
import pandas as pd


def contains_non_numeric(column):
    """

    >>> contains_non_numeric(['nan', '2'])
    False
    >>> contains_non_numeric(['hello', 2])
    True
    """
    for item in column:
        try:
            float(item)
        except ValueError:
            return True
    return False


def get_generic_column_headers(data):
    """Returns x0, x1, ... x{n} for the number of columns in the data

    :param ~numpy.array data: input data

    :return: list[str] labels

    >>> import numpy as np
    >>> get_generic_column_headers(np.array([[1, 4],[2, 3],[24, 31]]))
    ['x0', 'x1']
    """
    if len(data.shape) == 1:
        # Data is one dimentional (i.e. there's only one row)
        return [f"x{n}" for n in range(data.shape[0])]
    return [f"x{n}" for n in range(data.shape[1])]

def validate_data(data, labels=None):
    return data if isinstance(data, RealkdArrayLike) else RealkdArrayLike(data, labels)

class RealkdArrayLike:
    @staticmethod
    def get_labels(data, labels=None):
        if labels is not None:
            return labels
        elif hasattr(data, "columns"):
            # TODO: Probably fine to just have data.columns here
            return data.columns.to_list()
        elif hasattr(data, "keys"):
            return data.keys()
        else:
            return get_generic_column_headers(data)

    def __init__(self, data, labels=None):
        self._raw = data
        self.labels = RealkdArrayLike.get_labels(data, labels)

    def reset_index(self, **kwargs):
        if hasattr(self._raw, 'reset_index'):
            self._raw.reset_index(**kwargs)
        return self

    def __repr__(self):
        return f"RealkdArrayLike<_raw={self._raw}, labels={self.labels}>"

    def __getitem__(self, keys):
        """
        Facilitates numpy-style indexing, but optionally the first key can be a string,
        in which case the corresponding column from labels is selected

        >>> data, target = np.array([[1, 7],[2, 8],[3, 9]]), np.array([1,2,3])
        >>> a = RealkdArrayLike(data)
        >>> a[1]
        RealkdArrayLike<_raw=[2 8], labels=['x0', 'x1']>
        >>> a[:, 1]
        RealkdArrayLike<_raw=[7 8 9], labels=['x0', 'x1']>
        >>> b = RealkdArrayLike(pd.DataFrame(data, columns=a.labels))
        >>> b[1]
        RealkdArrayLike<_raw=x0    2
        x1    8
        Name: 1, dtype: int64, labels=['x0', 'x1']>
        >>> b[:, 1]
        RealkdArrayLike<_raw=0    7
        1    8
        2    9
        Name: x1, dtype: int64, labels=['x0', 'x1']>
        >>> a['x1'][1] + 1
        9
        >>> b['x1'][1] + 1
        9
        >>> len(a['x1'])
        3
        >>> c = RealkdArrayLike(np.array([[1., np.NaN],[2, 8],[3, 9]]))
        >>> col = c[:, 1]
        >>> col[~pd.isnull(col)]
        RealkdArrayLike<_raw=[8. 9.], labels=['x0', 'x1']>
        """
        data, labels = self._index(keys)
        if hasattr(data, "__len__"):
            return RealkdArrayLike(data, labels)
        return data

    def _index(self, first_key, *more_keys):
        if type(first_key) == str:
            if hasattr(self._raw, "iloc"):
                return self._raw.__getitem__(first_key), self.labels
            if hasattr(self._raw, 'shape'):
                # If indexing a numpy array with a string, the logic is:
                # If the array is 1D, treat it as an observation
                # If the array is 2D, treat it the first dimention (rows) as observations
                if self._raw.ndim == 1:
                    return self._raw[self.labels.index(first_key)], [first_key]
                return self._raw[:, self.labels.index(first_key)], [first_key]
            return self._raw.__getitem__(first_key), [first_key]

        else:
            if hasattr(self._raw, "iloc"):
                try:
                    return self._raw.iloc.__getitem__(first_key, *more_keys), self.labels
                except:
                    return self._raw.loc.__getitem__(first_key, *more_keys), self.labels
            return self._raw.__getitem__(first_key, *more_keys), self.labels

    def __len__(self):
        return len(self._raw)

    def __eq__(self, other):
        """
        >>> c = RealkdArrayLike(np.array([[1, 5],[2, 8],[3, 9]]))
        >>> 2 < c
        array([[False,  True],
               [False,  True],
               [ True,  True]])
        >>> d = RealkdArrayLike(pd.DataFrame(np.array([[1, 5],[2, 8],[3, 9]])))
        >>> d > 2
               0     1
        0  False  True
        1  False  True
        2   True  True
        """
        return self._raw.__eq__(other)

    def __gt__(self, other): return self._raw.__gt__(other)
    def __ge__(self, other): return self._raw.__ge__(other)
    def __lt__(self, other): return self._raw.__lt__(other)
    def __le__(self, other): return self._raw.__le__(other)
    def __neg__(self): return self._raw.__neg__()

    def __getattr__(self, name):
        """
        Delegate to the raw datastructure.
        This is to facilitate code like the following:
        # >>> from realkd.rules import GradientBoostingObjective
        # >>> data, target = np.array([[1],[2],[3]]), np.array([1,2,3])
        # >>> est = GradientBoostingObjective(data, target)
        # >>> est.data.sort()
        """
        try:
            return getattr(self._raw, name)
        except AttributeError:
            raise AttributeError("'Array' object has no attribute {}".format(name))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
