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


def validate_data_xx(data, labels=None):
    """Converts pandas Dataframe or numpy array to numpy array and a list of labels

    :param dict|~pandas.DataFrame|~numpy.array data: input data
    :param list[str] labels: labels for numpy features

    :return: Tuple of (~numpy.array array_data, list[str] labels)

    >>> import numpy as np
    >>> from pandas import DataFrame
    >>> validate_data(DataFrame({'petal_area': [1,2]}))
    (array([[1],
           [2]]), ['petal_area'])
    >>> validate_data(np.array([[1],[2]]))
    (array([[1],
           [2]]), ['x0'])
    >>> validate_data(np.array([[1],[2]]), ['petal_area'])
    (array([[1],
           [2]]), ['petal_area'])
    >>> validate_data({ 'sex': 'female', 'age': 10})
    (array(['female', '10'], dtype='<U21'), ['sex', 'age'])
    >>> validate_data({ 'sex': ['female', 'male'], 'age': [10, 12]})
    (array([['female', '10'],
           ['male', '12']], dtype='<U21'), ['sex', 'age'])
    """
    if hasattr(data, "iloc"):
        return data.to_numpy(), data.columns.tolist()
    elif hasattr(data, "shape"):
        return data, labels if labels is not None else get_generic_column_headers(data)
    elif hasattr(data, "values"):
        # Hopefully dictlike
        return np.array([*data.values()]).T, [*data.keys()]
    else:
        return np.array(
            data
        ), labels if labels is not None else get_generic_column_headers(np.array(data))


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

    def __repr__(self):
        return f"RealkdArrayLike<_raw={self._raw}, labels={self.labels}>"

    def __getitem__(self, keys):
        """
        Facilitates numpy-style indexing, but optionally the first key can be a string,
        in which case the corresponding column from labels is selected

        >>> data, target = np.array([[1, 7],[2, 8],[3, 9]]), np.array([1,2,3])
        >>> a = RealkdArrayLike(data)
        >>> a[:, 1]
        RealkdArrayLike<_raw=[7 8 9], labels=['x0', 'x1']>
        >>> b = RealkdArrayLike(pd.DataFrame(data, columns=a.labels))
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
                return self._raw[:, self.labels.index(first_key)], [first_key]
            return self._raw.__getitem__(first_key), [first_key]

        else:
            if hasattr(self._raw, "iloc"):
                return self._raw.iloc.__getitem__(first_key, *more_keys), self.labels
            return self._raw.__getitem__(first_key, *more_keys), self.labels

    def __len__(self):
        return len(self._raw)

    def __eq__(self, other):
        return self._raw.__eq__(other)

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
