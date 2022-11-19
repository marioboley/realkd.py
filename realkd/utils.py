import numpy as np

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
    if(len(data.shape) == 1):
        # Data is one dimentional (i.e. there's only one row)
        return [f'x{n}' for n in range(data.shape[0])]
    return [f'x{n}' for n in range(data.shape[1])]


def to_numpy_and_labels(data, labels=None):
    """Converts pandas Dataframe or numpy array to numpy array and a list of labels

    :param dict|~pandas.DataFrame|~numpy.array data: input data
    :param list[str] labels: labels for numpy features

    :return: Tuple of (~numpy.array array_data, list[str] labels)

    >>> import numpy as np 
    >>> from pandas import DataFrame
    >>> to_numpy_and_labels(DataFrame({'petal_area': [1,2]}))
    (array([[1],
           [2]]), ['petal_area'])
    >>> to_numpy_and_labels(np.array([[1],[2]]))
    (array([[1],
           [2]]), ['x0'])
    >>> to_numpy_and_labels(np.array([[1],[2]]), ['petal_area'])
    (array([[1],
           [2]]), ['petal_area'])
    >>> to_numpy_and_labels({ 'sex': 'female', 'age': 10})
    (array(['female', '10'], dtype='<U21'), ['sex', 'age'])
    >>> to_numpy_and_labels({ 'sex': ['female', 'male'], 'age': [10, 12]})
    (array([['female', '10'],
           ['male', '12']], dtype='<U21'), ['sex', 'age'])
    """
    if hasattr(data, "iloc"):
        return data.to_numpy(), data.columns.tolist()
    elif hasattr(data, 'shape'):
        return data, labels if labels is not None else get_generic_column_headers(data)
    elif hasattr(data, 'values'):
        # Hopefully dictlike
        return np.array([*data.values()]).T, [*data.keys()]
    else:
        return np.array(data), labels if labels is not None else get_generic_column_headers(np.array(data))


def validate_data(data, labels=None):
    return RealkdArrayLike(data, labels)

class RealkdArrayLike:
    @staticmethod
    def get_labels(self, data, labels=None):
        if labels is not None:
            return labels
        elif hasattr(data, 'columns'):
            # TODO: Probably fine to just have data.columns here
            return data.columns.to_list()
        elif hasattr(data, 'keys'):
            return data.keys()
        else:
            return get_generic_column_headers(data)

    def __init__(self, data, labels=None):
        self._raw = data
        self.labels = RealkdArrayLike.get_labels(data, labels)

    def __getitem__(self, key): 
        if hasattr(self._raw, "iloc"):
            return self._raw.iloc.__getitem__(key)
        return self._raw.__getitem__(key)

    # def __setitem__(self, key, value):
    #     self.np_array.__setitem__(key, value)
    #     self._lib.set_arr(new_arr.ctypes)

    def __getattr__(self, name):
        """
            Delegate to the raw datastructure.
            This is to facilitate code like the following:
            >>> from realkd.rules import GradientBoostingObjective
            >>> data, target = np.array([[1],[2],[3]]), np.array([1,2,3])
            >>> est = GradientBoostingObjective(data, target)
            >>> est.data.sort()
        """
        try:
            return getattr(self._raw, name)
        except AttributeError:
            raise AttributeError(
                 "'Array' object has no attribute {}".format(name))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
