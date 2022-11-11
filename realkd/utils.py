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

    :param ~pandas.DataFrame|~numpy.array data: input data
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
    """
    if hasattr(data, "iloc"):
        return data.to_numpy(), data.columns.tolist()
    else:
        return data, labels if labels is not None else get_generic_column_headers(data)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
