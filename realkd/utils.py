def get_generic_column_headers(data):
    """Returns x0, x1, ... x{n} for the number of columns in the data
    
    :param ~numpy.array data: input data

    :return: list[str] labels

    >>> import numpy as np
    >>> get_generic_column_headers(np.array([[1, 4],[2, 3],[24, 31]]))
    ['x0', 'x1']
    """
    return [f'x{n}' for n in range(data.shape[1])]


def validate_data(data, labels=None):
    """Converts pandas Dataframe or numpy array to numpy array and a list of labels

    :param ~pandas.DataFrame|~numpy.array data: input data
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
    """
    if hasattr(data, "iloc"):
        return data.to_numpy(), data.columns.tolist()
    else:
        return data, labels if labels is not None else get_generic_column_headers(data)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
