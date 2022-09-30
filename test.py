import numpy as np
from numba import njit


# @njit
def intersect_sorted_array(A, B):
    i = 0
    j = 0
    intersection = set()

    while i < len(A) and j < len(B):
        if A[i] == B[j]:
            intersection.add(A[i])
            i += 1
            j += 1
        elif A[i] < B[j]:
            i += 1
        else:
            j += 1
    return intersection


n = 1000000
# Unique, because extents are unique
test_A = np.unique(np.sort(np.random.randint(0, n, n)))
test_B = np.unique(np.sort(np.random.randint(0, n, n)))

intersect_sorted_array(test_A, test_B)