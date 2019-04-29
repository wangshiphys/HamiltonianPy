"""
Permutation and combination
"""


__all__ = [
    "Permutation",
    "Combination",
]


from math import factorial


def Permutation(n, k, repeat=False):
    """
    Permutation with/without repetition

    The order matters.

    Parameters
    ----------
    n : int
        The total number of items for choice
    k : int
        The number of items to choose
    repeat : boolean, optional
        Whether repetition is allowed
        default: False

    Returns
    -------
    res : int
        The total number of permutations
    """

    assert isinstance(n, int) and n >= 0
    assert isinstance(k, int) and 0 <= k <= n

    if repeat:
        return n ** k
    else:
        return int(factorial(n) / factorial(n - k))


def Combination(n, k, repeat=False):
    """
    Combination with/without repetition

    The order does not matter.

    Parameters
    ----------
    n : int
        The total number of items for choice
    k : int
        The number of items to choose
    repeat : boolean, optional
        Whether repetition is allowed
        default: False

    Returns
    -------
    res : int
        The total number of combinations
    """

    assert isinstance(n, int) and n >= 0
    assert isinstance(k, int) and 0 <= k <= n

    if repeat:
        return int(factorial(n + k - 1) / factorial(k) / factorial(n - 1))
    else:
        return int(factorial(n) / factorial(k) / factorial(n - k))
