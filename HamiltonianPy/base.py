"""
The function to generate basis of Hilbert space in occupation representation.
"""

from itertools import combinations, product

import numpy as np

__all__ = ['base_table']

def _nchoosek(n, k):# {{{
    """
    Return integers that represents the result of choosing k items from n items.

    Parameter:
    ----------
    n: int
        The total number of items.
    k: int
        The number of items should be choosen.

    Return:
    -------
    table: list
        The collection of integers stored in ascending order.
    """

    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("The wrong data type of n or k!")

    if n <= 0:
        raise ValueError("Got a non-positive n: {0}!".format(n))
    if k < 0 or k > n:
        raise ValueError("Got an invalid k: {0}!".format(k))
    elif k == 0:
        table = [0]
    else:
        table = []
        for cfg in combinations(range(0, n), k):
            res = 0
            for pos in cfg:
                res += 1 << pos
            table.append(res)
    return table
# }}}


def base_table(ns, ks=None, toarr=False):# {{{
    """
    Return integers that represents the permutation specified by ns and ks.

    Here, ns represents the number of elements which are available to be
    choosen and ks represents the number of elements desired. ns can be either
    integer or tuple/list of integers. If ns is integer, then ks must be either
    None or ineger. If ns is tuple/list of integers, ks must also be tuple/list
    of integer with the same length. If ns is tuple or list, that means we are
    choosing from multiple containers, the entries of ns are the number of
    elements in these containers and entries of ks are the number of elements
    to be choosen from these containers.

    Parameter:
    ----------
    ns: tuple or int
        The number of elements which are available to be choosen
    ks: tuple or int, optional
        The number of elements desired.
        default: None
    toarr: boolean, optional
        Determine whether to convert the result to a numpy array.
        default: False

    Return:
    -------
    res: tuple or np.ndarray
        The collection of integers stored in ascending order.
    """

    if isinstance(ns, int):
        if ks is None:
            res = range(0, 1<<ns)
        elif isinstance(ks, int):
            res = _nchoosek(ns, ks)
            res.sort()
        else:
            raise TypeError("The ks parameter is not of type int or None!")
    elif isinstance(ns, (tuple, list)) and isinstance(ks, (tuple, list)):
        length = len(ns)
        if len(ks) != length:
            raise ValueError("The length of ns and ks does not match!")

        allchoice = []
        for n, k in zip(ns, ks):
            allchoice.append(_nchoosek(n, k))

        res = []
        for cfg in product(*allchoice):
            ket = 0
            for i, state in enumerate(cfg):
                ket += state << sum(ns[:i])
            res.append(ket)
        res.sort()
    else:
        raise TypeError("The wrong type of ns parameter!")

    if toarr:
        return np.array(res)
    else:
        return tuple(res)
# }}}

if __name__ == "__main__":
    ns = 4
    ks = None
    print("ns = {0}, ks = {1}".format(ns, ks))
    print(base_table(ns, ks))
    print("=" * 60)
    ns = 4
    ks = 2
    print("ns = {0}, ks = {1}".format(ns, ks))
    print(base_table(ns, ks))
    print("=" * 60)
    ns = [2, 2, 2]
    ks = [1, 1, 1]
    print("ns = {0}, ks = {1}".format(ns, ks))
    print(base_table(ns, ks))
