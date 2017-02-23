"""
This module implement the common search algorithm.
"""

from bisect import bisect_left
from exception import TargetError

__all__ = ['bisect_search', 'binary_search', 'sequential_search']

def binary_search(target, container):# {{{
    """
    Return the position of the target in the container which is sorted in ascending order.

    Parameter:
    ----------
    target: int or float
        The object whose position is to be searched.
    container : sequence or ndarray
        The object that might contain the item.

    Return:
    -------
    mid: int
        The position of the target in the container.

    Raise:
    ------
    TargetError:
        raise when the target is not in the container.
    """

    low = 0
    high = len(container) - 1
    while low <= high:
        mid = (low + high) // 2
        buff = container[mid]
        if target > buff:
            low = mid + 1
        elif target < buff:
            high = mid - 1
        else:
            return mid
    raise TargetError('The target does not contained in the container!')
# }}}

def bisect_search(target, container):# {{{
    """
    Return the position of the target in the container which is sorted in ascending order.

    Parameter:
    ----------
    target: int or float
        The object whose position is to be searched.
    container : sequence or ndarray
        The object that might contain the item.

    Return:
    -------
    res: int
        The position of the target in the container.

    Raise:
    ------
    TargetError:
        raise when the target is not in the container.
    """

    res = bisect_left(container, target)
    if res != len(container) and container[res] == target:
        return res
    raise TargetError('The target does not contained in the container!')
# }}}

def sequential_search(target, container):# {{{
    """
    Return the position of the target in the container.

    Parameter:
    ----------
    target: int or float
        The object whose position is to be searched.
    container: sequence or ndarray
        The object that might contain the item.

    Return:
    -------
    result: list
        The position(s) of the target in the container.

    Raise:
    ------
    TargetError:
        raise when the target is not in the container.
    """

    res = []
    for index, item in enumerate(container):
        if target == item:
            res.append(index)

    if len(res) == 0:
        raise TargetError('The target does not contained in the container!')
    else:
        return res
# }}}
