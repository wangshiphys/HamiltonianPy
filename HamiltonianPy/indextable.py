"""Mapping hashable objects to integers in a continuous range
"""

__all__ = ["IndexTable"]

class IndexTable:
    """
    A table that manages hashable objects and their indices

    Examples
    --------
    >>> objs = [(x, y) for x in range(10) for y in range(10)]
    >>> table = IndexMap(objs)
    >>> len(table)
    100
    >>> table((3, 5))
    35
    >>> table(23)
    (2, 3)
    """

    def __init__(self, objs, *, start=0):# {{{
        """Customize the newly created instance

        Paramters
        ---------
        objs: iterable
            A collection of hashable objects that to be managed by the table
        start: int, keyword-only, optional
            The start of the index
            default: 0
        """

        # _objtype is the type of the input objects
        # _obj2index is a dict
        # The keys are the input objects and values are the indices
        # _index2obj is a dict
        # The keys are the indices and values are the objects

        self._obj2index = dict()
        self._index2obj = dict()
        for index, obj in enumerate(objs, start):
            if index == start:
                objtype = type(obj)
            elif not isinstance(obj, objtype):
                raise TypeError(
                        "All the input objects should be of the same type!")

            if obj in self._obj2index:
                raise ValueError(
                        "There are duplicate items in the given objects.")
            else:
                self._obj2index[obj] = index
                self._index2obj[index] = obj

        self._objtype = objtype
        self._length = len(self._obj2index)
    # }}}

    def __str__(self):# {{{
        """Return a string that describe the content the object
        """

        fmt = "Index: {0}\tKey: {1!r}"
        return "\n".join(fmt.format(i, k) for i, k in self._index2obj.items())
    # }}}

    def __len__(self):# {{{
        """The number of entries in the table
        """

        return self._length
    # }}}

    def __call__(self, key):# {{{
        """
        Return the object or index according to the given key

        Parameters
        ----------
        key : object or int
            The key for which the corresponding object or index is queried

        Returns
        -------
        res : The corresponding object or index.

        Raises
        ------
        KeyError :
            The queried object or index is not managed by this table
        TypeError :
            The given key is neither of the same type as the objects managed
            by this table nor of type int
        """

        if isinstance(key, self._objtype):
            res = self._obj2index[key]
        elif isinstance(key, int):
            res = self._index2obj[key]
        else:
            raise TypeError("The given key is neither of the same type as the"
                    "objects managed by this table nor of type int")
        return res
    # }}}


#This is a test of the IndexMap class.
if __name__ == "__main__":
    from random import randrange
    numx = 10
    numy = 3
    objs = ((x, y) for x in range(numx) for y in range(numy))
    table = IndexTable(objs)
    key0 = (randrange(numx), randrange(numy))
    key1 = randrange(numx * numy)
    print("key = {0}\t value = {1}".format(key0, table(key0)))
    print("key = {0}\t value = {1}".format(key1, table(key1)))
    assert len(table) == numx * numy
    assert table(key0) == key0[0] * numy + key0[1]
    assert table(key1) == divmod(key1, numy)
