"""
Mapping hashable objects to integers in a continuous range
"""


from collections import OrderedDict


__all__ = [
    "IndexTable",
]


class IndexTable:
    """
    A table that manages hashable objects and their indices

    Attributes
    ----------
    object_type
        The type of the objects managed by this table

    Examples
    --------
    >>> from HamiltonianPy.indextable import IndexTable
    >>> objects = [(x, y) for x in range(10) for y in range(10)]
    >>> table = IndexTable(objects)
    >>> len(table)
    100
    >>> table((3, 5))
    35
    >>> table(23)
    (2, 3)
    """

    def __init__(self, objects, *, start=0):
        """
        Customize the newly created instance

        Parameters
        ----------
        objects: iterable
            A collection of hashable objects that is to be managed by the table
        start: int, optional, keyword-only
            The start of the index
            default: 0
        """

        # object_type is the type of the input objects
        # object2index is an OrderedDict
        # The keys are the input objects and values are the indices
        # index2object is an OrderedDict
        # The keys are the indices and values are the objects

        object2index = OrderedDict()
        index2object = OrderedDict()
        for index, item in enumerate(objects, start):
            if index == start:
                object_type = type(item)
            else:
                if not isinstance(item, object_type):
                    raise TypeError(
                        "The {0}th object has different type from the "
                        "previous ones".format(index - start)
                    )

            if item in object2index:
                raise ValueError(
                    "The {0}th object already exists".format(index - start)
                )
            else:
                object2index[item] = index
                index2object[index] = item

        self._object2index = object2index
        self._index2object = index2object
        self._object_type = object_type
        self._length = len(object2index)

    @property
    def object_type(self):
        """
        The `object_type` attribute
        """

        return self._object_type

    def __str__(self):
        """
        Return a string that describe the content the object
        """

        info = "Index\tObject\n==============\n" + "\n".join(
            "{0}\t{1!r}".format(i, k) for i, k in self._index2object.items()
        )
        return info

    def indices(self):
        """
        Return a new view of the objects' indices
        """

        return self._index2object.keys()

    def objects(self):
        """
        Return a new view of the managed objects
        """

        return self._object2index.keys()

    def __iter__(self):
        """
        Make instance of this class iterable

        Iterate over the indices and the corresponding objects: (index, obj)
        """

        return iter(self._index2object.items())

    def __len__(self):
        """
        The number of entries in the table
        """

        return self._length

    def __call__(self, key):
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

        if isinstance(key, self._object_type):
            res = self._object2index[key]
        elif isinstance(key, int):
            res = self._index2object[key]
        else:
            raise TypeError(
                "The given key is neither of the same type as the objects "
                "managed by this table nor of type int."
            )
        return res


# This is a test of the IndexMap class
if __name__ == "__main__":
    from random import randrange

    num0 = 7
    num1 = 3
    objects = ((x, y) for x in range(num0) for y in range(num1))
    table = IndexTable(objects)
    assert len(table) == num0 * num1

    for i in range(10):
        key0 = (randrange(num0), randrange(num1))
        key1 = randrange(num0 * num1)
        assert table(key0) == key0[0] * num1 + key0[1]
        assert table(key1) == divmod(key1, num1)
        print("key = {0}\t value = {1}".format(key0, table(key0)))
        print("key = {0}\t value = {1}".format(key1, table(key1)))
        print("=" * 80)

    for index in table.indices():
        print(index)
    print("=" * 80)
    for obj in table.objects():
        print(obj)
    print("=" * 80)
    for index, obj in table:
        print(index, obj)
