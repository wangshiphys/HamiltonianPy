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

    def query_index(self, key):
        """
        Return the index corresponding to the given `key`.

        Parameters
        ----------
        key :
            The index corresponding to which is queried.

        Returns
        -------
        index : int
            The corresponding index.

        Raises
        ------
        KeyError :
            The queried key is not managed by this table.
        TypeError :
            The given key isn't of the same type as the objects managed by
            this table.
        """

        if isinstance(key, self._object_type):
            return self._object2index[key]
        else:
            raise TypeError(
                "The given key isn't of the same type "
                "as the objects managed by this table."
            )

    __call__ = query_index

    def query_object(self, index):
        """
        Return the object corresponding to the given `index`.

        Parameters
        ----------
        index : int
            The index.

        Returns
        -------
        res :
            The object corresponding to the given `index`.

        Raises
        ------
        KeyError :
            The queried index is not managed by this table.
        TypeError :
            The given index is not integer.
        """

        if isinstance(index, int):
            return self._index2object[index]
        else:
            raise TypeError("The given `index` should be integer!")
