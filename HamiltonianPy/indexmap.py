"""    
A class to map a collection of hashable objects to integer index.
"""

from collections import OrderedDict
from copy import deepcopy

__all__ = ['IndexMap']

class IndexMap:
    """
    This class construct a map between a collection of hashable objects and
    integer index.

    Attribute:
    ----------
    obj_type: 
        The type of the input objects.
    obj2index_table: dict
        A dictionary whose keys are the input objects and values are integer
        numbers represent the index of the obj.
    index2obj_table: tuple
        A tuple of the input objects, which is sorted according to the
        obj2index_table.
    length: int
        The length of the table.

    Method:
    -------
    __init__(objs)
    __str__()
    __len__()
    __call__(arg)
    """

    def __init__(self, objs):# {{{
        """
        Initilize instance of this class!

        Paramter:
        ---------
        objs:
            The objects which are asked to construct a map.
        """

        obj2index_table = OrderedDict()
        index2obj_table = []
        obj_type = objs[0].__class__
        for index, obj in enumerate(objs):
            if not isinstance(obj, obj_type):
                raise TypeError("The input objs are not all of the same type!")
            hash(obj)
            obj2index_table[deepcopy(obj)] = index
            index2obj_table.append(deepcopy(obj))

        self.obj_type = obj_type
        self.obj2index_table = obj2index_table
        self.index2obj_table = index2obj_table
        self.length = len(index2obj_table)
    # }}}

    def __str__(self):# {{{
        """
        Return the printing string of instance  of this class.
        """

        info = "\nKeys:"
        for key, val in self.obj2index_table.items():
            info += str(key)
        info += "=" * 50
        info += "\nItems:"
        for item in self.index2obj_table:
            info += str(item)
        return info
    # }}}

    def __len__(self):# {{{
        """
        The length of the table!
        """
        return self.length
    # }}}

    def __call__(self, arg):# {{{
        """
        Given the arg, return the corresponding items or index!
        """

        if isinstance(arg, self.obj_type):
            res = self.obj2index_table[arg]
        elif isinstance(arg, int) and arg >= 0:
            res = self.index2obj_table[arg]
        else:
            raise TypeError("The invalid arg!")
        return deepcopy(res)
    # }}}
