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

    Methods
    -------
    __init__(objs, start=0)
    __str__()
    __len__()
    __call__(key)
    add(obj)
    """

    def __init__(self, objs=None, start=0):# {{{
        """
        Initialize instance of this class!

        Paramters
        ---------
        objs: seqence, optional
            The hashable objects which are added to the map system.
            default: None
        start: int, optional
            The start of the index.
            default: 0
        """

        #_objtype is the type of the input objects.
        #_obj2index is an OrderedDict whose keys are the input objects and
        #values are corresponding indices.
        #_index2obj is an OrderedDict whose keys are the indices and values are
        #corresponding objects.
        #_current is the integer which will be assigned as the index of the
        #next object added to this map system.
        
        if isinstance(start, int) and start >= 0:
            self._current = start
        else:
            raise ValueError("The start index must be none negative integer.")

        self._obj2index = OrderedDict()
        self._index2obj = OrderedDict()
        self._objtype = None

        if objs is not None:
            for obj in objs:
                self.add(obj)
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describe the content of instance of this class.
        """

        if self.__len__() == 0:
            info = "The map system is empty.\n"
        else:
            info = ""
            for key, index in self._obj2index.items():
                info += "Key:\n" + str(key) + '\n'
                info += "Index:\n" + str(index) + '\n\n'
        return info
    # }}}

    def __len__(self):# {{{
        """
        The length of the dictionary!
        """

        return len(self._obj2index)
    # }}}

    def __call__(self, key):# {{{
        """
        Given the key, return the deepcopy of the corresponding object or index!

        Parameters
        ----------
        key : hashable object or int
            The key for which the corresponding object or index is queried.

        Returns
        -------
        res : A deepcopy of the corresponding object or index.

        Raises
        ------
        KeyError :
            The queried object or index is not managed by this map system.
        LookupError :
            The map system is queried before effectively construct.
        """

        if self.__len__() > 0:
            if isinstance(key, self._objtype):
                res = self._obj2index[key]
            elif isinstance(key, int):
                res = deepcopy(self._index2obj[key])
            else:
                raise TypeError("The invalid key!")
        else:
            errmsg = "The map system is empty. "
            errmsg += "Please add members to it before query!"
            raise LookupError(errmsg)
        return res
    # }}}

    def add(self, obj):# {{{
        """
        Add the given obj to the map system and associate it with an index.

        Parameters
        ----------
        obj : 
            The obj parameter must be:
            1. hashable
            2. the same type as the existing object
            3. not exist in the map system
        """

        num = self.__len__()
        if num == 0:
            self._objtype = obj.__class__
        elif num > 0:
            if not isinstance(obj, self._objtype):
                errmsg = "The input object is not the same type "
                errmsg += "as the existing objects."
                raise TypeError(errmsg)
        
        key = deepcopy(obj)
        if key not in self._obj2index:
            self._obj2index[key] = self._current
            self._index2obj[self._current] = key
            self._current += 1
        else:
            errmsg = "The object has already exist in the map system."
            raise ValueError(errmsg)
    # }}}


#This is a test of the IndexMap class.
if __name__ == "__main__":
    objects = [(0, 0), (0, 1), (1, 0), (1, 1)]
    testmap0 = IndexMap(objs=objects, start=0)
    print(testmap0)
    print("The length of this map: ", len(testmap0))
    for obj in sorted(objects):
        index = testmap0(obj)
        print("The index of ", obj, " is: ", index)
    print("=" * 60)
    for index in range(len(testmap0)):
        print("The {0}th object is{1}".format(index, testmap0(index)))
    print("=" * 60)


    testmap1 = IndexMap()
    print(testmap1)
    for obj in objects:
        testmap1.add(obj)
    print(testmap1)
    print("The length of this map: ", len(testmap1))
    for obj in sorted(objects):
        index = testmap1(obj)
        print("The index of ", obj, " is: ", index)
    print("=" * 60)
    for index in range(len(testmap1)):
        print("The {0}th object is{1}".format(index, testmap1(index)))
    print("=" * 60)
