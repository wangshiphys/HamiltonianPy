"""
A test script for the `indextable` module
"""


from random import randrange

import pytest

from HamiltonianPy.indextable import IndexTable


def test_IndexTable():
    with pytest.raises(TypeError):
        IndexTable([(1, 2), "34"])
    with pytest.raises(ValueError):
        IndexTable([(1, 2), (3, 4), (1, 2)])

    num0 = 7
    num1 = 3
    objects = ((x, y) for x in range(num0) for y in range(num1))
    table = IndexTable(objects)

    assert table.object_type == tuple
    assert len(table) == num0 * num1

    for i in range(5):
        key0 = (randrange(num0), randrange(num1))
        key1 = randrange(num0 * num1)
        assert table(key0) == key0[0] * num1 + key0[1]
        assert table(key1) == divmod(key1, num1)

    with pytest.raises(TypeError):
        table([1, 2])
    with pytest.raises(KeyError):
        table((num0, num1))
    with pytest.raises(KeyError):
        table(num0 * num1)
