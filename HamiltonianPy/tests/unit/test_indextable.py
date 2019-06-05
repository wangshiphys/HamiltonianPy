"""
A test script for the `indextable` module
"""


from random import randrange

import pytest

from HamiltonianPy.indextable import IndexTable


class TestIndexTable:
    def test_init(self):
        match0 = r"unhashable type"
        match1 = r"The .* has different type from the previous ones"
        match2 = r"The .* object already exists"
        with pytest.raises(TypeError, match=match0):
            IndexTable([[0, 1], [2, 3]])
        with pytest.raises(TypeError, match=match1):
            IndexTable([(0, 1), "ab"])
        with pytest.raises(ValueError, match=match2):
            IndexTable([(0, 1), (2, 3), (0, 1)])

    def test_object_type(self):
        table = IndexTable((x, y) for x in range(4) for y in range(4))
        assert table.object_type is tuple

    def test_str_and_iteration(self):
        separator = "*" * 80
        table = IndexTable((x, y) for x in range(2) for y in range(2))
        print(table)
        print(separator)

        for index in table.indices():
            print(index)
        print(separator)

        for item in table.objects():
            print(item)
        print(separator)

        for index, item in table:
            print(index, item)
        print(separator)

    def test_length(self):
        num0 = 4
        num1 = 7
        table = IndexTable((x, y) for x in range(num0) for y in range(num1))
        assert len(table) == num0 * num1

    def test_query_index(self):
        num0 = 7
        num1 = 3
        table = IndexTable((x, y) for x in range(num0) for y in range(num1))
        for i in range(5):
            key = (randrange(num0), randrange(num1))
            assert table(key) == key[0] * num1 + key[1]

    def test_query_object(self):
        num0 = 7
        num1 = 3
        table = IndexTable((x, y) for x in range(num0) for y in range(num1))
        for i in range(5):
            index = randrange(num0 * num1)
            assert table.query_object(index) == divmod(index, num1)
