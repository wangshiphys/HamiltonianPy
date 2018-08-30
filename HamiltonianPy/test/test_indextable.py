"""
A test script for the IndexTable class
"""


from random import randrange

from HamiltonianPy.indextable import IndexTable


num0 = 11
num1 = 3
objects = ((x, y) for x in range(num0) for y in range(num1))
table = IndexTable(objects)
for i in range(10):
    key0 = (randrange(num0), randrange(num1))
    key1 = randrange(num0 * num1)
    assert len(table) == num0 * num1
    assert table(key0) == key0[0] * num1 + key0[1]
    assert table(key1) == divmod(key1, num1)
    print("key = {0}\t value = {1}".format(key0, table(key0)))
    print("key = {0}\t value = {1}".format(key1, table(key1)))
    print("=" * 80)