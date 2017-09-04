import sys
import os.path
sys.path.append(os.path.abspath("../"))

import numpy as np

from indextable import IndexTable
from termofH import SiteID

sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])

siteids = []
for site in sites:
    siteid = SiteID(site=site)
    siteids.append(siteid)
    print(siteid)
    print("The hash value: ", hash(siteid))
    print("The tupleform: ", siteid._tupleform)
    print("The corresponding site: ", siteid.site)
    print("=" * 30)
indices_table = IndexTable(siteids)

for siteid in siteids:
    print("The index: ", siteid.getIndex(indices_table))
    print("=" * 40)

for id0 in siteids:
    for id1 in siteids:
        print("id0 = {0}, id1 = {1}, id0 < id1 is {2}".format(id0, id1, id0 < id1))
        print("id0 = {0}, id1 = {1}, id0 == id1 is {2}".format(id0, id1, id0 == id1))
        print("id0 = {0}, id1 = {1}, id0 > id1 is {2}".format(id0, id1, id0 > id1))
        print("id0 = {0}, id1 = {1}, id0 <= id1 is {2}".format(id0, id1, id0 <= id1))
        print("id0 = {0}, id1 = {1}, id0 >= id1 is {2}".format(id0, id1, id0 >= id1))
        print("id0 = {0}, id1 = {1}, id0 != id1 is {2}".format(id0, id1, id0 != id1))
        print("=" * 80)
