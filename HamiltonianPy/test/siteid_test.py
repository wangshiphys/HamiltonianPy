import numpy as np

from HamiltonianPy.indexmap import IndexMap
from HamiltonianPy.termofH import SiteID

#sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])

siteids = []
for site in sites:
    siteid = SiteID(site=site)
    siteids.append(siteid)
    print(siteid)
    print("The hash value: ", hash(siteid))
    print("The tupleform: ", siteid.tupleform())
    print("The corresponding site: ", siteid.getSite())
    print("=" * 30)
sitemap = IndexMap(siteids)
for siteid in siteids:
    print("The index: ", siteid.getIndex(sitemap))
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
