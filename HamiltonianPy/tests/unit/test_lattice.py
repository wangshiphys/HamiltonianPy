"""
A test script for the lattice subpackage.
"""


import numpy as np
import pytest

from HamiltonianPy import lattice


@pytest.fixture(scope="module")
def points():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)

@pytest.fixture(scope="module")
def vectors():
    return np.array([[2, 0], [0, 2]], dtype=np.int64)

@pytest.fixture(scope="module")
def cluster(points, vectors):
    tmp = lattice.Lattice(points=points, vectors=vectors)
    print(tmp)
    return tmp


def test_set_float_point_precision():
    with pytest.raises(AssertionError):
        lattice.set_float_point_precision(1.0)
    with pytest.raises(AssertionError):
        lattice.set_float_point_precision(-1)

    lattice.set_float_point_precision(precision=4)
    assert lattice.lattice._ZOOM == 10000
    assert lattice.lattice._VIEW_AS_ZERO == 1E-4

    lattice.set_float_point_precision(precision=6)
    assert lattice.lattice._ZOOM == 1000000
    assert lattice.lattice._VIEW_AS_ZERO == 1E-6

    # Restore the default value
    lattice.set_float_point_precision()


class TestLattice:
    def test_init(self, cluster):
        with pytest.raises(ValueError, match="Not supported space dimension"):
            points = np.arange(16).reshape((4, 4))
            vectors = 2 * np.identity(4)
            lattice.Lattice(points=points, vectors=vectors)
        with pytest.raises(ValueError, match="duplicate points"):
            points = np.array([[0, 0], [0, 1], [1, 0], [1, 0]])
            vectors = np.array([[2, 0], [0, 2]])
            lattice.Lattice(points=points, vectors=vectors)
        with pytest.raises(ValueError, match="duplicate vectors"):
            points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            vectors = np.array([[2, 0], [2, 0]])
            lattice.Lattice(points=points, vectors=vectors)

        assert cluster.point_num == 4
        assert cluster.space_dim == 2
        assert cluster.trans_dim == 2
        assert np.all(cluster.bs == np.array([[np.pi, 0], [0, np.pi]]))

    def test_getIndex(self, points, vectors, cluster):
        with pytest.raises(AssertionError):
            cluster.getIndex([0, 0])
        with pytest.raises(AssertionError):
            cluster.getIndex(np.array([0, 0, 0]))

        for i in range(cluster.point_num):
            dR = np.matmul(np.random.randint(-5, 6, 2), vectors)
            assert cluster.getIndex(points[i], fold=False) == i
            assert cluster.getIndex(points[i] + dR, fold=True) == i

        with pytest.raises(KeyError, match="does not belong the lattice"):
            cluster.getIndex(np.array([3, 2]))

    def test_getSite(self, points, vectors, cluster):
        with pytest.raises(AssertionError):
            cluster.getSite(1.0)
        with pytest.raises(AssertionError):
            cluster.getSite(-1)
        with pytest.raises(AssertionError):
            cluster.getSite(8)

        for i in range(cluster.point_num):
            assert np.all(cluster.getSite(i) == points[i])

    def test_decompose(self, points, vectors, cluster):
        with pytest.raises(AssertionError):
            cluster.decompose([0, 0])
        with pytest.raises(AssertionError):
            cluster.decompose(np.array([0, 0, 0]))

        for i in range(cluster.point_num):
            site0 = points[i]
            dR0 = np.matmul(np.random.randint(-5, 5, 2), vectors)
            site1, dR1 = cluster.decompose(site0)
            site2, dR2 = cluster.decompose(site0 + dR0)
            assert np.all(site0 == site1) and np.all(dR1 == [0, 0])
            assert np.all(site0 == site2) and np.all(dR2 == dR0)

        with pytest.raises(RuntimeError, match="Failed to decompose"):
            cluster.decompose(np.array([0.5, 0.5]))

    def test_neighbor_distance(self, points, vectors, cluster):
        assert cluster.neighbor_distance(0) == 0.0
        assert cluster.neighbor_distance(1) == 1.0
        assert cluster.neighbor_distance(2) == 1.4142

    def test_bonds(self, points, vectors, cluster):
        def helper(bonds):
            bonds_indices = []
            for bond in bonds:
                p0, p1 = bond.endpoints
                index0 = cluster.getIndex(p0, fold=True)
                index1 = cluster.getIndex(p1, fold=True)
                bonds_indices.append(
                    (index0, index1) if index0 < index1 else (index1, index0)
                )
            bonds_indices.sort()
            return bonds_indices

        intra_bonds, inter_bonds = cluster.bonds(nth=1)
        intra_bonds_indices = helper(intra_bonds)
        inter_bonds_indices = helper(inter_bonds)
        assert len(intra_bonds) == 4
        assert len(inter_bonds) == 4
        assert intra_bonds_indices == [(0, 1), (0, 2), (1, 3), (2, 3)]
        assert inter_bonds_indices == [(0, 1), (0, 2), (1, 3), (2, 3)]

        intra_bonds, inter_bonds = cluster.bonds(nth=2)
        intra_bonds_indices = helper(intra_bonds)
        inter_bonds_indices = helper(inter_bonds)
        assert len(intra_bonds) == 2
        assert len(inter_bonds) == 6
        assert intra_bonds_indices == [(0, 3), (1, 2)]
        assert inter_bonds_indices == [
            (0, 3), (0, 3), (0, 3), (1, 2), (1, 2), (1, 2)
        ]

        intra_bonds, inter_bonds = cluster.bonds(nth=2, only=False)
        intra_bonds_indices = helper(intra_bonds)
        inter_bonds_indices = helper(inter_bonds)
        assert len(intra_bonds) == 6
        assert len(inter_bonds) == 10
        assert intra_bonds_indices == [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
        ]
        assert inter_bonds_indices == [
            (0, 1), (0, 2), (0, 3), (0, 3), (0, 3),
            (1, 2), (1, 2), (1, 2), (1, 3), (2, 3),
        ]

    def test_in_cluster(self, points, vectors, cluster):
        for i in range(cluster.point_num):
            assert cluster.in_cluster(points[i])
        assert not cluster.in_cluster(np.array([0.5, 0.5]))


def test_KPath():
    anchor_points = np.array([[0, 0], [1, 0], [1, 1]])
    with pytest.raises(AssertionError):
        lattice.KPath(anchor_points, min_num=0)
    with pytest.raises(AssertionError, match="At least two points"):
        lattice.KPath(np.array([[0, 0]]))
    with pytest.raises(AssertionError):
        lattice.KPath(np.array([[0, 0, 0, 0], [1, 0, 0, 0]]))

    kpoints, indices = lattice.KPath(anchor_points, min_num=1, loop=False)
    assert np.all(kpoints == anchor_points) and indices == [0, 1, 2]
    kpoints, indices = lattice.KPath(anchor_points, min_num=1, loop=True)
    assert np.all(kpoints == np.array([[0, 0], [1, 0], [1, 1], [0, 0]]))
    assert indices == [0, 1, 2, 3]


    kpoints, indices = lattice.KPath(anchor_points, min_num=2, loop=False)
    assert np.all(
        kpoints == np.array([[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1]])
    )
    assert indices == [0, 2, 4]

    kpoints, indices = lattice.KPath(anchor_points, min_num=2, loop=True)
    assert np.all(
        kpoints == np.array(
            [[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 0.5], [0, 0]]
        )
    )
    assert indices == [0, 2, 4, 6]
