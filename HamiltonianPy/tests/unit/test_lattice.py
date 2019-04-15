"""
A test script for the lattice module
"""


import numpy as np
import pytest

from HamiltonianPy import lattice


def test_set_float_point_precision():
    with pytest.raises(AssertionError):
        lattice.set_float_point_precision(-1)

    lattice.set_float_point_precision(precision=4)
    assert lattice._ZOOM == 10000
    assert lattice._VIEW_AS_ZERO == 1E-4

    lattice.set_float_point_precision(precision=6)
    assert lattice._ZOOM == 1000000
    assert lattice._VIEW_AS_ZERO == 1E-6


def test_init():
    with pytest.raises(ValueError):
        points = np.array([[0, 0], [0, 1], [1, 0], [1, 0]])
        vectors = np.array([[2, 0], [0, 2]])
        lattice.Lattice(points=points, vectors=vectors)
    with pytest.raises(ValueError):
        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        vectors = np.array([[2, 0], [2, 0]])
        lattice.Lattice(points=points, vectors=vectors)

    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    vectors = np.array([[2, 0], [0, 2]])
    cluster = lattice.Lattice(points=points, vectors=vectors)

    assert np.allclose(cluster._all_distances, np.array([0, 1, np.sqrt(2)]))
    assert cluster.point_num == 4
    assert cluster.space_dim == 2
    assert cluster.trans_dim == 2
    assert np.all(cluster.points == points)
    assert np.all(cluster.vectors == vectors)
    assert np.all(cluster.bs == np.array([[np.pi, 0], [0, np.pi]]))


def test_getIndex():
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    vectors = np.array([[2, 0], [0, 2]])
    cluster = lattice.Lattice(points=points, vectors=vectors)

    for i in range(cluster.point_num):
        dR = np.matmul(np.random.randint(-5, 5, 2), vectors)
        assert cluster.getIndex(points[i], fold=False) == i
        assert cluster.getIndex(points[i] + dR, fold=True) == i

    with pytest.raises(KeyError):
        cluster.getIndex(np.array([3, 2]))


def test_getSite():
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    vectors = np.array([[2, 0], [0, 2]])
    cluster = lattice.Lattice(points=points, vectors=vectors)

    with pytest.raises(AssertionError, match="should be an integer"):
        cluster.getSite(1.0)
    with pytest.raises(ValueError, match="none negative integer"):
        cluster.getSite(-1)
    with pytest.raises(ValueError, match="larger than the number of sites"):
        cluster.getSite(4)

    for i in range(cluster.point_num):
        assert np.all(cluster.getSite(i) == points[i])


def test_decompose():
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    vectors = np.array([[2, 0], [0, 2]])
    cluster = lattice.Lattice(points=points, vectors=vectors)

    with pytest.raises(AssertionError):
        cluster.decompose([0, 0])
    with pytest.raises(AssertionError):
        cluster.decompose(np.array([0, 0, 0]))

    for i in range(cluster.point_num):
        dR0 = np.matmul(np.random.randint(-5, 5, 2), vectors)
        site, dR1 = cluster.decompose(points[i] + dR0)
        assert np.all(site == points[i]) and np.all(dR0 == dR1)

    with pytest.raises(RuntimeError, match="Failed to decompose"):
        cluster.decompose(np.array([0.5, 0.5]))


def test_neighbor_distance():
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    vectors = np.array([[2, 0], [0, 2]])
    cluster = lattice.Lattice(points=points, vectors=vectors)

    assert cluster.neighbor_distance(0) == 0.0
    assert cluster.neighbor_distance(1) == 1.0
    assert np.allclose(cluster.neighbor_distance(2), np.sqrt(2))


def _test_bonds_helper(bonds, cluster):
    bonds_indices = []
    for bond in bonds:
        p0, p1 = bond.getEndpoints()
        index0 = cluster.getIndex(p0, fold=True)
        index1 = cluster.getIndex(p1, fold=True)
        tmp = (index0, index1) if index0 < index1 else (index1, index0)
        bonds_indices.append(tmp)
    bonds_indices.sort()
    return bonds_indices


def test_bonds():
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    vectors = np.array([[2, 0], [0, 2]])
    cluster = lattice.Lattice(points=points, vectors=vectors)

    intra_bonds, inter_bonds = cluster.bonds(nth=1)
    intra_bonds_indices = _test_bonds_helper(intra_bonds, cluster)
    inter_bonds_indices = _test_bonds_helper(inter_bonds, cluster)
    assert len(intra_bonds)  == 4
    assert len(inter_bonds) == 4
    assert intra_bonds_indices == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert inter_bonds_indices == [(0, 1), (0, 2), (1, 3), (2, 3)]

    intra_bonds, inter_bonds = cluster.bonds(nth=2)
    intra_bonds_indices = _test_bonds_helper(intra_bonds, cluster)
    inter_bonds_indices = _test_bonds_helper(inter_bonds, cluster)
    assert len(intra_bonds)  == 2
    assert len(inter_bonds) == 6
    assert intra_bonds_indices == [(0, 3), (1, 2)]
    assert inter_bonds_indices == [
        (0, 3), (0, 3), (0, 3), (1, 2), (1, 2), (1, 2)
    ]

    intra_bonds, inter_bonds = cluster.bonds(nth=2, only=False)
    intra_bonds_indices = _test_bonds_helper(intra_bonds, cluster)
    inter_bonds_indices = _test_bonds_helper(inter_bonds, cluster)
    assert len(intra_bonds)  == 6
    assert len(inter_bonds) == 10
    assert intra_bonds_indices == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
    ]
    assert inter_bonds_indices == [
        (0, 1), (0, 2), (0, 3), (0, 3), (0, 3),
        (1, 2), (1, 2), (1, 2), (1, 3), (2, 3)
    ]


def test_in_cluster():
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    vectors = np.array([[2, 0], [0, 2]])
    cluster = lattice.Lattice(points=points, vectors=vectors)

    for i in range(cluster.point_num):
        assert cluster.in_cluster(points[i])

    assert not cluster.in_cluster(np.array([0.5, 0.5]))