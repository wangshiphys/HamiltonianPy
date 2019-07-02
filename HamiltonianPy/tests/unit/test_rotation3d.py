"""
A test script for the `rotation3d` module.
"""


import numpy as np
import pytest

from HamiltonianPy.rotation3d import *


@pytest.fixture(scope="module")
def RX0():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

@pytest.fixture(scope="module")
def RX30():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.sqrt(3) / 2, -0.5],
            [0.0, 0.5, np.sqrt(3) / 2],
        ]
    )

@pytest.fixture(scope="module")
def RX45():
    return np.array(
        [
            [1.0, 0.0,0.0],
            [0.0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
            [0.0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
        ]
    )

@pytest.fixture(scope="module")
def RY0():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

@pytest.fixture(scope="module")
def RY30():
    return np.array(
        [
            [np.sqrt(3) / 2, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [-0.5, 0.0, np.sqrt(3) / 2],
        ]
    )

@pytest.fixture(scope="module")
def RY45():
    return np.array(
        [
            [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)],
            [0.0, 1.0, 0.0],
            [-1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)],
        ]
    )

@pytest.fixture(scope="module")
def RZ0():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

@pytest.fixture(scope="module")
def RZ30():
    return np.array(
        [
            [np.sqrt(3) / 2, -0.5, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

@pytest.fixture(scope="module")
def RZ45():
    return np.array(
        [
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_RotationX(RX0, RX30, RX45):
    assert np.allclose(RX0, RotationX(0))
    assert np.allclose(RX0, RotationX(0, deg=True))
    assert np.allclose(-RX0, RotationX(0, inversion=True))

    assert np.allclose(RX30, RotationX(np.pi / 6))
    assert np.allclose(RX30, RotationX(30, deg=True))
    assert np.allclose(-RX30, RotationX(np.pi / 6, inversion=True))

    assert np.allclose(RX45, RotationX(np.pi / 4))
    assert np.allclose(RX45, RotationX(45, deg=True))
    assert np.allclose(-RX45, RotationX(np.pi / 4, inversion=True))

    components = (RX45, RX45, RX45)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationX(3 * np.pi / 4)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationX(135, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationX(3 * np.pi / 4, inversion=True)
    )

    components = (RX45, RX30, RX45, RX30, RX45)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationX(13 * np.pi / 12)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationX(195, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationX(13 * np.pi / 12, inversion=True)
    )

    components = (RX45, RX30, RX45, RX30, RX45, RX30, RX45, RX30)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationX(5 * np.pi / 3)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationX(300, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationX(5 * np.pi / 3, inversion=True)
    )


def test_RotationY(RY0, RY30, RY45):
    assert np.allclose(RY0, RotationY(0))
    assert np.allclose(RY0, RotationY(0, deg=True))
    assert np.allclose(-RY0, RotationY(0, inversion=True))

    assert np.allclose(RY30, RotationY(np.pi / 6))
    assert np.allclose(RY30, RotationY(30, deg=True))
    assert np.allclose(-RY30, RotationY(np.pi / 6, inversion=True))

    assert np.allclose(RY45, RotationY(np.pi / 4))
    assert np.allclose(RY45, RotationY(45, deg=True))
    assert np.allclose(-RY45, RotationY(np.pi / 4, inversion=True))

    components = (RY45, RY45, RY45)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationY(3 * np.pi / 4)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationY(135, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationY(3 * np.pi / 4, inversion=True)
    )

    components = (RY45, RY30, RY45, RY30, RY45)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationY(13 * np.pi / 12)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationY(195, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationY(13 * np.pi / 12, inversion=True)
    )

    components = (RY45, RY30, RY45, RY30, RY45, RY30, RY45, RY30)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationY(5 * np.pi / 3)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationY(300, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationY(5 * np.pi / 3, inversion=True)
    )


def test_RotationZ(RZ0, RZ30, RZ45):
    assert np.allclose(RZ0, RotationZ(0))
    assert np.allclose(RZ0, RotationZ(0, deg=True))
    assert np.allclose(-RZ0, RotationZ(0, inversion=True))

    assert np.allclose(RZ30, RotationZ(np.pi / 6))
    assert np.allclose(RZ30, RotationZ(30, deg=True))
    assert np.allclose(-RZ30, RotationZ(np.pi / 6, inversion=True))

    assert np.allclose(RZ45, RotationZ(np.pi / 4))
    assert np.allclose(RZ45, RotationZ(45, deg=True))
    assert np.allclose(-RZ45, RotationZ(np.pi / 4, inversion=True))

    components = (RZ45, RZ45, RZ45)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationZ(3 * np.pi / 4)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationZ(135, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationZ(3 * np.pi / 4, inversion=True)
    )

    components = (RZ45, RZ30, RZ45, RZ30, RZ45)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationZ(13 * np.pi / 12)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationZ(195, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationZ(13 * np.pi / 12, inversion=True)
    )

    components = (RZ45, RZ30, RZ45, RZ30, RZ45, RZ30, RZ45, RZ30)
    assert np.allclose(
        np.linalg.multi_dot(components), RotationZ(5 * np.pi / 3)
    )
    assert np.allclose(
        np.linalg.multi_dot(components), RotationZ(300, deg=True)
    )
    assert np.allclose(
        -np.linalg.multi_dot(components),
        RotationZ(5 * np.pi / 3, inversion=True)
    )


def test_RotationGeneral():
    theta = (2 * np.random.random() - 1) * np.pi
    assert np.allclose(
        RotationX(theta), RotationGeneral((1, 0, 0), theta)
    )
    assert np.allclose(
        RotationY(theta), RotationGeneral((0, 1, 0), theta)
    )
    assert np.allclose(
        RotationZ(theta), RotationGeneral((0, 0, 1), theta)
    )

    alpha = (2 * np.random.random() - 1) * np.pi
    beta = np.random.random() * np.pi
    theta = (2 * np.random.random() - 1) * np.pi
    R0 = RotationGeneral((alpha, beta), theta)
    R1 = np.linalg.multi_dot(
        [
            RotationZ(alpha), RotationY(beta), RotationZ(theta),
            RotationY(-beta), RotationZ(-alpha),
        ]
    )
    assert np.allclose(R0, R1)


def test_RotationEuler():
    alpha, beta, gamma = (2 * np.random.random(3) - 1) * np.pi
    R0 = RotationEuler(alpha, beta, gamma)
    R1 = np.linalg.multi_dot(
        [RotationZ(alpha), RotationY(beta), RotationZ(gamma)]
    )
    assert np.allclose(R0, R1)
