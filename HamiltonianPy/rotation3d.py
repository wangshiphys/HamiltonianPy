"""
Orthogonal group in 3 dimensions.

Both proper rotations and improper rotations are included;
For proper rotations, the determinant of the transformation matrix R is 1;
For improper rotations, the determinant of the transformation matrix R is -1.

The convention we follow throughout this module is that a rotation operation
affects a physical system itself while the coordinate axes remain unchanged.
The rotation-angle '$\\theta$' is taken to be positive when the rotation in
question is counterclockwise in the plane perpendicular to rotation-axis, as
viewed from the positive side of the rotation-axis.
"""


__all__ = [
    "E", "INVERSION",
    "RotationX", "RotationY", "RotationZ",
    "RotationGeneral", "RotationEuler",
]


import numpy as np


# Useful global constant
_VIEW_AS_ZERO = 1E-10
################################################################################


def set_threshold(threshold=1E-10):
    """
    Set the threshold for viewing a number as zero.

    If the absolute value of a number is less than the given `threshold`,
    then the number is viewed as zero.

    Parameters
    ----------
    threshold : float, optional
        The threshold value.
        Default: 1E-10.
    """

    assert isinstance(threshold, float) and threshold >= 0
    global _VIEW_AS_ZERO
    _VIEW_AS_ZERO = threshold


# Identity operation
E = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
# Inversion transformation
INVERSION = np.array(
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)


def RotationX(theta, *, deg=False, inversion=False):
    """
    Rotation about the x-axis.

    Parameters
    ----------
    theta : int or float
        The rotation-angle.
    deg : bool, optional, keyword-only
        Whether the given `theta` is in degree or radian.
        Default: False.
    inversion : bool, optional, keyword-only
        Whether inversion transformation is involved.
        Default : False.

    Returns
    -------
    R : (3, 3) orthogonal matrix
        The corresponding transformation matrix.
    """

    theta = (theta * np.pi / 180) if deg else theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ]
    )
    return np.matmul(R, INVERSION) if inversion else R


def RotationY(theta, *, deg=False, inversion=False):
    """
    Rotation about the y-axis.

    Parameters
    ----------
    theta : int or float
        The rotation-angle.
    deg : bool, optional, keyword-only
        Whether the given `theta` is in degree or radian.
        Default: False.
    inversion : bool, optional, keyword-only
        Whether inversion transformation is involved.
        Default : False.

    Returns
    -------
    R : (3, 3) orthogonal matrix
        The corresponding transformation matrix.
    """

    theta = (theta * np.pi / 180) if deg else theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    R = np.array(
        [
            [cos_theta, 0.0, sin_theta],
            [0.0, 1.0, 0.0],
            [-sin_theta, 0.0, cos_theta],
        ]
    )
    return np.matmul(R, INVERSION) if inversion else R


def RotationZ(theta, *, deg=False, inversion=False):
    """
    Rotation about the z-axis.

    Parameters
    ----------
    theta : int or float
        The rotation-angle.
    deg : bool, optional, keyword-only
        Whether the given `theta` is in degree or radian.
        Default: False.
    inversion : bool, optional, keyword-only
        Whether inversion transformation is involved.
        Default : False.

    Returns
    -------
    R : (3, 3) orthogonal matrix
        The corresponding transformation matrix.
    """

    theta = (theta * np.pi / 180) if deg else theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    R = np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return np.matmul(R, INVERSION) if inversion else R


def RotationGeneral(axis, theta, *, deg=False, inversion=False):
    """
    Rotation about the given `axis`.

    Parameters
    ----------
    axis : list, tuple or 1D np.ndarray
        The rotation-axis.
        The rotation-axis can be specified in two forms:
        1. In Cartesian coordinate system: (x, y, z)
           specify the vector along the rotation-axis;
        2. Spherical coordinate system: (alpha, beta)
           where alpha is the angle between positive-x and the projection of
           the rotation-axis in the xy-plane; beta is the angle between
           positive-z and the rotation-axis.
    theta : int or float
        The rotation-angle.
    deg : bool, optional, keyword-only
        Whether the given angles are in degree or radian.
        Default: False.
    inversion : bool, optional, keyword-only
        Whether inversion transformation is involved.
        Default : False.

    Returns
    -------
    R : (3, 3) orthogonal matrix
        The corresponding transformation matrix.
    """

    assert isinstance(axis, (list, tuple, np.ndarray))

    if len(axis) == 2:
        alpha, beta = axis
        angles = np.array([alpha, beta, theta])
        angles = (angles * np.pi / 180) if deg else angles
        sin_alpha, sin_beta, sin_theta = np.sin(angles)
        cos_alpha, cos_beta, cos_theta = np.cos(angles)
    elif len(axis) == 3:
        x, y, z = axis
        dxy = np.sqrt(x ** 2 + y ** 2)
        dr = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if dr < _VIEW_AS_ZERO:
            raise ValueError(
                "The given `axis` is a zero vector: {0}".format(axis)
            )

        if dxy < _VIEW_AS_ZERO:
            sin_alpha = 0.0
            cos_alpha = 1.0
        else:
            sin_alpha = y / dxy
            cos_alpha = x / dxy
        sin_beta = dxy / dr
        cos_beta = z / dr
        theta = (theta * np.pi / 180) if deg else theta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
    else:
        raise ValueError("Invalid `axis` parameter!")

    RzAlpha = np.array(
        [
            [cos_alpha, -sin_alpha, 0.0],
            [sin_alpha, cos_alpha, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    RyBeta = np.array(
        [
            [cos_beta, 0.0, sin_beta],
            [0.0, 1.0, 0.0],
            [-sin_beta, 0.0, cos_beta],
        ]
    )
    RzTheta = np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = np.linalg.multi_dot([RzAlpha, RyBeta, RzTheta, RyBeta.T, RzAlpha.T])
    return np.matmul(R, INVERSION) if inversion else R


def RotationEuler(alpha, beta, gamma, *, deg=False, inversion=False):
    """
    Euler rotation.

    Parameters
    ----------
    alpha, beta, gamma : int or float
        The three Euler angles.
    deg : bool, optional, keyword-only
        Whether the given angles are in degree or radian.
        Default: False.
    inversion : bool, optional, keyword-only
        Whether inversion transformation is involved.
        Default : False.

    Returns
    -------
    R : (3, 3) orthogonal matrix
        The corresponding transformation matrix.
    """

    angles = np.array([alpha, beta, gamma])
    angles = (angles * np.pi / 180) if deg else angles
    sin_alpha, sin_beta, sin_gamma = np.sin(angles)
    cos_alpha, cos_beta, cos_gamma = np.cos(angles)

    RzAlpha = np.array(
        [
            [cos_alpha, -sin_alpha, 0.0],
            [sin_alpha, cos_alpha, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    RyBeta = np.array(
        [
            [cos_beta, 0.0, sin_beta],
            [0.0, 1.0, 0.0],
            [-sin_beta, 0.0, cos_beta],
        ]
    )
    RzGamma = np.array(
        [
            [cos_gamma, -sin_gamma, 0.0],
            [sin_gamma, cos_gamma, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = np.linalg.multi_dot([RzAlpha, RyBeta, RzGamma])
    return np.matmul(R, INVERSION) if inversion else R
