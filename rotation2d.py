"""
Rotate 2D vectors about the axis perpendicular to the plane.
"""


__all__ = ["Rotation2D"]


import numpy as np


def Rotation2D(theta, *, deg=False):
    """
    Rotation about the axis perpendicular to the plane by theta angle.

    Parameters
    ----------
    theta : float
        The rotation angle.
    deg : bool, optional, keyword-only
        Whether the given `theta` is in degree or radian.
        Default: False.

    Returns
    -------
    R : (2, 2) orthogonal matrix
        The corresponding transformation matrix.
    """

    theta = (theta * np.pi / 180) if deg else theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
