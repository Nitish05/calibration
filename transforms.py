"""
Pixel-to-world coordinate transforms using AprilTag pose.

Single source of truth for camera intrinsics and the projection math
that converts pixel contours into real-world mm coordinates on the
Z=0 board plane (relative to the AprilTag origin).
"""

import numpy as np

# Camera intrinsics from calibration (Calib_board_outer.toml, 1920x1080)
FX = 1510.838352442521
FY = 1505.9514145965773
CX = 959.5
CY = 539.5

# AprilTag settings
TAG_SIZE = 0.05684  # 56.84 mm in meters

CAMERA_MATRIX = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0],
])


def pixels_to_world(contour_pixels: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert pixel coordinates to world XY (mm) on the Z=0 plane.

    Parameters
    ----------
    contour_pixels : (N, 2) array of (u, v) pixel coordinates
    R : (3, 3) rotation matrix  (camera <- tag)
    t : (3, 1) translation vector (camera <- tag)

    Returns
    -------
    (N, 2) array of (x_mm, y_mm) in AprilTag world frame
    """
    pts = np.asarray(contour_pixels, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)

    # Un-project pixels to normalized camera rays
    rays_cam = np.stack([
        (pts[:, 0] - CX) / FX,
        (pts[:, 1] - CY) / FY,
        np.ones(len(pts)),
    ], axis=1)  # (N, 3)

    # Transform rays into tag frame: r_tag = R^T (r_cam - t)
    t_flat = t.flatten()  # (3,)
    R_inv = R.T  # orthogonal so R^-1 = R^T

    # Camera origin in tag frame
    origin_tag = -R_inv @ t_flat  # (3,)

    # Ray directions in tag frame
    dirs_tag = (R_inv @ rays_cam.T).T  # (N, 3)

    # Intersect with Z=0 plane:  origin_tag.z + s * dirs_tag[:, 2] = 0
    s = -origin_tag[2] / dirs_tag[:, 2]  # (N,)

    world_pts = origin_tag[np.newaxis, :2] + s[:, np.newaxis] * dirs_tag[:, :2]

    # Convert meters to millimeters
    return world_pts * 1000.0
