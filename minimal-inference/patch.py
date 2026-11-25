"""The head patch: what FlexGaze sees of one head.

A virtual camera is turned to look straight at the head box's centre and its view is resampled into a
224 x 224 patch. The patch depends only on the image, the camera and the box, never on how far away
the head is:

1. ``c``: the unit ray through the box centre.
2. ``R``: the rotation whose rows are ``right``, ``down``, ``c``, with ``down = c x (1, 0, 0)``, so
   the patch keeps the camera's own x axis level (no head pose is used).
3. The window: the box's four edge midpoints are turned into rays, rotated by ``R`` and projected on
   the plane ``z = 1``; the window side is the shorter of the box's width and height measured there.
4. Patch pixel ``(j, i)`` looks along ``R^T (s (j + 0.5 - 112), s (i + 0.5 - 112), 1)`` with
   ``s = window / 224`` and reads the image bilinearly (``cv2.remap``) where that ray lands.

With lens distortion, the box points become rays through the lens model (``cv2.undistortPoints``)
and every patch ray is projected through it onto the image as taken (``cv2.projectPoints``), which is
then read once; the image is never undistorted first.

``box`` is ``(x0, y0, x1, y1)`` in continuous image coordinates where pixel ``u`` spans ``[u, u + 1)``,
which is what head detectors return. The camera's own conventions are described in :mod:`camera`.
"""

from typing import NamedTuple

import cv2
import numpy as np

from camera import Camera

PATCH_SIDE = 224
"""Side of the square patch the model reads, in pixels."""


class HeadPatch(NamedTuple):
    """One head's patch and the frame it was cut in.

    Attributes:
        pixels: ``(224, 224, 3)`` uint8, same channel order as the image it was cut from.
        rotation: ``(3, 3)`` float64 ``R`` taking camera-frame vectors into the patch frame (rows
            ``right``, ``down``, ``forward``). A direction predicted in the patch frame is
            ``R.T @ direction`` in the camera frame.
    """

    pixels: np.ndarray
    rotation: np.ndarray


def head_patch(image: np.ndarray, camera: Camera, box) -> HeadPatch:
    """Cut the patch of one head box out of an image.

    Args:
        image: ``(height, width, 3)`` uint8 image.
        camera: The camera that took ``image``. With lens distortion, the patch is read from
            ``image`` as taken, through the lens model.
        box: ``(x0, y0, x1, y1)`` head box, continuous image coordinates.

    Raises:
        ValueError: The image is not ``(height, width, 3)`` uint8, or the box is not four finite
            numbers with positive width and height.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a (height, width, 3) uint8 image, got {image.dtype} {image.shape}")
    rotation, window = patch_frame(camera, box)
    columns, rows = _source_positions(camera, rotation, window)
    # cv2.remap uses fixed-point weights, so patches differ slightly from the paper's float64 sampling.
    pixels = cv2.remap(
        image,
        columns.astype(np.float32),
        rows.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return HeadPatch(pixels, rotation)


def patch_frame(camera: Camera, box) -> tuple[np.ndarray, float]:
    """The rotation ``R`` and the window side (on the plane ``z = 1``) of one head box.

    Raises:
        ValueError: The box is not four finite numbers with positive width and height.
    """
    box = np.asarray(box, dtype=np.float64).reshape(4)
    x0, y0, x1, y1 = box
    if not (np.isfinite(box).all() and x1 > x0 and y1 > y0):
        raise ValueError(f"a head box needs positive width and height, got {box.tolist()}")
    middle_x, middle_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    forward = camera.ray(middle_x, middle_y)
    forward /= np.linalg.norm(forward)
    down = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    rotation = np.stack((right, down, forward))

    left_edge, right_edge, top_edge, bottom_edge = (
        rotation @ camera.ray(u, v)
        for u, v in ((x0, middle_y), (x1, middle_y), (middle_x, y0), (middle_x, y1))
    )
    width = abs(right_edge[0] / right_edge[2] - left_edge[0] / left_edge[2])
    height = abs(bottom_edge[1] / bottom_edge[2] - top_edge[1] / top_edge[2])
    return rotation, float(min(width, height))


def _source_positions(camera: Camera, rotation: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Where each patch pixel centre reads the image, as ``(columns, rows)`` with pixel centres at
    integers, through the camera's lens model. Rays pointing behind the camera read far outside the
    image."""
    focal = PATCH_SIDE / window
    j, i = np.meshgrid(np.arange(PATCH_SIDE, dtype=np.float64), np.arange(PATCH_SIDE, dtype=np.float64))
    u = (j + 0.5 - PATCH_SIDE / 2.0) / focal
    v = (i + 0.5 - PATCH_SIDE / 2.0) / focal
    rays = np.stack([u, v, np.ones_like(u)], -1) @ rotation  # R^T applied to every ray
    behind = ~(rays[..., 2] > 0.0)
    rays[behind] = (0.0, 0.0, 1.0)  # any ray in front, replaced below
    projected = camera.project(rays)
    projected[behind] = -1e6
    return projected[..., 0], projected[..., 1]
