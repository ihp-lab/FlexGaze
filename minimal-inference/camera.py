"""The camera that took an image: its pinhole matrix, its lens distortion, and the conversions between
image positions and camera rays.

FlexGaze needs the camera to know the direction in which it sees each head. Without a calibration,
:meth:`Camera.from_fov` and :meth:`Camera.from_focal_length` give a pinhole camera with its principal
point at the image centre. A wrong focal length tilts every prediction by roughly the error in the
angle at which the head is seen. The tilt is zero for a head at the image centre and several degrees
for a head near the edge of a wide or narrow lens.

Coordinate conventions:

- ``matrix`` is in OpenCV's convention, where the centre of pixel ``(u, v)`` is at ``(u, v)``. This
  is what ``cv2.calibrateCamera`` returns.
- ``distortion`` holds OpenCV's lens coefficients ``(k1, k2, p1, p2[, k3[, ...]])``, as
  ``cv2.calibrateCamera`` returns them; ``None`` or all zeros is a pinhole camera.
- Rays are camera-frame directions scaled to the plane ``z = 1``: x to the right of the image, y
  down, z forward from the camera.
"""

import math

import cv2
import numpy as np


class Camera:
    """A pinhole camera matrix and, optionally, lens distortion coefficients.

    Args:
        matrix: ``(3, 3)`` camera matrix in OpenCV's convention.
        distortion: OpenCV lens coefficients of the camera, or ``None`` for a pinhole camera or an
            image already undistorted. All zeros is the same as ``None``.

    Attributes:
        matrix: ``(3, 3)`` float64 camera matrix, read-only.
        distortion: Flat float64 lens coefficients, read-only, or ``None`` for a pinhole camera.
    """

    def __init__(self, matrix, distortion=None) -> None:
        self.matrix = _read_only(np.array(matrix, dtype=np.float64).reshape(3, 3))
        lens = None if distortion is None else np.array(distortion, dtype=np.float64).reshape(-1)
        self.distortion = _read_only(lens) if lens is not None and np.any(lens != 0) else None

    @classmethod
    def from_fov(cls, width: int, height: int, diagonal_fov_degrees: float) -> "Camera":
        """A pinhole camera for a ``width`` x ``height`` image with the given diagonal field of view,
        its principal point at the image centre.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            diagonal_fov_degrees: Field of view across the image diagonal.

        Raises:
            ValueError: The field of view is not a finite number between 0 and 180 degrees.
        """
        fov = float(diagonal_fov_degrees)
        if not (math.isfinite(fov) and 0.0 < fov < 180.0):
            raise ValueError(f"a field of view is between 0 and 180 degrees, got {fov}")
        focal_length = math.hypot(width, height) / 2.0 / math.tan(math.radians(fov) / 2.0)
        return cls.from_focal_length(width, height, focal_length)

    @classmethod
    def from_focal_length(cls, width: int, height: int, focal_length: float) -> "Camera":
        """A pinhole camera for a ``width`` x ``height`` image with the given focal length, its
        principal point at the image centre.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            focal_length: Focal length in pixels, the same on both axes.

        Raises:
            ValueError: The focal length is not a positive finite number.
        """
        focal_length = float(focal_length)
        if not (math.isfinite(focal_length) and focal_length > 0.0):
            raise ValueError(f"a focal length is a positive finite number, got {focal_length}")
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        return cls(np.array([[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]]))

    def ray(self, u, v) -> np.ndarray:
        """The ``(3,)`` ray, on the plane ``z = 1``, through the image position ``(u, v)``.

        ``(u, v)`` is in continuous image coordinates, where pixel ``u`` spans ``[u, u + 1)``: the
        convention of head boxes. With lens distortion, the position is undistorted through the lens
        model (``cv2.undistortPoints``).
        """
        if self.distortion is not None:
            centred = np.array([[[u - 0.5, v - 0.5]]])
            x, y = cv2.undistortPoints(centred, self.matrix, self.distortion).reshape(2)
            return np.array([x, y, 1.0])
        return np.linalg.solve(self.matrix, np.array([u - 0.5, v - 0.5, 1.0]))

    def project(self, points: np.ndarray) -> np.ndarray:
        """Where camera-frame ``points`` of shape ``(..., 3)``, all in front of the camera, appear in
        the image, as ``(..., 2)`` positions in OpenCV's convention (pixel centres at integers).

        With lens distortion, the points are projected through the lens model
        (``cv2.projectPoints``).
        """
        points = np.asarray(points, dtype=np.float64)
        if self.distortion is not None:
            zero = np.zeros(3)
            projected, _ = cv2.projectPoints(
                points.reshape(-1, 1, 3), zero, zero, self.matrix, self.distortion
            )
            return projected.reshape(points.shape[:-1] + (2,))
        K = self.matrix
        x = points[..., 0] / points[..., 2]
        y = points[..., 1] / points[..., 2]
        return np.stack([K[0, 0] * x + K[0, 1] * y + K[0, 2], K[1, 1] * y + K[1, 2]], -1)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array
