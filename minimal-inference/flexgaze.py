"""FlexGaze: an image, its camera and head boxes in, one 3D gaze prediction per head out."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike

import numpy as np
import onnxruntime as ort

from camera import Camera
from patch import head_patch
from weights import flexgaze_onnx_path


@dataclass(frozen=True)
class GazePrediction:
    """The gaze of one head.

    The camera frame is OpenCV's: x to the right of the image, y down, z forward from the camera.

    Attributes:
        box: ``(x0, y0, x1, y1)``, the head box the prediction was made from.
        direction: ``(x, y, z)`` unit vector from the head towards what it looks at, camera frame.
        kappa: Concentration of the predicted von Mises-Fisher distribution around ``direction``;
            larger is more certain.
    """

    box: tuple[float, float, float, float]
    direction: tuple[float, float, float]
    kappa: float

    @property
    def yaw_degrees(self) -> float:
        """Horizontal gaze angle: 0 when looking back along the optical axis towards the camera,
        positive when looking towards the image's right."""
        x, _, z = self.direction
        return math.degrees(math.atan2(x, -z))

    @property
    def pitch_degrees(self) -> float:
        """Vertical gaze angle: positive when looking up."""
        return math.degrees(math.asin(max(-1.0, min(1.0, -self.direction[1]))))


class FlexGaze:
    """A FlexGaze model ready to predict.

    Args:
        model: ``"base"`` (FlexGaze-B) or ``"small"`` (FlexGaze-S) for a released model, downloaded
            from Hugging Face on first use, or the path of a FlexGaze ``.onnx`` file.
        seed: Which of the three released training seeds (0, 1 or 2) to download. Required when
            ``model`` is a released model, and ignored when it is a path.
        cache_dir: The folder downloaded models are kept in. Required when ``model`` is a released
            model, and ignored when it is a path.
        providers: onnxruntime execution providers, CPU by default.
    """

    def __init__(
        self,
        model: str | PathLike,
        *,
        seed: int | None = None,
        cache_dir: str | PathLike | None = None,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.model_path = flexgaze_onnx_path(model, seed, cache_dir)
        self.session = ort.InferenceSession(str(self.model_path), providers=list(providers))

    def predict(self, image: np.ndarray, camera: Camera, boxes) -> list[GazePrediction]:
        """Predict the gaze of every head box in one image.

        Args:
            image: ``(height, width, 3)`` uint8 RGB image.
            camera: The camera that took ``image``. With lens distortion, each patch is read from
                ``image`` through the lens model, so the image is passed as taken.
            boxes: Head boxes ``(x0, y0, x1, y1)`` in pixels, one per row; a single box is accepted.

        Returns:
            One prediction per box, in the order given.

        Raises:
            ValueError: ``boxes`` is not ``(n, 4)`` rows, a box has no width or height, or ``image``
                is not ``(height, width, 3)`` uint8.
        """
        boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
        if boxes.size == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"boxes are (n, 4) rows x0, y0, x1, y1, got shape {boxes.shape}; "
                "keep the first four columns of a detector's (n, 5) output"
            )
        patches = [head_patch(image, camera, box) for box in boxes]
        stack = np.ascontiguousarray(np.stack([p.pixels.transpose(2, 0, 1) for p in patches]))
        directions, kappas = self.session.run(None, {"image": stack})
        predictions = []
        for box, patch, direction, kappa in zip(boxes, patches, directions, kappas, strict=True):
            in_camera = patch.rotation.T @ direction.astype(np.float64)
            predictions.append(
                GazePrediction(
                    box=tuple(float(v) for v in box),
                    direction=tuple(float(v) for v in in_camera / np.linalg.norm(in_camera)),
                    kappa=float(kappa),
                )
            )
        return predictions
