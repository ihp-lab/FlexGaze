# FlexGaze minimal inference

This folder holds the smallest complete code for running FlexGaze. Given an image, it returns a 3D
gaze direction and a confidence (von Mises-Fisher concentration) for every head in it.

## Setting up

The code needs Python 3.10 or newer, NumPy, OpenCV and ONNX Runtime.

```bash
git clone https://github.com/ihp-lab/FlexGaze.git
cd FlexGaze/minimal-inference
pip install -r requirements.txt
```

Models are downloaded on first use into the folder given by `--cache-dir` (by default a `flexgaze`
folder in the system's temporary directory) or by `cache_dir` in Python.

## Running from the command line

```console
$ python cli.py photo.jpg
head 0: box=(151.6, 6.9, 308.3, 192.0) direction=(-0.0881, +0.2940, -0.9517) yaw=-5.3 pitch=-17.1 kappa=951.62
```

| Option | Meaning |
| --- | --- |
| `--model small\|base\|PATH`, `--seed 0\|1\|2` | FlexGaze model |
| `--head-crop` | the image is already a head crop |
| `--fov DEG`, `--intrinsics FX,FY,CX,CY`, `--distortion K1,K2,P1,P2[,K3,...]` | camera (see [Camera intrinsics](#camera-intrinsics)) |
| `--cache-dir PATH` | folder for downloaded models |

`python cli.py --help` lists all options.

## Head boxes

FlexGaze works with head boxes from any detector. As in training, a box should cover the whole head
including the hair, even for a head seen from behind.

An image that is already a head crop can skip detection, with `--head-crop` on the command line or
the box `(0, 0, width, height)` in Python. The predicted direction is then relative to a camera
looking straight at the head. To get it in the frame of the photo the crop was cut from, give that
photo and the head box instead. A crop that is not square is read through its central square.

The command-line tool finds heads with the `Head` class of DEIM-Wholebody28 (see
[Third-party models](#third-party-models)), which was chosen for its permissive license and is not
used for the paper's results. Of its two sizes, `s` finds more heads in crowds and more small heads,
while `n` (`--detector-size n`, or `"n"` in Python) runs about twice as fast.

## Camera intrinsics

- **Default.** Without a calibration the command-line tool assumes a pinhole camera with its
  principal point at the image centre and an 80 degree diagonal field of view, typical of webcams
  and phone main cameras.
- **Field of view or focal length.** `--fov 70` (diagonal, degrees) or `--focal 1000` (pixels); in
  Python, `Camera.from_fov(width, height, 70)` or `Camera.from_focal_length(width, height, 1000)`.
- **Calibration.** `--intrinsics fx,fy,cx,cy`, or `Camera(matrix)` with the matrix as
  `cv2.calibrateCamera` returns it (OpenCV's convention, where the centre of the top-left pixel is
  at `(0, 0)`). For a lens with noticeable distortion, also give its distortion coefficients from
  `cv2.calibrateCamera`, with `--distortion k1,k2,p1,p2[,k3,...]` or `Camera(matrix, distortion)`,
  and pass the image as taken, not undistorted.

## Using the parts in your own code

The command-line tool is built from three parts that can also be used on their own. `Camera`
(`camera.py`) holds the camera matrix and lens distortion, `DeimHeadDetector` (`detector.py`) finds
head boxes, and `FlexGaze` (`flexgaze.py`) predicts the gaze for given head boxes. The example below
puts them together as `cli.py` does.

```python
# run from the minimal-inference folder
import tempfile

import cv2

from camera import Camera
from detector import DeimHeadDetector
from flexgaze import FlexGaze

image = cv2.imread("photo.jpg")[:, :, ::-1].copy()  # BGR -> RGB
camera = Camera.from_fov(image.shape[1], image.shape[0], 80)  # or Camera(your calibration)
cache = f"{tempfile.gettempdir()}/flexgaze"  # same download folder as the command-line tool
detector = DeimHeadDetector("s", cache_dir=cache, score_threshold=0.35, duplicate_iou=0.6)
boxes = detector.detect(image)[:, :4]
predictions = FlexGaze("base", seed=0, cache_dir=cache).predict(image, camera, boxes)
```

`image` is an `(height, width, 3)` uint8 RGB array, `camera` holds the camera matrix of that image
and, if any, its lens distortion coefficients, and each row of `boxes` is a head box
`(x0, y0, x1, y1)` in pixels. Each `GazePrediction` in `predictions` has

- `box`: the head box the prediction was made from;
- `direction`: unit vector from the head towards where it looks, in the camera frame (x right, y
  down, z forward from the camera);
- `kappa`: concentration of the predicted von Mises-Fisher distribution around `direction`, larger
  meaning more certain;
- `yaw_degrees`, `pitch_degrees`: the direction as two angles, zero when looking straight back at
  the camera along the optical axis, yaw positive towards the image's right, pitch positive upwards.

`"base"` and `"small"` are FlexGaze-B and FlexGaze-S in the paper. `seed` (0, 1 or 2) picks one of
the three released training runs. A released model needs both `seed` and `cache_dir`.
`FlexGaze("path/to/model.onnx")` loads a local file instead.

## Third-party models

**DEIM-Wholebody28**, by Katsuya Hyodo (PINTO0309),
[PINTO_model_zoo 465_DEIM-Wholebody28](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/465_DEIM-Wholebody28),
licensed under the
[Apache License 2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/465_DEIM-Wholebody28/LICENSE).
It is built on [DEIM](https://github.com/ShihuaHuang95/DEIM) (Apache License 2.0). The ONNX files
are not distributed with FlexGaze but downloaded from the author's own release at first use, and
their use is subject to that license.