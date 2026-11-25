"""Command line: ``python cli.py IMAGE`` prints the gaze of every head in the image, one line each."""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from camera import Camera
from detector import DEIM_MODELS, DeimHeadDetector
from flexgaze import FlexGaze


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    # Decoded from the file's bytes, which also opens Windows paths with non-ASCII characters.
    try:
        bgr = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):  # missing, a folder, or empty
        bgr = None
    if bgr is None:
        print(f"cannot read image {args.image}", file=sys.stderr)
        return 1
    image = np.ascontiguousarray(bgr[:, :, ::-1])
    try:
        camera = _camera(args, image.shape[1], image.shape[0])
    except ValueError as error:
        parser.error(str(error))
    if args.detector_model is not None and not args.head_crop:
        if not Path(args.detector_model).expanduser().is_file():
            parser.error(f"no DEIM-Wholebody28 model file at {args.detector_model}")
    try:
        model = FlexGaze(args.model, seed=args.seed, cache_dir=args.cache_dir)
        detector = (
            None
            if args.head_crop
            else DeimHeadDetector(
                args.detector_size,
                model_path=args.detector_model,
                cache_dir=args.cache_dir,
                score_threshold=args.score_threshold,
                duplicate_iou=args.duplicate_iou,
            )
        )
    except FileNotFoundError as error:
        # a missing --model or --detector-model file, or a --cache-dir that cannot be made
        parser.error(str(error))
    if args.head_crop:
        boxes = np.array([[0.0, 0.0, image.shape[1], image.shape[0]]])
    else:
        boxes = detector.detect(image)[:, :4]
    predictions = model.predict(image, camera, boxes)
    if not predictions:
        print(f"no heads found in {args.image}", file=sys.stderr)
    for n, p in enumerate(predictions):
        x, y, z = p.direction
        print(
            f"head {n}: box=({', '.join(f'{v:.1f}' for v in p.box)}) direction=({x:+.4f}, {y:+.4f}, "
            f"{z:+.4f}) yaw={p.yaw_degrees:+.1f} pitch={p.pitch_degrees:+.1f} kappa={p.kappa:.2f}"
        )
    return 0


def _camera(args, width: int, height: int) -> Camera:
    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
        matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    elif args.focal is not None:
        matrix = Camera.from_focal_length(width, height, args.focal).matrix
    else:
        matrix = Camera.from_fov(width, height, args.fov).matrix
    return Camera(matrix, args.distortion)


def _numbers(text: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from None
    if not all(math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError(f"expected finite numbers, got {text!r}")
    return values


def _intrinsics(text: str) -> tuple[float, float, float, float]:
    values = _numbers(text)
    if len(values) != 4 or values[0] <= 0.0 or values[1] <= 0.0:
        raise argparse.ArgumentTypeError("expected fx,fy,cx,cy with positive fx and fy")
    return tuple(values)


def _distortion(text: str) -> np.ndarray:
    values = _numbers(text)
    if len(values) not in (4, 5, 8, 12, 14):
        raise argparse.ArgumentTypeError("expected 4, 5, 8, 12 or 14 coefficients, k1,k2,p1,p2[,k3,...]")
    return np.array(values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="FlexGaze: print the 3D gaze of every head in an image."
    )
    parser.add_argument("image", type=Path, help="the image")
    parser.add_argument(
        "--model",
        default="base",
        help="FlexGaze model, base or small (released models), or the path of an .onnx file (default base)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="which of the three released training runs to use (default 0)",
    )
    parser.add_argument(
        "--head-crop",
        action="store_true",
        help="the image is a head crop, skip detection and use the whole image as the head box",
    )
    parser.add_argument(
        "--detector-size",
        default="s",
        choices=sorted(DEIM_MODELS),
        help="DEIM-Wholebody28 head detector size, n runs about twice as fast and s finds more heads "
        "(default s)",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        metavar="PATH",
        help="local DEIM-Wholebody28 .onnx file, used instead of downloading --detector-size",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        metavar="T",
        help="heads the detector scores below this are dropped (default 0.35)",
    )
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.6,
        metavar="T",
        help="a head box overlapping a better-scoring one by more than this intersection over union is "
        "dropped (default 0.6)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "flexgaze",
        metavar="PATH",
        help=(
            "folder downloaded models are kept in "
            "(default a flexgaze folder in the system's temporary directory)"
        ),
    )
    given = parser.add_mutually_exclusive_group()
    given.add_argument(
        "--fov",
        type=float,
        default=80.0,
        metavar="DEG",
        help="diagonal field of view in degrees of a camera without calibration, with the principal "
        "point at the image centre (default 80)",
    )
    given.add_argument(
        "--focal",
        type=float,
        metavar="PX",
        help="focal length in pixels, instead of --fov, with the principal point at the image centre",
    )
    given.add_argument(
        "--intrinsics",
        type=_intrinsics,
        metavar="FX,FY,CX,CY",
        help="calibrated camera matrix in OpenCV's convention (the centre of the top-left pixel at 0,0), "
        "instead of --fov",
    )
    parser.add_argument(
        "--distortion",
        type=_distortion,
        metavar="K1,K2,P1,P2[,K3,...]",
        help="lens distortion coefficients of the calibrated camera, in the order cv2.calibrateCamera "
        "returns them, with the image given as taken, not undistorted",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
