"""Head detection with DEIM-Wholebody28. Any detector's head boxes can be passed to FlexGaze.

The detector here is DEIM-Wholebody28 by Katsuya Hyodo (PINTO_model_zoo 465, Apache-2.0), a
28-class person-part detector of which only the head class is used. Its ONNX files are downloaded
from the author's own release at first use and checked against pinned SHA-256 digests; they are not
redistributed with FlexGaze.
"""

import os
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import onnxruntime as ort

from weights import download_pinned_file, sha256_of_file


@dataclass(frozen=True)
class ArchivedFile:
    """One file inside a release archive, pinned by URL and digests.

    Attributes:
        url: The archive's download address.
        archive_sha256: SHA-256 of the archive.
        member: The file's name inside the archive.
        member_sha256: SHA-256 of the file.
    """

    url: str
    archive_sha256: str
    member: str
    member_sha256: str


DEIM_MODELS = {
    "n": ArchivedFile(
        url="https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/465_DEIM-Wholebody28/resources_n.tar.gz",
        archive_sha256="b72f4302e029f357a78a68e9c6e205f10791b2429b441b8cc1fdc1bffeea93e6",
        member="deim_hgnetv2_n_wholebody28_1250query.onnx",
        member_sha256="86e002a179349fb0a60747ffd378c0f026ee5ec4677d66edd63fc0d4eb2ca40f",
    ),
    "s": ArchivedFile(
        url="https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/465_DEIM-Wholebody28/resources_s.tar.gz",
        archive_sha256="0d7b1d483431252c60604e71490d9280895b9bbf50a4eff762791dddf6e4ad6e",
        member="deim_hgnetv2_s_wholebody28_ft_1250query.onnx",
        member_sha256="8fb5633f433f10728d831158dee19fbcae3b654b231371b59f2e7a405d4642a0",
    ),
}
"""The DEIM-Wholebody28 sizes: ``n`` (15 MB, about twice as fast) and ``s`` (42 MB, finds more heads
in crowds and more small heads)."""

DEIM_HEAD_CLASS = 7
"""Class index of ``Head`` in DEIM-Wholebody28 (class table in the README of the release page). The
head box covers the whole head from any side; classes 8-15 repeat it with a facing direction."""

DEIM_INPUT_PAD_LEVEL = 114
"""Grey level the image is padded with to a square before detection."""


class DeimHeadDetector:
    """Heads found by DEIM-Wholebody28.

    The graph stretches whatever it is given to a square, so the image is first padded on the right
    or at the bottom to a square; a tall or wide image stretched directly loses most heads seen from
    behind. Boxes come back in the image's own pixels.

    Args:
        size: ``"n"`` or ``"s"``, the released size to download. Required unless ``model_path`` is
            given, and ignored then.
        model_path: A local copy of a DEIM-Wholebody28 ONNX file.
        cache_dir: The folder the download is kept in. Required unless ``model_path`` is given, and
            ignored then.
        score_threshold: Heads scoring below this are dropped.
        duplicate_iou: A head box overlapping a better-scoring one by more than this intersection
            over union is dropped; the detector occasionally returns one head twice.
        providers: onnxruntime execution providers, CPU by default.

    Raises:
        FileNotFoundError: ``model_path`` is not an existing file.
        ValueError: Neither ``size`` nor ``model_path`` is given, ``size`` is not a released size, or
            ``cache_dir`` is missing for a download.
    """

    def __init__(
        self,
        size: str | None = None,
        *,
        model_path: str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        score_threshold: float,
        duplicate_iou: float,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        if model_path is not None:
            self.model_path = Path(model_path).expanduser()
            if not self.model_path.is_file():
                raise FileNotFoundError(f"no DEIM-Wholebody28 model file at {self.model_path}")
        elif size is None:
            raise ValueError(f"give a size {sorted(DEIM_MODELS)} or a model_path")
        else:
            self.model_path = deim_onnx_path(size, cache_dir)
        self.score_threshold = float(score_threshold)
        self.duplicate_iou = float(duplicate_iou)
        self.session = ort.InferenceSession(str(self.model_path), providers=list(providers))
        self._input = self.session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Head boxes in an RGB image.

        Args:
            image: ``(height, width, 3)`` uint8 RGB image.

        Returns:
            ``(n, 5)`` float rows ``(x0, y0, x1, y1, score)`` in pixels, best first. The first four
            columns are the boxes :meth:`FlexGaze.predict` takes.

        Raises:
            ValueError: The image is not ``(height, width, 3)`` uint8.
        """
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected a (height, width, 3) uint8 image, got {image.dtype} {image.shape}")
        height, width = image.shape[:2]
        side = max(height, width)
        square = np.full((side, side, 3), DEIM_INPUT_PAD_LEVEL, dtype=np.uint8)
        square[:height, :width] = image[:, :, ::-1]  # the graph reads BGR
        frame = square.transpose(2, 0, 1)[None].astype(np.float32)
        rows = self.session.run(None, {self._input: frame})[0][0]
        heads = rows[(np.rint(rows[:, 0]) == DEIM_HEAD_CLASS) & (rows[:, 5] >= self.score_threshold)]
        boxes = np.clip(heads[:, 1:5].astype(np.float64), 0.0, 1.0) * side
        boxes[:, [0, 2]] = np.minimum(boxes[:, [0, 2]], width)
        boxes[:, [1, 3]] = np.minimum(boxes[:, [1, 3]], height)
        found = np.concatenate([boxes, heads[:, 5:6].astype(np.float64)], axis=1)
        found = found[(found[:, 2] > found[:, 0]) & (found[:, 3] > found[:, 1])]  # left empty by clipping
        found = found[np.argsort(-found[:, 4], kind="stable")]
        kept: list[int] = []
        for n in range(len(found)):
            if all(_iou(found[n, :4], found[k, :4]) <= self.duplicate_iou for k in kept):
                kept.append(n)
        return found[kept]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    overlap = width * height
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def deim_onnx_path(size: str, cache_dir: str | PathLike | None = None) -> Path:
    """The local path of a DEIM-Wholebody28 ONNX file, downloaded from the author on first use.

    Raises:
        ValueError: ``size`` is not one of :data:`DEIM_MODELS`, or ``cache_dir`` is missing.
    """
    if size not in DEIM_MODELS:
        raise ValueError(f"DEIM-Wholebody28 sizes are {sorted(DEIM_MODELS)}, got {size!r}")
    if cache_dir is None:
        raise ValueError("give a cache_dir to download the detector into")
    pinned = DEIM_MODELS[size]
    return fetch_archived_file(pinned, Path(cache_dir).expanduser() / "deim-wholebody28" / pinned.member)


def fetch_archived_file(pinned: ArchivedFile, destination: Path) -> Path:
    """Download ``pinned``'s archive, check it, and extract the pinned file to ``destination``.

    A file already at ``destination`` is used if its digest matches.

    Raises:
        OSError: The archive or the file inside it does not match its pinned digest, or the file is
            missing from the archive.
    """
    destination = Path(destination)
    if destination.is_file() and sha256_of_file(destination) == pinned.member_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        archive = download_pinned_file(pinned.url, pinned.archive_sha256, Path(scratch) / "archive.tar.gz")
        extracted = Path(scratch) / "member"
        with tarfile.open(archive, "r:gz") as bundle:
            try:
                source = bundle.extractfile(pinned.member)
            except KeyError:
                source = None
            if source is None:
                raise OSError(f"{pinned.url} holds no file {pinned.member}")
            with source, open(extracted, "wb") as out:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    out.write(chunk)
        found = sha256_of_file(extracted)
        if found != pinned.member_sha256:
            raise OSError(f"{pinned.member} has SHA-256 {found}, expected {pinned.member_sha256}")
        os.replace(extracted, destination)
    return destination
