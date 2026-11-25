"""Locating FlexGaze ONNX files (released models downloaded from Hugging Face, or local paths) and the
checked download the head detector also uses."""

import hashlib
import http.client
import os
import sys
import tempfile
import urllib.request
from os import PathLike
from pathlib import Path

HF_REPOSITORY = "ZongjianLi/FlexGaze"
"""The Hugging Face model repository holding the released ONNX files."""

DOWNLOAD_ROOT = f"https://huggingface.co/{HF_REPOSITORY}/resolve/main"
"""Address the repository's files are downloaded from."""

MODEL_SIZES = ("base", "small")
"""Released model sizes: ``base`` is FlexGaze-B and ``small`` is FlexGaze-S in the paper."""

SEEDS = (0, 1, 2)

RELEASED_SHA256 = {
    "flexgaze-v1-base-8set-seed0.onnx": "b9296fe93d02c41e2b0e3b33337515d8812e21c2a4bd79b0d8a693f0592103ba",
    "flexgaze-v1-base-8set-seed1.onnx": "716fa00443ae147a447eb50e5198457ad86331780bfbab2644005f326aef92f6",
    "flexgaze-v1-base-8set-seed2.onnx": "246736f3503bd26ab4005f33913bc80eea61bcd66423816713d4c78f3e6b347c",
    "flexgaze-v1-small-8set-seed0.onnx": "19bc97097c3d1a6798aaf530f15022d539e8702ae41f430f7445057a2e5b3afe",
    "flexgaze-v1-small-8set-seed1.onnx": "ebf40234b56764415da92d3bc9c8aaba87bc3f3513e09d962e7a080a8fa22d2e",
    "flexgaze-v1-small-8set-seed2.onnx": "1f43bfed2e34730e7d199f2887ef868a9a81ef30da828fa58d4de9ee6c0a7412",
}
"""SHA-256 of every released ONNX file, by file name."""


def released_file_path(size: str, seed: int) -> str:
    """The path of a released ONNX model inside the Hugging Face repository. In the file name, ``v1``
    is the model version and ``8set`` means fine-tuned jointly on the eight gaze datasets."""
    return f"onnx/flexgaze-v1-{size}-8set-seed{seed}.onnx"


def flexgaze_onnx_path(
    model: str | PathLike, seed: int | None = None, cache_dir: str | PathLike | None = None
) -> Path:
    """The local path of a FlexGaze ONNX file, downloading a released one when ``model`` names it.

    A released model is kept in ``cache_dir`` under its path in the repository.

    Args:
        model: ``"base"`` or ``"small"`` for a released model, or the path of an ``.onnx`` file.
        seed: The released training seed, 0, 1 or 2. Required when ``model`` names a released model,
            and ignored when it is a path.
        cache_dir: The folder downloaded files are kept in. Required when ``model`` names a released
            model, and ignored when it is a path.

    Raises:
        FileNotFoundError: ``model`` is neither a released model name nor an existing file.
        ValueError: ``model`` names a released model and ``seed`` is missing or not a released seed,
            or ``cache_dir`` is missing.
        OSError: The download fails or does not match its pinned digest.
    """
    if model in MODEL_SIZES:
        if seed not in SEEDS:
            raise ValueError(f"released seeds are {SEEDS}, got {seed!r}")
        if cache_dir is None:
            raise ValueError("give a cache_dir to download a released model into")
        name = released_file_path(model, seed)
        return download_pinned_file(
            f"{DOWNLOAD_ROOT}/{name}", RELEASED_SHA256[Path(name).name], Path(cache_dir).expanduser() / name
        )
    path = Path(model).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{model!r} is neither a released model ({' or '.join(MODEL_SIZES)}) nor an existing file"
        )
    return path


def download_pinned_file(url: str, sha256: str, destination: str | PathLike) -> Path:
    """Download ``url`` to ``destination`` and check it against its pinned SHA-256.

    A file already at ``destination`` is used if its digest matches. The download goes to a
    temporary file next to ``destination`` and is renamed into place only once checked, so
    ``destination`` never holds a partial or wrong file. A download prints
    ``downloading <url>`` to standard error when it starts, and fails once the server sends nothing
    for 60 seconds.

    Raises:
        OSError: The download fails or does not match ``sha256``.
    """
    destination = Path(destination)
    if destination.is_file() and sha256_of_file(destination) == sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".partial"
    )
    try:
        digest = hashlib.sha256()
        print(f"downloading {url}", file=sys.stderr)
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(url, timeout=60) as response:
            try:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    digest.update(chunk)
                    out.write(chunk)
            except http.client.HTTPException as error:
                raise OSError(f"the download of {url} was cut short") from error
        if digest.hexdigest() != sha256:
            raise OSError(
                f"{url} arrived with SHA-256 {digest.hexdigest()}, expected {sha256}. "
                "The download was cut short or the file has changed."
            )
        os.replace(partial, destination)
    finally:
        Path(partial).unlink(missing_ok=True)
    return destination


def sha256_of_file(path: str | PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
