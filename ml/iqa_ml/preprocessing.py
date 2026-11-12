"""Image loading and preprocessing (SDS 5.1.1 ImagePreprocessor).

Supports PNG, JPEG and DICOM. The file format is detected from the file content
(magic bytes), never from the extension.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from . import config

PathOrBytes = Union[str, Path, bytes]
log = logging.getLogger(__name__)


class ImageLoadError(ValueError):
    """Raised when a file is not a readable PNG/JPEG/DICOM image (FR-IMG-02)."""

    def __init__(self, message: str, code: str = "unreadable_image"):
        super().__init__(message)
        self.code = code


@dataclass
class LoadedImage:
    pixels: np.ndarray  # H x W x 3, uint8, RGB
    file_format: str  # "PNG" | "JPEG" | "DICOM"
    metadata_modality: str | None  # modality from DICOM tag, if any

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


def _read_bytes(source: PathOrBytes) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return Path(source).read_bytes()


def detect_format(data: bytes) -> str | None:
    """Return "PNG", "JPEG", "DICOM" or None from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if len(data) >= 132 and data[128:132] == b"DICM":
        return "DICOM"
    return None


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Min-max scale an arbitrary numeric array to uint8 (no-op for uint8 input)."""
    if array.dtype == np.uint8:
        return array
    arr = array.astype(np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.where(finite, arr, 0.0)
    lo, hi = float(arr[finite].min()), float(arr[finite].max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo) * 255.0
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    """Convert grayscale / gray+alpha / RGB / RGBA arrays of any dtype to H x W x 3 uint8."""
    arr = to_uint8(np.asarray(array))
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1:
            return np.repeat(arr, 3, axis=2)
        if channels == 2:  # gray + alpha
            return np.repeat(arr[:, :, :1], 3, axis=2)
        if channels >= 3:
            return np.ascontiguousarray(arr[:, :, :3])
    raise ImageLoadError(f"Unsupported image shape {arr.shape}", code="unsupported_shape")


def _load_raster(data: bytes, file_format: str) -> LoadedImage:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if img.mode in ("P", "PA", "CMYK", "YCbCr", "LAB", "HSV", "1"):
                img = img.convert("RGB")
            array = np.array(img)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageLoadError(f"Corrupt or unreadable {file_format} file: {exc}") from exc
    return LoadedImage(pixels=to_rgb_uint8(array), file_format=file_format, metadata_modality=None)


def _voi_functions():
    try:  # pydicom >= 3
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError:  # pydicom 2.x
        from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    return apply_modality_lut, apply_voi_lut


def _load_dicom(data: bytes) -> LoadedImage:
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        ds = pydicom.dcmread(io.BytesIO(data))
    except (InvalidDicomError, OSError, ValueError, EOFError) as exc:
        raise ImageLoadError(f"Corrupt or unreadable DICOM file: {exc}") from exc

    if "PixelData" not in ds:
        raise ImageLoadError("DICOM file contains no pixel data", code="no_pixel_data")
    try:
        pixels = ds.pixel_array
    except Exception as exc:  # missing decoder plugins, truncated data, ...
        raise ImageLoadError(
            f"Cannot decode DICOM pixel data (possibly unsupported compression): {exc}",
            code="unsupported_dicom",
        ) from exc

    samples = int(getattr(ds, "SamplesPerPixel", 1))
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if frames > 1:  # multi-frame: use the middle frame (2D-only system, SRS 2.3)
        pixels = pixels[frames // 2]

    if samples == 1:
        apply_modality_lut, apply_voi_lut = _voi_functions()
        try:
            pixels = apply_modality_lut(pixels, ds)
            pixels = apply_voi_lut(pixels, ds)  # window/level (FR-QA-19)
        except Exception as exc:
            log.warning("DICOM modality/VOI LUT not applied (%s); using raw pixel values", exc)
            pixels = np.asarray(pixels, dtype=np.float64)
        pixels = to_uint8(np.asarray(pixels))
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            pixels = 255 - pixels
    else:
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        if photometric.startswith("YBR"):
            try:
                from pydicom.pixels import convert_color_space
            except ImportError:
                from pydicom.pixel_data_handlers.util import convert_color_space
            pixels = convert_color_space(pixels, photometric, "RGB")

    modality_tag = str(getattr(ds, "Modality", "")).upper().strip()
    return LoadedImage(
        pixels=to_rgb_uint8(pixels),
        file_format="DICOM",
        metadata_modality=config.DICOM_MODALITY_MAP.get(modality_tag),
    )


class ImagePreprocessor:
    """Load, validate and convert medical images to model-ready tensors."""

    def __init__(
        self,
        target_size: int = config.IMAGE_SIZE,
        mean: tuple[float, float, float] = config.IMAGENET_MEAN,
        std: tuple[float, float, float] = config.IMAGENET_STD,
        min_side: int = config.MIN_IMAGE_SIDE,
    ):
        self.target_size = target_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.min_side = min_side

    def load(self, source: PathOrBytes, validate_size: bool = True) -> LoadedImage:
        data = _read_bytes(source)
        if not data:
            raise ImageLoadError("File is empty", code="empty_file")
        file_format = detect_format(data)
        if file_format is None:
            raise ImageLoadError(
                "Unsupported file type: only DICOM, PNG and JPEG are accepted",
                code="unsupported_format",
            )
        loaded = _load_dicom(data) if file_format == "DICOM" else _load_raster(data, file_format)
        if validate_size and min(loaded.height, loaded.width) < self.min_side:
            raise ImageLoadError(
                f"Image is {loaded.width}x{loaded.height}; minimum is "
                f"{self.min_side}x{self.min_side} pixels",
                code="image_too_small",
            )
        return loaded

    def resize(self, pixels: np.ndarray) -> np.ndarray:
        size = (self.target_size, self.target_size)
        if pixels.shape[0] == self.target_size and pixels.shape[1] == self.target_size:
            return pixels
        shrinking = pixels.shape[0] > self.target_size or pixels.shape[1] > self.target_size
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return cv2.resize(pixels, size, interpolation=interp)

    def to_tensor(self, pixels: np.ndarray) -> torch.Tensor:
        """H x W x 3 uint8 (already target size) -> 3 x H x W float tensor, ImageNet-normalised."""
        arr = pixels.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    def preprocess(self, pixels: np.ndarray) -> torch.Tensor:
        return self.to_tensor(self.resize(to_rgb_uint8(pixels)))


def make_thumbnail(pixels: np.ndarray, out: Union[str, Path, BinaryIO], size: int = config.THUMBNAIL_SIZE) -> None:
    """Save a PNG thumbnail whose longer side is `size` pixels (aspect ratio preserved)."""
    img = Image.fromarray(to_rgb_uint8(pixels))
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    img.save(out, format="PNG")
