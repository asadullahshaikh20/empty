import io

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import png_bytes, synthetic_scan
from iqa_ml import config
from iqa_ml.preprocessing import ImageLoadError, ImagePreprocessor, detect_format, make_thumbnail, to_uint8


def _dicom_bytes(pixels: np.ndarray, modality="CR", photometric="MONOCHROME2", window=None, with_pixels=True) -> bytes:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.PatientName = "Anonymous"
    if with_pixels:
        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = photometric
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        if window:
            ds.WindowCenter, ds.WindowWidth = window
        ds.PixelData = pixels.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


@pytest.fixture
def pre():
    return ImagePreprocessor()


def test_detect_format():
    assert detect_format(png_bytes(synthetic_scan())) == "PNG"
    assert detect_format(b"\xff\xd8\xff\xe0rest") == "JPEG"
    assert detect_format(b"\0" * 128 + b"DICM" + b"\0" * 10) == "DICOM"
    assert detect_format(b"hello world") is None


def test_load_png_rgb(pre):
    loaded = pre.load(png_bytes(synthetic_scan(1, color=True)))
    assert loaded.file_format == "PNG"
    assert loaded.pixels.shape == (224, 224, 3) and loaded.pixels.dtype == np.uint8
    assert loaded.metadata_modality is None


def test_load_grayscale_16bit_png(pre):
    arr = (np.linspace(0, 65535, 200 * 300).reshape(200, 300)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    loaded = pre.load(buf.getvalue())
    assert loaded.pixels.shape == (200, 300, 3)
    assert loaded.pixels.min() == 0 and loaded.pixels.max() == 255


def test_load_rgba_and_jpeg(pre):
    rgba = np.dstack([synthetic_scan(2), np.full((224, 224), 255, np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    assert pre.load(buf.getvalue()).pixels.shape == (224, 224, 3)
    buf = io.BytesIO()
    Image.fromarray(synthetic_scan(3)).save(buf, format="JPEG")
    loaded = pre.load(buf.getvalue())
    assert loaded.file_format == "JPEG" and loaded.pixels.shape == (224, 224, 3)


def test_extension_is_ignored(pre, tmp_path):
    path = tmp_path / "actually_png.jpg"
    path.write_bytes(png_bytes(synthetic_scan()))
    assert pre.load(path).file_format == "PNG"


@pytest.mark.parametrize(
    "data, code",
    [
        (b"", "empty_file"),
        (b"just some text, not an image", "unsupported_format"),
        (b"\x89PNG\r\n\x1a\n" + b"garbage" * 20, "unreadable_image"),
        (b"\xff\xd8\xff" + b"\x00" * 50, "unreadable_image"),
    ],
)
def test_rejects_bad_files(pre, data, code):
    with pytest.raises(ImageLoadError) as exc:
        pre.load(data)
    assert exc.value.code == code


def test_minimum_size(pre):
    with pytest.raises(ImageLoadError) as exc:
        pre.load(png_bytes(synthetic_scan(size=224)[:127, :]))
    assert exc.value.code == "image_too_small"
    assert pre.load(png_bytes(synthetic_scan(size=128))).width == 128
    # size validation can be disabled (training data)
    assert pre.load(png_bytes(synthetic_scan(size=64)), validate_size=False).width == 64


def test_dicom_monochrome1_window_and_modality(pre):
    pixels = np.zeros((160, 160), dtype=np.uint16)
    pixels[:, 80:] = 4000  # right half "bright" in raw values
    loaded = pre.load(_dicom_bytes(pixels, modality="CR", photometric="MONOCHROME1", window=(2000, 4000)))
    assert loaded.file_format == "DICOM"
    assert loaded.metadata_modality == "xray"
    assert loaded.pixels.shape == (160, 160, 3)
    # MONOCHROME1: high raw values are displayed dark
    assert loaded.pixels[0, 159, 0] < loaded.pixels[0, 0, 0]


@pytest.mark.parametrize("tag, expected", [("CT", "ct"), ("MR", "mri"), ("OP", "retinal"), ("US", None)])
def test_dicom_modality_map(pre, tag, expected):
    pixels = (np.arange(160 * 160).reshape(160, 160) % 4096).astype(np.uint16)
    assert pre.load(_dicom_bytes(pixels, modality=tag)).metadata_modality == expected


def test_dicom_without_pixels(pre):
    with pytest.raises(ImageLoadError) as exc:
        pre.load(_dicom_bytes(None, with_pixels=False))
    assert exc.value.code == "no_pixel_data"


def test_preprocess_tensor(pre):
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    t = pre.preprocess(img)
    assert isinstance(t, torch.Tensor) and t.shape == (3, 224, 224) and t.dtype == torch.float32
    expected = (1.0 - np.array(config.IMAGENET_MEAN)) / np.array(config.IMAGENET_STD)
    assert np.allclose(t[:, 0, 0].numpy(), expected, atol=1e-5)


def test_to_uint8_constant_and_nan():
    assert to_uint8(np.full((4, 4), 7.0)).max() == 0
    arr = np.array([[0.0, np.nan], [1.0, 2.0]])
    out = to_uint8(arr)
    assert out.dtype == np.uint8 and out[1, 1] == 255


def test_thumbnail(tmp_path):
    out = tmp_path / "t.png"
    make_thumbnail(np.zeros((600, 300, 3), np.uint8), out)
    with Image.open(out) as im:
        assert im.size == (128, 256)
