import numpy as np
import pytest
from PIL import Image

def test_read_image_returns_rgb_uint8(tmp_path):
    from retina_seg.dataset.utils import read_image
    img_path = str(tmp_path / "test.png")
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:20, 10:20] = [255, 0, 0]
    Image.fromarray(arr).save(img_path)
    result = read_image(img_path)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8

def test_read_mask_returns_binary(tmp_path):
    from retina_seg.dataset.utils import read_mask
    mask_path = str(tmp_path / "mask.png")
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[10:50, 10:50] = 255
    Image.fromarray(arr).save(mask_path)
    result = read_mask(mask_path)
    assert result.shape == (64, 64)
    assert set(np.unique(result)).issubset({0, 255})

def test_list_files_sorted(tmp_path):
    from retina_seg.dataset.utils import list_files
    for name in ["03.png", "01.png", "02.png"]:
        (tmp_path / name).write_bytes(b"")
    result = list_files(str(tmp_path), ".png")
    assert result == sorted(result)
    assert len(result) == 3
