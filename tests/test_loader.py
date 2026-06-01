import numpy as np
import pytest
import os
from PIL import Image


def _make_fake_image(path, size=(64, 64), mode="RGB"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.full((*size, 3) if mode == "RGB" else size, 128, dtype=np.uint8)
    img = Image.fromarray(arr if mode == "RGB" else arr, mode=mode)
    img.save(path)


def test_load_drive_returns_list_of_dicts(tmp_path):
    from retina_seg.dataset.loader import load_drive
    # Build fake DRIVE structure
    for split in ["training", "test"]:
        for i in range(1, 3):
            name = f"0{i}_{split}"
            _make_fake_image(str(tmp_path / split / "images" / f"{name}.png"))
            _make_fake_image(str(tmp_path / split / "mask" / f"{name}_mask.png"), mode="L")
            _make_fake_image(str(tmp_path / split / "1st_manual" / f"{name}_manual1.png"), mode="L")

    samples = load_drive(str(tmp_path), split="training")
    assert len(samples) == 2
    assert "image" in samples[0]
    assert "mask" in samples[0]
    assert "gt1" in samples[0]
    assert samples[0]["image"].shape[2] == 3
    assert samples[0]["mask"].ndim == 2


def test_load_stare_returns_list_of_dicts(tmp_path):
    from retina_seg.dataset.loader import load_stare
    imgs_dir = tmp_path / "images"
    ah_dir = tmp_path / "labels-ah"
    imgs_dir.mkdir(); ah_dir.mkdir()
    for i in range(1, 3):
        _make_fake_image(str(imgs_dir / f"im000{i}.png"))
        _make_fake_image(str(ah_dir / f"im000{i}.ah.png"), mode="L")

    samples = load_stare(str(tmp_path))
    assert len(samples) == 2
    assert samples[0]["image"].shape[2] == 3


def test_load_chase_returns_list_of_dicts(tmp_path):
    from retina_seg.dataset.loader import load_chase
    for i in range(1, 3):
        _make_fake_image(str(tmp_path / f"Image_0{i}L.jpg"))
        _make_fake_image(str(tmp_path / f"Image_0{i}L_1stHO.png"), mode="L")
        _make_fake_image(str(tmp_path / f"Image_0{i}L_2ndHO.png"), mode="L")

    samples = load_chase(str(tmp_path))
    assert len(samples) == 2
    assert "gt2" in samples[0]
