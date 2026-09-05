from pathlib import Path

import pytest
from PIL import Image

from src.errors import AppError, ErrorCode
from src.services import image_service


def test_rejects_excessive_pixel_count(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), "black")
    original_limit = image_service.MAX_INPUT_PIXELS
    image_service.MAX_INPUT_PIXELS = 9_999
    try:
        with pytest.raises(AppError) as error:
            image_service.persist_input_image(image, tmp_path, 1024)
    finally:
        image_service.MAX_INPUT_PIXELS = original_limit

    assert error.value.code is ErrorCode.INVALID_INPUT


def test_disk_write_failure_is_retriable_internal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (100, 100), "black")

    def fail_save(*args, **kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(AppError) as error:
        image_service.persist_input_image(image, tmp_path, 1024)

    assert error.value.code is ErrorCode.INTERNAL
    assert error.value.retriable is True


def test_canonical_image_hash_uses_normalized_pixels() -> None:
    first = Image.new("RGB", (40, 20), "navy")
    same_pixels = first.copy()
    changed = first.copy()
    changed.putpixel((0, 0), (255, 255, 255))

    first_hash = image_service.canonical_image_sha256(first, 1024)

    assert image_service.canonical_image_sha256(same_pixels, 1024) == first_hash
    assert image_service.canonical_image_sha256(changed, 1024) != first_hash
