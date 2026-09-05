from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from src.errors import AppError, ErrorCode, PipelineStage

MAX_INPUT_PIXELS = 40_000_000
IMAGE_NORMALIZATION_VERSION = "exif-rgb-thumbnail-lanczos-v1"


def _normalized_input_image(image: Image.Image, max_side: int) -> Image.Image:
    if image.width * image.height > MAX_INPUT_PIXELS:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            stage=PipelineStage.INPUT,
            user_message="图片像素尺寸过大，请缩小到 4000 万像素以内再试。",
        )
    try:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        normalized.load()
        return normalized
    except (OSError, SyntaxError, ValueError, MemoryError) as exc:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            stage=PipelineStage.INPUT,
            user_message="图片无法读取，请换一张 JPG、PNG 或 HEIC 图片重试。",
            detail=repr(exc),
        ) from exc


def canonical_image_sha256(image: Image.Image, max_side: int) -> str:
    normalized = _normalized_input_image(image, max_side)
    digest = hashlib.sha256()
    digest.update(IMAGE_NORMALIZATION_VERSION.encode("ascii"))
    digest.update(normalized.mode.encode("ascii"))
    digest.update(normalized.width.to_bytes(4, "big"))
    digest.update(normalized.height.to_bytes(4, "big"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def persist_input_image(
    image: Image.Image | None,
    output_dir: Path,
    max_side: int,
) -> Path | None:
    if image is None:
        return None
    normalized = _normalized_input_image(image, max_side)

    output_path = output_dir / f"scan_{uuid4().hex}_before.jpg"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized.save(output_path, format="JPEG", quality=94, optimize=True)
        return output_path
    except OSError as exc:
        raise AppError(
            code=ErrorCode.INTERNAL,
            stage=PipelineStage.INPUT,
            user_message="无法保存输入图片，请检查磁盘空间和目录权限后重试。",
            retriable=True,
            detail=repr(exc),
        ) from exc
