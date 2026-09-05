from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Literal

from PIL import Image

from src.config import Settings
from src.errors import AppError, ErrorCode, PipelineStage
from src.providers.image_edit_base import ImageEditProvider
from src.providers.vlm_base import VLMProvider
from src.schemas import (
    AppraisalResult,
    ImageIdentityVerdict,
    InventoryItem,
    PipelineResult,
    PipelineStatus,
    ScanRequest,
)
from src.services.image_service import canonical_image_sha256, persist_input_image
from src.services.local_file_lock import LocalFileLock
from src.services.scan_cache_repository import CacheIdentity, ScanCacheRepository

LOGGER = logging.getLogger("PIPELINE")


class ScanPipeline:
    def __init__(
        self,
        settings: Settings,
        vlm_provider: VLMProvider,
        image_provider: ImageEditProvider,
        cache_repository: ScanCacheRepository | None = None,
    ) -> None:
        self.settings = settings
        self.vlm_provider = vlm_provider
        self.image_provider = image_provider
        self.cache_repository = cache_repository
        self._pipeline_lock = LocalFileLock(settings.ai_pipeline_lock_path)

    def scan(
        self,
        image: Image.Image | None,
        request: ScanRequest,
        *,
        cache_source: Literal["runtime", "prewarm"] = "runtime",
    ) -> PipelineResult:
        if image is None and not request.description.strip():
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                stage=PipelineStage.INPUT,
                user_message="请上传一张物品照片，或填写物品文字描述。",
            )

        request_started = perf_counter()
        cache_identity = self._cache_identity(image, request)
        cached = self._cache_lookup(cache_identity, request, request_started)
        if cached is not None:
            return cached
        try:
            with self._pipeline_lock.acquire(
                self.settings.ai_pipeline_lock_timeout_seconds
            ):
                # A page request and the prewarm worker can miss simultaneously.
                # Recheck after winning the global GPU lock so only one generates.
                cached = self._cache_lookup(cache_identity, request, request_started)
                if cached is not None:
                    return cached
                return self._scan_uncached(
                    image,
                    request,
                    request_started=request_started,
                    cache_identity=cache_identity,
                    cache_source=cache_source,
                )
        except TimeoutError as exc:
            raise AppError(
                code=ErrorCode.TIMEOUT,
                stage=PipelineStage.VLM,
                user_message=(
                    "本机 AI 正在处理另一项鉴定或缓存任务，请稍后重试。"
                ),
                retriable=True,
                detail=str(exc),
            ) from exc

    def _scan_uncached(
        self,
        image: Image.Image | None,
        request: ScanRequest,
        *,
        request_started: float,
        cache_identity: CacheIdentity | None,
        cache_source: Literal["runtime", "prewarm"],
    ) -> PipelineResult:
        image_path = persist_input_image(
            image,
            self.settings.output_dir,
            self.settings.image_max_side,
        )

        # Recover safely from a prior provider failure before loading either model.
        # Repeating an unload is cheap; allowing stale models to overlap is not.
        self.release_vlm(strict=True)
        self.release_image_provider(strict=True)

        vlm_started = perf_counter()
        LOGGER.info("[VLM] provider=%s scan started", self.vlm_provider.name)
        try:
            appraisal = self.vlm_provider.analyze(image_path, request)
        except Exception:
            self.release_vlm()
            raise
        self.release_vlm(strict=True)
        vlm_ms = round((perf_counter() - vlm_started) * 1000)

        edit_started = perf_counter()
        after_path = None
        identity_attempts = 0
        identity_verdict: ImageIdentityVerdict | None = None
        status = PipelineStatus.SUCCESS
        try:
            if image_path is None:
                LOGGER.info(
                    "[IMAGE_GENERATE] provider=%s text-to-image started",
                    self.image_provider.name,
                )
                after_path = self._generate_text_image(appraisal, request)
            else:
                LOGGER.info(
                    "[IMAGE_EDIT] provider=%s edit started", self.image_provider.name
                )
                after_path, identity_attempts, identity_verdict = (
                    self._generate_verified_image(image_path, appraisal, request)
                )
            if after_path is None:
                appraisal.warnings.append(
                    "AI 文字生图服务未返回图片，可稍后单独重试。"
                    if image_path is None
                    else "AI 图像编辑服务未返回 AFTER 图片，可稍后单独重试。"
                )
                status = PipelineStatus.PARTIAL
            elif not Path(after_path).is_file():
                raise AppError(
                    code=ErrorCode.IMAGE_EDIT_FAILED,
                    stage=PipelineStage.IMAGE_EDIT,
                    user_message=(
                        "AI 图像服务未生成有效图片，可稍后单独重试。"
                    ),
                    retriable=True,
                    detail=f"missing output path: {after_path}",
                )
        except AppError as exc:
            LOGGER.exception("[IMAGE_EDIT] partial result: %s", exc.detail or exc)
            appraisal.warnings.append(exc.user_message)
            status = PipelineStatus.PARTIAL
        except Exception:
            LOGGER.exception("[IMAGE_EDIT] unexpected partial result")
            appraisal.warnings.append(
                "AI 文字生图服务暂时不可用，可稍后单独重试。"
                if image_path is None
                else "AI 图像编辑服务暂时不可用，可稍后单独重试。"
            )
            status = PipelineStatus.PARTIAL
        edit_ms = round((perf_counter() - edit_started) * 1000)

        item = InventoryItem(
            appraisal=appraisal,
            original_image=str(image_path) if image_path else None,
            apocalypse_image=str(after_path) if after_path else None,
            source_description=request.description.strip(),
            scenario=request.scenario,
            apocalypse_years=request.apocalypse_years,
        )
        total_ms = round((perf_counter() - request_started) * 1000)
        LOGGER.info("[PIPELINE] scan complete total_ms=%s", total_ms)
        result = PipelineResult(
            status=status,
            item=item,
            original_image=item.original_image,
            apocalypse_image=item.apocalypse_image,
            timings_ms={"vlm": vlm_ms, "image_edit": edit_ms, "total": total_ms},
            provider_metadata={
                "vlm": self.vlm_provider.name,
                "image_edit": self.image_provider.name,
                "mock_mode": self.settings.mock_mode,
                "generation_mode": (
                    "text_to_image" if image_path is None else "image_edit"
                ),
                "identity_verified": (
                    None
                    if image_path is None
                    else bool(
                        identity_verdict
                        and identity_verdict.is_acceptable(
                            self.settings.image_identity_min_confidence
                        )
                    )
                ),
                "identity_not_applicable": image_path is None,
                "identity_attempts": identity_attempts,
                "identity_confidence": (
                    identity_verdict.confidence if identity_verdict else None
                ),
                "cache_hit": False,
            },
        )
        self._cache_store(cache_identity, result, source=cache_source)
        return result

    def retry_image(self, item: InventoryItem) -> InventoryItem:
        try:
            with self._pipeline_lock.acquire(
                self.settings.ai_pipeline_lock_timeout_seconds
            ):
                return self._retry_image_locked(item)
        except TimeoutError as exc:
            raise AppError(
                code=ErrorCode.TIMEOUT,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="本机 AI 正在处理另一项鉴定或缓存任务，请稍后重试。",
                retriable=True,
                detail=str(exc),
            ) from exc

    def _retry_image_locked(self, item: InventoryItem) -> InventoryItem:
        request = ScanRequest(
            scenario=item.scenario,
            apocalypse_years=item.apocalypse_years,
            description=item.source_description or item.appraisal.original_item,
        )
        identity_attempts = 0
        identity_verdict: ImageIdentityVerdict | None = None
        try:
            if item.original_image is None:
                after_path = self._generate_text_image(item.appraisal, request)
            else:
                after_path, identity_attempts, identity_verdict = self._generate_verified_image(
                    Path(item.original_image),
                    item.appraisal,
                    request,
                )
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="灾后图片生成失败，请稍后重试。",
                retriable=True,
                detail=repr(exc),
            ) from exc
        if after_path is None:
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="灾后图片没有返回有效结果，请稍后重试。",
                retriable=True,
            )
        if not Path(after_path).is_file():
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="灾后图片没有生成有效文件，请稍后重试。",
                retriable=True,
                detail=f"missing output path: {after_path}",
            )

        updated = item.model_copy(deep=True)
        updated.apocalypse_image = str(after_path)
        image_warning_markers = ("图像", "图片", "生图", "AFTER", "显存", "GPU")
        updated.appraisal.warnings = [
            warning
            for warning in updated.appraisal.warnings
            if not any(marker in warning for marker in image_warning_markers)
        ]
        cache_identity = self._cache_identity_for_item(updated, request)
        retry_result = PipelineResult(
            status=PipelineStatus.SUCCESS,
            item=updated,
            original_image=updated.original_image,
            apocalypse_image=updated.apocalypse_image,
            provider_metadata={
                "vlm": self.vlm_provider.name,
                "image_edit": self.image_provider.name,
                "mock_mode": self.settings.mock_mode,
                "generation_mode": (
                    "text_to_image" if updated.original_image is None else "image_edit"
                ),
                "identity_verified": (
                    None
                    if updated.original_image is None
                    else bool(
                        identity_verdict
                        and identity_verdict.is_acceptable(
                            self.settings.image_identity_min_confidence
                        )
                    )
                ),
                "identity_not_applicable": updated.original_image is None,
                "identity_attempts": identity_attempts,
                "identity_confidence": (
                    identity_verdict.confidence if identity_verdict else None
                ),
                "cache_hit": False,
            },
        )
        self._cache_store(cache_identity, retry_result, source="runtime")
        return updated

    def _cache_identity(
        self,
        image: Image.Image | None,
        request: ScanRequest,
    ) -> CacheIdentity | None:
        if self.cache_repository is None or not self.settings.scan_cache_enabled:
            return None
        try:
            image_sha256 = (
                canonical_image_sha256(image, self.settings.image_max_side)
                if image is not None
                else None
            )
            return self.cache_repository.make_identity(
                request,
                input_image_sha256=image_sha256,
            )
        except AppError as exc:
            LOGGER.warning("[CACHE] key preparation failed: %s", exc.detail or exc)
            return None

    def _cache_identity_for_item(
        self,
        item: InventoryItem,
        request: ScanRequest,
    ) -> CacheIdentity | None:
        if self.cache_repository is None or not self.settings.scan_cache_enabled:
            return None
        try:
            image_sha256 = None
            if item.original_image is not None:
                with Image.open(item.original_image) as image:
                    image_sha256 = canonical_image_sha256(
                        image,
                        self.settings.image_max_side,
                    )
            return self.cache_repository.make_identity(
                request,
                input_image_sha256=image_sha256,
            )
        except (AppError, OSError) as exc:
            LOGGER.warning("[CACHE] retry key preparation failed: %r", exc)
            return None

    def _cache_lookup(
        self,
        identity: CacheIdentity | None,
        request: ScanRequest,
        request_started: float,
    ) -> PipelineResult | None:
        if self.cache_repository is None or identity is None:
            return None
        try:
            result = self.cache_repository.lookup(identity, request)
        except AppError as exc:
            LOGGER.warning("[CACHE] lookup failed: %s", exc.detail or exc)
            return None
        if result is None:
            return None
        total_ms = round((perf_counter() - request_started) * 1000)
        result.timings_ms = {"vlm": 0, "image_edit": 0, "total": total_ms}
        LOGGER.info("[CACHE] hit key=%s total_ms=%s", identity.cache_key, total_ms)
        return result

    def _cache_store(
        self,
        identity: CacheIdentity | None,
        result: PipelineResult,
        *,
        source: Literal["runtime", "prewarm"],
    ) -> None:
        if self.cache_repository is None or identity is None:
            return
        try:
            stored = self.cache_repository.store(identity, result, source=source)
            if stored:
                result.provider_metadata["cache_key"] = identity.cache_key
                result.provider_metadata["cache_source"] = source
                LOGGER.info("[CACHE] stored key=%s source=%s", identity.cache_key, source)
        except (AppError, OSError, ValueError) as exc:
            LOGGER.warning("[CACHE] store failed: %r", exc)

    def _generate_text_image(
        self,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path | None:
        self.release_vlm(strict=True)
        self.release_image_provider(strict=True)
        try:
            candidate_path = self.image_provider.edit(None, appraisal, request)
        except Exception:
            self.release_image_provider()
            raise
        self.release_image_provider(strict=True)
        if candidate_path is None:
            return None
        candidate_path = Path(candidate_path)
        if not candidate_path.is_file():
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="AI 文字生图服务未生成有效图片，可稍后单独重试。",
                retriable=True,
                detail=f"missing output path: {candidate_path}",
            )
        return candidate_path

    def _generate_verified_image(
        self,
        image_path: Path,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> tuple[Path | None, int, ImageIdentityVerdict | None]:
        last_verdict: ImageIdentityVerdict | None = None
        for attempt in range(1, self.settings.image_identity_max_attempts + 1):
            # An earlier failed request may have left either backend resident. Never
            # enter the GPU-heavy image stage until both cleanup calls succeed.
            self.release_vlm(strict=True)
            self.release_image_provider(strict=True)
            candidate_path: Path | None = None
            try:
                candidate_path = self.image_provider.edit(
                    image_path,
                    appraisal,
                    request,
                )
            except Exception:
                self.release_image_provider()
                raise
            self.release_image_provider(strict=True)

            if candidate_path is None:
                return None, attempt, None
            candidate_path = Path(candidate_path)
            if not candidate_path.is_file():
                raise AppError(
                    code=ErrorCode.IMAGE_EDIT_FAILED,
                    stage=PipelineStage.IMAGE_EDIT,
                    user_message=(
                        "AI 图像编辑服务未生成有效的 AFTER 图片，可稍后单独重试。"
                    ),
                    retriable=True,
                    detail=f"missing output path: {candidate_path}",
                )

            LOGGER.info("[VLM] identity check attempt=%s started", attempt)
            try:
                last_verdict = self.vlm_provider.verify_identity(
                    image_path,
                    candidate_path,
                    appraisal,
                    request,
                )
            except Exception:
                self.release_vlm()
                raise
            self.release_vlm(strict=True)

            if last_verdict.is_acceptable(
                self.settings.image_identity_min_confidence
            ):
                LOGGER.info(
                    "[VLM] identity check accepted attempt=%s confidence=%.2f",
                    attempt,
                    last_verdict.confidence,
                )
                return candidate_path, attempt, last_verdict
            LOGGER.warning(
                "[VLM] identity check rejected attempt=%s confidence=%.2f "
                "same=%s category=%s silhouette=%s camera=%s functional=%s "
                "surface_only=%s damage_visible=%s added=%s missing=%s moved=%s issues=%s",
                attempt,
                last_verdict.confidence,
                last_verdict.same_physical_object,
                last_verdict.category_preserved,
                last_verdict.silhouette_and_proportions_preserved,
                last_verdict.camera_and_composition_preserved,
                last_verdict.functional_features_preserved,
                last_verdict.only_surface_condition_changed,
                last_verdict.post_apocalyptic_damage_clearly_visible,
                last_verdict.added_features,
                last_verdict.missing_features,
                last_verdict.moved_or_duplicated_features,
                last_verdict.issues,
            )
            self._discard_rejected_candidate(candidate_path, image_path)

        detail = last_verdict.model_dump_json() if last_verdict else "no verdict"
        raise AppError(
            code=ErrorCode.IMAGE_EDIT_FAILED,
            stage=PipelineStage.IMAGE_EDIT,
            user_message=(
                "生成的 AFTER 图片未通过同一实物结构或灾后损伤强度核验，已拦截；"
                "可点击按钮单独重试图片。"
            ),
            retriable=True,
            detail=detail,
        )

    def _discard_rejected_candidate(
        self,
        candidate_path: Path,
        source_path: Path,
    ) -> None:
        try:
            candidate = candidate_path.resolve(strict=True)
            source = source_path.resolve(strict=True)
            output_root = self.settings.output_dir.resolve(strict=True)
            candidate.relative_to(output_root)
            if candidate == source:
                raise ValueError("candidate resolves to source image")
            candidate.unlink()
            LOGGER.info("[IMAGE_EDIT] removed rejected candidate=%s", candidate.name)
        except (OSError, ValueError) as exc:
            LOGGER.warning(
                "[IMAGE_EDIT] rejected candidate retained because safe cleanup failed: %r",
                exc,
            )

    def release_vlm(self, *, strict: bool = False) -> None:
        try:
            self.vlm_provider.unload()
        except Exception as exc:
            LOGGER.warning("[VLM] unload failed: %r", exc)
            if strict:
                raise AppError(
                    code=ErrorCode.PROVIDER_UNAVAILABLE,
                    stage=PipelineStage.VLM,
                    user_message=(
                        "无法确认视觉模型已释放；为避免显存冲突，本次流程已安全停止。"
                    ),
                    retriable=True,
                    detail=repr(exc),
                ) from exc

    def release_image_provider(self, *, strict: bool = False) -> None:
        try:
            self.image_provider.unload()
        except Exception as exc:
            LOGGER.warning("[IMAGE_EDIT] unload failed: %r", exc)
            if strict:
                raise AppError(
                    code=ErrorCode.PROVIDER_UNAVAILABLE,
                    stage=PipelineStage.IMAGE_EDIT,
                    user_message=(
                        exc.user_message
                        if isinstance(exc, AppError)
                        else "无法确认图像模型已释放；为避免显存冲突，本次流程已安全停止。"
                    ),
                    retriable=True,
                    detail=repr(exc),
                ) from exc
