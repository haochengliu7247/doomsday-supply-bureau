from pathlib import Path

import pytest
from PIL import Image

from src.config import Settings
from src.errors import AppError, ErrorCode, PipelineStage
from src.pipeline import ScanPipeline
from src.providers.mock_provider import MockVLMProvider
from src.schemas import (
    ApocalypseScenario,
    AppraisalResult,
    ImageIdentityVerdict,
    PipelineStatus,
    ScanRequest,
)
from src.services.scan_cache_repository import ScanCacheRepository


class FailingImageProvider:
    name = "failing-image"

    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path | None:
        raise AppError(
            code=ErrorCode.IMAGE_EDIT_FAILED,
            stage=PipelineStage.IMAGE_EDIT,
            user_message="图像编辑暂时不可用。",
            retriable=True,
        )

    def unload(self) -> None:
        return None


class IdentityImageProvider:
    name = "identity-image"

    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path | None:
        return image_path

    def unload(self) -> None:
        return None


class FlakyImageProvider:
    name = "flaky-image"

    def __init__(self) -> None:
        self.calls = 0

    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path | None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary backend failure")
        return image_path

    def unload(self) -> None:
        return None


class TextImageProvider:
    name = "text-image"

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.calls = 0
        self.descriptions: list[str] = []

    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path:
        assert image_path is None
        self.calls += 1
        self.descriptions.append(request.description)
        candidate = self.output_dir / f"text-candidate-{self.calls}.png"
        Image.new("RGB", (20, 20), (0, self.calls, 0)).save(candidate)
        return candidate

    def unload(self) -> None:
        return None


class FlakyTextImageProvider(TextImageProvider):
    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path:
        if self.calls == 0:
            self.calls += 1
            self.descriptions.append(request.description)
            raise AppError(
                code=ErrorCode.IMAGE_EDIT_FAILED,
                stage=PipelineStage.IMAGE_EDIT,
                user_message="文字生图暂时不可用。",
                retriable=True,
            )
        return super().edit(image_path, appraisal, request)


class FailingVLMProvider:
    name = "failing-vlm"

    def __init__(self) -> None:
        self.unload_calls = 0

    def analyze(
        self,
        image_path: Path | None,
        request: ScanRequest,
    ) -> AppraisalResult:
        raise AppError(
            code=ErrorCode.TIMEOUT,
            stage=PipelineStage.VLM,
            user_message="视觉鉴定超时。",
            retriable=True,
        )

    def unload(self) -> None:
        self.unload_calls += 1


class CountingVLMProvider(MockVLMProvider):
    name = "counting-vlm"

    def __init__(self) -> None:
        self.analyze_calls = 0
        self.unload_calls = 0

    def analyze(
        self,
        image_path: Path | None,
        scan_request: ScanRequest,
    ) -> AppraisalResult:
        self.analyze_calls += 1
        return super().analyze(image_path, scan_request)

    def unload(self) -> None:
        self.unload_calls += 1


def identity_verdict(
    *,
    accepted: bool,
    damage_visible: bool | None = None,
) -> ImageIdentityVerdict:
    differences = [] if accepted else ["new side opening"]
    return ImageIdentityVerdict(
        same_physical_object=True,
        category_preserved=True,
        silhouette_and_proportions_preserved=True,
        camera_and_composition_preserved=True,
        functional_features_preserved=accepted,
        only_surface_condition_changed=accepted,
        post_apocalyptic_damage_clearly_visible=(
            accepted if damage_visible is None else damage_visible
        ),
        before_functional_features=["one side button"],
        after_functional_features=(
            ["one side button"] if accepted else ["one side button", "one side opening"]
        ),
        added_features=differences,
        missing_features=[],
        moved_or_duplicated_features=[],
        issues=differences,
        confidence=0.95,
    )


class SequencedVLMProvider(MockVLMProvider):
    name = "sequenced-vlm"

    def __init__(
        self,
        verdicts: list[ImageIdentityVerdict],
        events: list[str],
    ) -> None:
        self.verdicts = verdicts
        self.events = events

    def analyze(
        self,
        image_path: Path | None,
        scan_request: ScanRequest,
    ) -> AppraisalResult:
        self.events.append("analyze")
        return super().analyze(image_path, scan_request)

    def verify_identity(
        self,
        source_path: Path,
        candidate_path: Path,
        appraisal: AppraisalResult,
        scan_request: ScanRequest,
    ) -> ImageIdentityVerdict:
        self.events.append("verify")
        return self.verdicts.pop(0)

    def unload(self) -> None:
        self.events.append("unload-vlm")


class SequencedImageProvider:
    name = "sequenced-image"

    def __init__(self, output_dir: Path, events: list[str]) -> None:
        self.output_dir = output_dir
        self.events = events
        self.calls = 0

    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        scan_request: ScanRequest,
    ) -> Path:
        assert image_path is not None
        self.calls += 1
        self.events.append(f"edit-{self.calls}")
        candidate = self.output_dir / f"candidate-{self.calls}.png"
        Image.new("RGB", (20, 20), (self.calls, 0, 0)).save(candidate)
        return candidate

    def unload(self) -> None:
        self.events.append("unload-image")


def request() -> ScanRequest:
    return ScanRequest(
        scenario=ApocalypseScenario.CITY_BLACKOUT,
        apocalypse_years=3,
        description="测试物品",
    )


def test_pipeline_rejects_empty_input(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, image_max_side=1024)
    pipeline = ScanPipeline(
        settings,
        MockVLMProvider(),
        IdentityImageProvider(),
    )
    with pytest.raises(AppError) as error:
        pipeline.scan(
            None,
            ScanRequest(
                scenario=ApocalypseScenario.CITY_BLACKOUT,
                apocalypse_years=3,
                description="",
            ),
        )
    assert error.value.code == ErrorCode.INVALID_INPUT


def test_text_only_scan_generates_after_without_identity_check(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, image_max_side=1024)
    image_provider = TextImageProvider(tmp_path)
    pipeline = ScanPipeline(
        settings,
        MockVLMProvider(),
        image_provider,
    )

    result = pipeline.scan(None, request())

    assert result.status is PipelineStatus.SUCCESS
    assert result.original_image is None
    assert result.apocalypse_image == str(tmp_path / "text-candidate-1.png")
    assert image_provider.calls == 1
    assert result.provider_metadata["generation_mode"] == "text_to_image"
    assert result.provider_metadata["identity_verified"] is None
    assert result.provider_metadata["identity_not_applicable"] is True
    assert result.provider_metadata["identity_attempts"] == 0


def test_complete_text_result_cache_skips_both_models_on_repeat(
    tmp_path: Path,
) -> None:
    settings = Settings(
        mock_mode=True,
        vlm_provider="mock",
        image_provider="mock",
        output_dir=tmp_path / "outputs",
        database_url=f"sqlite:///{(tmp_path / 'cache.db').as_posix()}",
        ai_pipeline_lock_path=tmp_path / "pipeline.lock",
        image_max_side=1024,
    )
    settings.ensure_directories()
    cache = ScanCacheRepository(settings)
    cache.initialize()
    vlm = CountingVLMProvider()
    image_provider = TextImageProvider(settings.output_dir)
    pipeline = ScanPipeline(
        settings,
        vlm,
        image_provider,
        cache_repository=cache,
    )

    first = pipeline.scan(None, request())
    unloads_after_first = vlm.unload_calls
    second_request = request().model_copy(update={"description": "  测试物品　"})
    second = pipeline.scan(None, second_request)

    assert first.provider_metadata["cache_hit"] is False
    assert second.provider_metadata["cache_hit"] is True
    assert first.provider_metadata["cache_key"] == second.provider_metadata["cache_key"]
    assert vlm.analyze_calls == 1
    assert vlm.unload_calls == unloads_after_first
    assert image_provider.calls == 1
    assert first.item.item_id != second.item.item_id


def test_text_only_generation_failure_returns_retryable_partial(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, image_max_side=1024)
    pipeline = ScanPipeline(
        settings,
        MockVLMProvider(),
        FailingImageProvider(),
    )

    result = pipeline.scan(None, request())

    assert result.status is PipelineStatus.PARTIAL
    assert result.original_image is None
    assert result.apocalypse_image is None
    assert "图像编辑暂时不可用。" in result.item.appraisal.warnings


def test_text_only_partial_can_retry_without_rescanning(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, image_max_side=1024)
    provider = FlakyTextImageProvider(tmp_path)
    pipeline = ScanPipeline(settings, MockVLMProvider(), provider)
    detailed_request = ScanRequest(
        scenario=ApocalypseScenario.CITY_BLACKOUT,
        apocalypse_years=3,
        description="透明的 750 毫升塑料运动水瓶，蓝色旋盖，瓶身有纵向防滑凹槽",
    )

    result = pipeline.scan(None, detailed_request)
    result.item.appraisal.warnings.append(
        "GPU 可用显存未恢复到安全阈值；已停止以避免冲突。"
    )
    updated = pipeline.retry_image(result.item)

    assert result.status is PipelineStatus.PARTIAL
    assert provider.calls == 2
    assert provider.descriptions == [
        detailed_request.description,
        detailed_request.description,
    ]
    assert updated.source_description == detailed_request.description
    assert updated.item_id == result.item.item_id
    assert updated.original_image is None
    assert updated.apocalypse_image == str(tmp_path / "text-candidate-2.png")
    assert "文字生图暂时不可用。" not in updated.appraisal.warnings
    assert not any("显存" in warning for warning in updated.appraisal.warnings)


def test_vlm_is_unloaded_when_analysis_fails(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, image_max_side=1024)
    provider = FailingVLMProvider()
    pipeline = ScanPipeline(settings, provider, IdentityImageProvider())

    with pytest.raises(AppError) as error:
        pipeline.scan(None, request())

    assert error.value.code is ErrorCode.TIMEOUT
    # One preflight release plus one best-effort cleanup after the failed call.
    assert provider.unload_calls == 2


def test_pipeline_persists_bounded_image_and_returns_card(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, image_max_side=1024)
    pipeline = ScanPipeline(
        settings,
        MockVLMProvider(),
        IdentityImageProvider(),
    )
    image = Image.new("RGB", (2400, 1200), "gray")
    result = pipeline.scan(image, request())

    assert result.status is PipelineStatus.SUCCESS
    assert result.item.appraisal.apocalypse_name == "便携式能源储备核心"
    assert result.original_image is not None
    with Image.open(result.original_image) as persisted:
        assert max(persisted.size) == 1024


def test_image_failure_returns_partial_appraisal(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, image_max_side=1024)
    pipeline = ScanPipeline(
        settings,
        MockVLMProvider(),
        FailingImageProvider(),
    )
    result = pipeline.scan(Image.new("RGB", (200, 200), "black"), request())
    assert result.status is PipelineStatus.PARTIAL
    assert result.apocalypse_image is None
    assert result.item.appraisal.apocalypse_name
    assert "图像编辑暂时不可用。" in result.item.appraisal.warnings


def test_partial_image_can_retry_without_rescanning(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, image_max_side=1024)
    provider = FlakyImageProvider()
    pipeline = ScanPipeline(settings, MockVLMProvider(), provider)

    result = pipeline.scan(Image.new("RGB", (200, 200), "black"), request())
    original_id = result.item.item_id
    original_appraisal = result.item.appraisal.model_copy(deep=True)

    updated = pipeline.retry_image(result.item)

    assert result.status is PipelineStatus.PARTIAL
    assert provider.calls == 2
    assert updated.item_id == original_id
    assert updated.apocalypse_image == updated.original_image
    assert updated.appraisal.apocalypse_name == original_appraisal.apocalypse_name
    assert result.item.apocalypse_image is None


def test_identity_rejection_retries_with_new_candidate(tmp_path: Path) -> None:
    events: list[str] = []
    settings = Settings(
        output_dir=tmp_path,
        image_max_side=1024,
        image_identity_max_attempts=2,
    )
    vlm = SequencedVLMProvider(
        [identity_verdict(accepted=False), identity_verdict(accepted=True)],
        events,
    )
    image_provider = SequencedImageProvider(tmp_path, events)
    pipeline = ScanPipeline(settings, vlm, image_provider)

    result = pipeline.scan(Image.new("RGB", (40, 40), "black"), request())

    assert result.status is PipelineStatus.SUCCESS
    assert result.apocalypse_image == str(tmp_path / "candidate-2.png")
    assert result.provider_metadata["identity_verified"] is True
    assert result.provider_metadata["identity_attempts"] == 2
    assert image_provider.calls == 2
    assert events == [
        "unload-vlm",
        "unload-image",
        "analyze",
        "unload-vlm",
        "unload-vlm",
        "unload-image",
        "edit-1",
        "unload-image",
        "verify",
        "unload-vlm",
        "unload-vlm",
        "unload-image",
        "edit-2",
        "unload-image",
        "verify",
        "unload-vlm",
    ]


def test_insufficient_damage_retries_even_when_identity_is_preserved(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    settings = Settings(
        output_dir=tmp_path,
        image_max_side=1024,
        image_identity_max_attempts=2,
    )
    image_provider = SequencedImageProvider(tmp_path, events)
    pipeline = ScanPipeline(
        settings,
        SequencedVLMProvider(
            [
                identity_verdict(accepted=True, damage_visible=False),
                identity_verdict(accepted=True, damage_visible=True),
            ],
            events,
        ),
        image_provider,
    )

    result = pipeline.scan(Image.new("RGB", (40, 40), "black"), request())

    assert result.status is PipelineStatus.SUCCESS
    assert result.provider_metadata["identity_attempts"] == 2
    assert image_provider.calls == 2
    assert not (tmp_path / "candidate-1.png").exists()
    assert (tmp_path / "candidate-2.png").is_file()


def test_rejected_candidates_are_never_exposed(tmp_path: Path) -> None:
    events: list[str] = []
    settings = Settings(
        output_dir=tmp_path,
        image_max_side=1024,
        image_identity_max_attempts=2,
    )
    pipeline = ScanPipeline(
        settings,
        SequencedVLMProvider(
            [identity_verdict(accepted=False), identity_verdict(accepted=False)],
            events,
        ),
        SequencedImageProvider(tmp_path, events),
    )

    result = pipeline.scan(Image.new("RGB", (40, 40), "black"), request())

    assert result.status is PipelineStatus.PARTIAL
    assert result.apocalypse_image is None
    assert result.item.apocalypse_image is None
    assert result.provider_metadata["identity_verified"] is False
    assert any("结构或灾后损伤强度核验" in warning for warning in result.item.appraisal.warnings)
    assert not (tmp_path / "candidate-1.png").exists()
    assert not (tmp_path / "candidate-2.png").exists()
