import sqlite3
from pathlib import Path

from PIL import Image

from src.config import Settings
from src.providers.mock_provider import MockVLMProvider
from src.schemas import (
    ApocalypseScenario,
    InventoryItem,
    PipelineResult,
    PipelineStatus,
    ScanRequest,
)
from src.services.scan_cache_repository import (
    ScanCacheRepository,
    normalize_description,
)


def settings_for(tmp_path: Path, *, model: str = "cache-model") -> Settings:
    return Settings(
        mock_mode=True,
        vlm_provider="mock",
        image_provider="mock",
        ollama_model=model,
        database_url=f"sqlite:///{(tmp_path / 'cache.db').as_posix()}",
        output_dir=tmp_path / "outputs",
        ai_pipeline_lock_path=tmp_path / "pipeline.lock",
    )


def request(
    description: str = "透明塑料水瓶",
    *,
    scenario: ApocalypseScenario = ApocalypseScenario.CITY_BLACKOUT,
    years: float = 3,
) -> ScanRequest:
    return ScanRequest(
        scenario=scenario,
        apocalypse_years=years,
        description=description,
    )


def complete_result(
    settings: Settings,
    scan_request: ScanRequest,
    *,
    photo: bool = False,
    identity_verified: bool | None = None,
) -> PipelineResult:
    settings.ensure_directories()
    before = settings.output_dir / "before.jpg"
    after = settings.output_dir / "after.png"
    if photo:
        Image.new("RGB", (30, 20), "white").save(before)
    Image.new("RGB", (30, 20), "gray").save(after)
    appraisal = MockVLMProvider().analyze(before if photo else None, scan_request)
    item = InventoryItem(
        appraisal=appraisal,
        original_image=str(before) if photo else None,
        apocalypse_image=str(after),
        source_description=scan_request.description,
        scenario=scan_request.scenario,
        apocalypse_years=scan_request.apocalypse_years,
    )
    return PipelineResult(
        status=PipelineStatus.SUCCESS,
        item=item,
        original_image=item.original_image,
        apocalypse_image=item.apocalypse_image,
        provider_metadata={
            "generation_mode": "image_edit" if photo else "text_to_image",
            "identity_verified": identity_verified,
        },
    )


def repository_for(tmp_path: Path) -> tuple[Settings, ScanCacheRepository]:
    settings = settings_for(tmp_path)
    repository = ScanCacheRepository(settings)
    repository.initialize()
    return settings, repository


def test_text_cache_round_trip_validates_files_and_returns_fresh_items(
    tmp_path: Path,
) -> None:
    settings, repository = repository_for(tmp_path)
    scan_request = request()
    identity = repository.make_identity(scan_request, input_image_sha256=None)

    assert repository.store(identity, complete_result(settings, scan_request)) is True
    first = repository.lookup(identity, scan_request)
    second = repository.lookup(identity, scan_request)

    assert first is not None and second is not None
    assert first.provider_metadata["cache_hit"] is True
    assert first.provider_metadata["cache_key"] == identity.cache_key
    assert first.item.item_id != second.item.item_id
    assert first.item.appraisal == second.item.appraisal
    with sqlite3.connect(repository.database_path) as connection:
        hit_count = connection.execute(
            "SELECT hit_count FROM scan_cache WHERE cache_key = ?",
            (identity.cache_key,),
        ).fetchone()[0]
    assert hit_count == 2


def test_normalization_aliases_and_request_dimensions_control_hits(
    tmp_path: Path,
) -> None:
    settings, repository = repository_for(tmp_path)
    repository.register_prewarm_manifest(
        lexicon_version="test-v1",
        items=[(1, "electronics", "移动电源")],
        aliases={"充电宝": "移动电源"},
        pipeline_signature=repository.make_identity(
            request("移动电源"), input_image_sha256=None
        ).pipeline_signature,
    )
    canonical = repository.make_identity(request("移动电源"), input_image_sha256=None)
    alias = repository.make_identity(request("  充电宝　"), input_image_sha256=None)
    different_year = repository.make_identity(
        request("移动电源", years=10), input_image_sha256=None
    )
    different_scenario = repository.make_identity(
        request("移动电源", scenario=ApocalypseScenario.FLOOD),
        input_image_sha256=None,
    )

    assert normalize_description("  充电宝　") == "充电宝"
    assert alias.cache_key == canonical.cache_key
    assert different_year.cache_key != canonical.cache_key
    assert different_scenario.cache_key != canonical.cache_key
    assert repository.store(canonical, complete_result(settings, request("移动电源")))
    assert repository.lookup(alias, request("充电宝")) is not None


def test_missing_or_modified_cached_image_invalidates_row(tmp_path: Path) -> None:
    settings, repository = repository_for(tmp_path)
    scan_request = request()
    identity = repository.make_identity(scan_request, input_image_sha256=None)
    result = complete_result(settings, scan_request)
    assert repository.store(identity, result)
    Path(result.apocalypse_image or "").write_bytes(b"not the cached image")

    assert repository.lookup(identity, scan_request) is None
    assert repository.cache_count() == 0


def test_cache_signature_changes_with_model_tag(tmp_path: Path) -> None:
    settings, repository = repository_for(tmp_path)
    scan_request = request()
    first = repository.make_identity(scan_request, input_image_sha256=None)
    assert repository.store(first, complete_result(settings, scan_request))

    changed = ScanCacheRepository(settings_for(tmp_path, model="changed-model"))
    changed.initialize()
    second = changed.make_identity(scan_request, input_image_sha256=None)

    assert second.cache_key != first.cache_key
    assert changed.lookup(second, scan_request) is None


def test_only_complete_verified_results_are_cached(tmp_path: Path) -> None:
    settings, repository = repository_for(tmp_path)
    scan_request = request()
    text_identity = repository.make_identity(scan_request, input_image_sha256=None)
    partial = complete_result(settings, scan_request)
    partial.status = PipelineStatus.PARTIAL
    assert repository.store(text_identity, partial) is False

    photo_identity = repository.make_identity(
        scan_request,
        input_image_sha256="a" * 64,
    )
    unverified = complete_result(
        settings,
        scan_request,
        photo=True,
        identity_verified=False,
    )
    assert repository.store(photo_identity, unverified) is False

    verified = complete_result(
        settings,
        scan_request,
        photo=True,
        identity_verified=True,
    )
    assert repository.store(photo_identity, verified) is True
    cached = repository.lookup(photo_identity, scan_request)
    assert cached is not None
    assert cached.original_image is not None


def test_prewarm_manifest_is_resumable_and_signature_aware(tmp_path: Path) -> None:
    _, repository = repository_for(tmp_path)
    items = [(1, "home", "枕头"), (2, "home", "被子")]
    repository.register_prewarm_manifest(
        lexicon_version="test-v1",
        items=items,
        aliases={},
        pipeline_signature="signature-one",
    )

    first = repository.claim_next_prewarm("test-v1", max_attempts=3)
    assert first is not None and first.canonical_description == "枕头"
    assert repository.prewarm_progress("test-v1")["running"] == 1
    assert repository.recover_interrupted_prewarm("test-v1") == 1
    resumed = repository.claim_next_prewarm("test-v1", max_attempts=3)
    assert resumed is not None and resumed.canonical_description == "枕头"
    repository.finish_prewarm(
        resumed,
        succeeded=True,
        cache_key="b" * 64,
        duration_ms=1200,
    )
    assert repository.prewarm_progress("test-v1") == {
        "pending": 1,
        "running": 0,
        "ready": 1,
        "failed": 0,
        "total": 2,
    }

    repository.register_prewarm_manifest(
        lexicon_version="test-v1",
        items=items,
        aliases={},
        pipeline_signature="signature-two",
    )
    assert repository.prewarm_progress("test-v1")["pending"] == 2
