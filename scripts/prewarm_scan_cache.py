from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from src.config import get_settings
from src.data.common_items import COMMON_ITEMS_PATH, load_common_item_lexicon
from src.logging_config import configure_logging
from src.pipeline import ScanPipeline
from src.providers import create_image_provider, create_vlm_provider
from src.schemas import ApocalypseScenario, PipelineStatus, ScanRequest
from src.services.local_file_lock import LocalFileLock
from src.services.scan_cache_repository import ScanCacheRepository

LOGGER = logging.getLogger("CACHE.PREWARM")
DEFAULT_SCENARIO = ApocalypseScenario.CITY_BLACKOUT
DEFAULT_YEARS = 3.0
MINIMUM_FREE_DISK_BYTES = 10 * 1024**3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="可续跑地预生成常用物品完整结果缓存。"
    )
    parser.add_argument("--lexicon", type=Path, default=COMMON_ITEMS_PATH)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _status_payload(
    repository: ScanCacheRepository,
    lexicon_version: str,
) -> dict[str, int | str]:
    return {
        "lexicon_version": lexicon_version,
        **repository.prewarm_progress(lexicon_version),
        "cache_rows": repository.cache_count(),
    }


def _print_status(repository: ScanCacheRepository, lexicon_version: str) -> None:
    print(json.dumps(_status_payload(repository, lexicon_version), ensure_ascii=False))


def _close_provider(provider: object) -> None:
    close = getattr(provider, "close", None)
    if callable(close):
        close()


def run() -> int:
    args = _parser().parse_args()
    if args.max_attempts < 1 or args.max_attempts > 10:
        raise SystemExit("--max-attempts 必须在 1 到 10 之间。")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须大于 0。")
    if args.delay_seconds < 0 or args.delay_seconds > 60:
        raise SystemExit("--delay-seconds 必须在 0 到 60 秒之间。")

    settings = get_settings()
    configure_logging(settings.log_level)
    lexicon = load_common_item_lexicon(args.lexicon.resolve())
    repository = ScanCacheRepository(settings)
    repository.initialize()

    sample_request = ScanRequest(
        scenario=DEFAULT_SCENARIO,
        apocalypse_years=DEFAULT_YEARS,
        description=lexicon.categories[0].items[0].canonical,
    )
    signature = repository.make_identity(
        sample_request,
        input_image_sha256=None,
    ).pipeline_signature
    repository.register_prewarm_manifest(
        lexicon_version=lexicon.version,
        items=lexicon.manifest_rows(),
        aliases=lexicon.alias_map(),
        pipeline_signature=signature,
    )

    if args.status or args.validate_only:
        _print_status(repository, lexicon.version)
        return 0
    if settings.mock_mode:
        raise SystemExit("预生成正式缓存要求 MOCK_MODE=false。")

    worker_lock = LocalFileLock(
        settings.ai_pipeline_lock_path.with_name("scan_cache_prewarm.lock")
    )
    try:
        worker_context = worker_lock.acquire(0.2)
        worker_context.__enter__()
    except TimeoutError:
        LOGGER.error("另一个完整缓存预生成任务已经在运行。")
        return 3

    vlm_provider = create_vlm_provider(settings)
    image_provider = create_image_provider(settings)
    pipeline = ScanPipeline(
        settings,
        vlm_provider,
        image_provider,
        cache_repository=repository,
    )
    processed = 0
    consecutive_failures = 0
    recovered = repository.recover_interrupted_prewarm(lexicon.version)
    if recovered:
        LOGGER.warning("已恢复 %s 个上次中断的任务。", recovered)

    try:
        while args.limit is None or processed < args.limit:
            if processed % 10 == 0:
                free_bytes = shutil.disk_usage(settings.output_dir).free
                if free_bytes < MINIMUM_FREE_DISK_BYTES:
                    LOGGER.error("磁盘剩余空间低于 10GiB，批量任务已安全停止。")
                    return 4

            item = repository.claim_next_prewarm(
                lexicon.version,
                max_attempts=args.max_attempts,
            )
            if item is None:
                break
            request = ScanRequest(
                scenario=DEFAULT_SCENARIO,
                apocalypse_years=DEFAULT_YEARS,
                description=item.canonical_description,
            )
            started = time.perf_counter()
            try:
                result = pipeline.scan(None, request, cache_source="prewarm")
                duration_ms = round((time.perf_counter() - started) * 1000)
                cache_key = result.provider_metadata.get("cache_key")
                succeeded = (
                    result.status is PipelineStatus.SUCCESS
                    and isinstance(cache_key, str)
                    and len(cache_key) == 64
                )
                error = None
                if not succeeded:
                    error = " ".join(result.item.appraisal.warnings) or (
                        "完整结果未进入缓存"
                    )
                repository.finish_prewarm(
                    item,
                    succeeded=succeeded,
                    cache_key=cache_key if isinstance(cache_key, str) else None,
                    duration_ms=duration_ms,
                    error=error,
                )
                if succeeded:
                    consecutive_failures = 0
                    LOGGER.info(
                        "[%s/1000] %s 已缓存，耗时 %.1f 秒%s",
                        item.ordinal,
                        item.canonical_description,
                        duration_ms / 1000,
                        "（已有缓存）"
                        if result.provider_metadata.get("cache_hit") is True
                        else "",
                    )
                else:
                    consecutive_failures += 1
                    LOGGER.error(
                        "[%s/1000] %s 未得到完整结果：%s",
                        item.ordinal,
                        item.canonical_description,
                        error,
                    )
            except Exception as exc:
                duration_ms = round((time.perf_counter() - started) * 1000)
                repository.finish_prewarm(
                    item,
                    succeeded=False,
                    cache_key=None,
                    duration_ms=duration_ms,
                    error=repr(exc),
                )
                consecutive_failures += 1
                LOGGER.exception(
                    "[%s/1000] %s 生成失败",
                    item.ordinal,
                    item.canonical_description,
                )
            processed += 1
            _print_status(repository, lexicon.version)
            sys.stdout.flush()
            if consecutive_failures >= 3:
                LOGGER.error("连续 3 项失败，已触发保护性停止；修复后可直接续跑。")
                return 5
            if args.delay_seconds:
                time.sleep(args.delay_seconds)
    except KeyboardInterrupt:
        LOGGER.warning("收到停止请求，将从当前检查点安全退出。")
        return 130
    finally:
        pipeline.release_vlm()
        pipeline.release_image_provider()
        _close_provider(vlm_provider)
        _close_provider(image_provider)
        worker_context.__exit__(None, None, None)

    progress = repository.prewarm_progress(lexicon.version)
    _print_status(repository, lexicon.version)
    if progress["ready"] == progress["total"] == 1000:
        LOGGER.info("1000 项完整结果缓存已全部生成。")
        return 0
    if args.limit is not None and processed >= args.limit:
        LOGGER.info(
            "已达到本轮 --limit=%s；其余项目保留在检查点中，可继续生成。",
            args.limit,
        )
        return 0
    LOGGER.warning("本轮结束，但仍有达到重试上限的失败项。")
    return 2


if __name__ == "__main__":
    raise SystemExit(run())
