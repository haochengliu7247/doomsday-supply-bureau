from __future__ import annotations

import logging
from collections.abc import Callable
from html import escape
from pathlib import Path

import gradio as gr
from PIL import Image

from src.config import Settings
from src.data.market_catalog import MARKET_ASSET_DIR, MARKET_CATALOG
from src.errors import AppError
from src.game.balance import GRADE_MORALE_REWARDS, GRADE_WATER_REWARDS
from src.game.engine import (
    abandon_inventory_item,
    abandon_market_inventory_item,
    abandon_pending_item,
    apply_inventory_item,
    apply_market_inventory_item,
    apply_pending_item,
    buy_market_item,
    can_scavenge,
    register_scanned_item,
    rest,
    sell_inventory_item,
    sell_pending_item,
    store_pending_item,
)
from src.pipeline import ScanPipeline
from src.providers import create_image_provider, create_vlm_provider
from src.providers.mock_provider import MockVLMProvider
from src.schemas import (
    ApocalypseScenario,
    GameState,
    InventoryItem,
    MarketItem,
    PipelineStatus,
    ScanRequest,
    StateTransition,
)
from src.services.game_repository import GameRepository
from src.services.scan_cache_repository import ScanCacheRepository
from src.ui.renderers import (
    MARKET_CATEGORY_LABELS,
    active_inventory_items,
    inventory_gallery_entries,
    inventory_gallery_value,
    market_effect_text,
    market_gallery_value,
    render_appraisal,
    render_effects,
    render_inventory,
    render_log,
    render_market,
    render_notice,
    render_profile_summary,
    render_status,
)

LOGGER = logging.getLogger("UI")
ROOT = Path(__file__).resolve().parents[2]
DEMO_BEFORE = ROOT / "assets" / "demo" / "powerbank_before.jpg"
DEMO_AFTER = ROOT / "assets" / "demo" / "powerbank_after.jpg"

DEFAULT_PROGRESS = gr.Progress(track_tqdm=False)


def _hydrate_state(raw_state: dict | None) -> GameState:
    return GameState.model_validate(raw_state or {})


def _state_payload(state: GameState) -> dict:
    return state.model_dump(mode="json", exclude_computed_fields=True)


def _action_updates(
    item: InventoryItem | None,
) -> tuple[dict, dict, dict, dict, dict]:
    has_pending = item is not None
    needs_image_retry = bool(item is not None and item.apocalypse_image is None)
    actionable = bool(
        item is not None and not item.appraisal.is_fallback and not needs_image_retry
    )
    return (
        gr.update(interactive=actionable),
        gr.update(interactive=actionable),
        gr.update(interactive=actionable),
        gr.update(interactive=has_pending),
        gr.update(visible=needs_image_retry, interactive=needs_image_retry),
    )


def _inventory_gallery_update(state: GameState) -> dict:
    values = inventory_gallery_value(state)
    return gr.update(
        value=values,
        selected_index=None,
    )


def _market_items(repository: GameRepository, state: GameState) -> list[MarketItem]:
    try:
        return repository.get_market_items(state.market_item_ids)
    except AppError as exc:
        LOGGER.warning("[DATABASE] market catalog read failed: %s", exc.detail)
        return []


def _profile_choices(repository: GameRepository, fallback: str) -> list[str]:
    try:
        return repository.list_profiles()
    except AppError as exc:
        LOGGER.warning("[DATABASE] profile list read failed: %s", exc.detail)
        return [fallback]


def _market_gallery_update(items: list[MarketItem]) -> dict:
    values = market_gallery_value(items)
    return gr.update(
        value=values,
        selected_index=None,
    )


def _selection_placeholder(kind: str) -> str:
    return (
        '<div class="selection-readout empty">'
        f"请先点选一件{escape(kind)}的图片卡片。</div>"
    )


def _delete_confirmation_copy(profile_name: str) -> str:
    return (
        '<div class="delete-confirm-copy">'
        f"<strong>永久删除存档“{escape(profile_name)}”？</strong>"
        "<span>角色状态、背包与生存记录会从数据库移除，且无法恢复。"
        "删除后将自动切换到另一个存档。</span></div>"
    )


def _profile_manager_label(state: GameState) -> str:
    return f"存档管理 · 当前：{state.profile_name} · DAY {state.player.day}"


def _transition_outputs(
    transition: StateTransition,
    mock_mode: bool,
    repository: GameRepository,
    *,
    notice_message: str | None = None,
    notice_tone: str | None = None,
) -> tuple:
    state = transition.state
    market_items = _market_items(repository, state)
    message = notice_message or " ".join(transition.messages) or "状态已更新。"
    tone = notice_tone or ("info" if transition.succeeded else "error")
    pending_updates = _action_updates(state.pending_item)
    return (
        _state_payload(state),
        render_status(state, mock_mode),
        render_notice(message, tone),
        render_effects(transition),
        render_inventory(state),
        _inventory_gallery_update(state),
        "",
        _selection_placeholder("背包物资"),
        render_market(state, market_items),
        _market_gallery_update(market_items),
        "",
        _selection_placeholder("市场商品"),
        render_log(state),
        *pending_updates,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        render_profile_summary(state),
        gr.update(label=_profile_manager_label(state)),
    )


def _artifact_outputs(state: GameState) -> tuple:
    item = state.pending_item
    if item is None:
        return None, None, render_appraisal(None), None
    return (
        item.original_image,
        item.apocalypse_image,
        render_appraisal(item),
        item.appraisal.model_dump(mode="json"),
    )


def _year_label(value: float) -> str:
    if value == 0.25:
        return "3个月"
    return f"{value:g}年"


def stylesheet_path() -> Path:
    return ROOT / "assets" / "styles.css"


def reveal_script() -> str:
    return (ROOT / "assets" / "appraisal_reveal.js").read_text(encoding="utf-8")


def _reveal_signal(item: InventoryItem | None, *, status: str = "error") -> dict:
    if item is None or item.appraisal.is_fallback:
        return {"status": "error"}
    return {
        "status": status,
        "grade": item.appraisal.grade.value,
        "name": item.appraisal.apocalypse_name,
        "original_name": item.appraisal.original_item,
    }


def build_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
    ).set(
        body_background_fill="#080b09",
        body_background_fill_dark="#080b09",
        body_text_color="#f3eddd",
        body_text_color_dark="#f3eddd",
        body_text_color_subdued="#c8c1ae",
        body_text_color_subdued_dark="#c8c1ae",
        background_fill_primary="#0b0e0b",
        background_fill_primary_dark="#0b0e0b",
        background_fill_secondary="#1d231a",
        background_fill_secondary_dark="#1d231a",
        block_background_fill="#101410",
        block_background_fill_dark="#101410",
        block_border_color="#4d5446",
        block_border_color_dark="#4d5446",
        block_info_text_color="#d6cfbd",
        block_info_text_color_dark="#d6cfbd",
        block_label_background_fill="#171c16",
        block_label_background_fill_dark="#171c16",
        block_label_border_color="#58604f",
        block_label_border_color_dark="#58604f",
        block_label_text_color="#f3eddd",
        block_label_text_color_dark="#f3eddd",
        block_title_text_color="#f3eddd",
        block_title_text_color_dark="#f3eddd",
        input_background_fill="#171c16",
        input_background_fill_dark="#171c16",
        input_background_fill_hover="#1c221a",
        input_background_fill_hover_dark="#1c221a",
        input_background_fill_focus="#20271e",
        input_background_fill_focus_dark="#20271e",
        input_border_color="#626b58",
        input_border_color_dark="#626b58",
        input_border_color_hover="#88916f",
        input_border_color_hover_dark="#88916f",
        input_border_color_focus="#ffc447",
        input_border_color_focus_dark="#ffc447",
        input_placeholder_color="#b7b5aa",
        input_placeholder_color_dark="#b7b5aa",
        accordion_text_color="#f3eddd",
        accordion_text_color_dark="#f3eddd",
    )


def build_app(settings: Settings) -> gr.Blocks:
    settings.ensure_directories()
    repository = GameRepository(settings.database_url)
    repository.initialize(MARKET_CATALOG)
    initial_state = repository.load_or_create_initial_profile()

    vlm_provider = create_vlm_provider(settings)
    image_provider = create_image_provider(settings)
    cache_repository = None
    if settings.scan_cache_enabled:
        try:
            cache_repository = ScanCacheRepository(settings)
            cache_repository.initialize()
        except AppError as exc:
            LOGGER.warning("[CACHE] initialization failed: %s", exc.detail or exc)
    pipeline = ScanPipeline(
        settings,
        vlm_provider,
        image_provider,
        cache_repository=cache_repository,
    )

    sample_request = ScanRequest(
        scenario=ApocalypseScenario.CITY_BLACKOUT,
        apocalypse_years=3,
        description="便携式充电宝",
    )
    sample_appraisal = MockVLMProvider().analyze(None, sample_request)
    if not settings.mock_mode:
        sample_appraisal.warnings = [
            "当前为静态示例档案；上传照片或填写描述后，将调用本机模型真实鉴定。"
        ]
    sample_item = InventoryItem(
        item_id="DSB-DEMO01",
        appraisal=sample_appraisal,
        original_image=str(DEMO_BEFORE),
        apocalypse_image=str(DEMO_AFTER),
        scenario=sample_request.scenario,
        apocalypse_years=sample_request.apocalypse_years,
    )
    display_item = initial_state.pending_item or sample_item
    initial_market = _market_items(repository, initial_state)
    profiles = repository.list_profiles()
    initial_notice = (
        f"已载入存档“{initial_state.profile_name}”。只有休息和鉴定会推进一天。"
    )

    def commit_transition(transition: StateTransition) -> StateTransition:
        if transition.succeeded:
            if transition.day_advanced:
                transition.state.market_item_ids = repository.sample_market_ids()
            repository.save_profile(transition.state)
        return transition

    def safe_transition(
        transition_fn: Callable[[GameState], StateTransition],
        raw_state: dict,
    ) -> StateTransition:
        state = _hydrate_state(raw_state)
        try:
            return commit_transition(transition_fn(state))
        except (AppError, ValueError) as exc:
            LOGGER.warning("[GAME] state transition failed: %s", exc)
            return StateTransition(state=state, messages=[str(exc)])

    def scan_item(
        image: Image.Image | None,
        description: str,
        scenario: str,
        apocalypse_years: float,
        raw_state: dict,
        progress: gr.Progress = DEFAULT_PROGRESS,
    ):
        state = _hydrate_state(raw_state)
        allowed, reason = can_scavenge(state)
        if not allowed:
            failure = StateTransition(state=state, messages=[reason])
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                *_transition_outputs(failure, settings.mock_mode, repository),
                _reveal_signal(None),
            )

        try:
            request = ScanRequest(
                scenario=ApocalypseScenario(scenario),
                apocalypse_years=float(apocalypse_years),
                description=description or "",
            )
            progress(
                0.08,
                desc=(
                    "正在根据描述建立末日物资"
                    if image is None
                    else "正在整理现实物资证据"
                ),
            )
            result = pipeline.scan(image, request)
            progress(0.82, desc="正在签发末日物资档案")
            transition = commit_transition(register_scanned_item(state, result.item))
            committed_item = transition.state.pending_item or result.item
            message = " ".join(transition.messages)
            if result.status is PipelineStatus.PARTIAL:
                message += (
                    " 鉴定卡已保存，但图片未完成；本次鉴定只推进一次时间，"
                    "可单独重试图片。"
                )
                notice_kind = "warning"
                progress(1.0, desc="鉴定卡已完成，图片待重试")
            elif result.provider_metadata.get("cache_hit") is True:
                message += (
                    " 已从本机完整缓存直接读取，无需启动 Ollama 或 FLUX，"
                    f"耗时 {result.timings_ms['total'] / 1000:.2f} 秒。"
                )
                notice_kind = "info"
                progress(1.0, desc="本地缓存命中，鉴定完成")
            elif result.original_image is None:
                message += (
                    f" 已根据文字描述直接生成灾后形态，共耗时 "
                    f"{result.timings_ms['total'] / 1000:.1f} 秒。"
                )
                notice_kind = "info"
                progress(1.0, desc="文字鉴定与图片生成完成")
            else:
                message += (
                    f" 鉴定完成，共耗时 {result.timings_ms['total'] / 1000:.1f} 秒。"
                )
                notice_kind = "info"
                progress(1.0, desc="鉴定完成")
            return (
                result.original_image,
                result.apocalypse_image,
                render_appraisal(committed_item),
                result.item.appraisal.model_dump(mode="json"),
                *_transition_outputs(
                    transition,
                    settings.mock_mode,
                    repository,
                    notice_message=message,
                    notice_tone=notice_kind,
                ),
                _reveal_signal(
                    committed_item if transition.succeeded else None,
                    status=result.status.value,
                ),
            )
        except AppError as exc:
            LOGGER.warning(
                "[%s] %s detail=%s",
                exc.stage.value,
                exc.code.value,
                exc.detail,
            )
            failure = StateTransition(state=state, messages=[exc.user_message])
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                *_transition_outputs(failure, settings.mock_mode, repository),
                _reveal_signal(None),
            )
        except Exception:
            LOGGER.exception("[PIPELINE] unexpected scan failure")
            failure = StateTransition(
                state=state,
                messages=["鉴定流程暂时不可用，请稍后重试。"],
            )
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                *_transition_outputs(failure, settings.mock_mode, repository),
                _reveal_signal(None),
            )

    def retry_after(raw_state: dict):
        state = _hydrate_state(raw_state)
        item = state.pending_item
        if item is None:
            failure = StateTransition(state=state, messages=["当前没有等待补图的物资。"])
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                *_transition_outputs(failure, settings.mock_mode, repository),
            )
        try:
            updated_item = pipeline.retry_image(item)
            updated_state = state.model_copy(deep=True)
            updated_state.pending_item = updated_item
            transition = commit_transition(
                StateTransition(
                    state=updated_state,
                    messages=["图片补充完成；补图不推进时间。"],
                    succeeded=True,
                )
            )
            return (
                updated_item.apocalypse_image,
                render_appraisal(updated_item),
                updated_item.appraisal.model_dump(mode="json"),
                *_transition_outputs(transition, settings.mock_mode, repository),
            )
        except AppError as exc:
            LOGGER.warning(
                "[%s] retry image failed: %s detail=%s",
                exc.stage.value,
                exc.code.value,
                exc.detail,
            )
            failure = StateTransition(state=state, messages=[exc.user_message])
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                *_transition_outputs(failure, settings.mock_mode, repository),
            )

    def run_transition(
        transition_fn: Callable[[GameState], StateTransition],
        raw_state: dict,
    ):
        transition = safe_transition(transition_fn, raw_state)
        return _transition_outputs(transition, settings.mock_mode, repository)

    def use_item(raw_state: dict):
        return run_transition(apply_pending_item, raw_state)

    def store_item(raw_state: dict):
        return run_transition(store_pending_item, raw_state)

    def sell_item(raw_state: dict):
        return run_transition(sell_pending_item, raw_state)

    def discard_item(raw_state: dict):
        return run_transition(abandon_pending_item, raw_state)

    def rest_player(raw_state: dict):
        return run_transition(rest, raw_state)

    def use_stored_item(item_id: str | None, raw_state: dict):
        selected_id = item_id or ""

        def apply_selected(state: GameState) -> StateTransition:
            if any(
                entry.inventory_id == selected_id
                for entry in state.market_inventory
            ):
                return apply_market_inventory_item(state, selected_id)
            return apply_inventory_item(state, selected_id)

        return run_transition(
            apply_selected,
            raw_state,
        )

    def sell_stored_item(item_id: str | None, raw_state: dict):
        return run_transition(
            lambda state: sell_inventory_item(state, item_id or ""),
            raw_state,
        )

    def discard_stored_item(item_id: str | None, raw_state: dict):
        selected_id = item_id or ""

        def abandon_selected(state: GameState) -> StateTransition:
            if any(
                entry.inventory_id == selected_id
                for entry in state.market_inventory
            ):
                return abandon_market_inventory_item(state, selected_id)
            return abandon_inventory_item(state, selected_id)

        return run_transition(
            abandon_selected,
            raw_state,
        )

    def buy_selected_market_item(item_id: str | None, raw_state: dict):
        item = repository.get_market_item(item_id or "")
        return run_transition(lambda state: buy_market_item(state, item), raw_state)

    def select_inventory(raw_state: dict, evt: gr.SelectData):
        state = _hydrate_state(raw_state)
        entries = inventory_gallery_entries(state)
        index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
        if not isinstance(index, int) or not 0 <= index < len(entries):
            return (
                "",
                _selection_placeholder("背包物资"),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        entry = entries[index]
        item_id = entry.inventory_id
        if entry.kind == "market":
            purchased = next(
                item
                for item in state.market_inventory
                if item.inventory_id == item_id
            )
            market_item = purchased.item
            detail = (
                '<div class="selection-readout">'
                f"<strong>{escape(market_item.name)}</strong>"
                f"<span>{escape(MARKET_CATEGORY_LABELS[market_item.category])} · "
                f"{escape(market_effect_text(market_item))}<br>"
                "市场购入物资 · 可使用或放弃 · 不可转售</span></div>"
            )
            return (
                item_id,
                detail,
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )

        item = next(
            item for item in active_inventory_items(state) if item.item_id == item_id
        )
        grade = item.appraisal.grade.value
        reward = (
            f"出售可得 {GRADE_WATER_REWARDS[grade]:g}L 净水"
            if grade in GRADE_WATER_REWARDS
            else f"出售可得 +{GRADE_MORALE_REWARDS[grade]} 士气"
        )
        detail = (
            '<div class="selection-readout">'
            f"<strong>{escape(item.appraisal.apocalypse_name)}</strong>"
            f"<span>{escape(item.appraisal.original_item)} · {grade}级 · "
            f"{escape(reward)}</span></div>"
        )
        return (
            item_id,
            detail,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def select_market(raw_state: dict, evt: gr.SelectData):
        state = _hydrate_state(raw_state)
        items = _market_items(repository, state)
        index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
        if not isinstance(index, int) or not 0 <= index < len(items):
            return "", _selection_placeholder("市场商品"), gr.update(interactive=False)
        item = items[index]
        effects = []
        if item.satiety_gain:
            effects.append(f"饱食 +{item.satiety_gain}")
        if item.health_gain:
            effects.append(f"生命 +{item.health_gain}")
        if item.energy_gain:
            effects.append(f"体力 +{item.energy_gain}")
        if item.morale_gain:
            effects.append(f"士气 +{item.morale_gain}")
        effect_text = " · ".join(effects) if effects else "没有属性收益"
        detail = (
            '<div class="selection-readout">'
            f"<strong>{escape(item.name)} · {item.water_price:g}L</strong>"
            f"<span>{escape(MARKET_CATEGORY_LABELS[item.category])} · "
            f"{escape(effect_text)}<br>{escape(item.description)}</span></div>"
        )
        return item.item_id, detail, gr.update(interactive=True)

    def switch_profile(name: str | None, raw_state: dict):
        current = _hydrate_state(raw_state)
        if not name:
            failure = StateTransition(state=current, messages=["请选择一个存档。"])
            return (
                "",
                gr.update(visible=False),
                *_artifact_outputs(current),
                *_transition_outputs(failure, settings.mock_mode, repository),
            )
        try:
            state = repository.load_profile(name)
            repository.set_active_profile_name(state.profile_name)
            transition = StateTransition(
                state=state,
                messages=[f"已切换到存档“{state.profile_name}”。"],
                succeeded=True,
            )
        except (AppError, ValueError) as exc:
            transition = StateTransition(state=current, messages=[str(exc)])
        return (
            "",
            gr.update(visible=False),
            *_artifact_outputs(transition.state),
            *_transition_outputs(transition, settings.mock_mode, repository),
        )

    def create_profile(name: str, raw_state: dict):
        current = _hydrate_state(raw_state)
        try:
            state = repository.create_profile(name)
            choices = _profile_choices(repository, state.profile_name)
            transition = StateTransition(
                state=state,
                messages=[f"已创建并切换到存档“{state.profile_name}”。"],
                succeeded=True,
            )
            profile_update = gr.update(choices=choices, value=state.profile_name)
            name_update = gr.update(value="")
        except (AppError, ValueError) as exc:
            transition = StateTransition(state=current, messages=[str(exc)])
            choices = _profile_choices(repository, current.profile_name)
            profile_update = gr.update(
                choices=choices,
                value=current.profile_name,
            )
            name_update = gr.update()
        return (
            profile_update,
            name_update,
            gr.update(interactive=len(choices) > 1),
            "",
            gr.update(visible=False),
            *_artifact_outputs(transition.state),
            *_transition_outputs(transition, settings.mock_mode, repository),
        )

    def restart_profile(raw_state: dict):
        current = _hydrate_state(raw_state)
        try:
            state = repository.restart_profile(current.profile_name)
            transition = StateTransition(
                state=state,
                messages=[f"存档“{state.profile_name}”已从 DAY 1 重新开始。"],
                succeeded=True,
            )
        except (AppError, ValueError) as exc:
            transition = StateTransition(state=current, messages=[str(exc)])
        return (
            "",
            gr.update(visible=False),
            *_artifact_outputs(transition.state),
            *_transition_outputs(transition, settings.mock_mode, repository),
        )

    def prepare_delete_profile(raw_state: dict):
        current = _hydrate_state(raw_state)
        return (
            current.profile_name,
            gr.update(visible=True),
            _delete_confirmation_copy(current.profile_name),
        )

    def cancel_delete_profile():
        return "", gr.update(visible=False)

    def confirm_delete_profile(target_name: str, raw_state: dict):
        current = _hydrate_state(raw_state)
        deleted_name = target_name.strip()
        if deleted_name != current.profile_name:
            transition = StateTransition(
                state=current,
                messages=["存档选择已经变化，删除已取消，请重新操作。"],
            )
        else:
            try:
                state = repository.delete_profile(deleted_name)
                transition = StateTransition(
                    state=state,
                    messages=[
                        f"已永久删除存档“{deleted_name}”，"
                        f"并切换到“{state.profile_name}”。"
                    ],
                    succeeded=True,
                )
            except (AppError, ValueError) as exc:
                transition = StateTransition(state=current, messages=[str(exc)])
        choices = _profile_choices(repository, transition.state.profile_name)
        return (
            gr.update(choices=choices, value=transition.state.profile_name),
            gr.update(interactive=len(choices) > 1),
            "",
            gr.update(visible=False),
            *_artifact_outputs(transition.state),
            *_transition_outputs(transition, settings.mock_mode, repository),
        )

    def load_session():
        try:
            state = repository.load_or_create_initial_profile()
            choices = repository.list_profiles()
            transition = StateTransition(
                state=state,
                messages=[f"已载入存档“{state.profile_name}”。"],
                succeeded=True,
            )
        except (AppError, ValueError) as exc:
            state = initial_state
            choices = [initial_state.profile_name]
            transition = StateTransition(state=state, messages=[str(exc)])
        return (
            gr.update(
                choices=choices,
                value=transition.state.profile_name,
            ),
            gr.update(interactive=len(choices) > 1),
            "",
            gr.update(visible=False),
            *_artifact_outputs(transition.state),
            *_transition_outputs(transition, settings.mock_mode, repository),
        )

    with gr.Blocks(title="末日物资鉴定局", fill_width=True) as demo:
        game_state = gr.State(_state_payload(initial_state))
        selected_inventory_id = gr.State("")
        selected_market_id = gr.State("")
        delete_profile_target = gr.State("")

        with gr.Column(elem_id="dsb-shell"):
            gr.HTML(
                """
                <header class="dsb-brand">
                  <div>
                    <div class="brand-code">DSB // REALITY TRANSLATION TERMINAL</div>
                    <h1>末日物资鉴定局</h1>
                    <p>DOOMSDAY SUPPLY BUREAU</p>
                  </div>
                  <div class="brand-manifesto">
                    <strong>当文明消失，万物都需要被重新翻译。</strong>
                    拍下或描述身边的东西，在末日活下去。
                  </div>
                </header>
                """
            )
            with gr.Accordion(
                _profile_manager_label(initial_state),
                open=False,
                elem_id="profile-manager",
            ) as profile_manager:
                with gr.Column(elem_id="profile-manager-content"):
                    profile_summary_html = gr.HTML(
                        render_profile_summary(initial_state),
                        elem_id="profile-summary",
                    )
                    with gr.Row(elem_id="profile-switch-row"):
                        profile_selector = gr.Dropdown(
                            choices=profiles,
                            value=initial_state.profile_name,
                            label="当前存档",
                            interactive=True,
                            filterable=False,
                            scale=6,
                            elem_id="profile-selector",
                        )
                        restart_button = gr.Button(
                            "重新开始",
                            variant="secondary",
                            scale=1,
                            min_width=118,
                            elem_id="restart-button",
                        )
                        delete_profile_button = gr.Button(
                            "删除存档…",
                            variant="stop",
                            interactive=len(profiles) > 1,
                            scale=1,
                            min_width=118,
                            elem_id="delete-profile-button",
                        )
                    with gr.Row(elem_id="profile-create-row"):
                        profile_name = gr.Textbox(
                            label="新建存档",
                            placeholder="输入新角色名称",
                            max_lines=1,
                            scale=6,
                            elem_id="profile-name-input",
                        )
                        create_profile_button = gr.Button(
                            "＋ 新建并进入",
                            variant="primary",
                            scale=2,
                            min_width=180,
                            elem_id="create-profile-button",
                        )
                    gr.HTML(
                        '<p class="profile-manager-foot">存档自动保存在本机 · '
                        "至少保留 1 个存档 · 切换不消耗游戏时间</p>"
                    )
                    with gr.Group(
                        visible=False,
                        elem_id="delete-confirm-panel",
                    ) as delete_confirm_group:
                        delete_confirmation_html = gr.HTML(
                            _delete_confirmation_copy(initial_state.profile_name)
                        )
                        with gr.Row(elem_id="delete-confirm-actions"):
                            cancel_delete_button = gr.Button(
                                "取消",
                                variant="secondary",
                            )
                            confirm_delete_button = gr.Button(
                                "永久删除此存档",
                                variant="stop",
                                elem_id="confirm-delete-button",
                            )

            status_html = gr.HTML(render_status(initial_state, settings.mock_mode))
            effects_html = gr.HTML(render_effects(), elem_id="global-effects")
            reveal_signal = gr.JSON(value=None, visible=False)

            with gr.Tabs(elem_id="main-tabs"):
                with gr.Tab("扫描现实物资"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=8, min_width=620, elem_id="scan-workbench"):
                            gr.HTML(
                                """
                                <div class="section-heading">
                                  <h2>现实物资翻译台</h2>
                                  <p>上传照片：同物改造 · 仅填描述：直接生成</p>
                                </div>
                                """
                            )
                            with gr.Row():
                                with gr.Column():
                                    gr.HTML(
                                        '<p class="evidence-label">'
                                        "BEFORE <span>原始证物（照片模式）</span></p>"
                                    )
                                    before_output = gr.Image(
                                        value=display_item.original_image,
                                        type="filepath",
                                        show_label=False,
                                        interactive=False,
                                        height=430,
                                        elem_id="before-image",
                                    )
                                with gr.Column():
                                    gr.HTML(
                                        '<p class="evidence-label">'
                                        "AFTER <span>灾后形态</span></p>"
                                    )
                                    after_output = gr.Image(
                                        value=display_item.apocalypse_image,
                                        type="filepath",
                                        show_label=False,
                                        interactive=False,
                                        height=430,
                                        elem_id="after-image",
                                    )

                            with gr.Group(elem_id="control-panel"):
                                with gr.Row():
                                    upload_image = gr.Image(
                                        value=None,
                                        label="现实物品照片（可选）",
                                        type="pil",
                                        sources=["upload", "webcam", "clipboard"],
                                        height=190,
                                        elem_id="upload-image",
                                    )
                                    with gr.Column():
                                        description = gr.Textbox(
                                            value="",
                                            label="物品描述（无照片时必填）",
                                            placeholder="例如：透明塑料水瓶",
                                            max_lines=2,
                                            elem_id="description-input",
                                        )
                                        scenario = gr.Dropdown(
                                            choices=[
                                                (entry.label, entry.value)
                                                for entry in ApocalypseScenario
                                            ],
                                            value=ApocalypseScenario.CITY_BLACKOUT.value,
                                            label="末日场景",
                                            elem_id="scenario-select",
                                        )
                                        apocalypse_years = gr.Dropdown(
                                            choices=[
                                                ("3个月", 0.25),
                                                ("1年", 1),
                                                ("3年", 3),
                                                ("10年", 10),
                                                ("30年", 30),
                                            ],
                                            value=3,
                                            label="末日时间",
                                            elem_id="years-select",
                                        )
                                scan_button = gr.Button(
                                    "鉴定并生成",
                                    variant="primary",
                                    elem_id="scan-button",
                                )

                            notice_html = gr.HTML(render_notice(initial_notice))
                            retry_image_button = gr.Button(
                                "重试生成图片",
                                visible=bool(
                                    initial_state.pending_item
                                    and not initial_state.pending_item.apocalypse_image
                                ),
                                elem_id="retry-image-button",
                            )

                        with gr.Column(
                            scale=4,
                            min_width=370,
                            elem_id="appraisal-column",
                        ):
                            appraisal_html = gr.HTML(render_appraisal(display_item))
                            with gr.Accordion("查看结构化鉴定数据", open=False):
                                raw_json = gr.JSON(
                                    value=display_item.appraisal.model_dump(mode="json"),
                                    show_label=False,
                                )

                    initial_actions = _action_updates(initial_state.pending_item)
                    with gr.Row(elem_classes=["action-row"]):
                        use_button = gr.Button(
                            "立即使用",
                            variant="primary",
                            interactive=initial_actions[0]["interactive"],
                        )
                        store_button = gr.Button(
                            "放入背包",
                            interactive=initial_actions[1]["interactive"],
                        )
                        sell_pending_button = gr.Button(
                            "出售鉴定物资",
                            interactive=initial_actions[2]["interactive"],
                        )
                        discard_button = gr.Button(
                            "放弃",
                            variant="stop",
                            interactive=initial_actions[3]["interactive"],
                        )
                        rest_button = gr.Button("休息（+1 天）")

                with gr.Tab("背包"):
                    inventory_html = gr.HTML(render_inventory(initial_state))
                    inventory_gallery = gr.Gallery(
                        value=inventory_gallery_value(initial_state),
                        label="背包物资 · 点选图片",
                        show_label=False,
                        columns=3,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                        fit_columns=False,
                        allow_preview=True,
                        selected_index=None,
                        interactive=False,
                        type="filepath",
                        elem_id="inventory-gallery",
                    )
                    inventory_selection_html = gr.HTML(
                        _selection_placeholder("背包物资")
                    )
                    with gr.Row(elem_id="inventory-actions"):
                        inventory_use_button = gr.Button(
                            "使用选中物资",
                            variant="primary",
                            interactive=False,
                        )
                        inventory_sell_button = gr.Button(
                            "出售选中物资",
                            interactive=False,
                        )
                        inventory_discard_button = gr.Button(
                            "放弃选中物资",
                            variant="stop",
                            interactive=False,
                        )

                with gr.Tab("废土市场"):
                    market_html = gr.HTML(render_market(initial_state, initial_market))
                    market_gallery = gr.Gallery(
                        value=market_gallery_value(initial_market),
                        label="本轮随机商品 · 点选图片",
                        show_label=False,
                        columns=5,
                        rows=1,
                        height="auto",
                        object_fit="contain",
                        fit_columns=False,
                        allow_preview=True,
                        selected_index=None,
                        interactive=False,
                        type="filepath",
                        elem_id="market-gallery",
                    )
                    market_selection_html = gr.HTML(
                        _selection_placeholder("市场商品")
                    )
                    market_buy_button = gr.Button(
                        "购买选中商品",
                        variant="primary",
                        interactive=False,
                        elem_id="market-buy-button",
                    )

                with gr.Tab("生存记录"):
                    log_html = gr.HTML(render_log(initial_state))

            gr.HTML(
                """
                <footer class="dsb-footer">
                  <span>DSB SYSTEM // LOCAL-FIRST AI DEMO</span>
                  <span>现实世界就是玩家的物资库</span>
                </footer>
                """
            )

        transition_outputs = [
            game_state,
            status_html,
            notice_html,
            effects_html,
            inventory_html,
            inventory_gallery,
            selected_inventory_id,
            inventory_selection_html,
            market_html,
            market_gallery,
            selected_market_id,
            market_selection_html,
            log_html,
            use_button,
            store_button,
            sell_pending_button,
            discard_button,
            retry_image_button,
            inventory_use_button,
            inventory_sell_button,
            inventory_discard_button,
            market_buy_button,
            profile_summary_html,
            profile_manager,
        ]
        artifact_outputs = [
            before_output,
            after_output,
            appraisal_html,
            raw_json,
        ]

        demo.load(
            fn=load_session,
            inputs=None,
            outputs=[
                profile_selector,
                delete_profile_button,
                delete_profile_target,
                delete_confirm_group,
                *artifact_outputs,
                *transition_outputs,
            ],
            api_name="load_session",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )

        scan_event = scan_button.click(
            fn=scan_item,
            inputs=[upload_image, description, scenario, apocalypse_years, game_state],
            outputs=[*artifact_outputs, *transition_outputs, reveal_signal],
            api_name="scan_supply",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
            js="""(...args) => {
                try { window.dsbPackReveal?.start(); } catch (_) {}
                return args;
            }""",
        )
        scan_event.success(
            fn=None,
            inputs=[reveal_signal, after_output],
            outputs=None,
            queue=False,
            api_visibility="private",
            js="""(result, image) => {
                try { window.dsbPackReveal?.finish(result, image); }
                catch (_) { window.dsbPackReveal?.abort(); }
                return [];
            }""",
        )
        scan_event.failure(
            fn=None,
            inputs=None,
            outputs=None,
            queue=False,
            api_visibility="private",
            js="""() => {
                window.dsbPackReveal?.abort();
                return [];
            }""",
        )
        retry_image_button.click(
            fn=retry_after,
            inputs=[game_state],
            outputs=[
                after_output,
                appraisal_html,
                raw_json,
                *transition_outputs,
            ],
            api_name="retry_after_image",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )

        for button, function, api_name in (
            (use_button, use_item, "use_supply"),
            (store_button, store_item, "store_supply"),
            (sell_pending_button, sell_item, "sell_pending_supply"),
            (discard_button, discard_item, "discard_supply"),
            (rest_button, rest_player, "rest"),
        ):
            button.click(
                fn=function,
                inputs=[game_state],
                outputs=transition_outputs,
                api_name=api_name,
                concurrency_limit=1,
                concurrency_id="dsb-state",
                trigger_mode="once",
            )

        inventory_gallery.select(
            fn=select_inventory,
            inputs=[game_state],
            outputs=[
                selected_inventory_id,
                inventory_selection_html,
                inventory_use_button,
                inventory_sell_button,
                inventory_discard_button,
            ],
            concurrency_limit=1,
            concurrency_id="dsb-state",
        )
        inventory_use_button.click(
            fn=use_stored_item,
            inputs=[selected_inventory_id, game_state],
            outputs=transition_outputs,
            api_name="use_inventory_supply",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )
        inventory_sell_button.click(
            fn=sell_stored_item,
            inputs=[selected_inventory_id, game_state],
            outputs=transition_outputs,
            api_name="sell_inventory_supply",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )
        inventory_discard_button.click(
            fn=discard_stored_item,
            inputs=[selected_inventory_id, game_state],
            outputs=transition_outputs,
            api_name="discard_inventory_supply",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )

        market_gallery.select(
            fn=select_market,
            inputs=[game_state],
            outputs=[selected_market_id, market_selection_html, market_buy_button],
            concurrency_limit=1,
            concurrency_id="dsb-state",
        )
        market_buy_button.click(
            fn=buy_selected_market_item,
            inputs=[selected_market_id, game_state],
            outputs=transition_outputs,
            api_name="buy_market_item",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )

        profile_selector.change(
            fn=switch_profile,
            inputs=[profile_selector, game_state],
            outputs=[
                delete_profile_target,
                delete_confirm_group,
                *artifact_outputs,
                *transition_outputs,
            ],
            api_name="switch_profile",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )
        create_profile_button.click(
            fn=create_profile,
            inputs=[profile_name, game_state],
            outputs=[
                profile_selector,
                profile_name,
                delete_profile_button,
                delete_profile_target,
                delete_confirm_group,
                *artifact_outputs,
                *transition_outputs,
            ],
            api_name="create_profile",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )
        restart_button.click(
            fn=restart_profile,
            inputs=[game_state],
            outputs=[
                delete_profile_target,
                delete_confirm_group,
                *artifact_outputs,
                *transition_outputs,
            ],
            api_name="restart_profile",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )
        delete_profile_button.click(
            fn=prepare_delete_profile,
            inputs=[game_state],
            outputs=[
                delete_profile_target,
                delete_confirm_group,
                delete_confirmation_html,
            ],
            api_name="prepare_delete_profile",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )
        cancel_delete_button.click(
            fn=cancel_delete_profile,
            inputs=None,
            outputs=[delete_profile_target, delete_confirm_group],
            api_name="cancel_delete_profile",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )
        confirm_delete_button.click(
            fn=confirm_delete_profile,
            inputs=[delete_profile_target, game_state],
            outputs=[
                profile_selector,
                delete_profile_button,
                delete_profile_target,
                delete_confirm_group,
                *artifact_outputs,
                *transition_outputs,
            ],
            api_name="delete_profile",
            concurrency_limit=1,
            concurrency_id="dsb-state",
            trigger_mode="once",
        )

    return demo


def allowed_asset_paths(settings: Settings) -> list[str]:
    return [
        str(settings.output_dir.resolve()),
        str((ROOT / "assets").resolve()),
        str(MARKET_ASSET_DIR.resolve()),
    ]
