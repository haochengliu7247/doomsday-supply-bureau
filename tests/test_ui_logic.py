from pathlib import Path

import pytest

from src.config import Settings
from src.data.market_catalog import MARKET_CATALOG, MARKET_CATALOG_BY_ID
from src.pipeline import ScanPipeline
from src.providers.mock_provider import MockVLMProvider
from src.schemas import (
    ApocalypseScenario,
    GameState,
    InventoryItem,
    PipelineResult,
    PipelineStatus,
    PurchasedMarketItem,
    ScanRequest,
    StateTransition,
)
from src.services.game_repository import GameRepository
from src.ui.gradio_app import (
    _action_updates,
    _delete_confirmation_copy,
    _profile_manager_label,
    _transition_outputs,
    build_app,
    build_theme,
    stylesheet_path,
)
from src.ui.renderers import (
    CATEGORY_LABELS,
    inventory_gallery_ids,
    inventory_gallery_value,
    market_gallery_value,
    render_effects,
    render_inventory,
    render_market,
    render_profile_summary,
    render_status,
)


def text_item(*, apocalypse_image: str | None) -> InventoryItem:
    request = ScanRequest(
        scenario=ApocalypseScenario.CITY_BLACKOUT,
        apocalypse_years=3,
        description="透明塑料水瓶",
    )
    return InventoryItem(
        appraisal=MockVLMProvider().analyze(None, request),
        original_image="before.png",
        apocalypse_image=apocalypse_image,
        scenario=request.scenario,
        apocalypse_years=request.apocalypse_years,
    )


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        mock_mode=True,
        output_dir=tmp_path / "outputs",
        database_url=f"sqlite:///{(tmp_path / 'game.db').as_posix()}",
    )


def repository_for(tmp_path: Path) -> GameRepository:
    repository = GameRepository(f"sqlite:///{(tmp_path / 'ui.db').as_posix()}")
    repository.initialize(MARKET_CATALOG)
    return repository


def test_text_only_partial_disables_actions_and_enables_image_retry() -> None:
    use, store, sell, discard, retry = _action_updates(text_item(apocalypse_image=None))

    assert use["interactive"] is False
    assert store["interactive"] is False
    assert sell["interactive"] is False
    assert discard["interactive"] is True
    assert retry["visible"] is True
    assert retry["interactive"] is True


def test_text_only_success_enables_actions_and_hides_retry() -> None:
    item = text_item(apocalypse_image="generated-after.png")
    use, store, sell, discard, retry = _action_updates(item)

    assert use["interactive"] is True
    assert store["interactive"] is True
    assert sell["interactive"] is True
    assert discard["interactive"] is True
    assert retry["visible"] is False
    assert retry["interactive"] is False


@pytest.mark.parametrize("status", [PipelineStatus.SUCCESS, PipelineStatus.PARTIAL])
def test_reveal_reports_current_item_and_scan_still_advances_only_one_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: PipelineStatus,
) -> None:
    app = build_app(settings_for(tmp_path))
    scan = next(fn for fn in app.fns.values() if fn.api_name == "scan_supply")
    item = text_item(
        apocalypse_image="generated-after.png" if status is PipelineStatus.SUCCESS else None
    )
    calls = []

    def scan_once(*args, **kwargs):
        calls.append(1)
        return PipelineResult(
            status=status, item=item,
            original_image=item.original_image, apocalypse_image=item.apocalypse_image,
            timings_ms={"total": 1}, provider_metadata={"cache_hit": True},
        )

    monkeypatch.setattr(ScanPipeline, "scan", scan_once)
    state = GameState()
    values = scan.fn(
        None, "水瓶", "CITY_BLACKOUT", 3,
        state.model_dump(exclude_computed_fields=True),
    )
    assert len(values) == len(scan.outputs)
    assert values[-1] == {
        "status": status.value, "grade": item.appraisal.grade.value,
        "name": item.appraisal.apocalypse_name,
        "original_name": item.appraisal.original_item,
    }
    assert values[4]["player"]["day"] == state.player.day + 1
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["game_over", "invalid_input", "unexpected_error"])
def test_reveal_failure_cannot_reveal_previous_pending_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    app = build_app(settings_for(tmp_path))
    scan = next(fn for fn in app.fns.values() if fn.api_name == "scan_supply")
    old_item = text_item(apocalypse_image="previous-after.png")
    state = GameState(pending_item=old_item)
    description = ""
    if failure == "game_over":
        state.player.health = 0
    elif failure == "unexpected_error":
        description = "水瓶"

        def unavailable(*args, **kwargs):
            raise RuntimeError("test service unavailable")

        monkeypatch.setattr(ScanPipeline, "scan", unavailable)
    values = scan.fn(
        None, description, "CITY_BLACKOUT", 3,
        state.model_dump(exclude_computed_fields=True),
    )
    assert len(values) == len(scan.outputs)
    assert values[-1] == {"status": "error"}
    assert values[4]["pending_item"]["item_id"] == old_item.item_id
    assert values[4]["player"]["day"] == state.player.day


def test_initial_controls_allow_description_only_scan_and_named_profiles(
    tmp_path: Path,
) -> None:
    app = build_app(settings_for(tmp_path))
    config = app.get_config_file()
    by_label = {
        component.get("props", {}).get("label"): component
        for component in config["components"]
    }

    upload = by_label["现实物品照片（可选）"]["props"]
    description = by_label["物品描述（无照片时必填）"]["props"]
    assert upload.get("value") is None
    assert description["value"] == ""
    assert by_label["当前存档"]["props"]["value"] == "幸存者A"
    assert by_label["新建存档"]["props"].get("value", "") == ""

    by_elem_id = {
        component.get("props", {}).get("elem_id"): component.get("props", {})
        for component in config["components"]
    }
    assert by_elem_id["upload-image"]["label"] == "现实物品照片（可选）"
    assert by_elem_id["description-input"]["label"] == "物品描述（无照片时必填）"
    assert (
        by_elem_id["scenario-select"]["value"]
        == ApocalypseScenario.CITY_BLACKOUT.value
    )
    assert by_elem_id["years-select"]["value"] == 3


def test_status_explains_time_and_previews_toxic_damage() -> None:
    html = render_status(GameState(profile_name="A 人"), False)

    assert "饱食度" in html
    assert ">80<" in html
    assert "仅休息 / 鉴定 +1 天" in html
    assert "下一天 -4 生命" in html
    assert "存档 · A 人" in html


def test_market_explains_twenty_item_catalog_and_random_five() -> None:
    items = list(MARKET_CATALOG[:5])
    html = render_market(GameState(), items)
    gallery = market_gallery_value(items)

    assert "交易开放" in html
    assert "总目录固定 20 件" in html
    assert "3 件食物、3 件医疗用品" in html
    assert "随机刷新 5 件" in html
    assert "书本、玩具" in html
    assert "购买后先放入背包" in html
    assert "当场使用" not in html
    assert "B：2.0L · A：5.0L · S：15.0L" in html
    assert len(gallery) == 5
    assert all(Path(image_path).is_absolute() for image_path, _ in gallery)


def test_backpack_gallery_contains_item_images() -> None:
    item = text_item(apocalypse_image="generated-after.png")

    gallery = inventory_gallery_value(GameState(inventory=[item]))

    assert gallery == [
        (
            "generated-after.png",
            f"{item.appraisal.apocalypse_name}｜"
            f"{CATEGORY_LABELS[item.appraisal.category.value]}｜"
            f"{item.appraisal.grade.value}级",
        )
    ]


def test_backpack_gallery_combines_appraised_and_market_items_with_matching_ids() -> None:
    appraised = text_item(apocalypse_image="generated-after.png")
    market_item = MARKET_CATALOG_BY_ID["medical_first_aid"]
    purchased = PurchasedMarketItem(item=market_item)
    state = GameState(inventory=[appraised], market_inventory=[purchased])

    gallery = inventory_gallery_value(state)
    item_ids = inventory_gallery_ids(state)

    assert item_ids == [appraised.item_id, purchased.inventory_id]
    assert len(gallery) == len(item_ids) == 2
    assert gallery[1][0] == market_item.image_path
    assert market_item.name in gallery[1][1]
    assert "市场购入" in gallery[1][1]
    assert "生命 +20" in gallery[1][1]
    assert "背包 2 / 6" in render_inventory(state)
    assert "市场购入物资不可转售" in render_inventory(state)


def test_backpack_and_market_use_image_selection_instead_of_sell_dropdown(
    tmp_path: Path,
) -> None:
    app = build_app(settings_for(tmp_path))
    config = app.get_config_file()
    labels = {
        component.get("props", {}).get("label")
        for component in config["components"]
    }
    values = {
        component.get("props", {}).get("value")
        for component in config["components"]
        if isinstance(component.get("props", {}).get("value"), str)
    }

    assert "背包物资 · 点选图片" in labels
    assert "本轮随机商品 · 点选图片" in labels
    assert "选择要使用的背包物资" not in labels
    assert "选择要出售的背包物资" not in labels
    assert "使用选中物资" in values
    assert "出售选中物资" in values
    assert "放弃选中物资" in values
    assert "购买选中商品" in values
    assert "重新开始" in values
    assert "删除存档…" in values
    assert "永久删除此存档" in values
    assert "＋ 新建并进入" in values

    by_elem_id = {
        component.get("props", {}).get("elem_id"): component.get("props", {})
        for component in config["components"]
    }
    profile_manager = by_elem_id["profile-manager"]
    assert profile_manager["label"] == "存档管理 · 当前：幸存者A · DAY 1"
    assert profile_manager["open"] is False
    assert by_elem_id["profile-selector"]["filterable"] is False
    assert by_elem_id["delete-profile-button"]["interactive"] is False
    assert by_elem_id["delete-confirm-panel"]["visible"] is False
    for gallery_id, columns in (("inventory-gallery", 3), ("market-gallery", 5)):
        gallery = by_elem_id[gallery_id]
        assert gallery["fit_columns"] is False
        assert gallery["show_label"] is False
        assert gallery["columns"] == columns
        assert gallery["height"] == "auto"


def test_survival_log_and_effect_styles_are_explicit_and_readable() -> None:
    css = stylesheet_path().read_text(encoding="utf-8")
    log_body_rule = css.split(".log-row p {", 1)[1].split("}", 1)[0]

    assert "font-size: 16px" in log_body_rule
    assert "line-height: 1.7" in log_body_rule
    assert "color: #f3eddd" in log_body_rule
    assert "Microsoft YaHei UI" in log_body_rule
    assert "@keyframes morale-page-shake" in css
    assert "@keyframes page-red-hit" in css
    assert ".falling-dust" in css
    assert "--body-text-color: #f3eddd" in css
    assert "#dsb-shell .secondary-surface h2" in css
    assert "#inventory-gallery .grid-container" in css
    assert "grid-auto-rows: 260px" in css
    assert "grid-auto-rows: 235px" in css
    assert ".caption-label" in css
    assert "opacity: 1 !important" in css
    assert "--input-background-fill: #171c16" in css
    assert "--input-placeholder-color: #b7b5aa" in css
    assert '#dsb-shell [data-testid="block-label"]' in css
    assert '#dsb-shell [data-testid="block-info"]' in css
    assert ".profile-current-summary" in css
    assert "#delete-confirm-panel" in css


def test_dark_theme_synchronizes_form_and_label_tokens() -> None:
    config = build_theme().to_dict()["theme"]

    assert config["input_background_fill"] == "#171c16"
    assert config["input_placeholder_color"] == "#b7b5aa"
    assert config["block_label_background_fill"] == "#171c16"
    assert config["block_label_text_color"] == "#f3eddd"
    assert config["block_info_text_color"] == "#d6cfbd"


def test_profile_summary_and_delete_confirmation_name_the_target() -> None:
    state = GameState(profile_name='<远征者 & "A">')

    summary = render_profile_summary(state)
    confirmation = _delete_confirmation_copy(state.profile_name)

    assert "&lt;远征者 &amp; &quot;A&quot;&gt;" in summary
    assert "&lt;远征者 &amp; &quot;A&quot;&gt;" in confirmation
    assert "无法恢复" in confirmation
    assert _profile_manager_label(state) == '存档管理 · 当前：<远征者 & "A"> · DAY 1'


def test_effect_markup_uses_transition_flags() -> None:
    transition = StateTransition(
        state=GameState(),
        succeeded=True,
        health_damage=7,
        morale_shake=True,
    )

    html = render_effects(transition)

    assert "health-hit-trigger" in html
    assert "-7 HP" in html
    assert "morale-shock-trigger" in html
    assert html.count("<i>") == 24


def test_transition_output_refreshes_both_galleries_and_resets_selection(
    tmp_path: Path,
) -> None:
    repository = repository_for(tmp_path)
    item = text_item(apocalypse_image="generated-after.png")
    state = GameState(
        inventory=[item],
        market_item_ids=repository.sample_market_ids(),
    )
    transition = StateTransition(
        state=state,
        messages=["测试成功提示"],
        succeeded=True,
    )

    outputs = _transition_outputs(transition, False, repository)

    assert len(outputs) == 24
    assert "operation-notice info" in outputs[2]
    assert len(outputs[5]["value"]) == 1
    assert outputs[6] == ""
    assert len(outputs[9]["value"]) == 5
    assert outputs[10] == ""
    assert "profile-current-summary" in outputs[-2]
    assert outputs[-1]["label"] == "存档管理 · 当前：幸存者A · DAY 1"
