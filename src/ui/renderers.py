from __future__ import annotations

from dataclasses import dataclass
from html import escape
from uuid import uuid4

from src.game.balance import GRADE_MORALE_REWARDS, GRADE_WATER_REWARDS
from src.schemas import (
    AppraisalResult,
    GameState,
    InventoryItem,
    MarketItem,
    MarketItemCategory,
    StateTransition,
)

CATEGORY_LABELS = {
    "water": "水源",
    "food": "食物",
    "energy": "能源",
    "electronics": "电子",
    "tool": "工具",
    "medical": "医疗",
    "shelter": "庇护",
    "clothing": "衣物",
    "identity": "身份",
    "morale": "士气",
    "hazardous": "危险品",
    "other": "其他",
}

MARKET_CATEGORY_LABELS = {
    MarketItemCategory.FOOD: "食物",
    MarketItemCategory.MEDICAL: "医疗",
    MarketItemCategory.SURVIVAL: "生存用品",
    MarketItemCategory.BOOK: "书本",
    MarketItemCategory.TOY: "玩具",
    MarketItemCategory.JUNK: "杂物",
}

LOG_CATEGORY_LABELS = {
    "GAME": "生存",
    "SCAN": "鉴定",
    "REST": "休息",
    "USE": "使用",
    "INVENTORY": "背包",
    "MARKET": "市场",
}


@dataclass(frozen=True, slots=True)
class InventoryGalleryEntry:
    inventory_id: str
    kind: str
    image_path: str
    caption: str


def _metric(label: str, value: str, icon: str, tone: str = "") -> str:
    return (
        f'<div class="status-metric {tone}">'
        f'<span class="metric-icon">{icon}</span>'
        f'<span class="metric-label">{escape(label)}</span>'
        f'<strong>{escape(value)}</strong>'
        "</div>"
    )


def render_status(state: GameState, mock_mode: bool) -> str:
    player = state.player
    satiety = 100 - player.hunger
    mode = "MOCK / 离线演示" if mock_mode else "REAL / AI 在线"
    game_over = (
        '<span class="mode-badge danger">GAME OVER</span>'
        if player.is_game_over
        else ""
    )
    return (
        '<section class="status-ribbon" aria-label="玩家生存状态">'
        '<div class="day-block"><small>SURVIVAL DAY</small>'
        f'<strong>DAY {player.day}</strong><span>仅休息 / 鉴定 +1 天</span></div>'
        '<div class="status-metrics">'
        + _metric("生命", str(player.health), "♥", "health")
        + _metric("体力", str(player.energy), "↯", "energy")
        + _metric("饱食度", str(satiety), "◒", "hunger")
        + _metric("士气", str(player.morale), "◆", "morale")
        + _metric("净水", f"{player.water:.1f}L", "◉", "water")
        + "</div>"
        '<div class="mode-stack">'
        f'<span class="profile-badge">存档 · {escape(state.profile_name)}</span>'
        f'<span class="toxic-forecast">☣ 下一天 -{player.toxic_damage_forecast} 生命</span>'
        f'<span class="mode-badge">{mode}</span>{game_over}</div>'
        "</section>"
    )


def render_profile_summary(state: GameState) -> str:
    return (
        '<section class="profile-current-summary">'
        '<div class="profile-current-rail" aria-hidden="true"></div>'
        '<div class="profile-current-copy"><small>ACTIVE SURVIVOR RECORD</small>'
        f"<strong>{escape(state.profile_name)}</strong>"
        "<span>本地自动保存 · 切换存档不会推进时间</span></div>"
        '<div class="profile-current-meta">'
        f"<b>DAY {state.player.day}</b>"
        f"<span>背包 {state.inventory_slots_used} / 6</span></div>"
        "</section>"
    )


def _stat_bar(label: str, value: int, inverse: bool = False) -> str:
    tone = "risk" if inverse else ""
    return (
        f'<div class="rating-row {tone}"><span>{escape(label)}</span>'
        f'<div class="rating-track"><i style="width:{value * 20}%"></i></div>'
        f"<b>{value}/5</b></div>"
    )


def render_appraisal(
    item: InventoryItem | None,
    fallback: AppraisalResult | None = None,
) -> str:
    if item is None and fallback is None:
        return (
            '<article class="dossier-card empty-dossier">'
            '<div class="dossier-kicker">等待鉴定</div>'
            "<h2>尚未建立物资档案</h2>"
            "<p>上传照片或填写描述，鉴定局会在这里签发末日物资卡。</p>"
            "</article>"
        )

    appraisal = item.appraisal if item else fallback
    assert appraisal is not None
    item_id = item.item_id if item else "DSB-DEMO01"
    warning_html = ""
    if appraisal.warnings:
        warning_html = (
            '<div class="dossier-warning">'
            + " ".join(escape(value) for value in appraisal.warnings)
            + "</div>"
        )

    grade = appraisal.grade.value
    if grade in GRADE_WATER_REWARDS:
        sale_copy = f"实际售价：净水 {GRADE_WATER_REWARDS[grade]:g}L"
    else:
        sale_copy = f"实际售价：士气 +{GRADE_MORALE_REWARDS[grade]}"

    return (
        '<article class="dossier-card">'
        '<header class="dossier-header">'
        '<div><div class="dossier-kicker">WASTELAND APPRAISAL FILE</div>'
        f"<code>{escape(item_id)}</code></div>"
        '<span class="archive-stamp">鉴定通过</span>'
        "</header>"
        '<div class="translated-name"><small>末日译名</small>'
        f"<h2>{escape(appraisal.apocalypse_name)}</h2>"
        f'<p>旧文明名称：{escape(appraisal.original_item)}</p></div>'
        '<div class="grade-value-row">'
        f'<div class="grade-seal"><strong>{grade}</strong><span>级物资</span></div>'
        '<div class="water-value"><small>交易规则</small>'
        f'<strong class="sale-value">{escape(sale_copy)}</strong>'
        "<span>D 级及以上换净水；E/F 级换士气</span></div>"
        "</div>"
        '<section class="rating-panel"><div class="panel-title">六维鉴定</div>'
        + _stat_bar("生存价值", appraisal.stats.survival)
        + _stat_bar("稀缺程度", appraisal.stats.scarcity)
        + _stat_bar("交易价值", appraisal.stats.trade)
        + _stat_bar("保存能力", appraisal.stats.storage)
        + _stat_bar("多用途性", appraisal.stats.versatility)
        + _stat_bar("风险系数", appraisal.stats.risk, inverse=True)
        + "</section>"
        '<section class="dossier-section"><small>建议处置</small>'
        '<strong class="recommendation">'
        f"{escape(appraisal.recommended_action.value)}</strong></section>"
        '<section class="dossier-section"><small>隐藏用途</small>'
        f"<p>{escape(appraisal.hidden_use)}</p></section>"
        '<section class="verdict-block"><small>鉴定判词</small>'
        f"<blockquote>“{escape(appraisal.verdict)}”</blockquote></section>"
        f"{warning_html}"
        "</article>"
    )


def active_inventory_items(state: GameState) -> list[InventoryItem]:
    return [item for item in state.inventory if not item.consumed]


def inventory_gallery_entries(state: GameState) -> list[InventoryGalleryEntry]:
    entries: list[InventoryGalleryEntry] = []
    for item in active_inventory_items(state):
        image_path = item.apocalypse_image or item.original_image
        if image_path:
            entries.append(
                InventoryGalleryEntry(
                    inventory_id=item.item_id,
                    kind="appraised",
                    image_path=image_path,
                    caption=(
                        f"{item.appraisal.apocalypse_name}｜"
                        f"{CATEGORY_LABELS[item.appraisal.category.value]}｜"
                        f"{item.appraisal.grade.value}级"
                    ),
                )
            )
    for purchased in state.market_inventory:
        item = purchased.item
        entries.append(
            InventoryGalleryEntry(
                inventory_id=purchased.inventory_id,
                kind="market",
                image_path=item.image_path,
                caption=(
                    f"{item.name}｜市场购入｜{MARKET_CATEGORY_LABELS[item.category]}｜"
                    f"{market_effect_text(item)}"
                ),
            )
        )
    return entries


def inventory_gallery_value(state: GameState) -> list[tuple[str, str]]:
    return [(entry.image_path, entry.caption) for entry in inventory_gallery_entries(state)]


def inventory_gallery_ids(state: GameState) -> list[str]:
    return [entry.inventory_id for entry in inventory_gallery_entries(state)]


def render_inventory(state: GameState) -> str:
    count = state.inventory_slots_used
    return (
        '<section class="secondary-surface inventory-surface">'
        '<header><div><small>FIELD STORAGE</small>'
        f"<h2>背包 {count} / 6</h2></div>"
        "<p>点选图片后可使用或放弃；只有鉴定物资可以出售。操作不推进时间。</p>"
        "</header>"
        '<div class="inventory-rule">只有鉴定物资能出售：'
        "D 级 0.1L、C 级 0.5L、B 级 2L、A 级 5L、S 级 15L；"
        "E/F 级只换士气。市场购入物资不可转售。</div>"
        "</section>"
    )


def render_log(state: GameState) -> str:
    if not state.log:
        rows = (
            '<div class="empty-state">尚无生存记录。完成第一次鉴定后，'
            "行动会记录在这里。</div>"
        )
    else:
        rows = "".join(
            '<div class="log-row" role="listitem">'
            f'<span>第 {entry.day} 天</span>'
            f'<b>[{LOG_CATEGORY_LABELS.get(entry.category, escape(entry.category))}]</b>'
            f"<p>{escape(entry.message)}</p></div>"
            for entry in reversed(state.log[-20:])
        )
    return (
        '<section class="secondary-surface"><header><div><small>SURVIVAL ARCHIVE</small>'
        "<h2>生存记录</h2></div></header>"
        f'<div class="log-list" role="list">{rows}</div></section>'
    )


def market_effect_text(item: MarketItem) -> str:
    effects: list[str] = []
    if item.satiety_gain:
        effects.append(f"饱食 +{item.satiety_gain}")
    if item.health_gain:
        effects.append(f"生命 +{item.health_gain}")
    if item.energy_gain:
        effects.append(f"体力 +{item.energy_gain}")
    if item.morale_gain:
        effects.append(f"士气 +{item.morale_gain}")
    return " · ".join(effects) if effects else "没有属性收益"


def market_gallery_value(items: list[MarketItem]) -> list[tuple[str, str]]:
    return [
        (
            item.image_path,
            f"{item.name}｜{MARKET_CATEGORY_LABELS[item.category]}｜"
            f"{item.water_price:g}L｜{market_effect_text(item)}",
        )
        for item in items
    ]


def render_market(state: GameState, items: list[MarketItem] | None = None) -> str:
    item_count = len(items or [])
    return (
        '<section class="secondary-surface market-surface"><header><div>'
        "<small>WASTELAND EXCHANGE</small><h2>废土市场</h2></div>"
        '<span class="market-open-badge">交易开放</span></header>'
        '<div class="market-summary">'
        f"<strong>当前净水 {state.player.water:.1f}L · 当日剩余 {item_count} / 5</strong>"
        "<p>市场总目录固定 20 件：3 件食物、3 件医疗用品，以及生存用品、"
        "书本、玩具和杂物。每次休息或鉴定后随机刷新 5 件。</p>"
        "<p>直接点选图片购买。食物与医疗用品耗水较多；书本、玩具和部分杂物"
        "可能没有任何属性收益。购买后先放入背包，需手动使用；购买与使用都不推进时间。</p>"
        "</div>"
        '<div class="market-sell-note"><strong>鉴定物资收购价</strong>'
        "<p>D：0.1L · C：0.5L · B：2.0L · A：5.0L · S：15.0L；"
        "E/F 只能换取士气。</p></div>"
        "</section>"
    )


def render_notice(message: str, tone: str = "info") -> str:
    symbol = "!" if tone == "error" else "✓"
    return (
        f'<div class="operation-notice {escape(tone)}" role="status">'
        f"<span>{symbol}</span><p>{escape(message)}</p></div>"
    )


def render_effects(transition: StateTransition | None = None) -> str:
    if transition is None or (not transition.health_damage and not transition.morale_shake):
        return '<div class="effect-idle" aria-hidden="true"></div>'
    token = uuid4().hex
    parts = [f'<div class="effect-event" data-event="{token}" aria-hidden="true">']
    if transition.health_damage:
        parts.append(
            '<div class="health-hit-trigger">'
            f'<strong>-{transition.health_damage} HP</strong></div>'
        )
    if transition.morale_shake:
        dust = "".join("<i></i>" for _ in range(24))
        parts.append(
            '<div class="morale-shock-trigger">'
            '<strong>士气崩落</strong>'
            f'<div class="falling-dust">{dust}</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)
