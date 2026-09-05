from __future__ import annotations

from src.game.balance import (
    DAILY_ENERGY_COST,
    DAILY_HUNGER_GAIN,
    DAILY_MORALE_COST,
    DAILY_WATER_COST,
    EFFECT_MAGNITUDES,
    EXTREME_HUNGER_HEALTH_COST,
    GENERATED_FOOD_HEALTH_COST,
    GENERATED_FOOD_HUNGER_RELIEF,
    GENERATED_FOOD_SAFE_YEARS,
    GRADE_MORALE_REWARDS,
    GRADE_WATER_REWARDS,
    INVENTORY_CAPACITY,
    REST_ENERGY_GAIN,
    SCAVENGE_ENERGY_COST,
    SCAVENGE_HUNGER_GAIN,
    SCAVENGE_MORALE_COST,
    TOXIC_DAMAGE_MAX,
    TOXIC_DAMAGE_MIN,
    WATER_SHORTAGE_HEALTH_COST,
)
from src.schemas import (
    GameLogEntry,
    GameState,
    Grade,
    InventoryItem,
    ItemCategory,
    MarketItem,
    PurchasedMarketItem,
    StateTransition,
)

GRADE_ORDER = {
    Grade.S: 7,
    Grade.A: 6,
    Grade.B: 5,
    Grade.C: 4,
    Grade.D: 3,
    Grade.E: 2,
    Grade.F: 1,
}


def _bounded(value: int | float, lower: int | float, upper: int | float):
    return max(lower, min(upper, value))


def _copy(state: GameState) -> GameState:
    return state.model_copy(deep=True)


def _record(state: GameState, category: str, message: str) -> None:
    state.log.append(
        GameLogEntry(category=category, message=message, day=state.player.day)
    )
    state.log = state.log[-100:]


def can_scavenge(state: GameState) -> tuple[bool, str]:
    if state.player.is_game_over:
        return False, "生存记录已经终止。"
    if state.pending_item is not None:
        return False, "请先使用、入库、出售或放弃当前待处理物资。"
    if state.player.energy <= 0:
        return False, "你已经精疲力尽，必须先休息。"
    return True, ""


def _next_toxic_forecast(state: GameState) -> int:
    span = TOXIC_DAMAGE_MAX - TOXIC_DAMAGE_MIN + 1
    step = 2 + (state.scanned_items % 2)
    offset = (state.player.toxic_damage_forecast - TOXIC_DAMAGE_MIN + step) % span
    return TOXIC_DAMAGE_MIN + offset


def _settle_day(state: GameState, *, consume_satiety: bool) -> list[str]:
    player = state.player
    previous_water = player.water
    previous_hunger = player.hunger
    previous_energy = player.energy
    previous_morale = player.morale
    toxic_damage = player.toxic_damage_forecast
    water_shortage = player.water < DAILY_WATER_COST

    player.water = round(max(0.0, player.water - DAILY_WATER_COST), 1)
    if consume_satiety:
        player.hunger = int(_bounded(player.hunger + DAILY_HUNGER_GAIN, 0, 100))
    player.energy = int(_bounded(player.energy - DAILY_ENERGY_COST, 0, 100))
    player.morale = int(_bounded(player.morale - DAILY_MORALE_COST, 0, 100))
    player.health = int(_bounded(player.health - toxic_damage, 0, 100))

    water_cost = round(previous_water - player.water, 1)
    satiety_cost = player.hunger - previous_hunger
    energy_cost = previous_energy - player.energy
    morale_cost = previous_morale - player.morale
    player.day += 1
    player.actions_today = 0
    player.toxic_damage_forecast = _next_toxic_forecast(state)

    messages = [
        f"时间推进至 DAY {player.day}：净水 -{water_cost:g}L，"
        f"饱食度 -{satiety_cost}，体力 -{energy_cost}，士气 -{morale_cost}。",
        f"毒气侵袭：生命 -{toxic_damage}；下一天预计损失 "
        f"{player.toxic_damage_forecast} 点生命。",
    ]
    if player.hunger >= 95:
        player.health = int(
            _bounded(player.health - EXTREME_HUNGER_HEALTH_COST, 0, 100)
        )
        messages.append(f"极端饥饿：生命 -{EXTREME_HUNGER_HEALTH_COST}。")
    if water_shortage:
        player.health = int(
            _bounded(player.health - WATER_SHORTAGE_HEALTH_COST, 0, 100)
        )
        messages.append(f"净水不足：生命 -{WATER_SHORTAGE_HEALTH_COST}。")

    _record(
        state,
        "GAME",
        f"进入 DAY {player.day}，下一天毒气预计造成 "
        f"{player.toxic_damage_forecast} 点伤害。",
    )
    return messages


def _feedback_flags(
    state: GameState,
    *,
    previous_health: int,
    previous_morale: int,
) -> tuple[int, bool]:
    player = state.player
    if player.morale > previous_morale:
        player.morale_shake_anchor = player.morale

    morale_shake = False
    lost_since_anchor = player.morale_shake_anchor - player.morale
    if lost_since_anchor >= 5:
        player.morale_shake_anchor -= (lost_since_anchor // 5) * 5
        morale_shake = True

    return max(0, previous_health - player.health), morale_shake


def _complete_action(
    state: GameState,
    messages: list[str],
    category: str,
    log_message: str,
    *,
    previous_health: int,
    previous_morale: int,
    advance_day: bool = False,
    consume_satiety: bool = True,
) -> StateTransition:
    _record(state, category, log_message)
    if advance_day:
        messages.extend(_settle_day(state, consume_satiety=consume_satiety))
    health_damage, morale_shake = _feedback_flags(
        state,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )
    return StateTransition(
        state=state,
        messages=messages,
        succeeded=True,
        day_advanced=advance_day,
        health_damage=health_damage,
        morale_shake=morale_shake,
    )


def _game_over_transition(state: GameState, action: str) -> StateTransition | None:
    if not state.player.is_game_over:
        return None
    return StateTransition(
        state=_copy(state),
        messages=[f"生存记录已经终止，无法{action}。"],
    )


def register_scanned_item(state: GameState, item: InventoryItem) -> StateTransition:
    allowed, reason = can_scavenge(state)
    if not allowed:
        return StateTransition(state=_copy(state), messages=[reason])

    result = _copy(state)
    player = result.player
    previous_health = player.health
    previous_morale = player.morale
    energy_cost = SCAVENGE_ENERGY_COST
    morale_cost = SCAVENGE_MORALE_COST

    if player.hunger >= 80:
        energy_cost += 4
    if player.morale < 20:
        energy_cost += 2

    player.energy = int(_bounded(player.energy - energy_cost, 0, 100))
    previous_hunger = player.hunger
    player.hunger = int(_bounded(player.hunger + SCAVENGE_HUNGER_GAIN, 0, 100))
    player.morale = int(_bounded(player.morale - morale_cost, 0, 100))
    result.scanned_items += 1
    result.pending_item = item

    grade = item.appraisal.grade
    if result.highest_grade is None or GRADE_ORDER[grade] > GRADE_ORDER[result.highest_grade]:
        result.highest_grade = grade

    messages = [
        f"鉴定完成：体力 -{energy_cost}，"
        f"饱食度 -{player.hunger - previous_hunger}，士气 -{morale_cost}。"
    ]
    return _complete_action(
        result,
        messages,
        "SCAN",
        f"发现 {item.appraisal.apocalypse_name}（{grade}级）。",
        previous_health=previous_health,
        previous_morale=previous_morale,
        advance_day=True,
    )


def rest(state: GameState) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "休息")
    if blocked:
        return blocked

    player = result.player
    previous_health = player.health
    previous_morale = player.morale
    previous_energy = player.energy
    player.energy = int(_bounded(player.energy + REST_ENERGY_GAIN, 0, 100))
    energy_gain = player.energy - previous_energy
    message = f"休息完成：体力 +{energy_gain}；休息不会降低饱食度。"
    return _complete_action(
        result,
        [message],
        "REST",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
        advance_day=True,
        consume_satiety=False,
    )


def store_pending_item(state: GameState) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "整理背包")
    if blocked:
        return blocked
    item = result.pending_item
    if item is None:
        return StateTransition(state=result, messages=["当前没有待处理物资。"])
    if result.inventory_slots_used >= INVENTORY_CAPACITY:
        return StateTransition(
            state=result,
            messages=["背包已经装满，请先使用、出售或放弃一件物资。"],
        )

    previous_health = result.player.health
    previous_morale = result.player.morale
    result.inventory.append(item)
    result.pending_item = None
    message = f"{item.appraisal.apocalypse_name} 已放入背包；时间未推进。"
    return _complete_action(
        result,
        [message],
        "INVENTORY",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def _apply_item_effects(state: GameState, item: InventoryItem) -> str:
    profile = item.appraisal.player_effect_profile
    player = state.player
    previous_health = player.health
    previous_energy = player.energy
    previous_hunger = player.hunger
    previous_morale = player.morale

    if item.appraisal.category is ItemCategory.FOOD:
        health_cost = (
            GENERATED_FOOD_HEALTH_COST
            if item.apocalypse_years > GENERATED_FOOD_SAFE_YEARS
            else 0
        )
        player.health = int(_bounded(player.health - health_cost, 0, 100))
        player.hunger = int(
            _bounded(player.hunger - GENERATED_FOOD_HUNGER_RELIEF, 0, 100)
        )
    else:
        player.health = int(
            _bounded(player.health + EFFECT_MAGNITUDES[profile.health], 0, 100)
        )

    player.energy = int(
        _bounded(player.energy + EFFECT_MAGNITUDES[profile.energy], 0, 100)
    )
    player.morale = int(
        _bounded(player.morale + EFFECT_MAGNITUDES[profile.morale], 0, 100)
    )

    health_delta = player.health - previous_health
    energy_delta = player.energy - previous_energy
    satiety_delta = previous_hunger - player.hunger
    morale_delta = player.morale - previous_morale
    food_note = (
        f"；这是经历 {item.apocalypse_years:g} 年的生成食物，补充较慢且过期会伤身"
        if item.appraisal.category is ItemCategory.FOOD
        else "；非食物不补充饱食度"
    )
    return (
        f"已使用 {item.appraisal.apocalypse_name}：生命 {health_delta:+d}，"
        f"体力 {energy_delta:+d}，饱食度 {satiety_delta:+d}，"
        f"士气 {morale_delta:+d}{food_note}；时间未推进。"
    )


def apply_pending_item(state: GameState) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "使用物资")
    if blocked:
        return blocked
    item = result.pending_item
    if item is None:
        return StateTransition(state=result, messages=["当前没有可使用的物资。"])

    previous_health = result.player.health
    previous_morale = result.player.morale
    result.pending_item = None
    message = _apply_item_effects(result, item)
    return _complete_action(
        result,
        [message],
        "USE",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def apply_inventory_item(state: GameState, item_id: str) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "使用背包物资")
    if blocked:
        return blocked
    item_index = next(
        (
            index
            for index, item in enumerate(result.inventory)
            if item.item_id == item_id and not item.consumed
        ),
        None,
    )
    if item_index is None:
        return StateTransition(state=result, messages=["请先点选一件有效的背包物资。"])

    previous_health = result.player.health
    previous_morale = result.player.morale
    item = result.inventory.pop(item_index)
    message = _apply_item_effects(result, item)
    return _complete_action(
        result,
        [message],
        "USE",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def abandon_pending_item(state: GameState) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "放弃物资")
    if blocked:
        return blocked
    item = result.pending_item
    if item is None:
        return StateTransition(state=result, messages=["当前没有待放弃的物资。"])
    previous_health = result.player.health
    previous_morale = result.player.morale
    result.pending_item = None
    message = f"你放弃了 {item.appraisal.apocalypse_name}；时间未推进。"
    return _complete_action(
        result,
        [message],
        "INVENTORY",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def abandon_inventory_item(state: GameState, item_id: str) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "放弃背包物资")
    if blocked:
        return blocked
    item_index = next(
        (
            index
            for index, item in enumerate(result.inventory)
            if item.item_id == item_id and not item.consumed
        ),
        None,
    )
    if item_index is None:
        return StateTransition(state=result, messages=["请先点选一件有效的背包物资。"])
    previous_health = result.player.health
    previous_morale = result.player.morale
    item = result.inventory.pop(item_index)
    message = f"已从背包放弃 {item.appraisal.apocalypse_name}；时间未推进。"
    return _complete_action(
        result,
        [message],
        "INVENTORY",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def _sell_item(state: GameState, item: InventoryItem) -> str:
    grade = item.appraisal.grade.value
    if grade in GRADE_WATER_REWARDS:
        value = GRADE_WATER_REWARDS[grade]
        state.player.water = round(state.player.water + value, 1)
        return (
            f"已出售 {item.appraisal.apocalypse_name}（{grade}级），"
            f"获得净水 {value:g}L；时间未推进。"
        )
    morale = GRADE_MORALE_REWARDS[grade]
    previous_morale = state.player.morale
    state.player.morale = int(_bounded(previous_morale + morale, 0, 100))
    actual_gain = state.player.morale - previous_morale
    return (
        f"已处理 {item.appraisal.apocalypse_name}（{grade}级），"
        f"获得士气 {actual_gain:+d}；E/F 级不换净水，时间未推进。"
    )


def sell_pending_item(state: GameState) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "出售物资")
    if blocked:
        return blocked
    item = result.pending_item
    if item is None:
        return StateTransition(state=result, messages=["当前没有可出售的物资。"])
    if item.appraisal.is_fallback:
        return StateTransition(state=result, messages=["鉴定失败的物资不能出售。"])
    previous_health = result.player.health
    previous_morale = result.player.morale
    result.pending_item = None
    message = _sell_item(result, item)
    return _complete_action(
        result,
        [message],
        "MARKET",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def sell_inventory_item(state: GameState, item_id: str) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "出售背包物资")
    if blocked:
        return blocked
    if any(entry.inventory_id == item_id for entry in result.market_inventory):
        return StateTransition(
            state=result,
            messages=["市场购入物资不能转售，只能使用或放弃。"],
        )
    item_index = next(
        (
            index
            for index, item in enumerate(result.inventory)
            if item.item_id == item_id and not item.consumed
        ),
        None,
    )
    if item_index is None:
        return StateTransition(state=result, messages=["请先点选一件有效的背包物资。"])
    if result.inventory[item_index].appraisal.is_fallback:
        return StateTransition(state=result, messages=["鉴定失败的物资不能出售。"])
    previous_health = result.player.health
    previous_morale = result.player.morale
    item = result.inventory.pop(item_index)
    message = _sell_item(result, item)
    return _complete_action(
        result,
        [message],
        "MARKET",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def _apply_market_item_effects(state: GameState, item: MarketItem) -> str:
    player = state.player
    previous_health = player.health
    previous_morale = player.morale
    previous_hunger = player.hunger
    previous_energy = player.energy
    player.health = int(_bounded(player.health + item.health_gain, 0, 100))
    player.hunger = int(_bounded(player.hunger - item.satiety_gain, 0, 100))
    player.energy = int(_bounded(player.energy + item.energy_gain, 0, 100))
    player.morale = int(_bounded(player.morale + item.morale_gain, 0, 100))
    return (
        f"已使用市场物资 {item.name}：生命 {player.health - previous_health:+d}，"
        f"体力 {player.energy - previous_energy:+d}，"
        f"饱食度 {previous_hunger - player.hunger:+d}，"
        f"士气 {player.morale - previous_morale:+d}；时间未推进。"
    )


def apply_market_inventory_item(state: GameState, inventory_id: str) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "使用背包物资")
    if blocked:
        return blocked
    item_index = next(
        (
            index
            for index, entry in enumerate(result.market_inventory)
            if entry.inventory_id == inventory_id
        ),
        None,
    )
    if item_index is None:
        return StateTransition(state=result, messages=["请先点选一件有效的背包物资。"])

    previous_health = result.player.health
    previous_morale = result.player.morale
    entry = result.market_inventory.pop(item_index)
    message = _apply_market_item_effects(result, entry.item)
    return _complete_action(
        result,
        [message],
        "USE",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def abandon_market_inventory_item(state: GameState, inventory_id: str) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "放弃背包物资")
    if blocked:
        return blocked
    item_index = next(
        (
            index
            for index, entry in enumerate(result.market_inventory)
            if entry.inventory_id == inventory_id
        ),
        None,
    )
    if item_index is None:
        return StateTransition(state=result, messages=["请先点选一件有效的背包物资。"])

    previous_health = result.player.health
    previous_morale = result.player.morale
    entry = result.market_inventory.pop(item_index)
    message = f"已从背包放弃市场物资 {entry.item.name}；时间未推进。"
    return _complete_action(
        result,
        [message],
        "INVENTORY",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )


def buy_market_item(state: GameState, item: MarketItem | None) -> StateTransition:
    result = _copy(state)
    blocked = _game_over_transition(result, "进行市场交易")
    if blocked:
        return blocked
    if item is None or item.item_id not in result.market_item_ids:
        return StateTransition(state=result, messages=["该商品不在当前市场中。"])
    if result.inventory_slots_used >= INVENTORY_CAPACITY:
        return StateTransition(
            state=result,
            messages=["背包已经装满，请先使用、出售或放弃一件物资。"],
        )
    if result.player.water < item.water_price:
        return StateTransition(
            state=result,
            messages=[f"净水不足 {item.water_price:g}L，无法换取{item.name}。"],
        )

    player = result.player
    previous_health = player.health
    previous_morale = player.morale
    player.water = round(player.water - item.water_price, 1)
    result.market_item_ids.remove(item.item_id)
    result.market_inventory.append(
        PurchasedMarketItem(item=item.model_copy(deep=True))
    )
    message = (
        f"市场交易：支付 {item.water_price:g}L 净水换取 {item.name}；"
        "商品已放入背包，需在背包中手动使用；时间未推进。"
    )
    return _complete_action(
        result,
        [message],
        "MARKET",
        message,
        previous_health=previous_health,
        previous_morale=previous_morale,
    )
