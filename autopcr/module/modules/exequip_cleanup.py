from collections import Counter, defaultdict

from ..modulebase import *
from ..config import *
from ...core.pcrclient import pcrclient
from ...model.error import *
from ...model.common import ExtraEquipChangeSlot, ExtraEquipChangeUnit, ExtraEquipProtectInfo, InventoryInfoPost
from ...model.enums import eInventoryType, eSystemId
from .exequip_cleanup_analyzer import ExEquipCleanupAnalyzer, compute_max_possible_r2


EX_EQUIP_SHOP_MONTHLY_LIMIT = 3
EX_EQUIP_SHOP_SURPLUS_R2_TARGET = 2
EX_EQUIP_SHOP_BY_SLOT = {
    1: eSystemId.EX_EQUIPMENT_WEAPON_SHOP,
    2: eSystemId.EX_EQUIPMENT_ARMOR_SHOP,
    3: eSystemId.EX_EQUIPMENT_ACCESSORY_SHOP,
}
EX_EQUIP_SHOP_CURRENCY_NAMES = {
    eSystemId.EX_EQUIPMENT_WEAPON_SHOP: '武器币',
    eSystemId.EX_EQUIPMENT_ARMOR_SHOP: '防具币',
    eSystemId.EX_EQUIPMENT_ACCESSORY_SHOP: '饰品币',
}


def _is_force_locked(ex_id: int) -> bool:
    return bool(db.ex_equipment_data[ex_id].is_force_protected)


def _is_rainbow(ex_id: int) -> bool:
    return db.get_ex_equip_rarity(ex_id) == 5


def _choose_pink_progression_targets(items):
    return sorted(items, key=lambda ex: (-(1 if ex.rank == 1 else 0), -ex.enhancement_pt, -ex.rank))


def _equipped_ex_slots(client):
    equipped = {}
    for unit_id, unit in client.data.unit.items():
        for frame, ex_slots in enumerate([unit.ex_equip_slot, unit.cb_ex_equip_slot], start=1):
            for ex_slot in ex_slots or []:
                if ex_slot.serial_id:
                    equipped[ex_slot.serial_id] = (unit_id, frame, ex_slot.slot)
    return equipped


def _restricted_ex_serials(client):
    return set((getattr(client.data, 'user_clan_battle_ex_equip_restriction', None) or {}).keys())


def _can_modify_ex(ex, restricted_serials):
    return ex.protection_flag != 2 and ex.serial_id not in restricted_serials


def _can_consume_ex(ex, blocked_serials):
    return ex.protection_flag != 2 and ex.serial_id not in blocked_serials


def _stats_for_ex(client, ex_id: int):
    equips = [ex for ex in client.data.ex_equips.values() if ex.ex_equipment_id == ex_id]
    return {
        'full': [ex for ex in equips if ex.rank >= 2 and ex.enhancement_pt >= 6000],
        'r2': [ex for ex in equips if ex.rank >= 2 and ex.enhancement_pt < 6000],
        'r1': [ex for ex in equips if ex.rank == 1],
        'r0': [ex for ex in equips if ex.rank == 0],
        'all': equips,
    }


def _plan_missing_normal_best_ex_equip_buys(report):
    planned_by_ex_id = {}
    for slot_report in report.slot_reports:
        for row in slot_report.equip_reports:
            if row.is_clan_battle or db.get_ex_equip_rarity(row.ex_equipment_id) != 3:
                continue
            # shared_best_count is apportioned across tied best equips for cleanup
            # retention. Purchases must cover every unit for which this normal
            # equip is optimal, even when a clan-battle equip is tied with it.
            best_demand = row.best_count
            if best_demand <= 0:
                continue
            target = best_demand + EX_EQUIP_SHOP_SURPLUS_R2_TARGET
            possible = compute_max_possible_r2({
                'current_full_r2': row.current_full_r2,
                'current_r2_not_full': row.current_r2_not_full,
                'current_r1': row.current_r1,
                'current_r0': row.current_r0,
            })
            shortage = target - possible
            if shortage <= 0:
                continue
            planned = {
                'row': row,
                'buy_count': EX_EQUIP_SHOP_MONTHLY_LIMIT,
                'best_demand': best_demand,
                'target': target,
                'possible': possible,
                'shortage': shortage,
            }
            old = planned_by_ex_id.get(row.ex_equipment_id)
            if old is None or shortage > old['shortage']:
                planned_by_ex_id[row.ex_equipment_id] = planned
    return sorted(planned_by_ex_id.values(), key=lambda item: (item['row'].slot_index, item['row'].category, item['row'].ex_equipment_id))


def _log_planned_shop_buys(module: Module, planned_buys):
    if not planned_buys:
        module._log('预计商店购买: 无')
        return
    module._log('预计商店购买:')
    for item in planned_buys:
        row = item['row']
        module._log(
            f"{row.equip_name} +{item['buy_count']} "
            f"(当前可成品{item['possible']} / 目标{item['target']} = 最优需求{item['best_demand']}+{EX_EQUIP_SHOP_SURPLUS_R2_TARGET})"
        )


def _empty_ex_equip_shop_buy_result():
    return {
        'items': [],
        'skipped': [],
        'cost_by_shop': {},
    }


def _shop_item_purchase_count(item) -> int:
    if item.purchase_count is not None:
        return item.purchase_count
    return item.exchange_count or 0


def _shop_item_remaining_count(item) -> int:
    if item.sold:
        return 0
    purchased = _shop_item_purchase_count(item)
    if item.is_unlimited_stock:
        remaining = EX_EQUIP_SHOP_MONTHLY_LIMIT
    elif item.stock_count is not None:
        remaining = max(0, item.stock_count - purchased)
    else:
        remaining = max(0, EX_EQUIP_SHOP_MONTHLY_LIMIT - purchased)
    return remaining


def _shop_item_buy_cost(item, buy_count: int) -> int:
    if buy_count <= 0:
        return 0
    purchased = _shop_item_purchase_count(item)
    if item.price_group:
        return sum(
            db.get_shop_item_price_info(item.price_group, purchased + offset).count
            for offset in range(buy_count)
        )
    if item.price and item.price.currency_num is not None:
        return item.price.currency_num * buy_count
    raise ValueError(f'商店槽位{item.slot_id}缺少价格信息')


def _affordable_shop_item_count(item, wanted_count: int, currency: int):
    for count in range(wanted_count, 0, -1):
        cost = _shop_item_buy_cost(item, count)
        if cost <= currency:
            return count, cost
    return 0, 0


async def _buy_planned_normal_best_ex_equips(client: pcrclient, planned_buys):
    result = _empty_ex_equip_shop_buy_result()
    if not planned_buys:
        return result

    shop_response = await client.get_shop_item_list()
    shops = {shop.system_id: shop for shop in (shop_response.shop_list or [])}
    candidates_by_shop = defaultdict(list)

    for plan in planned_buys:
        row = plan['row']
        shop_id = EX_EQUIP_SHOP_BY_SLOT.get(row.slot_index)
        shop = shops.get(shop_id)
        if shop is None:
            result['skipped'].append(f'{row.equip_name}: 对应EX商店未开启')
            continue
        shop_item = next((
            item for item in (shop.item_list or [])
            if item.type == eInventoryType.ExtraEquip and item.item_id == row.ex_equipment_id
        ), None)
        if shop_item is None:
            result['skipped'].append(f'{row.equip_name}: 商店中没有该装备')
            continue
        remaining = _shop_item_remaining_count(shop_item)
        wanted = min(plan['buy_count'], remaining)
        if wanted <= 0:
            result['skipped'].append(f'{row.equip_name}: 本月已无可购买次数')
            continue
        candidates_by_shop[shop_id].append((plan, shop_item, wanted))

    for shop_id in sorted(candidates_by_shop):
        currency = client.data.get_shop_gold(shop_id)
        selected = Counter()
        selected_items = []
        estimated_cost = 0
        for plan, shop_item, wanted in candidates_by_shop[shop_id]:
            buy_count, cost = _affordable_shop_item_count(shop_item, wanted, currency)
            row = plan['row']
            if buy_count <= 0:
                result['skipped'].append(
                    f'{row.equip_name}: {EX_EQUIP_SHOP_CURRENCY_NAMES[shop_id]}不足'
                )
                continue
            selected[shop_item.slot_id] = buy_count
            selected_items.append({
                'row': row,
                'buy_count': buy_count,
                'shop_id': shop_id,
            })
            estimated_cost += cost
            currency -= cost
            if buy_count < wanted:
                result['skipped'].append(
                    f'{row.equip_name}: {EX_EQUIP_SHOP_CURRENCY_NAMES[shop_id]}不足，仅购买{buy_count}/{wanted}'
                )

        if not selected:
            continue
        currency_before = client.data.get_shop_gold(shop_id)
        await client.shop_buy_bulk(shop_id, selected)
        currency_after = client.data.get_shop_gold(shop_id)
        actual_cost = currency_before - currency_after
        result['cost_by_shop'][shop_id] = actual_cost if actual_cost > 0 or estimated_cost == 0 else estimated_cost
        result['items'].extend(selected_items)

    return result


def _log_ex_equip_shop_buy_result(module: Module, planned_buys, result):
    if not planned_buys:
        module._log('商店实际购买: 无缺口')
        return
    if result['items']:
        module._log('商店实际购买:')
        for item in result['items']:
            module._log(f"{item['row'].equip_name} +{item['buy_count']}")
        for shop_id, cost in result['cost_by_shop'].items():
            module._log(f'{EX_EQUIP_SHOP_CURRENCY_NAMES[shop_id]}花费: {cost}')
    else:
        module._log('商店实际购买: 0')
    for reason in result['skipped']:
        module._warn(f'购买跳过: {reason}')


async def _unlock_equips(client: pcrclient, equips):
    unlocked = []
    skipped = []
    for ex in equips:
        if _is_rainbow(ex.ex_equipment_id) or _is_force_locked(ex.ex_equipment_id):
            skipped.append((ex.serial_id, ex.ex_equipment_id, 'force_locked_or_rainbow'))
            continue
        try:
            await client.equipment_protect_ex([ExtraEquipProtectInfo(serial_id=ex.serial_id, protection_flag=1)])
            if ex.serial_id in client.data.ex_equips:
                client.data.ex_equips[ex.serial_id].protection_flag = 1
            unlocked.append(ex.serial_id)
        except Exception as e:
            skipped.append((ex.serial_id, ex.ex_equipment_id, str(e)))
    return unlocked, skipped


async def _unequip_all_ex(client: pcrclient):
    removed = 0
    for unit_id, unit in client.data.unit.items():
        normal = [ExtraEquipChangeSlot(slot=i + 1, serial_id=0) for i, ex in enumerate(unit.ex_equip_slot or []) if ex.serial_id]
        clan = [ExtraEquipChangeSlot(slot=i + 1, serial_id=0) for i, ex in enumerate(unit.cb_ex_equip_slot or []) if ex.serial_id]
        if normal or clan:
            await client.unit_equip_ex([ExtraEquipChangeUnit(unit_id=unit_id, ex_equip_slot=normal or None, cb_ex_equip_slot=clan or None)])
            removed += len(normal) + len(clan)
    return removed


async def _rankup_to_target(client: pcrclient, ex_id: int, target_total: int):
    actions = 0
    rarity = db.get_ex_equip_rarity(ex_id)
    while True:
        equipped_slots = _equipped_ex_slots(client)
        restricted_serials = _restricted_ex_serials(client)
        blocked_consume_serials = set(equipped_slots) | restricted_serials
        stats = _stats_for_ex(client, ex_id)
        current_total = len(stats['full']) + len(stats['r2'])
        if current_total >= target_total:
            break
        candidate = None
        need = 0
        if rarity == 4:
            rankup_candidates = _choose_pink_progression_targets([ex for ex in stats['all'] if ex.rank < db.get_ex_equip_max_rank(ex_id) and _can_modify_ex(ex, restricted_serials)])
            if not rankup_candidates:
                break
            candidate = rankup_candidates[0]
            if candidate.rank == 1:
                need = 1
            elif candidate.rank == 0 and len(stats['r0']) >= 2:
                need = 2
            else:
                break
        else:
            r1_candidates = [ex for ex in stats['r1'] if _can_modify_ex(ex, restricted_serials)]
            r0_candidates = [ex for ex in stats['r0'] if _can_modify_ex(ex, restricted_serials)]
            if r1_candidates:
                candidate = sorted(r1_candidates, key=lambda ex: ex.enhancement_pt, reverse=True)[0]
                need = 1
            elif r0_candidates:
                candidate = sorted(r0_candidates, key=lambda ex: ex.serial_id)[0]
                need = 2
            else:
                break
        fodder_candidates = [ex for ex in stats['r0'] if _can_consume_ex(ex, blocked_consume_serials)]
        fodder = [ex.serial_id for ex in (_choose_pink_progression_targets(fodder_candidates) if rarity == 4 else sorted(fodder_candidates, key=lambda ex: ex.serial_id)) if ex.serial_id != candidate.serial_id][:need]
        if len(fodder) < need:
            break
        final_rank = candidate.rank + len(fodder)
        mana = db.get_ex_equip_rankup_cost(ex_id, candidate.rank, final_rank)
        await client.prepare_mana(mana)
        unit_id, frame, slot = equipped_slots.get(candidate.serial_id, (0, 0, 0))
        await client.equipment_rankup_ex(serial_id=candidate.serial_id, unit_id=unit_id, frame=frame, slot=slot, before_rank=candidate.rank, after_rank=final_rank, consume_gold=mana, from_view=2, item_list=[], consume_ex_serial_id_list=fodder)
        actions += 1
    return actions


async def _enhance_to_target(client: pcrclient, ex_id: int, target_full: int, enhance_mode: str = '强化一半', clan_full_cap: int = 20):
    actions = 0
    rarity = db.get_ex_equip_rarity(ex_id)
    while True:
        equipped_slots = _equipped_ex_slots(client)
        restricted_serials = _restricted_ex_serials(client)
        stats = _stats_for_ex(client, ex_id)
        full_cnt = sum(1 for ex in stats['all'] if db.get_ex_equip_star_from_pt(ex.ex_equipment_id, ex.enhancement_pt) >= db.get_ex_equip_max_star(ex.ex_equipment_id, ex.rank)) if rarity == 4 else len(stats['full'])
        if full_cnt >= target_full:
            break
        if rarity == 4:
            candidates = [ex for ex in _choose_pink_progression_targets(stats['all']) if _can_modify_ex(ex, restricted_serials) and db.get_ex_equip_star_from_pt(ex.ex_equipment_id, ex.enhancement_pt) < db.get_ex_equip_max_star(ex.ex_equipment_id, ex.rank)]
            if not candidates:
                break
            candidate = candidates[0]
        else:
            candidates = [ex for ex in stats['r2'] if _can_modify_ex(ex, restricted_serials)]
            candidate = sorted(candidates, key=lambda ex: ex.enhancement_pt, reverse=True)[0] if candidates else None
            if not candidate:
                break
        max_star = db.get_ex_equip_max_star(ex_id, candidate.rank)
        if db.get_ex_equip_star_from_pt(ex_id, candidate.enhancement_pt) >= max_star:
            break
        demand_pt = db.get_ex_equip_enhance_pt(ex_id, candidate.enhancement_pt, max_star)
        mana = db.get_ex_equip_enhance_mana(ex_id, candidate.enhancement_pt, max_star)
        await client.prepare_mana(mana)
        unit_id, frame, slot = equipped_slots.get(candidate.serial_id, (0, 0, 0))
        await client.equipment_enhance_ex(unit_id=unit_id, serial_id=candidate.serial_id, frame=frame, slot=slot, before_enhancement_pt=candidate.enhancement_pt, after_enhancement_pt=candidate.enhancement_pt + demand_pt, consume_gold=mana, from_view=2, item_list=[InventoryInfoPost(type=db.ex_pt[0], id=db.ex_pt[1], count=demand_pt)], consume_ex_serial_id_list=[])
        actions += 1
    return actions


async def _recycle_excess(client: pcrclient, ex_id: int, keep_total: int):
    rarity = db.get_ex_equip_rarity(ex_id)
    if rarity in (4, 5):
        return 0
    equipped_slots = _equipped_ex_slots(client)
    blocked_consume_serials = set(equipped_slots) | _restricted_ex_serials(client)
    stats = _stats_for_ex(client, ex_id)
    recycle_ids = []
    ready_total = len(stats['full']) + len(stats['r2'])
    needed_r2 = max(0, keep_total - len(stats['full']))
    r2_candidates = sorted(
        [ex for ex in stats['r2'] if _can_consume_ex(ex, blocked_consume_serials)],
        key=lambda ex: ex.enhancement_pt,
    )
    r2_excess_count = max(0, len(stats['r2']) - needed_r2)
    recycle_ids.extend(ex.serial_id for ex in r2_candidates[:r2_excess_count])
    if ready_total >= keep_total:
        recycle_ids.extend(
            ex.serial_id for ex in stats['r0']
            if _can_consume_ex(ex, blocked_consume_serials)
        )
    gap = client.data.settings.ex_equip.ex_equip_limit_consume_num
    actions = 0
    for i in range(0, len(recycle_ids), gap):
        chunk = recycle_ids[i:i+gap]
        if not chunk:
            continue
        await client.item_recycle_ex(chunk)
        actions += len(chunk)
    return actions


def _log_report_summary(module: Module, report, title: str):
    slot_totals = defaultdict(int)
    slot_full_totals = defaultdict(int)
    actual_total = 0
    actual_full = 0
    for slot_report in report.slot_reports:
        slot_total = 0
        slot_full = 0
        for eq in slot_report.equip_reports:
            slot_total += eq.current_total
            slot_full += eq.current_full_r2
        slot_totals[slot_report.slot_index] += slot_total
        slot_full_totals[slot_report.slot_index] += slot_full
        actual_total += slot_total
        actual_full += slot_full
    module._log(title)
    module._log(f"当前实际EX总数: {actual_total} / 当前满强总数: {actual_full}")
    module._log(f"槽位1实际EX总数: {slot_totals.get(1, 0)} / 实际满强总数: {slot_full_totals.get(1, 0)}")
    module._log(f"槽位2实际EX总数: {slot_totals.get(2, 0)} / 实际满强总数: {slot_full_totals.get(2, 0)}")
    module._log(f"槽位3实际EX总数: {slot_totals.get(3, 0)} / 实际满强总数: {slot_full_totals.get(3, 0)}")


def _build_detail_rows(report, planned_buy_counts=None):
    planned_buy_counts = planned_buy_counts or {}
    rows = []
    for slot_report in report.slot_reports:
        for item in slot_report.equip_reports:
            current_text = f"{item.current_full_r2}/{item.current_r2_not_full}/{item.current_r1}/{item.current_r0}"
            planned_buy_count = planned_buy_counts.get(item.ex_equipment_id, 0)
            if planned_buy_count:
                current_text += f"(+{planned_buy_count})"
            rows.append({
                '槽位': item.slot_index,
                '类别': item.category,
                '装备': item.equip_name,
                '类': '会战' if item.is_clan_battle else '普通',
                '现状(满/r2/r1/r0)': current_text,
                '目标(总/满/r2)': f"{item.keep_target_min}/{item.full_target}/{max(0, item.keep_target_min - item.full_target)}",
                '差值': item.evidence,
                '可分解': item.decompose_candidate_count,
            })
    return rows


@booltype('ex_equip_cleanup_execute_apply', '执行清理', False)
@booltype('ex_equip_cleanup_execute_prepare', '首次执行前解锁并脱装', True)
@inttype('ex_equip_cleanup_normal_floor_total', '普通最低保留总数', 5, list(range(0, 51)))
@inttype('ex_equip_cleanup_clan_floor_total', '会战最低保留总数', 10, list(range(0, 51)))
@singlechoice('ex_equip_cleanup_enhance_mode', '强化模式', '强化一半', ['不强化', '强化一半', '全强化'])
@inttype('ex_equip_cleanup_clan_full_cap', '会战最多强化数', 20, list(range(0, 51)))
@booltype('ex_equip_cleanup_buy_missing_normal_best', '购买缺口普通最优装', False)
@description('执行 EX 装清理：按照战力最优原则，先购买缺口普通最优金装，再尝试解锁可编辑金/粉装，按目标合成/强化、分解溢出金装；关闭执行清理时仅预览购买计划并在表格现状中标记(+3)')
@name('EX装清理')
@default(True)
class ex_equip_cleanup_execute(Module):
    async def do_task(self, client: pcrclient):
        do_apply = self.get_config('ex_equip_cleanup_execute_apply')
        do_prepare = self.get_config('ex_equip_cleanup_execute_prepare')
        normal_floor = self.get_config('ex_equip_cleanup_normal_floor_total')
        clan_floor = self.get_config('ex_equip_cleanup_clan_floor_total')
        enhance_mode = self.get_config('ex_equip_cleanup_enhance_mode')
        clan_full_cap = self.get_config('ex_equip_cleanup_clan_full_cap')
        buy_missing_normal_best = self.get_config('ex_equip_cleanup_buy_missing_normal_best')
        analyzer = ExEquipCleanupAnalyzer(client, getattr(self._parent, 'alias', 'unknown'), normal_floor_total=normal_floor, clan_floor_total=clan_floor, enhance_mode=enhance_mode, clan_full_cap=clan_full_cap)
        before = analyzer.analyze()

        unlocked = []
        unlock_skipped = []
        removed = 0
        rankup_actions = 0
        enhance_actions = 0
        recycle_actions = 0
        processed = 0
        pink_groups = 0
        pink_rankup = 0
        pink_enhance = 0
        planned_buys = _plan_missing_normal_best_ex_equip_buys(before) if buy_missing_normal_best else []
        shop_buy_result = _empty_ex_equip_shop_buy_result()
        execution_report = before

        if do_apply:
            if buy_missing_normal_best:
                shop_buy_result = await _buy_planned_normal_best_ex_equips(client, planned_buys)
                _log_ex_equip_shop_buy_result(self, planned_buys, shop_buy_result)
                if shop_buy_result['items']:
                    execution_report = ExEquipCleanupAnalyzer(client, getattr(self._parent, 'alias', 'unknown'), normal_floor_total=normal_floor, clan_floor_total=clan_floor, enhance_mode=enhance_mode, clan_full_cap=clan_full_cap).analyze()

            if do_prepare:
                lock_candidates = [ex for ex in client.data.ex_equips.values() if ex.protection_flag == 2 and db.get_ex_equip_rarity(ex.ex_equipment_id) in (3, 4)]
                unlocked, unlock_skipped = await _unlock_equips(client, lock_candidates)
                removed = await _unequip_all_ex(client)

            for slot_report in execution_report.slot_reports:
                for row in slot_report.equip_reports:
                    ex_id = row.ex_equipment_id
                    rarity = db.get_ex_equip_rarity(ex_id)
                    if rarity in (4, 5):
                        continue
                    keep_total = max(row.keep_target_min, clan_floor if row.is_clan_battle else normal_floor)
                    if buy_missing_normal_best and not row.is_clan_battle and row.best_count > 0:
                        keep_total = max(
                            keep_total,
                            row.best_count + EX_EQUIP_SHOP_SURPLUS_R2_TARGET,
                        )
                    full_target = row.full_target
                    rankup_actions += await _rankup_to_target(client, ex_id, keep_total)
                    enhance_actions += await _enhance_to_target(client, ex_id, full_target, enhance_mode=enhance_mode, clan_full_cap=clan_full_cap)
                    recycle_actions += await _recycle_excess(client, ex_id, keep_total)
                    processed += 1

            grouped = defaultdict(list)
            for ex in client.data.ex_equips.values():
                if db.get_ex_equip_rarity(ex.ex_equipment_id) == 4 and not db.ex_equipment_data[ex.ex_equipment_id].is_force_protected:
                    grouped[ex.ex_equipment_id].append(ex)
            pink_groups = len(grouped)
            for ex_id in grouped:
                pink_rankup += await _rankup_to_target(client, ex_id, 999999)
                pink_enhance += await _enhance_to_target(client, ex_id, 999999, enhance_mode='全强化', clan_full_cap=clan_full_cap)

        after = ExEquipCleanupAnalyzer(client, getattr(self._parent, 'alias', 'unknown'), normal_floor_total=normal_floor, clan_floor_total=clan_floor, enhance_mode=enhance_mode, clan_full_cap=clan_full_cap).analyze()

        if do_apply:
            _log_report_summary(self, before, '执行前汇总')
            _log_report_summary(self, after, '执行后汇总')
            self._log(f'解锁成功: {len(unlocked)} / 解锁跳过: {len(unlock_skipped)} / 脱下件数: {removed}')
            self._log(f'合成次数: {rankup_actions + pink_rankup} / 强化次数: {enhance_actions + pink_enhance} / 分解件数: {recycle_actions}')
            self._table_header(['项目', '数值'])
            if buy_missing_normal_best:
                self._table({'项目': '购买装备种类', '数值': len(shop_buy_result['items'])})
                self._table({'项目': '购买件数', '数值': sum(item['buy_count'] for item in shop_buy_result['items'])})
                for shop_id in EX_EQUIP_SHOP_BY_SLOT.values():
                    self._table({'项目': f'{EX_EQUIP_SHOP_CURRENCY_NAMES[shop_id]}花费', '数值': shop_buy_result['cost_by_shop'].get(shop_id, 0)})
                self._table({'项目': '购买跳过', '数值': len(shop_buy_result['skipped'])})
            self._table({'项目': '解锁成功', '数值': len(unlocked)})
            self._table({'项目': '解锁跳过', '数值': len(unlock_skipped)})
            self._table({'项目': '脱下件数', '数值': removed})
            self._table({'项目': '合成次数', '数值': rankup_actions + pink_rankup})
            self._table({'项目': '强化次数', '数值': enhance_actions + pink_enhance})
            self._table({'项目': '分解件数', '数值': recycle_actions})
            self._table({'项目': '粉装分组', '数值': pink_groups})
        else:
            _log_report_summary(self, before, '预览汇总')
            if buy_missing_normal_best:
                _log_planned_shop_buys(self, planned_buys)
            self._table_header(['槽位', '类别', '装备', '类', '现状(满/r2/r1/r0)', '目标(总/满/r2)', '差值', '可分解'])
            planned_buy_counts = {item['row'].ex_equipment_id: item['buy_count'] for item in planned_buys}
            for row in _build_detail_rows(before, planned_buy_counts):
                self._table(row)
