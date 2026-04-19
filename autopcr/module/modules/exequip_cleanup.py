from collections import defaultdict

from ..modulebase import *
from ..config import *
from ...core.pcrclient import pcrclient
from ...model.error import *
from ...model.common import ExtraEquipChangeSlot, ExtraEquipChangeUnit, ExtraEquipProtectInfo, InventoryInfoPost
from .exequip_cleanup_analyzer import ExEquipCleanupAnalyzer, summarize_retention_totals, summarize_full_totals


def _is_force_locked(ex_id: int) -> bool:
    return bool(db.ex_equipment_data[ex_id].is_force_protected)


def _is_rainbow(ex_id: int) -> bool:
    return db.get_ex_equip_rarity(ex_id) == 5


def _choose_pink_progression_targets(items):
    return sorted(items, key=lambda ex: (-(1 if ex.rank == 1 else 0), -ex.enhancement_pt, -ex.rank))


def _stats_for_ex(client, ex_id: int):
    equips = [ex for ex in client.data.ex_equips.values() if ex.ex_equipment_id == ex_id]
    return {
        'full': [ex for ex in equips if ex.rank >= 2 and ex.enhancement_pt >= 6000],
        'r2': [ex for ex in equips if ex.rank >= 2 and ex.enhancement_pt < 6000],
        'r1': [ex for ex in equips if ex.rank == 1],
        'r0': [ex for ex in equips if ex.rank == 0],
        'all': equips,
    }


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
        stats = _stats_for_ex(client, ex_id)
        current_total = len(stats['full']) + len(stats['r2'])
        if current_total >= target_total:
            break
        candidate = None
        need = 0
        if rarity == 4:
            rankup_candidates = _choose_pink_progression_targets([ex for ex in stats['all'] if ex.rank < db.get_ex_equip_max_rank(ex_id)])
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
            if stats['r1']:
                candidate = sorted(stats['r1'], key=lambda ex: ex.enhancement_pt, reverse=True)[0]
                need = 1
            elif stats['r0']:
                candidate = sorted(stats['r0'], key=lambda ex: ex.serial_id)[0]
                need = 2
            else:
                break
        fodder = [ex.serial_id for ex in (_choose_pink_progression_targets(stats['r0']) if rarity == 4 else sorted(stats['r0'], key=lambda ex: ex.serial_id)) if ex.serial_id != candidate.serial_id][:need]
        if len(fodder) < need:
            break
        final_rank = candidate.rank + len(fodder)
        mana = db.get_ex_equip_rankup_cost(ex_id, candidate.rank, final_rank)
        await client.prepare_mana(mana)
        await client.equipment_rankup_ex(serial_id=candidate.serial_id, unit_id=0, frame=0, slot=0, before_rank=candidate.rank, after_rank=final_rank, consume_gold=mana, from_view=2, item_list=[], consume_ex_serial_id_list=fodder)
        actions += 1
    return actions


async def _enhance_to_target(client: pcrclient, ex_id: int, target_full: int, enhance_mode: str = '强化一半', clan_full_cap: int = 20):
    actions = 0
    rarity = db.get_ex_equip_rarity(ex_id)
    while True:
        stats = _stats_for_ex(client, ex_id)
        full_cnt = sum(1 for ex in stats['all'] if db.get_ex_equip_star_from_pt(ex.ex_equipment_id, ex.enhancement_pt) >= db.get_ex_equip_max_star(ex.ex_equipment_id, ex.rank)) if rarity == 4 else len(stats['full'])
        if full_cnt >= target_full:
            break
        if rarity == 4:
            candidates = [ex for ex in _choose_pink_progression_targets(stats['all']) if db.get_ex_equip_star_from_pt(ex.ex_equipment_id, ex.enhancement_pt) < db.get_ex_equip_max_star(ex.ex_equipment_id, ex.rank)]
            if not candidates:
                break
            candidate = candidates[0]
        else:
            candidate = sorted(stats['r2'], key=lambda ex: ex.enhancement_pt, reverse=True)[0] if stats['r2'] else None
            if not candidate:
                break
        max_star = db.get_ex_equip_max_star(ex_id, candidate.rank)
        if db.get_ex_equip_star_from_pt(ex_id, candidate.enhancement_pt) >= max_star:
            break
        demand_pt = db.get_ex_equip_enhance_pt(ex_id, candidate.enhancement_pt, max_star)
        mana = db.get_ex_equip_enhance_mana(ex_id, candidate.enhancement_pt, max_star)
        await client.prepare_mana(mana)
        await client.equipment_enhance_ex(unit_id=0, serial_id=candidate.serial_id, frame=0, slot=0, before_enhancement_pt=candidate.enhancement_pt, after_enhancement_pt=candidate.enhancement_pt + demand_pt, consume_gold=mana, from_view=2, item_list=[InventoryInfoPost(type=db.ex_pt[0], id=db.ex_pt[1], count=demand_pt)], consume_ex_serial_id_list=[])
        actions += 1
    return actions


async def _recycle_excess(client: pcrclient, ex_id: int, keep_total: int, full_target: int):
    rarity = db.get_ex_equip_rarity(ex_id)
    if rarity in (4, 5):
        return 0
    stats = _stats_for_ex(client, ex_id)
    keep_r2 = max(0, keep_total - full_target)
    recycle_ids = []
    r2_excess = sorted(stats['r2'], key=lambda ex: ex.enhancement_pt)
    if len(r2_excess) > keep_r2:
        recycle_ids.extend(ex.serial_id for ex in r2_excess[keep_r2:])
    recycle_ids.extend(ex.serial_id for ex in stats['r0'])
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
    totals = summarize_retention_totals(report.slot_reports)
    full_totals = summarize_full_totals(report.slot_reports)
    actual_total = sum(eq.current_total for slot in report.slot_reports for eq in slot.equip_reports)
    actual_full = sum(eq.current_full_r2 for slot in report.slot_reports for eq in slot.equip_reports)
    module._log(title)
    module._log(f"当前实际EX总数: {actual_total} / 当前满强总数: {actual_full}")
    module._log(f"槽位1总保留数: {totals['slot_totals'].get(1, 0)} / 满强目标总数: {full_totals.get(1, 0)}")
    module._log(f"槽位2总保留数: {totals['slot_totals'].get(2, 0)} / 满强目标总数: {full_totals.get(2, 0)}")
    module._log(f"槽位3总保留数: {totals['slot_totals'].get(3, 0)} / 满强目标总数: {full_totals.get(3, 0)}")
    module._log(f"总计保留数: {totals['grand_total']} / 总计满强目标: {sum(full_totals.values())}")


def _build_detail_rows(report):
    rows = []
    for slot_report in report.slot_reports:
        for item in slot_report.equip_reports:
            rows.append({
                '槽位': item.slot_index,
                '类别': item.category,
                '装备': item.equip_name,
                '类': '会战' if item.is_clan_battle else '普通',
                '现状(满/r2/r1/r0)': f"{item.current_full_r2}/{item.current_r2_not_full}/{item.current_r1}/{item.current_r0}",
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
@description('执行 EX 装清理：按照战力最优原则，尝试解锁可编辑金/粉装，按目标合成/强化、分解溢出金装；关闭执行清理时仅返回执行前后的摘要，不真正操作')
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

        if do_apply:
            if do_prepare:
                lock_candidates = [ex for ex in client.data.ex_equips.values() if ex.protection_flag == 2 and db.get_ex_equip_rarity(ex.ex_equipment_id) in (3, 4)]
                unlocked, unlock_skipped = await _unlock_equips(client, lock_candidates)
                removed = await _unequip_all_ex(client)

            for slot_report in before.slot_reports:
                for row in slot_report.equip_reports:
                    ex_id = row.ex_equipment_id
                    rarity = db.get_ex_equip_rarity(ex_id)
                    if rarity in (4, 5):
                        continue
                    keep_total = max(row.keep_target_min, clan_floor if row.is_clan_battle else normal_floor)
                    full_target = row.full_target
                    rankup_actions += await _rankup_to_target(client, ex_id, keep_total)
                    enhance_actions += await _enhance_to_target(client, ex_id, full_target, enhance_mode=enhance_mode, clan_full_cap=clan_full_cap)
                    recycle_actions += await _recycle_excess(client, ex_id, keep_total, full_target)
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
            self._table({'项目': '解锁成功', '数值': len(unlocked)})
            self._table({'项目': '解锁跳过', '数值': len(unlock_skipped)})
            self._table({'项目': '脱下件数', '数值': removed})
            self._table({'项目': '合成次数', '数值': rankup_actions + pink_rankup})
            self._table({'项目': '强化次数', '数值': enhance_actions + pink_enhance})
            self._table({'项目': '分解件数', '数值': recycle_actions})
            self._table({'项目': '粉装分组', '数值': pink_groups})
        else:
            _log_report_summary(self, before, '预览汇总')
            self._table_header(['槽位', '类别', '装备', '类', '现状(满/r2/r1/r0)', '目标(总/满/r2)', '差值', '可分解'])
            for row in _build_detail_rows(before):
                self._table(row)
