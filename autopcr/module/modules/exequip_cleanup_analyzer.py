from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

from ...db.database import db
from .exequip_helpers import ExEquipConstants, ExEquipPowerCalculator, ExEquipRecommender


def summarize_inventory_item(ex) -> Dict[str, int]:
    if ex.rank >= ExEquipConstants.RANK_MAX and ex.enhancement_pt >= ExEquipConstants.ENHANCEMENT_PT_MAX:
        return {"full_r2": 1, "r2_not_full": 0, "r1": 0, "r0": 0, "points": 3}
    if ex.rank >= ExEquipConstants.RANK_MAX:
        return {"full_r2": 0, "r2_not_full": 1, "r1": 0, "r0": 0, "points": 3}
    if ex.rank == ExEquipConstants.RANK_ONE:
        return {"full_r2": 0, "r2_not_full": 0, "r1": 1, "r0": 0, "points": 2}
    return {"full_r2": 0, "r2_not_full": 0, "r1": 0, "r0": 1, "points": 1}


def merge_inventory_stats(items) -> Dict[str, int]:
    total = Counter({
        "current_total": 0,
        "current_full_r2": 0,
        "current_r2_not_full": 0,
        "current_r1": 0,
        "current_r0": 0,
        "equivalent_points": 0,
    })
    for ex in items:
        one = summarize_inventory_item(ex)
        total["current_total"] += 1
        total["current_full_r2"] += one["full_r2"]
        total["current_r2_not_full"] += one["r2_not_full"]
        total["current_r1"] += one["r1"]
        total["current_r0"] += one["r0"]
        total["equivalent_points"] += one["points"]
    return dict(total)


def clan_battle_keep_range(tier: str) -> tuple[int, int]:
    mapping = {
        "high": (12, 15),
        "medium": (6, 8),
        "low": (3, 5),
        "minimal": (1, 2),
    }
    return mapping.get(tier, (1, 2))


def collect_report_equipment_ids(recommended_ids: set[int], inventory_ids: set[int]) -> List[int]:
    return sorted(recommended_ids | inventory_ids)


def summarize_gap_statistics(gaps: List[int], best_count: int) -> Dict[str, int]:
    if not gaps:
        return {'avg_gap': 0, 'best_count': best_count}
    return {'avg_gap': round(sum(gaps) / len(gaps)), 'best_count': best_count}


def format_gap_summary(avg_gap: int, best_count: int) -> str:
    return f"{avg_gap} ({best_count})"


def build_power_evidence_text(counts: Dict[str, int], evidence: Dict[str, int | str]) -> str:
    avg_gap = evidence.get("avg_gap", 0)
    best_count = evidence.get("best_count", 0)
    return format_gap_summary(avg_gap, best_count)


def compute_slot_unit_counts(owned_units: Dict[int, object], unit_slot_map: Dict[int, object], slot_index: int, category: int) -> tuple[int, int, int]:
    db_unit_count = 0
    unit_count = 0
    for unit_id, slot_data in unit_slot_map.items():
        slot_category = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot_index - 1]
        if slot_category == category:
            db_unit_count += 1
            if unit_id in owned_units:
                unit_count += 1
    return unit_count, db_unit_count, db_unit_count - unit_count


def compute_keep_cap(unit_count: int, future_release_count: int) -> int:
    return unit_count + max(future_release_count, 2)


def allocate_reports_by_choice_clusters(reports: List[Dict], clusters: List[Dict], category_target: int) -> List[Dict]:
    alloc = {item['equip_name']: 0 for item in reports}
    by_ex_id = {item['ex_equipment_id']: item for item in reports}
    all_best_ids = set()
    remaining = 0

    cluster_alloc = allocate_cluster_shares(clusters, category_target)
    for cluster in clusters:
        best_ids = tuple(cluster['best_ids'])
        target = cluster_alloc.get(best_ids, 0)
        if target <= 0:
            continue
        all_best_ids.update(best_ids)
        need = target
        ordered_best_ids = sorted(best_ids, key=lambda ex_id: 1 if by_ex_id.get(ex_id, {}).get('is_clan_battle') else 0)
        for ex_id in ordered_best_ids:
            item = by_ex_id.get(ex_id)
            if not item:
                continue
            available = max(0, item.get('available_r2', item['current_total']) - alloc[item['equip_name']])
            take = min(available, need)
            alloc[item['equip_name']] += take
            need -= take
            if need <= 0:
                break
        remaining += need

    if remaining > 0:
        next_nonclan = sorted(
            [item for item in reports if not item['is_clan_battle'] and item['ex_equipment_id'] not in all_best_ids],
            key=lambda x: x.get('gap_avg', 999999)
        )
        for item in next_nonclan:
            available = max(0, item.get('available_r2', item['current_total']) - alloc[item['equip_name']])
            take = min(available, remaining)
            alloc[item['equip_name']] += take
            remaining -= take
            if remaining <= 0:
                break

    result = []
    for item in reports:
        new_item = dict(item)
        targets = finalize_targets({
            'ex_equipment_id': item['ex_equipment_id'],
            'is_clan_battle': item['is_clan_battle'],
            'allocated_mainline': alloc.get(item['equip_name'], 0),
        })
        new_item['keep_target_min'] = targets['keep_target_min']
        new_item['keep_target_max'] = targets['keep_target_min']
        new_item['full_target'] = targets['full_target']
        new_item['r2_hold_target'] = targets['r2_hold_target']
        result.append(new_item)
    return result


def compute_decompose_tiers(stats: Dict[str, int], reserve_r2_unenhanced: int, keep_target_total: int) -> Dict[str, int]:
    need_from_nonfull = max(0, keep_target_total - stats['current_full_r2'])
    r2_not_full_keep = min(stats['current_r2_not_full'], need_from_nonfull)
    extra_r2_not_full = max(0, stats['current_r2_not_full'] - r2_not_full_keep)

    crafts_needed = max(0, need_from_nonfull - stats['current_r2_not_full'])
    r1_used = min(stats['current_r1'], crafts_needed, stats['current_r0'])
    r0_used = r1_used + 3 * max(0, crafts_needed - r1_used)
    leftover_r0 = max(0, stats['current_r0'] - r0_used)
    decompose_r0 = max(0, leftover_r0 - (leftover_r0 % 3))

    return {
        'full_r2_keep': stats['current_full_r2'],
        'r2_not_full_keep': r2_not_full_keep,
        'r1_keep': max(0, stats['current_r1'] - r1_used),
        'r0_keep': max(0, stats['current_r0'] - decompose_r0),
        'decompose': extra_r2_not_full + decompose_r0,
    }


def compute_max_possible_r2(stats: Dict[str, int]) -> int:
    return stats['current_full_r2'] + stats['current_r2_not_full'] + (stats['current_r1'] + stats['current_r0']) // 3


def get_non_clan_retention_rule(ex_equipment_id: int, normal_floor_total: int = 5) -> Dict[str, int]:
    if ex_equipment_id in {4201303, 4202301}:
        return {'full_target': 0, 'r2_hold_target': 15}
    if ex_equipment_id == 4103303:
        return {'full_target': 6, 'r2_hold_target': 0}
    return {'full_target': 2, 'r2_hold_target': max(0, normal_floor_total - 2)}


def get_clan_retention_rule(ex_equipment_id: int, clan_floor_total: int = 10) -> Dict[str, int]:
    important = {
        4101351, 4109351, 4110351, 4201351, 4204351, 4301301, 4304351,
    }
    if ex_equipment_id in important:
        return {'full_target': 10, 'r2_hold_target': 10}
    return {'full_target': clan_floor_total // 2, 'r2_hold_target': clan_floor_total - clan_floor_total // 2}


def build_floor_targets(ex_equipment_id: int, is_clan_battle: bool, normal_floor_total: int = 5, clan_floor_total: int = 10, enhance_mode: str = '强化一半', clan_full_cap: int = 20) -> Dict[str, int]:
    rule = get_clan_retention_rule(ex_equipment_id, clan_floor_total) if is_clan_battle else get_non_clan_retention_rule(ex_equipment_id, normal_floor_total)
    floor_total = rule['full_target'] + rule['r2_hold_target']
    floor_full = rule['full_target']
    if ex_equipment_id in {4201303, 4202301}:
        floor_full = 0
    elif enhance_mode == '不强化':
        floor_full = 0
    elif enhance_mode == '强化一半':
        floor_full = floor_full // 2
    if is_clan_battle:
        floor_full = min(floor_full, clan_full_cap)
    floor_full = min(floor_full, floor_total)
    return {
        'floor_total': floor_total,
        'floor_full': floor_full,
    }


def merge_mainline_and_floor_targets(mainline: tuple[int, int], floor: tuple[int, int], surplus: int = 0) -> Dict[str, int]:
    keep_target_min = max(mainline[0] + surplus, floor[0])
    full_target = max(mainline[1], floor[1])
    full_target = min(full_target, keep_target_min)
    return {
        'keep_target_min': keep_target_min,
        'full_target': full_target,
        'r2_hold_target': max(0, keep_target_min - full_target),
    }


def apply_full_target_mode_cap(is_clan_battle: bool, keep_target: int, full_target: int, enhance_mode: str = '强化一半', clan_full_cap: int = 20) -> Dict[str, int]:
    if is_clan_battle:
        if enhance_mode == '不强化':
            full_target = 0
        elif enhance_mode == '强化一半':
            full_target = full_target // 2
        full_target = min(full_target, clan_full_cap)
    full_target = min(full_target, keep_target)
    return {'keep_target_min': keep_target, 'full_target': full_target}


def finalize_targets(item: Dict) -> Dict[str, int]:
    allocated = item.get('allocated_mainline', 0)
    mainline = (allocated, allocated)
    floor_info = build_floor_targets(
        item['ex_equipment_id'],
        item['is_clan_battle'],
        item.get('normal_floor_total', 5),
        item.get('clan_floor_total', 10),
        item.get('enhance_mode', '强化一半'),
        item.get('clan_full_cap', 20),
    )
    floor = (floor_info['floor_total'], floor_info['floor_full'])
    return merge_mainline_and_floor_targets(mainline, floor, item.get('surplus', 0))


def build_inventory_slot_category_index(inventory_items, ex_equipment_data: Dict[int, object], get_rarity) -> Dict[tuple[int, int], set[int]]:
    result = defaultdict(set)
    for ex in inventory_items:
        ex_data = ex_equipment_data.get(ex.ex_equipment_id)
        if not ex_data or get_rarity(ex.ex_equipment_id) != 3:
            continue
        slot_index = (ex.ex_equipment_id // 100000) % 10
        if slot_index not in (1, 2, 3):
            continue
        result[(slot_index, ex_data.category)].add(ex.ex_equipment_id)
    return result


def compute_gold_keep_targets_with_pink_override(counts: Dict[str, int], gold_best_power: int, pink_items: List[Dict[str, int]]) -> Dict[str, int]:
    covered = sum(1 for item in pink_items if item.get('current_power', 0) > gold_best_power)
    keep_min = max(0, counts['exclusive_best'] + counts['shared_best'] + counts['alt1'] + min(counts['alt2'], 1) - covered)
    full_target = max(0, min(keep_min, max(counts['exclusive_best'] + counts['shared_best'] - covered, 0)))
    return {'keep_min': keep_min, 'full_target': full_target, 'covered_by_pink': covered}


def match_pink_override_units(pink_edges: Dict[int, List[int]]) -> set[int]:
    matched_units: Dict[int, int] = {}

    def dfs(item_id: int, seen: set[int]) -> bool:
        for unit_idx in pink_edges.get(item_id, []):
            if unit_idx in seen:
                continue
            seen.add(unit_idx)
            if unit_idx not in matched_units or dfs(matched_units[unit_idx], seen):
                matched_units[unit_idx] = item_id
                return True
        return False

    for item_id in pink_edges:
        dfs(item_id, set())

    return set(matched_units.keys())


def filter_unit_slot_recommendations(unit_slot_recommendations: Dict, covered_unit_slots: set[tuple[int, int]]) -> Dict:
    return {
        key: value
        for key, value in unit_slot_recommendations.items()
        if key not in covered_unit_slots
    }


def summarize_retention_totals(slot_reports: List[Dict]) -> Dict[str, Dict[int, int] | int]:
    slot_totals: Dict[int, int] = defaultdict(int)
    category_totals: Dict[str, int] = {}
    grand_total = 0
    for slot_report in slot_reports:
        slot_total = 0
        for item in slot_report['equip_reports'] if isinstance(slot_report, dict) else slot_report.equip_reports:
            keep = item['keep_target_min'] if isinstance(item, dict) else item.keep_target_min
            slot_total += keep
            category_key = f"{slot_report['slot_index'] if isinstance(slot_report, dict) else slot_report.slot_index}-{slot_report['category'] if isinstance(slot_report, dict) else slot_report.category}"
            category_totals[category_key] = category_totals.get(category_key, 0) + keep
        slot_index = slot_report['slot_index'] if isinstance(slot_report, dict) else slot_report.slot_index
        slot_totals[slot_index] += slot_total
        grand_total += slot_total
    return {'slot_totals': dict(slot_totals), 'category_totals': category_totals, 'grand_total': grand_total}


def summarize_full_totals(slot_reports: List[Dict]) -> Dict[int, int]:
    slot_full_totals: Dict[int, int] = defaultdict(int)
    for slot_report in slot_reports:
        slot_index = slot_report['slot_index'] if isinstance(slot_report, dict) else slot_report.slot_index
        for item in slot_report['equip_reports'] if isinstance(slot_report, dict) else slot_report.equip_reports:
            full = item['full_target'] if isinstance(item, dict) else item.full_target
            slot_full_totals[slot_index] += full
    return {1: slot_full_totals.get(1, 0), 2: slot_full_totals.get(2, 0), 3: slot_full_totals.get(3, 0)}


def proportional_allocate_by_best_count(items: List[Dict], total_target: int) -> Dict[str, int]:
    total_best = sum(item['best_count'] for item in items)
    if total_best <= 0:
        return {item['equip_name']: 0 for item in items}
    alloc = {}
    remaining = total_target
    remainders = []
    for item in items:
        raw = item['best_count'] * total_target / total_best
        base = int(raw)
        alloc[item['equip_name']] = base
        remaining -= base
        remainders.append((raw - base, item['equip_name']))
    for _, name in sorted(remainders, reverse=True)[:remaining]:
        alloc[name] += 1
    return alloc


def cluster_units_by_top_choice(unit_slot_recommendations: Dict) -> Dict[tuple[int, tuple[int, ...]], Dict]:
    clusters: Dict[tuple[int, tuple[int, ...]], Dict] = {}
    for (unit_id, slot), recommendation in unit_slot_recommendations.items():
        best = tuple(sorted(ex_id for ex_id, _ in recommendation.get('best_equips', [])))
        if not best:
            continue
        top_choice = (best[0],)
        key = (slot, top_choice)
        if key not in clusters:
            clusters[key] = {
                'slot': slot,
                'best_ids': top_choice,
                'same_power_ids': best,
                'unit_ids': [],
            }
        clusters[key]['unit_ids'].append(unit_id)
    return clusters


def allocate_cluster_shares(clusters: List[Dict], total_target: int) -> Dict[tuple[int, ...], int]:
    total = sum(len(cluster['unit_ids']) for cluster in clusters)
    if total <= 0:
        return {tuple(cluster['best_ids']): 0 for cluster in clusters}
    alloc: Dict[tuple[int, ...], int] = {}
    remaining = total_target
    remainders = []
    for cluster in clusters:
        best_ids = tuple(cluster['best_ids'])
        raw = len(cluster['unit_ids']) * total_target / total
        base = int(raw)
        alloc[best_ids] = base
        remaining -= base
        remainders.append((raw - base, best_ids))
    for _, best_ids in sorted(remainders, reverse=True)[:remaining]:
        alloc[best_ids] += 1
    return alloc


def allocate_cluster_internal(cluster: Dict, reports_by_ex_id: Dict[int, Dict], target: int) -> Dict[str, int]:
    alloc = defaultdict(int)
    if target <= 0:
        return alloc
    same_power_ids = cluster.get('same_power_ids', cluster['best_ids'])
    non_clan_ids = [ex_id for ex_id in same_power_ids if ex_id in reports_by_ex_id and not reports_by_ex_id[ex_id]['is_clan_battle']]
    clan_ids = [ex_id for ex_id in same_power_ids if ex_id in reports_by_ex_id and reports_by_ex_id[ex_id]['is_clan_battle']]
    need = target
    for ex_id in non_clan_ids + clan_ids:
        item = reports_by_ex_id[ex_id]
        available = item.get('available_r2', item.get('current_total', 0))
        take = min(available, need)
        alloc[item['equip_name']] += take
        need -= take
        if need <= 0:
            break
    return alloc


def allocate_reports_by_choice_clusters(reports: List[Dict], clusters: List[Dict], category_target: int) -> List[Dict]:
    by_ex_id = {item['ex_equipment_id']: item for item in reports}
    alloc = {item['equip_name']: 0 for item in reports}
    cluster_targets = allocate_cluster_shares(clusters, category_target)
    remaining = category_target

    for cluster in sorted(clusters, key=lambda c: len(c['unit_ids']), reverse=True):
        target = cluster_targets.get(tuple(cluster['best_ids']), 0)
        if target <= 0:
            continue
        remaining -= target
        cluster_alloc = allocate_cluster_internal(cluster, by_ex_id, target)
        for name, cnt in cluster_alloc.items():
            alloc[name] += cnt

    if remaining > 0:
        next_nonclan = sorted(
            [item for item in reports if not item['is_clan_battle'] and item.get('best_count', 0) == 0],
            key=lambda x: x.get('gap_avg', 999999)
        )
        for item in next_nonclan:
            available = max(0, item.get('available_r2', item['current_total']) - alloc[item['equip_name']])
            take = min(available, remaining)
            alloc[item['equip_name']] += take
            remaining -= take
            if remaining <= 0:
                break

    result = []
    for item in reports:
        allocated = alloc.get(item['equip_name'], 0)
        targets = finalize_targets({
            'ex_equipment_id': item['ex_equipment_id'],
            'is_clan_battle': item['is_clan_battle'],
            'allocated_mainline': allocated,
        })
        result.append({
            **item,
            'keep_target_min': targets['keep_target_min'],
            'keep_target_max': targets['keep_target_min'],
            'full_target': targets['full_target'],
            'r2_hold_target': targets['r2_hold_target'],
        })
    return result


def render_cleanup_markdown(report: "ExEquipCleanupReport") -> str:
    totals = summarize_retention_totals(report.slot_reports)
    full_totals = summarize_full_totals(report.slot_reports)
    total_full = sum(full_totals.values())
    lines: List[str] = [
        f"# EX 装清理结果 - {report.account_alias}",
        "",
        f"生成时间：{report.generated_at}",
        f"普通金装备种类数：{report.summary.get('normal_gold_total', 0)}",
        f"彩装：{report.non_target_summary.get('rainbow_total', 0)} / 粉装：{report.non_target_summary.get('pink_total', 0)} / 银装：{report.non_target_summary.get('silver_total', 0)}",
        "",
        "## 总保留数汇总",
        "",
        f"- 槽位1总保留数：{totals['slot_totals'].get(1, 0)} / 满强目标总数：{full_totals.get(1, 0)}",
        f"- 槽位2总保留数：{totals['slot_totals'].get(2, 0)} / 满强目标总数：{full_totals.get(2, 0)}",
        f"- 槽位3总保留数：{totals['slot_totals'].get(3, 0)} / 满强目标总数：{full_totals.get(3, 0)}",
        f"- 总计保留数：{totals['grand_total']} / 总计满强目标：{total_full}",
        "",
        "## 槽位汇总",
        "",
        "| 槽位 | 类别 | 当前角色数 | 数据库角色数 | 未来实装数 | 装备种类 | 保留总数 | 满强化目标 | r2未强化目标 | 可分解 | 普通金 | 会战金 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for slot_report in report.slot_reports:
        keep_min = sum(item.keep_target_min for item in slot_report.equip_reports)
        full_target = sum(item.full_target for item in slot_report.equip_reports)
        r2_hold_target = sum(max(0, item.keep_target_min - item.full_target) for item in slot_report.equip_reports)
        decompose = sum(item.decompose_candidate_count for item in slot_report.equip_reports)
        normal_count = sum(1 for item in slot_report.equip_reports if not item.is_clan_battle)
        clan_count = sum(1 for item in slot_report.equip_reports if item.is_clan_battle)
        lines.append(
            f"| {slot_report.slot_index} | {slot_report.category} | {slot_report.unit_count} | {slot_report.db_unit_count} | {slot_report.reserve_unit_count} | {len(slot_report.equip_reports)} | {keep_min} | {full_target} | {r2_hold_target} | {decompose} | {normal_count} | {clan_count} |"
        )

    lines.extend([
        "",
        "## 装备明细",
        "",
    ])

    for slot_report in report.slot_reports:
        lines.extend([
            f"### 槽位 {slot_report.slot_index} / 类别 {slot_report.category}",
            "",
            f"当前角色数：{slot_report.unit_count} / 数据库角色数：{slot_report.db_unit_count} / 未来实装数：{slot_report.reserve_unit_count}",
            "",
            "| 装备 | 类 | 现状(满/r2/r1/r0) | 目标(总/满/r2) | 差值 | 可分解 | 动作 | 说明 |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ])
        for item in slot_report.equip_reports:
            lines.append(
                f"| {item.equip_name} | {'会战' if item.is_clan_battle else '普通'} | {item.current_full_r2}/{item.current_r2_not_full}/{item.current_r1}/{item.current_r0} | {item.keep_target_min}/{item.full_target}/{max(0, item.keep_target_min - item.full_target)} | {item.evidence} | {item.decompose_candidate_count} | {item.action} | {', '.join(item.reason_tags)} |"
            )
        lines.append("")

    return "\n".join(lines)


@dataclass
class ExEquipCleanupEquipReport:
    slot_index: int
    category: int
    ex_equipment_id: int
    equip_name: str
    is_clan_battle: bool
    current_total: int
    current_full_r2: int
    current_r2_not_full: int
    current_r1: int
    current_r0: int
    equivalent_points: int
    equipped_normal_count: int
    equipped_clan_count: int
    locked_count: int
    restricted_count: int
    exclusive_best_count: int
    shared_best_count: int
    alt1_count: int
    alt2_count: int
    provisional_cb_tier: str
    reserve_count: int
    keep_target_min: int
    keep_target_max: int
    full_target: int
    rankup_target_count: int
    enhance_target_count: int
    decompose_candidate_count: int
    action: str
    evidence: str = ""
    gap_avg: int = 0
    best_count: int = 0
    exclusive_units: List[str] = field(default_factory=list)
    reason_tags: List[str] = field(default_factory=list)


@dataclass
class ExEquipCleanupSlotReport:
    slot_index: int
    category: int
    unit_count: int
    db_unit_count: int = 0
    reserve_unit_count: int = 0
    equip_reports: List[ExEquipCleanupEquipReport] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ExEquipCleanupReport:
    account_alias: str
    generated_at: str
    summary: Dict[str, int] = field(default_factory=dict)
    slot_reports: List[ExEquipCleanupSlotReport] = field(default_factory=list)
    non_target_summary: Dict[str, int] = field(default_factory=dict)
    action_summary: Dict[str, int] = field(default_factory=dict)


class _LazyExEquipRecommender:
    def __init__(self, analyzer: "ExEquipCleanupAnalyzer"):
        self.analyzer = analyzer

    def calculate_recommendations(self):
        return self.analyzer._get_recommender().calculate_recommendations()


class ExEquipCleanupAnalyzer:
    def __init__(self, client, account_alias: str, normal_floor_total: int = 5, clan_floor_total: int = 10, enhance_mode: str = '强化一半', clan_full_cap: int = 20):
        self.client = client
        self.account_alias = account_alias
        self.normal_floor_total = normal_floor_total
        self.clan_floor_total = clan_floor_total
        self.enhance_mode = enhance_mode
        self.clan_full_cap = clan_full_cap
        self.calculator = None
        self._recommender = None
        self.recommender = _LazyExEquipRecommender(self)

    def _get_recommender(self) -> ExEquipRecommender:
        if self._recommender is None:
            if self.calculator is None:
                self.calculator = ExEquipPowerCalculator(self.client)
            self._recommender = ExEquipRecommender(self.client, self.calculator)
        return self._recommender

    def analyze(self) -> ExEquipCleanupReport:
        recommendations = self.recommender.calculate_recommendations()
        covered_unit_slots = self._collect_pink_override_unit_slots(recommendations["unit_slot_recommendations"])
        active_unit_slot_recommendations = filter_unit_slot_recommendations(recommendations["unit_slot_recommendations"], covered_unit_slots)
        normal_demands = self._collect_normal_demands(active_unit_slot_recommendations)
        choice_clusters = self._collect_choice_clusters(active_unit_slot_recommendations)
        equip_state = self._collect_inventory_state()
        slot_reports = self._build_slot_reports(normal_demands, choice_clusters, equip_state)
        return ExEquipCleanupReport(
            account_alias=self.account_alias,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary={"normal_gold_total": sum(len(v) for v in normal_demands.values())},
            slot_reports=slot_reports,
            non_target_summary=self._build_non_target_summary(),
            action_summary=self._build_action_summary(slot_reports),
        )

    def _collect_normal_demands(self, unit_slot_recommendations: Dict) -> Dict:
        demand = defaultdict(lambda: defaultdict(lambda: {
            "exclusive_best": 0,
            "shared_best": 0,
            "alt1": 0,
            "alt2": 0,
            "units": set(),
            "exclusive_units": set(),
            "gap_values": [],
            "best_count": 0,
            "reference_unit_id": None,
        }))
        shared_groups = defaultdict(Counter)
        for (unit_id, slot), recommendation in unit_slot_recommendations.items():
            slot_data = db.unit_ex_equipment_slot.get(unit_id)
            if not slot_data:
                continue
            category = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot - 1]
            if category is None:
                continue
            key = (slot, category)
            best_equips = sorted(ex_id for ex_id, _ in recommendation.get("best_equips", []))
            if len(best_equips) == 1:
                demand[key][best_equips[0]]["exclusive_best"] += 1
                demand[key][best_equips[0]].setdefault("exclusive_units", set()).add(unit_id)
            elif best_equips:
                shared_groups[key][tuple(best_equips)] += 1
            for ex_id, _ in recommendation.get("alt_level1", []):
                demand[key][ex_id]["alt1"] += 1
            for ex_id, _ in recommendation.get("alt_level2", []):
                demand[key][ex_id]["alt2"] += 1
            for ex_id in set(best_equips + [ex_id for ex_id, _ in recommendation.get("alt_level1", [])] + [ex_id for ex_id, _ in recommendation.get("alt_level2", [])]):
                demand[key][ex_id]["units"].add(unit_id)
            best_power = recommendation.get("best_power_5rank2", 0)
            best_ids = {ex_id for ex_id, _ in recommendation.get('best_equips', [])}
            for ex_id, power in recommendation.get("all_equips", []):
                gap = max(0, best_power - power)
                demand[key][ex_id]["gap_values"].append(gap)
                if ex_id in best_ids:
                    demand[key][ex_id]["best_count"] += 1
                if demand[key][ex_id]["reference_unit_id"] is None:
                    demand[key][ex_id]["reference_unit_id"] = unit_id

        for key, groups in shared_groups.items():
            for ex_ids, total_count in groups.items():
                base, remainder = divmod(total_count, len(ex_ids))
                for index, ex_id in enumerate(ex_ids):
                    demand[key][ex_id]["shared_best"] += base + (1 if index < remainder else 0)
        return demand

    def _collect_pink_override_unit_slots(self, unit_slot_recommendations: Dict) -> set[tuple[int, int]]:
        covered: set[tuple[int, int]] = set()
        pink_items = [
            ex for ex in self.client.data.ex_equips.values()
            if db.get_ex_equip_rarity(ex.ex_equipment_id) == 4 and ex.rank >= 1
        ]
        pink_edges: Dict[int, List[tuple[int, int]]] = defaultdict(list)
        for serial_ex in pink_items:
            ex_data = db.ex_equipment_data[serial_ex.ex_equipment_id]
            slot_index = (serial_ex.ex_equipment_id // 100000) % 10
            current_power_cache: Dict[tuple[int, int], int] = {}
            for (unit_id, slot), recommendation in unit_slot_recommendations.items():
                if slot != slot_index:
                    continue
                slot_data = db.unit_ex_equipment_slot.get(unit_id)
                if not slot_data:
                    continue
                category = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot - 1]
                if category != ex_data.category:
                    continue
                current_power = self.calculator.calculate_power_increase_at_current_state(
                    unit_id,
                    slot,
                    serial_ex.ex_equipment_id,
                    serial_ex.enhancement_pt,
                    serial_ex.rank,
                )
                if current_power >= recommendation.get('best_power_5rank2', 0):
                    pink_edges[serial_ex.serial_id].append((unit_id, slot))
        matched: Dict[tuple[int, int], int] = {}

        def dfs(item_id: int, seen: set[tuple[int, int]]) -> bool:
            for unit_slot in pink_edges.get(item_id, []):
                if unit_slot in seen:
                    continue
                seen.add(unit_slot)
                if unit_slot not in matched or dfs(matched[unit_slot], seen):
                    matched[unit_slot] = item_id
                    return True
            return False

        for item_id in pink_edges:
            dfs(item_id, set())

        return set(matched.keys())

    def _collect_choice_clusters(self, unit_slot_recommendations: Dict) -> Dict[tuple[int, int], List[Dict]]:
        cluster_map = defaultdict(lambda: defaultdict(lambda: {'best_ids': (), 'unit_ids': []}))
        for (unit_id, slot), recommendation in unit_slot_recommendations.items():
            slot_data = db.unit_ex_equipment_slot.get(unit_id)
            if not slot_data:
                continue
            category = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot - 1]
            if category is None:
                continue
            best_ids = tuple(sorted(ex_id for ex_id, _ in recommendation.get('best_equips', [])))
            if not best_ids:
                continue
            bucket = cluster_map[(slot, category)][best_ids]
            bucket['best_ids'] = best_ids
            bucket['unit_ids'].append(unit_id)
        return {key: list(group.values()) for key, group in cluster_map.items()}

    def _collect_inventory_state(self) -> Dict[int, Dict[str, int]]:
        state = defaultdict(lambda: {
            "equipped_normal_count": 0,
            "equipped_clan_count": 0,
            "locked_count": 0,
            "restricted_count": 0,
            "protected_serial_ids": set(),
        })

        for ex in self.client.data.ex_equips.values():
            if ex.protection_flag == 2:
                state[ex.ex_equipment_id]["locked_count"] += 1
                state[ex.ex_equipment_id]["protected_serial_ids"].add(ex.serial_id)

        for unit in self.client.data.unit.values():
            for ex_slot in unit.ex_equip_slot or []:
                ex = self.client.data.ex_equips.get(ex_slot.serial_id)
                if ex:
                    state[ex.ex_equipment_id]["equipped_normal_count"] += 1
                    state[ex.ex_equipment_id]["protected_serial_ids"].add(ex.serial_id)
            for ex_slot in unit.cb_ex_equip_slot or []:
                ex = self.client.data.ex_equips.get(ex_slot.serial_id)
                if ex:
                    state[ex.ex_equipment_id]["equipped_clan_count"] += 1
                    state[ex.ex_equipment_id]["protected_serial_ids"].add(ex.serial_id)

        for serial_id in self.client.data.user_clan_battle_ex_equip_restriction.keys():
            ex = self.client.data.ex_equips.get(serial_id)
            if ex:
                state[ex.ex_equipment_id]["restricted_count"] += 1
                state[ex.ex_equipment_id]["protected_serial_ids"].add(ex.serial_id)

        return state

    def _plan_targets(self, stats: Dict[str, int], state: Dict[str, int], keep_min: int, keep_max: int, full_target: int) -> Tuple[int, int, int]:
        retained_count = min(stats["current_total"], keep_max)
        keep_ready_count = min(retained_count, stats["current_full_r2"] + stats["current_r2_not_full"] + stats["current_r1"])
        full_ready_count = min(retained_count, stats["current_full_r2"])
        rankup_target = max(0, keep_min - keep_ready_count)
        enhance_target = max(0, full_target - full_ready_count)
        protected_count = len(state.get("protected_serial_ids", set()))
        available_count = max(0, stats["current_total"] - protected_count)
        decompose = max(0, available_count - keep_max)
        return rankup_target, enhance_target, decompose

    def _build_slot_reports(self, normal_demands: Dict, choice_clusters: Dict[tuple[int, int], List[Dict]], equip_state: Dict[int, Dict[str, int]]) -> List[ExEquipCleanupSlotReport]:
        reports = []
        inventory_by_ex_id = defaultdict(list)
        for ex in self.client.data.ex_equips.values():
            inventory_by_ex_id[ex.ex_equipment_id].append(ex)
        inventory_ids_by_slot_category = build_inventory_slot_category_index(
            self.client.data.ex_equips.values(),
            db.ex_equipment_data,
            db.get_ex_equip_rarity,
        )

        slot_keys = sorted(set(normal_demands.keys()) | set(inventory_ids_by_slot_category.keys()))
        for (slot, category) in slot_keys:
            equips = normal_demands.get((slot, category), {})
            recommended_ids = set(equips.keys())
            inventory_ids = inventory_ids_by_slot_category.get((slot, category), set())
            report_ids = collect_report_equipment_ids(recommended_ids, inventory_ids)
            equip_reports = []
            unit_count, db_unit_count, reserve_unit_count = compute_slot_unit_counts(self.client.data.unit, db.unit_ex_equipment_slot, slot, category)
            for ex_id in report_ids:
                raw_counts = equips.get(ex_id, {
                    'exclusive_best': 0,
                    'shared_best': 0,
                    'alt1': 0,
                    'alt2': 0,
                    'units': set(),
                    'exclusive_units': set(),
                    'gap_values': [],
                    'best_count': 0,
                    'reference_unit_id': None,
                })
                counts = raw_counts
                stats = merge_inventory_stats(inventory_by_ex_id.get(ex_id, []))
                state = equip_state[ex_id]
                keep_value = counts['exclusive_best'] + counts['shared_best'] + counts['alt1'] + min(counts['alt2'], 1)
                full_target = min(keep_value, max(counts['exclusive_best'] + counts['shared_best'], 1)) if keep_value else 0
                is_clan = bool(db.ex_equipment_data[ex_id].clan_battle_equip_flag)
                tier = self._clan_battle_tier(counts)
                keep_cap = compute_keep_cap(unit_count, reserve_unit_count)
                keep_min, keep_max = clan_battle_keep_range(tier) if is_clan else (min(keep_value, keep_cap), min(keep_value, keep_cap))
                full_target = min(full_target, keep_max)
                rankup_target, enhance_target, decompose = self._plan_targets(stats, state, keep_min, keep_max, full_target)
                action = self._pick_action(is_clan, rankup_target, enhance_target, decompose)
                ref_unit_id = raw_counts.get('reference_unit_id')
                evidence_stats = summarize_gap_statistics(raw_counts.get('gap_values', []), raw_counts.get('best_count', 0))
                evidence = build_power_evidence_text(raw_counts, evidence_stats)
                exclusive_units = [db.unit_data[uid].unit_name for uid in sorted(raw_counts.get('exclusive_units', set())) if uid in db.unit_data]
                equip_reports.append(ExEquipCleanupEquipReport(
                    slot_index=slot,
                    category=category,
                    ex_equipment_id=ex_id,
                    equip_name=db.ex_equipment_data[ex_id].name,
                    is_clan_battle=is_clan,
                    current_total=stats['current_total'],
                    current_full_r2=stats['current_full_r2'],
                    current_r2_not_full=stats['current_r2_not_full'],
                    current_r1=stats['current_r1'],
                    current_r0=stats['current_r0'],
                    equivalent_points=stats['equivalent_points'],
                    equipped_normal_count=state['equipped_normal_count'],
                    equipped_clan_count=state['equipped_clan_count'],
                    locked_count=state['locked_count'],
                    restricted_count=state['restricted_count'],
                    exclusive_best_count=counts['exclusive_best'],
                    shared_best_count=counts['shared_best'],
                    alt1_count=counts['alt1'],
                    alt2_count=counts['alt2'],
                    provisional_cb_tier=tier,
                    reserve_count=1 if keep_value else 0,
                    keep_target_min=keep_min,
                    keep_target_max=keep_max,
                    full_target=full_target,
                    rankup_target_count=rankup_target,
                    enhance_target_count=enhance_target,
                    decompose_candidate_count=decompose,
                    action='keep',
                    evidence=evidence,
                    gap_avg=evidence_stats['avg_gap'],
                    best_count=evidence_stats['best_count'],
                    exclusive_units=[db.unit_data[uid].unit_name for uid in sorted(raw_counts.get('exclusive_units', set())) if uid in db.unit_data],
                    reason_tags=[],
                ))
            category_target = unit_count + reserve_unit_count
            adjusted = allocate_reports_by_choice_clusters([
                {
                    'equip_name': item.equip_name,
                    'ex_equipment_id': item.ex_equipment_id,
                    'is_clan_battle': item.is_clan_battle,
                    'evidence': item.evidence,
                    'gap_avg': item.gap_avg,
                    'best_count': item.best_count,
                    'current_total': item.current_total,
                    'available_r2': compute_max_possible_r2({
                        'current_full_r2': item.current_full_r2,
                        'current_r2_not_full': item.current_r2_not_full,
                        'current_r1': item.current_r1,
                        'current_r0': item.current_r0,
                    }),
                    'normal_floor_total': self.normal_floor_total,
                    'clan_floor_total': self.clan_floor_total,
                    'enhance_mode': self.enhance_mode,
                    'clan_full_cap': self.clan_full_cap,
                    'surplus': 0,
                }
                for item in equip_reports
            ], choice_clusters.get((slot, category), []), category_target)
            adjusted_by_name = {item['equip_name']: item for item in adjusted}
            final_reports = []
            for item in equip_reports:
                update = adjusted_by_name[item.equip_name]
                stats_for_limit = {
                    'current_full_r2': item.current_full_r2,
                    'current_r2_not_full': item.current_r2_not_full,
                    'current_r1': item.current_r1,
                    'current_r0': item.current_r0,
                }
                max_possible_r2 = compute_max_possible_r2(stats_for_limit)
                item.keep_target_min = min(update['keep_target_min'], max_possible_r2)
                item.keep_target_max = item.keep_target_min
                item.full_target = min(update['full_target'], item.keep_target_min)
                r2_hold_target = max(0, item.keep_target_min - item.full_target)
                tiers = compute_decompose_tiers(
                    stats_for_limit,
                    reserve_r2_unenhanced=r2_hold_target,
                    keep_target_total=item.keep_target_min,
                )
                rankup_target = max(0, item.keep_target_min - (tiers['full_r2_keep'] + tiers['r2_not_full_keep'] + tiers['r1_keep']))
                enhance_target = max(0, item.full_target - tiers['full_r2_keep'])
                decompose = tiers['decompose']
                item.rankup_target_count = rankup_target
                item.enhance_target_count = enhance_target
                item.decompose_candidate_count = decompose
                item.action = self._pick_action(item.is_clan_battle, rankup_target, enhance_target, decompose)
                item.reason_tags = self._reason_tags(
                    {
                        'exclusive_best': item.exclusive_best_count,
                        'shared_best': item.shared_best_count,
                        'alt1': item.alt1_count,
                        'alt2': item.alt2_count,
                    },
                    item.is_clan_battle,
                    rankup_target,
                    enhance_target,
                    decompose,
                    equip_state[item.ex_equipment_id],
                )
                final_reports.append(item)
            equip_reports = final_reports

            reports.append(ExEquipCleanupSlotReport(
                slot_index=slot,
                category=category,
                unit_count=unit_count,
                db_unit_count=db_unit_count,
                reserve_unit_count=reserve_unit_count,
                equip_reports=equip_reports,
                notes=[],
            ))
        return reports

    def _build_non_target_summary(self) -> Dict[str, int]:
        rainbow_total = sum(1 for ex in self.client.data.ex_equips.values() if db.get_ex_equip_rarity(ex.ex_equipment_id) == 5)
        pink_total = sum(1 for ex in self.client.data.ex_equips.values() if db.get_ex_equip_rarity(ex.ex_equipment_id) == 4)
        silver_total = sum(1 for ex in self.client.data.ex_equips.values() if db.get_ex_equip_rarity(ex.ex_equipment_id) == 2)
        return {"rainbow_total": rainbow_total, "pink_total": pink_total, "silver_total": silver_total}

    def _build_action_summary(self, slot_reports: List[ExEquipCleanupSlotReport]) -> Dict[str, int]:
        counter = Counter()
        for slot_report in slot_reports:
            for equip_report in slot_report.equip_reports:
                counter[equip_report.action] += 1
        return dict(counter)

    def _clan_battle_tier(self, counts: Dict[str, int]) -> str:
        score = counts["exclusive_best"] * 3 + counts["shared_best"] * 2 + counts["alt1"] * 2 + counts["alt2"]
        if score >= 15:
            return "high"
        if score >= 8:
            return "medium"
        if score >= 3:
            return "low"
        return "minimal"

    def _pick_action(self, is_clan: bool, rankup_target: int, enhance_target: int, decompose: int) -> str:
        if is_clan:
            return "hold_review"
        if rankup_target > 0:
            return "rankup"
        if enhance_target > 0:
            return "enhance"
        if decompose > 0:
            return "decompose"
        return "keep"

    def _reason_tags(self, counts: Dict[str, int], is_clan: bool, rankup_target: int, enhance_target: int, decompose: int, state: Dict[str, int]) -> List[str]:
        tags = []
        if counts["exclusive_best"]:
            tags.append("exclusive_best")
        if counts["shared_best"]:
            tags.append("shared_best")
        if counts["alt1"]:
            tags.append("alt1_cover")
        if counts["alt2"]:
            tags.append("alt2_cover")
        if rankup_target:
            tags.append("needs_rankup")
        if enhance_target:
            tags.append("needs_enhance")
        if decompose:
            tags.append("over_target")
        if is_clan:
            tags.append("clan_review")
        if state["equipped_normal_count"]:
            tags.append("equipped_normal")
        if state["equipped_clan_count"]:
            tags.append("equipped_clan")
        if state["locked_count"]:
            tags.append("locked")
        if state["restricted_count"]:
            tags.append("restricted")
        return tags
