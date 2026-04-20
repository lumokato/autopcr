from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

from ...core.pcrclient import pcrclient
from ...db.database import db
from ...model.enums import eInventoryType
from ...model.error import AbortError
from ..config import *
from ..modulebase import *
from .unit import UnitController

SYNC_STAGE_ORDER = [6, 4, 2, 5, 3, 1]
SYNC_SLOT_INDEX = [slot_num - 1 for slot_num in SYNC_STAGE_ORDER]
SYNC_ANCHOR_COUNT = 20


@dataclass(frozen=True)
class RankStage:
    rank: int
    stage: int


@dataclass
class SlotSnapshot:
    id: int
    is_slot: bool
    enhancement_level: int
    enhancement_pt: int


@dataclass
class UnitSnapshot:
    rank: int
    slots: List[SlotSnapshot]


@dataclass
class StepCost:
    unit_id: int
    target: RankStage
    latest_demand: Counter
    latest_shortage: Counter
    mana: int
    stone_pt: int

    @property
    def latest_demand_total(self) -> int:
        return sum(self.latest_demand.values())

    @property
    def latest_shortage_total(self) -> int:
        return sum(self.latest_shortage.values())

    @property
    def latest_type_count(self) -> int:
        return len(self.latest_demand)


@dataclass
class StagePlan:
    target: RankStage
    selected: List[StepCost]
    latest_consumption: Counter
    latest_shortage: Counter
    mana: int
    stone_pt: int
    feasible_count: int
    mana_blocked: int


class SyncGrowthController(UnitController):

    def _promotion_value(self, promotion_level) -> int:
        return int(promotion_level.value if hasattr(promotion_level, "value") else promotion_level)

    def _state_key(self, state: RankStage) -> Tuple[int, int]:
        return state.rank, state.stage

    def _state_ge(self, left: RankStage, right: RankStage) -> bool:
        return self._state_key(left) >= self._state_key(right)

    def _format_state(self, state: RankStage) -> str:
        return f"{state.rank}-{state.stage}"

    def _parse_state(self, text: str) -> RankStage:
        text = text.strip()
        if not text:
            return self._current_sync_state()
        rank, stage = text.split("-", 1)
        state = RankStage(int(rank), int(stage))
        if state.stage < 0 or state.stage > 6:
            raise AbortError(f"非法状态 {text}，stage 必须在 0 到 6 之间")
        return state

    def _current_sync_state(self) -> RankStage:
        limit = self.client.data.get_synchro_parameter()
        rank = int(limit.promotion_level or 0)
        stage = 0
        for slot_num in SYNC_STAGE_ORDER:
            up = getattr(limit, f"equipment_{slot_num}", None)
            if up is None or up < 0:
                break
            stage += 1
        return RankStage(rank, stage)

    def _probe_unit_id(self) -> int:
        candidates = self._candidate_unit_ids()
        if not candidates:
            raise AbortError("没有可用角色，无法探测同步器终点")
        return candidates[0]

    def _global_stage_cap(self, rank: int) -> int:
        unit_id = self._probe_unit_id()
        if unit_id not in db.unit_promotion or rank not in db.unit_promotion[unit_id]:
            return 0
        stage = 0
        for slot_num in SYNC_STAGE_ORDER:
            equip_id = int(getattr(db.unit_promotion[unit_id][rank], f"equip_slot_{slot_num}"))
            if equip_id == 999999:
                break
            stage += 1
        return stage

    def _database_target_state(self) -> RankStage:
        final_rank = max(int(rank) for ranks in db.unit_promotion.values() for rank in ranks.keys())
        return RankStage(final_rank, 6)

    def _auto_target_state(self) -> RankStage:
        current = self._current_sync_state()
        clipped = current
        for step_target in self._state_path(current, self._database_target_state()):
            if step_target.stage > self._global_stage_cap(step_target.rank):
                break
            clipped = step_target
        return clipped

    def _state_path(self, start: RankStage, target: RankStage) -> List[RankStage]:
        if self._state_key(target) < self._state_key(start):
            raise AbortError(f"目标状态 {self._format_state(target)} 低于起始状态 {self._format_state(start)}")

        path: List[RankStage] = []
        current = start
        while current != target:
            if current.rank < target.rank:
                current = RankStage(current.rank, current.stage + 1) if current.stage < 6 else RankStage(current.rank + 1, 0)
            else:
                current = RankStage(current.rank, current.stage + 1)
            path.append(current)
        return path

    def _snapshot_stage(self, unit_id: int, snapshot: UnitSnapshot) -> int:
        stage = 0
        for slot_idx in SYNC_SLOT_INDEX:
            equip_id = int(getattr(db.unit_promotion[unit_id][snapshot.rank], f"equip_slot_{slot_idx + 1}"))
            if equip_id == 999999:
                break
            slot = snapshot.slots[slot_idx]
            if not slot.is_slot or slot.id != equip_id:
                break
            if slot.enhancement_level < db.get_equip_max_star(equip_id):
                break
            stage += 1
        return stage

    def _snapshot_state(self, unit_id: int, snapshot: UnitSnapshot) -> RankStage:
        return RankStage(snapshot.rank, self._snapshot_stage(unit_id, snapshot))

    def _build_target_snapshot(self, unit_id: int, state: RankStage) -> UnitSnapshot:
        slots: List[SlotSnapshot] = []
        filled = set(SYNC_SLOT_INDEX[:state.stage])
        for slot_idx in range(6):
            equip_id = int(getattr(db.unit_promotion[unit_id][state.rank], f"equip_slot_{slot_idx + 1}"))
            if slot_idx in filled and equip_id != 999999:
                max_star = db.get_equip_max_star(equip_id)
                enhancement_pt = db.get_equip_star_pt(equip_id, max_star)
                slots.append(SlotSnapshot(id=equip_id, is_slot=True, enhancement_level=max_star, enhancement_pt=enhancement_pt))
            else:
                slots.append(SlotSnapshot(id=equip_id, is_slot=False, enhancement_level=0, enhancement_pt=0))
        return UnitSnapshot(rank=state.rank, slots=slots)

    def _build_actual_snapshot(self, unit_id: int) -> UnitSnapshot:
        unit = self.client.data.unit[unit_id]
        return UnitSnapshot(
            rank=self._promotion_value(unit.promotion_level),
            slots=[
                SlotSnapshot(
                    id=slot.id,
                    is_slot=bool(slot.is_slot),
                    enhancement_level=slot.enhancement_level,
                    enhancement_pt=slot.enhancement_pt,
                )
                for slot in unit.equip_slot
            ],
        )

    def _effective_snapshot(
        self,
        unit_id: int,
        baseline: RankStage,
        actual: UnitSnapshot | None = None,
        actual_state: RankStage | None = None,
    ) -> UnitSnapshot:
        actual = actual if actual is not None else self._build_actual_snapshot(unit_id)
        actual_state = actual_state if actual_state is not None else self._snapshot_state(unit_id, actual)
        if self._state_ge(actual_state, baseline):
            return actual
        return self._build_target_snapshot(unit_id, baseline)

    def _latest_demand(self, target_equips: Counter) -> Counter:
        ret: Counter = Counter()
        for equip_item, equip_cnt in target_equips.items():
            materials = [
                (item, cnt)
                for item, cnt in db.equip_craft.get(equip_item, [])
                if cnt > 0 and item[0] == eInventoryType.Equip
            ]
            if not materials:
                ret[equip_item] += equip_cnt
                continue

            top_promotion = max(db.get_equip_promotion(item[1]) for item, _ in materials)
            for item, cnt in materials:
                if db.get_equip_promotion(item[1]) == top_promotion:
                    ret[item] += cnt * equip_cnt
        return ret

    def _latest_shortage(self, demand: Counter, remaining: Counter) -> Counter:
        return Counter({
            token: demand[token] - remaining[token]
            for token in demand
            if demand[token] > remaining[token]
        })

    def _calc_step_cost(
        self,
        unit_id: int,
        current: RankStage,
        target: RankStage,
        remaining: Counter,
        actual: UnitSnapshot | None = None,
        actual_state: RankStage | None = None,
    ) -> StepCost:
        snapshot = self._effective_snapshot(unit_id, current, actual, actual_state)
        state = actual_state if snapshot is actual and actual_state is not None else self._snapshot_state(unit_id, snapshot)
        if self._state_ge(state, target):
            return StepCost(unit_id, target, Counter(), Counter(), 0, 0)

        target_equips: Counter = Counter()
        mana = 0
        stone_pt = 0

        while not self._state_ge(state, target):
            now = state
            if now.rank < target.rank and now.stage >= self._global_stage_cap(now.rank):
                snapshot = self._build_target_snapshot(unit_id, RankStage(now.rank + 1, 0))
                state = RankStage(now.rank + 1, 0)
                continue

            if now.rank != target.rank or now.stage >= target.stage:
                raise AbortError(f"{db.get_unit_name(unit_id)} 无法从 {self._format_state(now)} 推进到 {self._format_state(target)}")

            slot_idx = SYNC_SLOT_INDEX[now.stage]
            slot_num = slot_idx + 1
            equip_id = int(getattr(db.unit_promotion[unit_id][now.rank], f"equip_slot_{slot_num}"))
            if equip_id == 999999:
                raise AbortError(f"{db.get_unit_name(unit_id)} 在 {self._format_state(now)} 的 {slot_num} 号位没有装备数据")

            slot = snapshot.slots[slot_idx]
            if not slot.is_slot or slot.id != equip_id:
                target_equips[(eInventoryType.Equip, equip_id)] += 1
                _, craft_mana = db.craft_equip(Counter({(eInventoryType.Equip, equip_id): 1}))
                mana += craft_mana
                slot.id = equip_id
                slot.is_slot = True
                slot.enhancement_level = 0
                slot.enhancement_pt = 0

            max_star = db.get_equip_max_star(equip_id)
            if slot.enhancement_level < max_star:
                target_pt = db.get_equip_star_pt(equip_id, max_star)
                delta_pt = target_pt - slot.enhancement_pt
                stone_pt += delta_pt
                mana += delta_pt * 200
                slot.enhancement_pt = target_pt
                slot.enhancement_level = max_star

            state = RankStage(now.rank, now.stage + 1)

        latest_demand = self._latest_demand(target_equips)
        latest_shortage = self._latest_shortage(latest_demand, remaining)
        return StepCost(unit_id, target, latest_demand, latest_shortage, mana, stone_pt)

    def _candidate_unit_ids(self) -> List[int]:
        return sorted(self.client.data.unit.keys())

    def _reprice_cost(self, cost: StepCost, remaining: Counter) -> StepCost:
        return StepCost(
            cost.unit_id,
            cost.target,
            cost.latest_demand,
            self._latest_shortage(cost.latest_demand, remaining),
            cost.mana,
            cost.stone_pt,
        )

    def _plan_steps(self, start: RankStage, target: RankStage) -> List[StagePlan]:
        remaining = Counter(self.client.data.inventory)
        remaining_mana = self.client.data.get_mana()
        steps: List[StagePlan] = []
        current = start
        candidate_ids = self._candidate_unit_ids()
        actual_snapshots = {unit_id: self._build_actual_snapshot(unit_id) for unit_id in candidate_ids}
        actual_states = {
            unit_id: self._snapshot_state(unit_id, snapshot)
            for unit_id, snapshot in actual_snapshots.items()
        }

        for step_target in self._state_path(start, target):
            costs = [
                self._calc_step_cost(
                    unit_id,
                    current,
                    step_target,
                    remaining,
                    actual_snapshots[unit_id],
                    actual_states[unit_id],
                )
                for unit_id in candidate_ids
            ]
            mana_blocked = sum(1 for cost in costs if cost.mana > remaining_mana)
            feasible = [cost for cost in costs if cost.mana <= remaining_mana]
            feasible.sort(
                key=lambda cost: (
                    cost.latest_shortage_total,
                    cost.latest_demand_total,
                    cost.latest_type_count,
                    cost.mana,
                    cost.unit_id,
                )
            )

            selected: List[StepCost] = []
            stage_remaining = remaining.copy()
            stage_remaining_mana = remaining_mana

            for cost in feasible:
                if len(selected) >= SYNC_ANCHOR_COUNT:
                    break
                priced = self._reprice_cost(cost, stage_remaining)
                if priced.mana > stage_remaining_mana:
                    continue
                selected.append(priced)
                stage_remaining -= priced.latest_demand
                stage_remaining_mana -= priced.mana

            stage_consumption = Counter()
            stage_shortage = Counter()
            stage_mana = 0
            stage_stone_pt = 0
            for cost in selected:
                stage_consumption += cost.latest_demand
                stage_shortage += cost.latest_shortage
                stage_mana += cost.mana
                stage_stone_pt += cost.stone_pt

            steps.append(StagePlan(
                target=step_target,
                selected=selected,
                latest_consumption=stage_consumption,
                latest_shortage=stage_shortage,
                mana=stage_mana,
                stone_pt=stage_stone_pt,
                feasible_count=len(feasible),
                mana_blocked=mana_blocked,
            ))

            remaining = stage_remaining
            remaining_mana = stage_remaining_mana
            current = step_target

        return steps

    async def _reach_state(self, target: RankStage, free_only: bool = False):
        while True:
            current = self._snapshot_state(self.unit_id, self._build_actual_snapshot(self.unit_id))
            if self._state_ge(current, target):
                return

            if current.rank < target.rank and current.stage >= self._global_stage_cap(current.rank):
                if free_only:
                    await self.client.unit_free_promotion(self.unit.id, current.rank + 1)
                else:
                    await self.unit_promotion_up_aware(current.rank + 1)
                continue

            slot_idx = SYNC_SLOT_INDEX[current.stage]
            slot_num = slot_idx + 1
            equip_id = int(getattr(db.unit_promotion[self.unit_id][current.rank], f"equip_slot_{slot_num}"))
            max_star = db.get_equip_max_star(equip_id)
            equip = self.unit.equip_slot[slot_idx]

            if not equip.is_slot:
                if free_only:
                    await self.client.unit_free_equip(self.unit_id, [slot_num])
                else:
                    await self.unit_equip_slot_aware(equip_id, slot_num)
                continue

            if equip.enhancement_level < max_star:
                if free_only:
                    await self.client.equipment_free_enhance(self.unit.id, slot_num, max_star)
                else:
                    await self.unit_equip_enhance_aware(slot_num, max_star)
                continue

    def _log_plan(self, start: RankStage, target: RankStage, steps: List[StagePlan]):
        self._log(f"练度规划: {self._format_state(start)} -> {self._format_state(target)}")
        for step in steps:
            self._log("")
            self._log(f"阶段 {self._format_state(step.target)}")

            keys = []
            seen = set()
            for item in step.latest_consumption:
                if item not in seen:
                    seen.add(item)
                    keys.append(item)
            for item in step.latest_shortage:
                if item not in seen:
                    seen.add(item)
                    keys.append(item)

            if keys:
                detail = "，".join(
                    f"{db.get_inventory_name_san(item)}{step.latest_consumption.get(item, 0)}（{step.latest_shortage.get(item, 0)}）"
                    for item in keys
                )
                self._log(f"碎片总消耗（缺口）：{detail}")

            if step.selected:
                names = "，".join(db.get_unit_name(cost.unit_id) for cost in step.selected)
                self._log(f"锚点角色：{names}")

            if step.feasible_count < 20:
                self._warn(f"本阶段无法选满20人，受玛娜限制：{step.mana_blocked}人")



@description("""自动读取当前同步器状态，自动推导当前可达终点，并按阶段规划20个锚点角色。
默认只输出规划结果；开启执行后，会先把每阶段锚点角色免费同步到当前同步器状态，再正式拉到下一阶段，全部阶段完成后再免费同步其余角色。""")
@name("同步器练度")
@booltype("sync_growth_do_execute", "执行", False)
@default(False)
class sync_growth(SyncGrowthController):

    def _pending_free_sync_unit_ids(self, target: RankStage) -> List[int]:
        return [
            unit_id
            for unit_id in self._candidate_unit_ids()
            if not self._state_ge(self._snapshot_state(unit_id, self._build_actual_snapshot(unit_id)), target)
        ]

    async def _free_sync_remaining(self, target: RankStage, pending: List[int] | None = None):
        pending = pending if pending is not None else self._pending_free_sync_unit_ids(target)

        if not pending:
            self._log(f"无需免费同步，所有角色已达到 {self._format_state(target)}")
            return

        self._log(f"开始免费同步到 {self._format_state(target)}，共 {len(pending)} 人")
        for unit_id in pending:
            self.unit_id = unit_id
            try:
                await self._reach_state(target, free_only=True)
            except Exception as e:
                self._warn(f"{self.unit_name} 免费同步到 {self._format_state(target)} 失败: {e}")

    async def _execute_steps(self, steps: List[StagePlan]):
        final_target = steps[-1].target if steps else self._current_sync_state()

        for step in steps:
            self._log(f"开始执行阶段 {self._format_state(step.target)}")
            sync_state = self._current_sync_state()

            for cost in step.selected:
                self.unit_id = cost.unit_id
                try:
                    if not self._state_ge(self._snapshot_state(self.unit_id, self._build_actual_snapshot(self.unit_id)), sync_state):
                        await self._reach_state(sync_state, free_only=True)
                    await self._reach_state(step.target, free_only=False)
                    self._log(f"{self.unit_name} 提升到 {self._format_state(step.target)}")
                except Exception as e:
                    self._warn(f"{self.unit_name} 提升到 {self._format_state(step.target)} 失败: {e}")

            sync_state = self._current_sync_state()
            if not self._state_ge(sync_state, step.target):
                self._warn(f"阶段 {self._format_state(step.target)} 完成后，同步器状态仅到 {self._format_state(sync_state)}")

        await self._free_sync_remaining(final_target)

    async def do_task(self, client: pcrclient):
        self.client = client
        start = self._current_sync_state()
        target = self._auto_target_state()
        steps = self._plan_steps(start, target)
        self._log_plan(start, target, steps)

        if not steps:
            pending = self._pending_free_sync_unit_ids(target)
            self._log(f"需免费同步角色：{len(pending)} 人")

        if self.get_config("sync_growth_do_execute"):
            if not steps:
                await self._free_sync_remaining(target, pending)
                return
            await self._execute_steps(steps)


class SyncGrowthPlanner(SyncGrowthController):

    def __init__(self, client: pcrclient):
        self.client = client


def build_sync_growth_plan(client: pcrclient) -> Tuple[RankStage, RankStage, List[StagePlan]]:
    planner = SyncGrowthPlanner(client)
    start = planner._current_sync_state()
    target = planner._auto_target_state()
    steps = planner._plan_steps(start, target)
    return start, target, steps
