from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ...core.pcrclient import pcrclient
from ...db.database import db
from ...db.models import GrowthParameterUnique
from ...model.common import InventoryInfoPost, ShopItem
from ...model.enums import eInventoryType, eSystemId
from ...model.error import AbortError
from ..config import booltype, inttype
from ..modulebase import default, description, name
from .unit import UnitController


EXCEED_ALREADY = "already"
EXCEED_SPECIFIC_RING = "specific_ring"
EXCEED_GENERAL_RING = "general_ring"
EXCEED_MEMORY = "memory"
EXCEED_WAIT = "wait"


@dataclass(frozen=True)
class StarExceedUnitInput:
    unit_id: int
    current_rarity: int
    memory_inventory: int
    star_memory_cost: int
    farmable: bool
    can_purchase_memory: bool
    star_supported: bool = True
    exceeded: bool = False
    exceed_available: bool = True
    exceed_memory_cost: int = 0
    specific_ring_id: int = 0


@dataclass
class StarExceedPlan:
    source: StarExceedUnitInput
    target_rarity: int
    star_memory_use: int = 0
    star_memory_buy: int = 0
    exceed_method: str = EXCEED_WAIT
    exceed_ring_id: int = 0
    exceed_memory_use: int = 0
    exceed_memory_buy: int = 0
    unique1_memory_buy: int = 0
    memory_after: int = 0
    will_exceed: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def unit_id(self) -> int:
        return self.source.unit_id

    @property
    def memory_purchase(self) -> int:
        return (
            self.star_memory_buy
            + self.exceed_memory_buy
            + self.unique1_memory_buy
        )


@dataclass
class StarExceedPlanningResult:
    units: List[StarExceedPlan]
    ring_use: Counter
    general_ring_id: int
    general_ring_inventory: int
    general_ring_keep: int

    @property
    def general_ring_use(self) -> int:
        return self.ring_use.get(self.general_ring_id, 0)

    @property
    def specific_ring_use(self) -> int:
        return sum(
            count for ring_id, count in self.ring_use.items()
            if ring_id != self.general_ring_id
        )


def plan_star_and_exceed(
    units: Sequence[StarExceedUnitInput],
    ring_inventory: Mapping[int, int],
    general_ring_id: int,
    general_ring_keep: int,
) -> StarExceedPlanningResult:
    """Plan star and exceed upgrades without mutating account state."""
    plans: List[StarExceedPlan] = []

    for source in sorted(units, key=lambda item: item.unit_id):
        memory = max(0, source.memory_inventory)
        plan = StarExceedPlan(
            source=source,
            target_rarity=source.current_rarity,
            memory_after=memory,
        )

        if source.current_rarity < 5:
            if not source.star_supported:
                plan.notes.append("缺少升五星主数据")
            elif memory >= source.star_memory_cost:
                plan.star_memory_use = source.star_memory_cost
                memory -= source.star_memory_cost
                plan.target_rarity = 5
            elif not source.farmable and source.can_purchase_memory:
                plan.star_memory_use = memory
                plan.star_memory_buy = source.star_memory_cost - memory
                memory = 0
                plan.target_rarity = 5
            elif source.farmable:
                plan.notes.append(
                    f"升至5星还缺{source.star_memory_cost - memory}片，等待地图刷取"
                )
            else:
                plan.notes.append(
                    f"升至5星还缺{source.star_memory_cost - memory}片，女神商店不可购买"
                )

        plan.memory_after = memory
        if source.exceeded:
            plan.exceed_method = EXCEED_ALREADY
            plan.will_exceed = True
        elif plan.target_rarity < 5:
            plan.notes.append("未到5星，暂不突破")
        elif not source.exceed_available or source.exceed_memory_cost <= 0:
            plan.notes.append("角色突破尚未开放")

        plans.append(plan)

    rings_left = Counter({ring_id: max(0, count) for ring_id, count in ring_inventory.items()})
    ring_use: Counter = Counter()
    candidates: List[StarExceedPlan] = []

    for plan in plans:
        source = plan.source
        if (
            plan.will_exceed
            or plan.target_rarity < 5
            or not source.exceed_available
            or source.exceed_memory_cost <= 0
        ):
            continue

        ring_id = source.specific_ring_id
        if ring_id and ring_id != general_ring_id and rings_left[ring_id] > 0:
            rings_left[ring_id] -= 1
            ring_use[ring_id] += 1
            plan.exceed_method = EXCEED_SPECIFIC_RING
            plan.exceed_ring_id = ring_id
            plan.will_exceed = True
        else:
            candidates.append(plan)

    expendable_general_rings = max(
        0, rings_left.get(general_ring_id, 0) - max(0, general_ring_keep)
    )
    candidates.sort(
        key=lambda plan: (
            -plan.source.exceed_memory_cost,
            plan.source.farmable,
            plan.unit_id,
        )
    )

    for plan in candidates:
        if expendable_general_rings <= 0:
            break
        expendable_general_rings -= 1
        rings_left[general_ring_id] -= 1
        ring_use[general_ring_id] += 1
        plan.exceed_method = EXCEED_GENERAL_RING
        plan.exceed_ring_id = general_ring_id
        plan.will_exceed = True

    for plan in candidates:
        if plan.will_exceed:
            continue

        source = plan.source
        demand = source.exceed_memory_cost
        if demand >= 120:
            plan.exceed_method = EXCEED_WAIT
            plan.notes.append(f"突破需要{demand}片，仅使用戒指，当前无可用戒指")
            continue

        if plan.memory_after >= demand:
            plan.exceed_method = EXCEED_MEMORY
            plan.exceed_memory_use = demand
            plan.memory_after -= demand
            plan.will_exceed = True
        elif not source.farmable and source.can_purchase_memory:
            plan.exceed_method = EXCEED_MEMORY
            plan.exceed_memory_use = plan.memory_after
            plan.exceed_memory_buy = demand - plan.memory_after
            plan.memory_after = 0
            plan.will_exceed = True
        elif source.farmable:
            plan.exceed_method = EXCEED_WAIT
            plan.notes.append(
                f"突破还缺{demand - plan.memory_after}片，等待地图刷取"
            )
        else:
            plan.exceed_method = EXCEED_WAIT
            plan.notes.append(
                f"突破还缺{demand - plan.memory_after}片，女神商店不可购买"
            )

    return StarExceedPlanningResult(
        units=plans,
        ring_use=ring_use,
        general_ring_id=general_ring_id,
        general_ring_inventory=max(0, ring_inventory.get(general_ring_id, 0)),
        general_ring_keep=max(0, general_ring_keep),
    )


@dataclass(frozen=True)
class Unique1Stage:
    target_rank: int
    target_level: int
    heart_cost: int
    memory_cost: int


@dataclass(frozen=True)
class Unique1UnitInput:
    unit_id: int
    eligible: bool
    current_rank: int
    current_level: int
    current_rank_max_level: int
    max_level: int
    memory_inventory: int
    stages: Tuple[Unique1Stage, ...]
    wait_reason: str = ""


@dataclass(frozen=True)
class Unique1Allocation:
    stage: Unique1Stage
    phase: str


@dataclass
class Unique1UnitPlan:
    source: Unique1UnitInput
    target_rank: int
    target_level: int
    memory_remaining: int
    allocations: List[Unique1Allocation] = field(default_factory=list)

    @property
    def unit_id(self) -> int:
        return self.source.unit_id

    @property
    def next_stage(self) -> Optional[Unique1Stage]:
        index = len(self.allocations)
        return self.source.stages[index] if index < len(self.source.stages) else None

    @property
    def heart_use(self) -> int:
        return sum(item.stage.heart_cost for item in self.allocations)

    @property
    def memory_use(self) -> int:
        return sum(item.stage.memory_cost for item in self.allocations)

    @property
    def full_heart_demand(self) -> int:
        return sum(stage.heart_cost for stage in self.source.stages)

    @property
    def full_memory_demand(self) -> int:
        return sum(stage.memory_cost for stage in self.source.stages)

    @property
    def memory_gap_to_max(self) -> int:
        return max(0, self.full_memory_demand - self.source.memory_inventory)

    @property
    def possible_heart_demand(self) -> int:
        memory = self.source.memory_inventory
        hearts = 0
        for stage in self.source.stages:
            if memory < stage.memory_cost:
                break
            memory -= stage.memory_cost
            hearts += stage.heart_cost
        return hearts


@dataclass
class Unique1PlanningResult:
    units: List[Unique1UnitPlan]
    heart_inventory: int
    heart_keep: int
    allocation_order: List[int]

    @property
    def heart_available(self) -> int:
        return max(0, self.heart_inventory - self.heart_keep)

    @property
    def heart_consumption(self) -> int:
        return sum(plan.heart_use for plan in self.units)

    @property
    def possible_heart_demand(self) -> int:
        return sum(
            plan.possible_heart_demand for plan in self.units
            if plan.source.eligible
        )

    @property
    def full_heart_demand(self) -> int:
        return sum(
            plan.full_heart_demand for plan in self.units
            if plan.source.eligible
        )

    @property
    def memory_gap_to_max(self) -> int:
        return sum(
            plan.memory_gap_to_max for plan in self.units
            if plan.source.eligible
        )


def plan_unique1(
    units: Sequence[Unique1UnitInput],
    heart_inventory: int,
    heart_keep: int,
    floor_level: int = 130,
) -> Unique1PlanningResult:
    """Allocate UE1 stages using the requested three-tier priority."""
    plans: List[Unique1UnitPlan] = []
    for source in sorted(units, key=lambda item: item.unit_id):
        target_level = source.current_level
        if source.eligible:
            target_level = max(
                source.current_level,
                min(source.max_level, source.current_rank_max_level),
            )
        plans.append(
            Unique1UnitPlan(
                source=source,
                target_rank=source.current_rank,
                target_level=target_level,
                memory_remaining=max(0, source.memory_inventory),
            )
        )

    hearts_left = max(0, heart_inventory - max(0, heart_keep))
    allocation_order: List[int] = []

    def allocate(plan: Unique1UnitPlan, phase: str) -> None:
        nonlocal hearts_left
        stage = plan.next_stage
        if stage is None:
            return
        hearts_left -= stage.heart_cost
        plan.memory_remaining -= stage.memory_cost
        plan.allocations.append(Unique1Allocation(stage=stage, phase=phase))
        plan.target_rank = stage.target_rank
        plan.target_level = max(
            plan.target_level,
            min(plan.source.max_level, stage.target_level),
        )
        if plan.unit_id not in allocation_order:
            allocation_order.append(plan.unit_id)

    finishers = sorted(
        (
            plan for plan in plans
            if plan.source.eligible
            and plan.full_heart_demand in (10, 20)
            and plan.full_memory_demand <= plan.memory_remaining
        ),
        key=lambda plan: (
            plan.full_heart_demand,
            -plan.target_level,
            plan.unit_id,
        ),
    )
    for plan in finishers:
        if plan.full_heart_demand > hearts_left:
            continue
        while plan.next_stage is not None:
            allocate(plan, "10/20心碎收尾")

    while True:
        candidates = []
        for plan in plans:
            stage = plan.next_stage
            floor = min(floor_level, plan.source.max_level)
            if (
                not plan.source.eligible
                or stage is None
                or plan.target_level >= floor
                or stage.memory_cost > plan.memory_remaining
                or stage.heart_cost > hearts_left
            ):
                continue
            candidates.append(plan)

        if not candidates:
            break
        selected = min(
            candidates,
            key=lambda plan: (
                plan.next_stage.target_level,
                plan.target_level,
                plan.unit_id,
            ),
        )
        allocate(selected, "平铺130级")

    while True:
        candidates = []
        for plan in plans:
            stage = plan.next_stage
            if (
                not plan.source.eligible
                or stage is None
                or stage.memory_cost > plan.memory_remaining
                or stage.heart_cost > hearts_left
            ):
                continue
            candidates.append(plan)

        if not candidates:
            break
        selected = min(
            candidates,
            key=lambda plan: (
                -plan.target_level,
                -plan.next_stage.target_level,
                plan.unit_id,
            ),
        )
        allocate(selected, "高专优先")

    return Unique1PlanningResult(
        units=plans,
        heart_inventory=max(0, heart_inventory),
        heart_keep=max(0, heart_keep),
        allocation_order=allocation_order,
    )


@dataclass(frozen=True)
class ShopPriceSegment:
    quantity: int
    unit_price: int
    total_price: int


@dataclass(frozen=True)
class GoddessPurchasePlan:
    unit_id: int
    memory_id: int
    slot_id: int
    quantity: int
    star_quantity: int
    exceed_quantity: int
    unique1_quantity: int
    exchange_count: int
    segments: Tuple[ShopPriceSegment, ...]

    @property
    def total_price(self) -> int:
        return sum(segment.total_price for segment in self.segments)


def build_shop_price_segments(
    price_group: int,
    exchange_count: int,
    quantity: int,
    price_lookup: Callable[[int, int], object],
) -> Tuple[ShopPriceSegment, ...]:
    segments: List[ShopPriceSegment] = []
    bought = max(0, exchange_count)
    left = max(0, quantity)

    while left > 0:
        price = price_lookup(price_group, bought)
        if price.buy_count_to == -1:
            count = left
        else:
            count = min(left, price.buy_count_to - bought)
        if count <= 0:
            raise ValueError(f"女神商店价格区间异常: group={price_group}, count={bought}")
        total = count * price.count
        segments.append(
            ShopPriceSegment(quantity=count, unit_price=price.count, total_price=total)
        )
        bought += count
        left -= count

    return tuple(segments)


@dataclass(frozen=True)
class Unique2UnitPlan:
    unit_id: int
    current_level: int
    target_level: int
    pure_memory_needed: int
    pure_memory_use: Counter
    wait_reason: str = ""


@dataclass(frozen=True)
class SmartEnhanceResources:
    amulets: int
    general_rings: int
    specific_rings: int
    hearts: int


@dataclass
class SmartEnhancePlan:
    star_exceed: StarExceedPlanningResult
    goddess_purchases: List[GoddessPurchasePlan]
    unique2: List[Unique2UnitPlan]
    unique1: Unique1PlanningResult
    amulet_inventory: int
    specific_ring_inventory: int
    large_heart_inventory: int

    @property
    def resources(self) -> SmartEnhanceResources:
        return SmartEnhanceResources(
            amulets=sum(item.total_price for item in self.goddess_purchases),
            general_rings=self.star_exceed.general_ring_use,
            specific_rings=self.star_exceed.specific_ring_use,
            hearts=self.unique1.heart_consumption,
        )

    @property
    def amulet_deficit(self) -> int:
        return max(0, self.resources.amulets - self.amulet_inventory)


@description(
    "自动规划全部已持有角色的五星、等级突破、专武2和专武1。"
    "\n地图可刷角色不使用母猪石；不可刷角色可购买碎片升五星，40片突破可继续购买，120片突破只使用戒指。"
    "\n专武1保留设置数量的心碎，依次优先完成仅差10/20心碎的角色、平铺到130级、再优先提升已有高等级专武；10/20心碎收尾角色缺少不可刷碎片时会使用母猪石补齐。"
    "\n默认仅预览，开启执行后会先从可可萝钱包将玛娜提至持有上限。"
)
@name("智能强化所有角色")
@booltype("smart_unit_enhance_execute", "执行强化", False)
@inttype("smart_unit_enhance_ring_keep", "通用戒指保留", 10, list(range(101)))
@inttype("smart_unit_enhance_heart_keep", "心碎保留", 500, list(range(5001)))
@default(False)
class smart_unit_enhance(UnitController):
    async def _growth_limits(
        self, unit_ids: Sequence[int]
    ) -> Dict[int, Optional[GrowthParameterUnique]]:
        limits: Dict[int, Optional[GrowthParameterUnique]] = {}
        for unit_id in unit_ids:
            self.unit_id = unit_id
            limits[unit_id] = await self.is_unique_growth_unit()
        return limits

    @staticmethod
    def _goddess_shop_items(shop_list) -> Dict[int, ShopItem]:
        goddess_shop = next(
            (
                shop for shop in shop_list
                if shop.system_id == eSystemId.MEMORY_PIECE_SHOP
            ),
            None,
        )
        if not goddess_shop:
            return {}
        return {
            item.item_id: item for item in goddess_shop.item_list
            if not item.sold and (item.num or 1) == 1
        }

    def _star_inputs(
        self,
        unit_ids: Sequence[int],
        shop_items: Mapping[int, ShopItem],
    ) -> Tuple[List[StarExceedUnitInput], Counter, int]:
        stage = db.exceed_level_stage.get(1)
        general_ring_id = stage.general_exceed_item_id if stage else 0
        ring_inventory: Counter = Counter()
        if general_ring_id:
            ring_inventory[general_ring_id] = self.client.data.get_inventory(
                (eInventoryType.Item, general_ring_id)
            )

        result: List[StarExceedUnitInput] = []
        for unit_id in unit_ids:
            if unit_id not in db.unit_to_memory:
                continue

            unit = self.client.data.unit[unit_id]
            memory_id = db.unit_to_memory[unit_id]
            token = (eInventoryType.Item, memory_id)
            required = db.rarity_up_required.get(unit_id, {})
            rarities = range(unit.unit_rarity + 1, 6)
            star_supported = unit.unit_rarity >= 5 or all(
                rarity in required for rarity in rarities
            )
            star_cost = sum(
                required.get(rarity, Counter()).get(token, 0)
                for rarity in rarities
            )

            exceed = db.exceed_level_unit_required.get(unit_id)
            specific_ring_id = exceed.exceed_item_id if exceed else 0
            if specific_ring_id and specific_ring_id != general_ring_id:
                ring_inventory[specific_ring_id] = self.client.data.get_inventory(
                    (eInventoryType.Item, specific_ring_id)
                )

            farmable = token in db.memory_hard_quest or token in db.memory_shiori_quest
            result.append(
                StarExceedUnitInput(
                    unit_id=unit_id,
                    current_rarity=unit.unit_rarity,
                    memory_inventory=self.client.data.get_inventory(token),
                    star_memory_cost=star_cost,
                    farmable=farmable,
                    can_purchase_memory=memory_id in shop_items,
                    star_supported=star_supported,
                    exceeded=bool(unit.exceed_stage),
                    exceed_available=exceed is not None,
                    exceed_memory_cost=exceed.consume_num_1 if exceed else 0,
                    specific_ring_id=specific_ring_id,
                )
            )

        return result, ring_inventory, general_ring_id

    @staticmethod
    def _purchase_plans(
        star_plan: StarExceedPlanningResult,
        shop_items: Mapping[int, ShopItem],
    ) -> List[GoddessPurchasePlan]:
        purchases: List[GoddessPurchasePlan] = []
        for plan in star_plan.units:
            if plan.memory_purchase <= 0:
                continue
            memory_id = db.unit_to_memory[plan.unit_id]
            item = shop_items[memory_id]
            exchange_count = item.exchange_count or 0
            segments = build_shop_price_segments(
                item.price_group,
                exchange_count,
                plan.memory_purchase,
                db.get_shop_item_price_info,
            )
            purchases.append(
                GoddessPurchasePlan(
                    unit_id=plan.unit_id,
                    memory_id=memory_id,
                    slot_id=item.slot_id,
                    quantity=plan.memory_purchase,
                    star_quantity=plan.star_memory_buy,
                    exceed_quantity=plan.exceed_memory_buy,
                    unique1_quantity=plan.unique1_memory_buy,
                    exchange_count=exchange_count,
                    segments=segments,
                )
            )
        return purchases

    def _unique1_inputs(
        self,
        star_plan: StarExceedPlanningResult,
        growth_limits: Mapping[int, Optional[GrowthParameterUnique]],
    ) -> List[Unique1UnitInput]:
        if 1 not in db.unique_equipment_max_level:
            return []
        max_level = db.unique_equipment_max_level[1]
        max_rank = db.get_unique_equip_rank_from_level(1, max_level)
        result: List[Unique1UnitInput] = []

        for plan in star_plan.units:
            unit_id = plan.unit_id
            if unit_id not in db.unit_unique_equip.get(1, {}):
                continue

            unit = self.client.data.unit[unit_id]
            slot = unit.unique_equip_slot[0] if unit.unique_equip_slot else None
            actual_rank = (slot.rank or 0) if slot else 0
            actual_level = (slot.enhancement_level or 0) if slot else 0
            actual_pt = (slot.enhancement_pt or 0) if slot else 0
            limit = growth_limits.get(unit_id)
            free_rank = (limit.unique_equip_rank_1 or 0) if limit else 0
            free_pt = (limit.unique_equip_strength_point_1 or 0) if limit else 0
            free_level = db.get_unique_equip_level_from_pt(1, free_pt) if free_pt else 0
            effective_rank = max(actual_rank, free_rank)
            effective_level = max(actual_level, free_level)
            current_cap = (
                min(max_level, db.get_unique_equip_max_level_from_rank(1, effective_rank))
                if effective_rank > 0 else 0
            )
            memory_token = (eInventoryType.Item, db.unit_to_memory[unit_id])
            stages: List[Unique1Stage] = []
            for rank in range(effective_rank, max_rank):
                demand = db.get_unique_equip_material_demand(
                    unit_id, 1, rank, rank + 1
                )
                if not demand:
                    break
                stages.append(
                    Unique1Stage(
                        target_rank=rank + 1,
                        target_level=min(
                            max_level,
                            db.get_unique_equip_max_level_from_rank(1, rank + 1),
                        ),
                        heart_cost=demand.get(db.xinsui, 0),
                        memory_cost=demand.get(memory_token, 0),
                    )
                )

            eligible = plan.target_rarity >= 5 and plan.will_exceed
            result.append(
                Unique1UnitInput(
                    unit_id=unit_id,
                    eligible=eligible,
                    current_rank=effective_rank,
                    current_level=effective_level,
                    current_rank_max_level=current_cap,
                    max_level=max_level,
                    memory_inventory=plan.memory_after,
                    stages=tuple(stages),
                    wait_reason="" if eligible else "未达到五星并完成突破",
                )
            )
        return result

    def _plan_unique1_with_finisher_purchases(
        self,
        star_plan: StarExceedPlanningResult,
        growth_limits: Mapping[int, Optional[GrowthParameterUnique]],
        heart_inventory: int,
        heart_keep: int,
    ) -> Unique1PlanningResult:
        base_inputs = self._unique1_inputs(star_plan, growth_limits)
        star_by_unit = {item.unit_id: item for item in star_plan.units}
        proposed: Dict[int, int] = {}
        boosted_inputs: List[Unique1UnitInput] = []

        for item in base_inputs:
            star = star_by_unit[item.unit_id]
            full_heart = sum(stage.heart_cost for stage in item.stages)
            full_memory = sum(stage.memory_cost for stage in item.stages)
            shortage = max(0, full_memory - item.memory_inventory)
            can_buy_finisher = (
                item.eligible
                and full_heart in (10, 20)
                and shortage > 0
                and not star.source.farmable
                and star.source.can_purchase_memory
            )
            if can_buy_finisher:
                proposed[item.unit_id] = shortage
                item = replace(
                    item,
                    memory_inventory=item.memory_inventory + shortage,
                )
            boosted_inputs.append(item)

        simulated = plan_unique1(
            boosted_inputs,
            heart_inventory,
            heart_keep,
        )
        selected = {
            item.unit_id: proposed[item.unit_id]
            for item in simulated.units
            if item.unit_id in proposed
            and len(item.allocations) == len(item.source.stages)
        }

        for unit_id, quantity in selected.items():
            star = star_by_unit[unit_id]
            star.unique1_memory_buy = quantity
            star.memory_after += quantity

        return plan_unique1(
            self._unique1_inputs(star_plan, growth_limits),
            heart_inventory,
            heart_keep,
        )

    def _unique2_plans(
        self,
        unit_ids: Sequence[int],
        growth_limits: Mapping[int, Optional[GrowthParameterUnique]],
    ) -> List[Unique2UnitPlan]:
        max_level = db.unique_equipment_max_level.get(2, 0)
        if max_level <= 0:
            return []

        inventory = Counter(self.client.data.inventory)
        result: List[Unique2UnitPlan] = []
        for unit_id in unit_ids:
            unit = self.client.data.unit[unit_id]
            if len(unit.unique_equip_slot or []) < 2:
                continue
            slot1 = unit.unique_equip_slot[0]
            slot2 = unit.unique_equip_slot[1]
            current_level = slot2.enhancement_level or 0
            if current_level >= max_level:
                continue

            target_pt = db.get_unique_equip_pt_from_level(2, max_level)
            current_pt = slot2.enhancement_pt or 0
            needed = max(0, target_pt - current_pt)
            if not slot1.is_slot and not growth_limits.get(unit_id):
                result.append(
                    Unique2UnitPlan(
                        unit_id=unit_id,
                        current_level=current_level,
                        target_level=max_level,
                        pure_memory_needed=needed,
                        pure_memory_use=Counter(),
                        wait_reason="需先装备专武1，为避免绕过心碎规划本次暂缓",
                    )
                )
                continue

            consume: Counter = Counter()
            left = needed
            kana = db.unit_data[unit_id].kana
            for kana_id in db.unit_kana_ids[kana]:
                token = db.unit_to_pure_memory.get(kana_id)
                if not token:
                    continue
                count = min(left, inventory[token])
                if count:
                    consume[token] += count
                    left -= count
                if left <= 0:
                    break

            if left > 0:
                result.append(
                    Unique2UnitPlan(
                        unit_id=unit_id,
                        current_level=current_level,
                        target_level=max_level,
                        pure_memory_needed=needed,
                        pure_memory_use=Counter(),
                        wait_reason=f"同名纯净记忆不足{left}片",
                    )
                )
                continue

            inventory.subtract(consume)
            result.append(
                Unique2UnitPlan(
                    unit_id=unit_id,
                    current_level=current_level,
                    target_level=max_level,
                    pure_memory_needed=needed,
                    pure_memory_use=consume,
                )
            )
        return result

    async def _build_plan(self) -> SmartEnhancePlan:
        unit_ids = sorted(self.client.data.unit)
        shop_response = await self.client.get_shop_item_list()
        shop_items = self._goddess_shop_items(shop_response.shop_list or [])
        growth_limits = await self._growth_limits(unit_ids)

        star_inputs, ring_inventory, general_ring_id = self._star_inputs(
            unit_ids, shop_items
        )
        star_plan = plan_star_and_exceed(
            star_inputs,
            ring_inventory,
            general_ring_id,
            self.get_config("smart_unit_enhance_ring_keep"),
        )
        heart_inventory = self.client.data.get_inventory(db.xinsui)
        heart_keep = self.get_config("smart_unit_enhance_heart_keep")
        unique1 = self._plan_unique1_with_finisher_purchases(
            star_plan,
            growth_limits,
            heart_inventory,
            heart_keep,
        )
        purchases = self._purchase_plans(star_plan, shop_items)
        unique2 = self._unique2_plans(unit_ids, growth_limits)

        relevant_specific_rings = {
            source.specific_ring_id for source in star_inputs
            if source.specific_ring_id
            and source.specific_ring_id != general_ring_id
        }
        return SmartEnhancePlan(
            star_exceed=star_plan,
            goddess_purchases=purchases,
            unique2=unique2,
            unique1=unique1,
            amulet_inventory=self.client.data.get_shop_gold(
                eSystemId.MEMORY_PIECE_SHOP
            ),
            specific_ring_inventory=sum(
                ring_inventory[ring_id] for ring_id in relevant_specific_rings
            ),
            large_heart_inventory=self.client.data.get_inventory(db.heart),
        )

    @staticmethod
    def _brief_unit_names(unit_ids: Sequence[int], limit: int = 6) -> str:
        unit_ids = sorted(set(unit_ids))
        names = [db.get_unit_name(unit_id) for unit_id in unit_ids[:limit]]
        suffix = f"等{len(unit_ids)}人" if len(unit_ids) > limit else ""
        return "、".join(names) + suffix

    def _log_plan(self, plan: SmartEnhancePlan, execute: bool) -> None:
        resources = plan.resources
        star = plan.star_exceed
        star_by_unit = {item.unit_id: item for item in star.units}
        unique1 = plan.unique1
        mode = "执行" if execute else "预览"
        self._log(f"智能强化：{mode}")
        self._log(
            f"资源：母猪石 库存{plan.amulet_inventory}/消耗{resources.amulets}/"
            f"缺口{plan.amulet_deficit}；通用戒指 库存{star.general_ring_inventory}/"
            f"消耗{resources.general_rings}/保留{star.general_ring_keep}；"
            f"专属戒指 库存{plan.specific_ring_inventory}/消耗{resources.specific_rings}"
        )
        heart_extra = f"；大心{plan.large_heart_inventory}未计" if plan.large_heart_inventory else ""
        self._log(
            f"心碎：库存{unique1.heart_inventory}/消耗{resources.hearts}/"
            f"保留{unique1.heart_keep}{heart_extra}"
        )
        bank = self.client.data.user_gold_bank_info
        bank_mana = bank.bank_gold if bank else 0
        mana = self.client.data.get_mana()
        mana_limit = self.client.data.settings.limit.limit_gold
        fill_to = min(mana_limit, mana + bank_mana)
        self._log(
            f"玛娜：持有{mana / self.E:.2f}亿，钱包{bank_mana / self.E:.2f}亿，"
            f"执行可提至{fill_to / self.E:.2f}亿"
        )

        star_up = [item for item in star.units if item.target_rarity > item.source.current_rarity]
        general_exceed = [item for item in star.units if item.exceed_method == EXCEED_GENERAL_RING]
        specific_exceed = [item for item in star.units if item.exceed_method == EXCEED_SPECIFIC_RING]
        memory_exceed = [item for item in star.units if item.exceed_method == EXCEED_MEMORY]
        wait_star = [
            item.unit_id for item in star.units
            if item.source.current_rarity < 5 and item.target_rarity < 5
        ]
        wait_exceed = [
            item.unit_id for item in star.units
            if item.target_rarity >= 5 and not item.will_exceed
        ]
        self._log(
            f"星级/突破：升星{len(star_up)}人（购片{sum(item.star_memory_buy for item in star_up)}）；"
            f"通用戒指{len(general_exceed)}人、专属戒指{len(specific_exceed)}人、"
            f"碎片突破{len(memory_exceed)}人；暂缓升星{len(wait_star)}/突破{len(wait_exceed)}"
        )

        unique2_run = [item for item in plan.unique2 if not item.wait_reason]
        unique2_wait = [item.unit_id for item in plan.unique2 if item.wait_reason]
        self._log(
            f"专武2：强化{len(unique2_run)}人，纯净记忆"
            f"{sum(item.pure_memory_needed for item in unique2_run)}；暂缓{len(unique2_wait)}"
        )

        unique1_run = [
            item for item in unique1.units
            if item.source.eligible and item.target_level > item.source.current_level
        ]
        phase_units = {
            phase: sum(
                any(allocation.phase == phase for allocation in item.allocations)
                for item in unique1_run
            )
            for phase in ("10/20心碎收尾", "平铺130级", "高专优先")
        }
        unique1_not_ready = [
            item.unit_id for item in unique1.units
            if not item.source.eligible and item.source.stages
        ]
        farm_wait: List[int] = []
        fragment_wait: List[int] = []
        heart_wait: List[int] = []
        for item in unique1.units:
            stage = item.next_stage
            if not item.source.eligible or stage is None:
                continue
            if stage.memory_cost > item.memory_remaining:
                star_item = star_by_unit[item.unit_id]
                purchasable_finisher = (
                    not star_item.source.farmable
                    and star_item.source.can_purchase_memory
                    and item.full_heart_demand in (10, 20)
                )
                if purchasable_finisher:
                    heart_wait.append(item.unit_id)
                elif star_item.source.farmable:
                    farm_wait.append(item.unit_id)
                else:
                    fragment_wait.append(item.unit_id)
            else:
                heart_wait.append(item.unit_id)

        unique1_buy = sum(item.unique1_memory_buy for item in star.units)
        self._log(
            f"专武1：强化{len(unique1_run)}人（收尾{phase_units['10/20心碎收尾']}/"
            f"平铺{phase_units['平铺130级']}/高专{phase_units['高专优先']}），"
            f"角色碎片{sum(item.memory_use for item in unique1_run)}（母猪石补{unique1_buy}）；"
            f"暂缓 未五星突破{len(unique1_not_ready)}/待刷{len(farm_wait)}/"
            f"碎片不足{len(fragment_wait)}/心碎预算{len(heart_wait)}"
        )

        blocked = wait_star + wait_exceed + unique2_wait + unique1_not_ready + farm_wait + fragment_wait + heart_wait
        if blocked:
            self._log(f"暂缓角色：{self._brief_unit_names(blocked)}")

    async def _execute_purchase(self, purchase: GoddessPurchasePlan) -> None:
        for segment in purchase.segments:
            await self.client.shop_buy(
                eSystemId.MEMORY_PIECE_SHOP,
                purchase.slot_id,
                segment.quantity,
                segment.total_price,
            )

    async def _execute_star_exceed(self, plan: SmartEnhancePlan) -> None:
        purchases = {item.unit_id: item for item in plan.goddess_purchases}
        for item in plan.star_exceed.units:
            has_action = (
                item.target_rarity > item.source.current_rarity
                or item.unit_id in purchases
                or item.exceed_method in (
                    EXCEED_SPECIFIC_RING,
                    EXCEED_GENERAL_RING,
                    EXCEED_MEMORY,
                )
            )
            if not has_action:
                continue

            self.unit_id = item.unit_id
            try:
                if item.unit_id in purchases:
                    await self._execute_purchase(purchases[item.unit_id])

                if self.unit.unit_rarity < item.target_rarity:
                    mana = sum(
                        10_000 * rarity
                        for rarity in range(
                            self.unit.unit_rarity + 1, item.target_rarity + 1
                        )
                    )
                    if not await self.client.prepare_mana(mana):
                        raise AbortError(f"升星需要Mana {mana}，当前不足")
                    memory_token = (
                        eInventoryType.Item,
                        db.unit_to_memory[item.unit_id],
                    )
                    await self.client.unit_multi_evolution(
                        unit_id=item.unit_id,
                        current_rarity=self.unit.unit_rarity,
                        after_rarity=item.target_rarity,
                        current_gold_num=self.client.data.get_mana(),
                        current_memory_piece_num=self.client.data.get_inventory(
                            memory_token
                        ),
                    )

                if self.unit.exceed_stage:
                    continue
                if not item.will_exceed or self.unit.unit_rarity < 5:
                    continue

                exceed = db.exceed_level_unit_required[item.unit_id]
                if not await self.client.prepare_mana(exceed.consume_num_2):
                    raise AbortError(
                        f"突破需要Mana {exceed.consume_num_2}，当前不足"
                    )
                if item.exceed_method in (
                    EXCEED_SPECIFIC_RING,
                    EXCEED_GENERAL_RING,
                ):
                    await self.client.unit_exceed_level_limit_with_exceed_item(
                        item.unit_id, 1, item.exceed_ring_id
                    )
                elif item.exceed_method == EXCEED_MEMORY:
                    memory_token = (
                        eInventoryType.Item,
                        db.unit_to_memory[item.unit_id],
                    )
                    await self.client.unit_exceed_level_limit(
                        unit_id=item.unit_id,
                        exceed_stage=1,
                        cost_item_list=[
                            InventoryInfoPost(
                                type=db.zmana[0],
                                id=db.zmana[1],
                                count=exceed.consume_num_2,
                            ),
                            InventoryInfoPost(
                                type=memory_token[0],
                                id=memory_token[1],
                                count=exceed.consume_num_1,
                            ),
                        ],
                    )
            except Exception as error:
                self._warn(f"{self.unit_name}星级/突破执行失败：{error}")

    async def _execute_unique2(self, plans: Sequence[Unique2UnitPlan]) -> None:
        for item in plans:
            if item.wait_reason:
                continue
            self.unit_id = item.unit_id
            try:
                slot = self.unit.unique_equip_slot[1]
                if (slot.enhancement_level or 0) < item.target_level:
                    log_start = len(self.log)
                    try:
                        await self.unit_unique_equip2_enhance_aware(item.target_level)
                    finally:
                        del self.log[log_start:]
            except Exception as error:
                self._warn(f"{self.unit_name}专武2强化失败：{error}")

    async def _execute_unique1(self, result: Unique1PlanningResult) -> None:
        by_unit = {item.unit_id: item for item in result.units}
        execution_order = list(result.allocation_order)
        execution_order.extend(
            item.unit_id for item in result.units
            if item.unit_id not in execution_order
            and item.source.eligible
            and item.target_level > 0
        )

        for unit_id in execution_order:
            item = by_unit[unit_id]
            self.unit_id = unit_id
            try:
                if self.unit.unit_rarity < 5 or not self.unit.exceed_stage:
                    self._warn(f"{self.unit_name}未完成五星突破，跳过专武1")
                    continue
                slot = self.unit.unique_equip_slot[0]
                current_level = slot.enhancement_level or 0
                if current_level >= item.target_level:
                    continue
                limit = await self.is_unique_growth_unit()
                log_start = len(self.log)
                try:
                    await self.unit_unique_equip1_enhance_aware(
                        item.target_level, limit
                    )
                finally:
                    del self.log[log_start:]
            except Exception as error:
                self._warn(f"{self.unit_name}专武1强化失败：{error}")

    async def do_task(self, client: pcrclient):
        self.client = client
        execute = self.get_config("smart_unit_enhance_execute")
        plan = await self._build_plan()
        self._log_plan(plan, execute)

        if not execute:
            return
        if plan.amulet_deficit:
            raise AbortError(
                f"母猪石不足，计划消耗{plan.resources.amulets}，"
                f"库存{plan.amulet_inventory}，缺口{plan.amulet_deficit}；未执行任何强化"
            )

        mana_drawn = await self.client.draw_from_bank_to_limit()
        if mana_drawn:
            self._log(
                f"玛娜：钱包提取{mana_drawn / self.E:.2f}亿，"
                f"当前{self.client.data.get_mana() / self.E:.2f}亿"
            )
        await self._execute_star_exceed(plan)
        await self._execute_unique2(plan.unique2)
        await self._execute_unique1(plan.unique1)
        self._log("执行结束，失败项见警告" if self.is_warn else "执行完成")
