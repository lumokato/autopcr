"""
EX装备自动装备执行器（重构版本）
负责执行自动装备操作的完整流程
"""
from typing import Dict, Set, Tuple, List
from collections import defaultdict
from ...db.database import db
from ...model.enums import eInventoryType
from ...model.common import ExtraEquipChangeSlot, ExtraEquipChangeUnit
from .exequip_helpers import (
    ExEquipPowerCalculator,
    ExEquipInventoryManager,
    ExEquipConstants,
)


class ExEquipAutoEquipper:
    """自动装备执行器：负责执行自动装备操作"""

    def __init__(self, client, power_calculator: ExEquipPowerCalculator, module):
        self.client = client
        self.calculator = power_calculator
        self.module = module
        self.inventory_mgr = ExEquipInventoryManager(client)

        # 分配结果记录
        self.allocation_results = {}  # {(unit_id, slot): {'ex_id': ..., 'rank': ..., 'pt': ...}}
        self.unequipped_count = 0
        self.unequipped_units = 0
        self.inventory_snapshots = {}  # 保存分配前的库存快照

    async def execute(self, recommendations: dict, auto_equip: bool):
        """
        执行自动装备流程

        Args:
            recommendations: 推荐结果字典
            auto_equip: 是否实际执行（False为模拟）
        """
        unit_slot_recommendations = recommendations['unit_slot_recommendations']

        # 步骤1: 撤下未锁定的非粉装
        await self._unequip_unlocked_equipment(auto_equip)

        # 步骤2: 构建库存池
        self.inventory_mgr.build_slot_category_pools(unit_slot_recommendations)

        # 步骤2.5: 保存分配前的库存快照
        self._save_inventory_snapshots()

        # 步骤3: 按槽位/类别分组执行分配
        await self._allocate_by_slot_category(unit_slot_recommendations, auto_equip)

        # 步骤4: 输出报告
        self._output_report(auto_equip)

    def _save_inventory_snapshots(self):
        """保存分配前的库存快照"""
        import copy
        self.inventory_snapshots = {}
        for key, pool in self.inventory_mgr.slot_category_pool.items():
            # 深拷贝每个pool的结构
            self.inventory_snapshots[key] = {}
            for ex_id, ranks in pool.items():
                self.inventory_snapshots[key][ex_id] = {
                    'rank2': len(ranks['rank2']),
                    'rank1': len(ranks['rank1']),
                    'rank0': len(ranks['rank0']),
                    'rank2_full': sum(1 for item in ranks['rank2'] if item['pt'] >= ExEquipConstants.ENHANCEMENT_PT_MAX),
                    'rank1_full': sum(1 for item in ranks['rank1'] if item['pt'] >= ExEquipConstants.ENHANCEMENT_PT_MAX),
                    'rank0_full': sum(1 for item in ranks['rank0'] if item['pt'] >= ExEquipConstants.ENHANCEMENT_PT_MAX),
                }

    async def _unequip_unlocked_equipment(self, auto_equip: bool):
        """
        撤下所有未锁定的非粉装EX装备

        Args:
            auto_equip: 是否实际执行（False为模拟）
        """
        removed_cnt = 0
        removed_units = 0

        for unit_id, unit in self.client.data.unit.items():
            exchange = []
            for slot in range(3):
                ex_slot = unit.ex_equip_slot[slot]
                if ex_slot.serial_id == 0:
                    continue

                ex = self.client.data.ex_equips.get(ex_slot.serial_id)
                if not ex:
                    continue

                rarity = db.get_ex_equip_rarity(ex.ex_equipment_id)
                # if ex.protection_flag == ExEquipConstants.PROTECTION_UNLOCKED and rarity < ExEquipConstants.RARITY_PINK:
                if rarity < ExEquipConstants.RARITY_PINK:
                    exchange.append(ExtraEquipChangeSlot(slot=slot + 1, serial_id=0))

            if exchange:
                removed_units += 1
                removed_cnt += len(exchange)
                if auto_equip:
                    await self.client.unit_equip_ex([ExtraEquipChangeUnit(
                        unit_id=unit_id, ex_equip_slot=exchange, cb_ex_equip_slot=None
                    )])

        self.unequipped_count = removed_cnt
        self.unequipped_units = removed_units

        if removed_cnt:
            if auto_equip:
                self.module._log(f"已撤下未锁定的非粉装EX：{removed_units}个角色，共{removed_cnt}件")
            else:
                self.module._log(f"将撤下未锁定的非粉装EX：{removed_units}个角色，共{removed_cnt}件")

        # 统计锁定的装备
        locked_count = len(self.inventory_mgr.locked_slots)
        if locked_count:
            self.module._log(f"锁定装备：{locked_count}个槽位已锁定，不会被重新分配")

    async def _allocate_by_slot_category(self, unit_slot_recommendations: dict, auto_equip: bool):
        """
        按槽位/类别分组执行分配

        Args:
            unit_slot_recommendations: {(unit_id, slot): recommendation_dict}
            auto_equip: 是否实际执行
        """
        # 按(slot, category)分组角色
        slot_category_groups = defaultdict(list)

        for (unit_id, slot), recommendation in unit_slot_recommendations.items():
            # 排除锁定的槽位
            if (unit_id, slot) in self.inventory_mgr.locked_slots:
                continue

            slot_data = db.unit_ex_equipment_slot.get(unit_id)
            if not slot_data:
                continue

            category = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot - 1]
            if category is None:
                continue

            # 按战力增加倒序排列
            power = recommendation['best_power_5rank2']
            slot_category_groups[(slot, category)].append((unit_id, power, recommendation))

        # 对每个组内的角色按战力降序排序
        for key in slot_category_groups:
            slot_category_groups[key].sort(key=lambda x: -x[1])

        # 逐组分配
        for (slot, category), roles in slot_category_groups.items():
            await self._allocate_one_slot_category(slot, category, roles, auto_equip)

    async def _allocate_one_slot_category(self, slot: int, category: int,
                                          roles: List[Tuple[int, int, dict]], auto_equip: bool):
        """
        为一个槽位/类别组分配装备

        Args:
            slot: 槽位
            category: 类别
            roles: [(unit_id, power, recommendation), ...] 已按战力降序排序
            auto_equip: 是否实际执行
        """
        pool = self.inventory_mgr.slot_category_pool.get((slot, category), {})
        if not pool:
            return

        unallocated_roles = []
        for unit_id, power, recommendation in roles:
            unallocated_roles.append((unit_id, recommendation))

        # Step1: 最优装备的rank2分配
        unallocated_roles = await self._step1_allocate_best_rank2(
            slot, category, unallocated_roles, pool, auto_equip
        )

        if not unallocated_roles:
            return

        # Step2: 1级替代的rank2分配
        unallocated_roles = await self._step2_allocate_alt1_rank2(
            slot, category, unallocated_roles, pool, auto_equip
        )

        if not unallocated_roles:
            return

        # Step3: 2级替代rank2 + 最优装备rank1/0混合分配
        unallocated_roles = await self._step3_allocate_alt2_and_best_rank10(
            slot, category, unallocated_roles, pool, auto_equip
        )

        if not unallocated_roles:
            return

        # Step4: 多种最优装备按比例分配
        await self._step4_allocate_multi_best_proportional(
            slot, category, unallocated_roles, pool, auto_equip
        )

    async def _step1_allocate_best_rank2(self, slot: int, category: int,
                                         roles: List[Tuple[int, dict]],
                                         pool: dict, auto_equip: bool) -> List[Tuple[int, dict]]:
        """
        Step1: 最优装备的rank2分配（考虑所有相同战力的最佳装备）

        Returns:
            未分配的角色列表
        """
        unallocated = []

        for unit_id, recommendation in roles:
            # 获取所有最佳装备（可能有多个相同战力）
            best_equips = recommendation['best_equips']
            
            # 尝试分配rank2装备
            allocated = False

            # 优先分配满强rank2
            for ex_id, _ in best_equips:
                if ex_id in pool and pool[ex_id]['rank2']:
                    items = pool[ex_id]['rank2']
                    for i, item in enumerate(items):
                        if item['pt'] >= ExEquipConstants.ENHANCEMENT_PT_MAX:
                            inst = items.pop(i)
                            await self._equip_to(unit_id, slot, inst, auto_equip)
                            allocated = True
                            break
                if allocated:
                    break

            # 然后分配未满强rank2
            if not allocated:
                for ex_id, _ in best_equips:
                    if ex_id in pool and pool[ex_id]['rank2']:
                        items = pool[ex_id]['rank2']
                        if items:
                            inst = items.pop(0)
                            await self._equip_to(unit_id, slot, inst, auto_equip)
                            allocated = True
                            break
            # 逻辑未优化好，暂时注释掉
            # # 尝试用rank1/0合成rank2
            # if not allocated:
            #     for ex_id, _ in best_equips:
            #         if self._can_synthesize_rank2(pool, ex_id):
            #             inst = self._synthesize_and_take_rank2(pool, ex_id)
            #             if inst:
            #                 await self._equip_to(unit_id, slot, inst, auto_equip)
            #                 allocated = True
            #                 break

            if not allocated:
                unallocated.append((unit_id, recommendation))

        return unallocated

    async def _step2_allocate_alt1_rank2(self, slot: int, category: int,
                                         roles: List[Tuple[int, dict]],
                                         pool: dict, auto_equip: bool) -> List[Tuple[int, dict]]:
        """
        Step2: 1级替代的rank2分配（考虑所有1级替代装备）

        Returns:
            未分配的角色列表
        """
        unallocated = []

        for unit_id, recommendation in roles:
            alt_level1 = recommendation['alt_level1']

            if not alt_level1:
                unallocated.append((unit_id, recommendation))
                continue

            allocated = False

            # 优先分配满强rank2
            for ex_id, power in alt_level1:
                if ex_id in pool and pool[ex_id]['rank2']:
                    items = pool[ex_id]['rank2']
                    for i, item in enumerate(items):
                        if item['pt'] >= ExEquipConstants.ENHANCEMENT_PT_MAX:
                            inst = items.pop(i)
                            await self._equip_to(unit_id, slot, inst, auto_equip)
                            allocated = True
                            break
                if allocated:
                    break

            # 然后分配未满强rank2
            if not allocated:
                for ex_id, power in alt_level1:
                    if ex_id in pool and pool[ex_id]['rank2']:
                        items = pool[ex_id]['rank2']
                        if items:
                            inst = items.pop(0)
                            await self._equip_to(unit_id, slot, inst, auto_equip)
                            allocated = True
                            break

            if not allocated:
                # 尝试用所有最优装备的rank1/0合成
                best_equips = recommendation['best_equips']
                for ex_id, _ in best_equips:
                    if self._can_synthesize_rank2(pool, ex_id):
                        inst = self._synthesize_and_take_rank2(pool, ex_id)
                        if inst:
                            await self._equip_to(unit_id, slot, inst, auto_equip)
                            allocated = True
                            break

            if not allocated:
                unallocated.append((unit_id, recommendation))

        return unallocated

    async def _step3_allocate_alt2_and_best_rank10(self, slot: int, category: int,
                                                   roles: List[Tuple[int, dict]],
                                                   pool: dict, auto_equip: bool) -> List[Tuple[int, dict]]:
        """
        Step3: 2级替代rank2 + 最优装备rank1/0混合分配（考虑所有最佳装备）

        Returns:
            未分配的角色列表
        """
        if not roles:
            return []

        # 检查是否有2级替代
        has_alt2 = any(recommendation['alt_level2'] for _, recommendation in roles)

        if not has_alt2:
            # 没有2级替代，直接用最优装备的rank1/0分配
            return await self._allocate_best_rank10_only(slot, category, roles, pool, auto_equip)

        # 计算所有最佳装备的rank1/0可以合成多少rank1
        best_equips = roles[0][1]['best_equips']
        best_rank1_available = 0
        for ex_id, _ in best_equips:
            best_rank1_available += self._count_synthesizable_rank1(pool, ex_id)

        # 统计2级替代的rank2数量（需匹配角色）
        alt2_rank2_count = 0
        for unit_id, recommendation in roles:
            for ex_id, power in recommendation['alt_level2']:
                if ex_id in pool and pool[ex_id]['rank2']:
                    alt2_rank2_count += len(pool[ex_id]['rank2'])
                    break  # 每个角色只计数一次

        total_available = best_rank1_available + alt2_rank2_count

        if total_available >= len(roles):
            # 足够：预留rank1，分配2级替代rank2，然后分配rank1/0
            return await self._allocate_alt2_with_best_rank1(
                slot, category, roles, pool, best_rank1_available, auto_equip
            )
        else:
            # 不够：全部分配rank1/0
            return await self._allocate_best_rank10_only(slot, category, roles, pool, auto_equip)

    async def _step4_allocate_multi_best_proportional(self, slot: int, category: int,
                                                     roles: List[Tuple[int, dict]],
                                                     pool: dict, auto_equip: bool) -> List[Tuple[int, dict]]:
        """
        Step4: 多种最优装备按比例分配（考虑所有相同战力的最佳装备）

        Returns:
            未分配的角色列表
        """
        if not roles:
            return []

        # 统计所有最优装备的点数（确保包含所有相同战力的装备）
        best_equips_points = {}
        for unit_id, recommendation in roles:
            for ex_id, power in recommendation['best_equips']:
                if ex_id not in best_equips_points:
                    best_equips_points[ex_id] = self._count_total_points(pool, ex_id)

        if not best_equips_points:
            return roles

        total_points = sum(best_equips_points.values())
        if total_points == 0:
            return roles

        # 按比例分配角色（按点数从高到低）
        unallocated = []
        role_index = 0

        sorted_equips = sorted(best_equips_points.items(), key=lambda x: -x[1])
        
        for i, (ex_id, points) in enumerate(sorted_equips):
            # 计算该装备应分配多少角色
            if i == len(sorted_equips) - 1:
                # 最后一个装备分配所有剩余角色
                num_roles = len(roles) - role_index
            else:
                num_roles = round(len(roles) * points / total_points)

            for _ in range(num_roles):
                if role_index >= len(roles):
                    break

                unit_id, recommendation = roles[role_index]
                role_index += 1

                # 尝试分配该装备的rank1/0
                allocated = await self._try_allocate_any_rank(unit_id, slot, ex_id, pool, auto_equip)

                if not allocated:
                    unallocated.append((unit_id, recommendation))

        # 处理剩余角色（如果有的话）
        while role_index < len(roles):
            unit_id, recommendation = roles[role_index]
            role_index += 1

            # 尝试任何可用的最优装备
            allocated = False
            for ex_id in best_equips_points.keys():
                allocated = await self._try_allocate_any_rank(unit_id, slot, ex_id, pool, auto_equip)
                if allocated:
                    break

            if not allocated:
                unallocated.append((unit_id, recommendation))

        return unallocated

    # ============ 辅助方法 ============

    def _can_synthesize_rank2(self, pool: dict, ex_id: int) -> bool:
        """判断是否可以用rank1/0合成rank2"""
        if ex_id not in pool:
            return False

        rank1_count = len(pool[ex_id]['rank1'])
        rank0_count = len(pool[ex_id]['rank0'])

        total_points = rank1_count * 2 + rank0_count
        return total_points >= 3

    def _synthesize_and_take_rank2(self, pool: dict, ex_id: int) -> dict:
        """
        合成并取出一个rank2装备（模拟）

        Returns:
            装备信息字典，如果无法合成则返回None
        """
        if not self._can_synthesize_rank2(pool, ex_id):
            return None

        # 贪心选择：优先使用rank1
        items_rank1 = pool[ex_id]['rank1']
        items_rank0 = pool[ex_id]['rank0']

        needed_points = 3
        used_items = []

        # 优先用rank1
        while needed_points > 0 and items_rank1:
            item = items_rank1.pop(0)
            used_items.append(item)
            needed_points -= 2

        # 然后用rank0
        while needed_points > 0 and items_rank0:
            item = items_rank0.pop(0)
            used_items.append(item)
            needed_points -= 1

        if needed_points > 0:
            # 合成失败，放回
            for item in used_items:
                if item in pool[ex_id]['rank1']:
                    items_rank1.append(item)
                else:
                    items_rank0.append(item)
            return None

        # 返回第一个item作为"合成结果"（模拟，pt设为0表示待合成）
        return {
            'pt': 0,
            'serial_id': used_items[0]['serial_id'],
            'is_clan': used_items[0]['is_clan'],
            'synthesized': True  # 标记为合成
        }

    def _count_synthesizable_rank1(self, pool: dict, ex_id: int) -> int:
        """计算可以合成多少个rank1"""
        if ex_id not in pool:
            return 0

        rank1_count = len(pool[ex_id]['rank1'])
        rank0_count = len(pool[ex_id]['rank0'])

        total_points = rank1_count * 2 + rank0_count
        return total_points // 2

    def _count_total_points(self, pool: dict, ex_id: int) -> int:
        """计算装备的总点数"""
        if ex_id not in pool:
            return 0

        rank2_count = len(pool[ex_id]['rank2'])
        rank1_count = len(pool[ex_id]['rank1'])
        rank0_count = len(pool[ex_id]['rank0'])

        return rank2_count * 3 + rank1_count * 2 + rank0_count

    async def _allocate_best_rank10_only(self, slot: int, category: int,
                                         roles: List[Tuple[int, dict]],
                                         pool: dict, auto_equip: bool) -> List[Tuple[int, dict]]:
        """只用最优装备的rank1/0分配（考虑所有最佳装备）"""
        unallocated = []

        for unit_id, recommendation in roles:
            best_equips = recommendation['best_equips']
            allocated = False
            
            # 尝试所有最佳装备
            for ex_id, _ in best_equips:
                allocated = await self._try_allocate_any_rank(unit_id, slot, ex_id, pool, auto_equip)
                if allocated:
                    break

            if not allocated:
                unallocated.append((unit_id, recommendation))

        return unallocated

    async def _allocate_alt2_with_best_rank1(self, slot: int, category: int,
                                            roles: List[Tuple[int, dict]],
                                            pool: dict, rank1_count: int,
                                            auto_equip: bool) -> List[Tuple[int, dict]]:
        """
        用2级替代rank2和最优装备rank1混合分配（考虑所有最佳装备）

        Args:
            rank1_count: 最优装备可合成的rank1数量
        """
        unallocated = []
        rank1_allocated = 0

        for unit_id, recommendation in roles:
            allocated = False

            # 先尝试分配所有最优装备的rank1
            if rank1_allocated < rank1_count:
                best_equips = recommendation['best_equips']
                for ex_id, _ in best_equips:
                    if ex_id in pool and pool[ex_id]['rank1']:
                        inst = pool[ex_id]['rank1'].pop(0)
                        await self._equip_to(unit_id, slot, inst, auto_equip)
                        rank1_allocated += 1
                        allocated = True
                        break

            # 然后尝试2级替代的rank2
            if not allocated:
                for ex_id, power in recommendation['alt_level2']:
                    if ex_id in pool and pool[ex_id]['rank2']:
                        inst = pool[ex_id]['rank2'].pop(0)
                        await self._equip_to(unit_id, slot, inst, auto_equip)
                        allocated = True
                        break

            # 最后尝试所有最优装备的rank0
            if not allocated:
                best_equips = recommendation['best_equips']
                for ex_id, _ in best_equips:
                    if ex_id in pool and pool[ex_id]['rank0']:
                        inst = pool[ex_id]['rank0'].pop(0)
                        await self._equip_to(unit_id, slot, inst, auto_equip)
                        allocated = True
                        break

            if not allocated:
                unallocated.append((unit_id, recommendation))

        return unallocated

    async def _try_allocate_any_rank(self, unit_id: int, slot: int, ex_id: int,
                                     pool: dict, auto_equip: bool) -> bool:
        """尝试分配任何rank的装备"""
        if ex_id not in pool:
            return False

        # 按优先级尝试：rank1 > rank0
        for rank_key in ['rank1', 'rank0']:
            if pool[ex_id][rank_key]:
                inst = pool[ex_id][rank_key].pop(0)
                await self._equip_to(unit_id, slot, inst, auto_equip)
                return True

        return False

    async def _equip_to(self, unit_id: int, slot: int, inst: dict, auto_equip: bool) -> bool:
        """
        实际装备或记录

        Args:
            inst: 装备信息字典，包含 pt, serial_id, is_clan 等字段
        """
        # 记录分配结果
        ex = self.client.data.ex_equips.get(inst['serial_id'])
        if ex:
            self.allocation_results[(unit_id, slot)] = {
                'ex_id': ex.ex_equipment_id,
                'rank': ex.rank,
                'pt': inst['pt'],
                'serial_id': inst['serial_id'],
                'is_synthesized': inst.get('synthesized', False)
            }

        if not auto_equip:
            return True

        try:
            await self.client.unit_equip_ex([ExtraEquipChangeUnit(
                unit_id=unit_id,
                ex_equip_slot=[ExtraEquipChangeSlot(slot=slot, serial_id=inst['serial_id'])],
                cb_ex_equip_slot=None
            )])
            return True
        except Exception as e:
            self.module._warn(f"装备失败: unit={unit_id}, slot={slot}, error={e}")
            return False

    def _output_report(self, auto_equip: bool):
        """输出装备分配报告"""
        if auto_equip:
            self.module._log(f"\n已为{len(set(uid for uid, _ in self.allocation_results.keys()))}个角色装备了{len(self.allocation_results)}件EX装备")
        else:
            self.module._log(f"\n模拟为{len(set(uid for uid, _ in self.allocation_results.keys()))}个角色装备{len(self.allocation_results)}件EX装备")

        # 按槽位/类别输出详细报告
        self._output_detailed_report(auto_equip)

    def _output_detailed_report(self, auto_equip: bool):
        """输出详细的分配报告"""
        # 按(slot, category)分组统计
        slot_category_stats = defaultdict(lambda: {
            'units': set(),
            'equip_counts': defaultdict(lambda: {'full': 0, 'r2': 0, 'r1': 0, 'r0': 0}),
            'to_enhance': defaultdict(lambda: {'r2': [], 'r1': [], 'r0': []})
        })

        for (unit_id, slot), alloc in self.allocation_results.items():
            slot_data = db.unit_ex_equipment_slot.get(unit_id)
            if not slot_data:
                continue

            category = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot - 1]
            if category is None:
                continue

            key = (slot, category)
            stats = slot_category_stats[key]
            stats['units'].add(unit_id)

            ex_name = db.ex_equipment_data[alloc['ex_id']].name
            rank = alloc['rank']
            pt = alloc['pt']

            # 统计数量
            if rank >= ExEquipConstants.RANK_MAX and pt >= ExEquipConstants.ENHANCEMENT_PT_MAX:
                stats['equip_counts'][ex_name]['full'] += 1
            elif rank == ExEquipConstants.RANK_MAX:
                stats['equip_counts'][ex_name]['r2'] += 1
                stats['to_enhance'][ex_name]['r2'].append(db.get_unit_name(unit_id))
            elif rank == ExEquipConstants.RANK_ONE:
                stats['equip_counts'][ex_name]['r1'] += 1
                stats['to_enhance'][ex_name]['r1'].append(db.get_unit_name(unit_id))
            else:
                stats['equip_counts'][ex_name]['r0'] += 1
                stats['to_enhance'][ex_name]['r0'].append(db.get_unit_name(unit_id))

        # 输出报告
        self.module._log("\n装备分配汇总（按槽位/类别）:\n")

        for (slot, category), stats in sorted(slot_category_stats.items()):
            self.module._log(f"槽位{slot}/类别{category}: {len(stats['units'])}个角色")

            # 输出库存信息（使用分配前的快照）
            snapshot = self.inventory_snapshots.get((slot, category), {})
            inventory_parts = []
            for ex_id, counts in snapshot.items():
                ex_name = db.ex_equipment_data[ex_id].name
                r2_total = counts['rank2']
                r2_full = counts['rank2_full']
                r1_total = counts['rank1']
                r1_full = counts['rank1_full']
                r0_total = counts['rank0']
                r0_full = counts['rank0_full']

                inventory_parts.append(f"{ex_name}: {r2_total}({r2_full})-{r1_total}({r1_full})-{r0_total}({r0_full})")

            if inventory_parts:
                self.module._log(f"**库存**（分配前）：{'; '.join(inventory_parts)}")

            # 输出分配信息（按类型汇总）
            allocation_by_type = {'full': [], 'r2': [], 'r1': [], 'r0': []}

            for ex_name, counts in stats['equip_counts'].items():
                if counts['full'] > 0:
                    allocation_by_type['full'].append(f"{counts['full']}件: {ex_name}")
                if counts['r2'] > 0:
                    allocation_by_type['r2'].append(f"{counts['r2']}件: {ex_name}")
                if counts['r1'] > 0:
                    allocation_by_type['r1'].append(f"{counts['r1']}件: {ex_name}")
                if counts['r0'] > 0:
                    allocation_by_type['r0'].append(f"{counts['r0']}件: {ex_name}")

            allocation_parts = []
            if allocation_by_type['full']:
                allocation_parts.append(f"满强: {', '.join(allocation_by_type['full'])}")
            if allocation_by_type['r2']:
                allocation_parts.append(f"待强化r2: {', '.join(allocation_by_type['r2'])}")
            if allocation_by_type['r1']:
                allocation_parts.append(f"待强化r1: {', '.join(allocation_by_type['r1'])}")
            if allocation_by_type['r0']:
                allocation_parts.append(f"待强化r0: {', '.join(allocation_by_type['r0'])}")

            if allocation_parts:
                self.module._log(f"**分配**: {'; '.join(allocation_parts)}")

            # 输出待强化清单
            to_enhance_parts = []
            for ex_name, ranks in stats['to_enhance'].items():
                for rank_key in ['r2', 'r1', 'r0']:
                    if ranks[rank_key]:
                        unit_names = '、'.join(ranks[rank_key][:10])  # 最多显示10个
                        if len(ranks[rank_key]) > 10:
                            unit_names += f"等{len(ranks[rank_key])}个"
                        to_enhance_parts.append(f"{rank_key}: {ex_name}: {unit_names}")

            if to_enhance_parts:
                self.module._log(f"**待强化**：{' | '.join(to_enhance_parts)}")

            self.module._log("")  # 空行分隔
