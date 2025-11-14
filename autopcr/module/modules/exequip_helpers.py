"""
EX装备相关的辅助类
用于支持 calc_best_3star_ex_equip 模块
"""
from dataclasses import dataclass
from typing import Dict, Set, Tuple
from ...db.database import db
from ...model.enums import eInventoryType
from ...model.common import ExtraEquipChangeSlot, ExtraEquipChangeUnit


class ExEquipConstants:
    """EX装备相关常量定义"""

    # 品质等级
    RARITY_SILVER = 2    # 银色（2星）
    RARITY_GOLD = 3      # 金色（3星）
    RARITY_PINK = 4      # 粉色（4星）

    # 突破等级
    RANK_ZERO = 0        # 未突破
    RANK_ONE = 1         # 1突破
    RANK_MAX = 2         # 最大突破（2突破）

    # 强化相关
    ENHANCEMENT_PT_MAX = 6000  # 满强化点数
    ENHANCE_LEVEL_MAX = 5      # 最大强化等级（5强）

    # 保护标志
    PROTECTION_UNLOCKED = 1   # 未锁定
    PROTECTION_LOCKED = 2     # 已锁定

    # 点数相关
    POINTS_PER_EQUIP = 3      # 每件装备满状态的点数


@dataclass
class SlotCategoryInfo:
    """槽位类别信息数据结构"""
    per_sc_roles: Dict[Tuple[int, int], Set[int]]  # {(slot, cat): set(unit_id)}
    per_sc_bestname_counts: Dict[Tuple[int, int], Dict[str, int]]  # {(slot, cat): {name: count}}
    per_sc_role_bestnames: Dict[Tuple[int, int], Dict[int, Set[str]]]  # {(slot, cat): {unit_id: set(names)}}
    role_best_power: Dict[Tuple[int, int], int]  # {(unit_id, slot): max_power}
    ref_role_by_sc: Dict[Tuple[int, int], int]  # {(slot, cat): unit_id}


class ExEquipPowerCalculator:
    """战力计算器：负责计算EX装备对角色战力的影响"""
    
    def __init__(self, client):
        self.client = client
        self.coeff = db.unit_status_coefficient[1]
        self._attr_mapping = [
            ('hp', 'max_hp'), ('atk', 'max_atk'), ('def_', 'max_def'), 
            ('magic_str', 'max_magic_str'), ('magic_def', 'max_magic_def'), 
            ('physical_critical', 'max_physical_critical'), ('magic_critical', 'max_magic_critical'),
            ('dodge', 'max_dodge'), ('accuracy', 'max_accuracy'), 
            ('energy_recovery_rate', 'max_energy_recovery_rate'),
            ('hp_recovery_rate', 'max_hp_recovery_rate'), ('energy_reduce_rate', 'max_energy_reduce_rate')
        ]
    
    def calculate_power_increase_at_level(self, unit_id: int, slot_index: int, ex_id: int, level: int) -> int:
        """
        计算指定强化等级的战力提升
        
        Args:
            unit_id: 角色ID
            slot_index: 槽位索引（1-3）
            ex_id: EX装备ID
            level: 强化等级（0-5）
        
        Returns:
            战力提升值
        """
        frac = 0 if level <= 0 else (1.0 if level >= 5 else level / 5.0)
        return self.calculate_power_increase_at_fraction(unit_id, slot_index, ex_id, frac)
    
    def calculate_power_increase_at_fraction(self, unit_id: int, slot_index: int, ex_id: int, frac: float) -> int:
        """
        按分数计算战力提升（支持线性插值）
        
        Args:
            unit_id: 角色ID
            slot_index: 槽位索引（未使用，保留用于扩展）
            ex_id: EX装备ID
            frac: 强化分数（0.0-1.0），例如0.8表示4强（4/5）
        
        Returns:
            战力提升值
        """
        unit = self.client.data.unit[unit_id]
        base_attr = db.calc_unit_attribute(unit, self.client.data.read_story_ids)
        ex_data = db.ex_equipment_data[ex_id]
        
        # 计算基础战力
        base_power = self._calc_power_from_attributes({
            'hp': base_attr.hp, 'atk': base_attr.atk, 'def_': base_attr.def_, 
            'magic_str': base_attr.magic_str, 'magic_def': base_attr.magic_def,
            'physical_critical': base_attr.physical_critical, 'magic_critical': base_attr.magic_critical, 
            'dodge': base_attr.dodge, 'accuracy': base_attr.accuracy,
            'energy_recovery_rate': base_attr.energy_recovery_rate, 
            'hp_recovery_rate': base_attr.hp_recovery_rate, 'energy_reduce_rate': base_attr.energy_reduce_rate
        })
        
        # 应用EX装备属性增强
        enhanced_attr = self._apply_ex_attributes(base_attr, ex_data, frac)
        
        # 计算增强后战力
        enhanced_power = self._calc_power_from_attributes(enhanced_attr)
        
        return int(enhanced_power - base_power)
    
    def _apply_ex_attributes(self, base_attr, ex_data, frac: float) -> dict:
        """
        应用EX装备属性到基础属性
        
        Args:
            base_attr: 基础属性对象
            ex_data: EX装备数据
            frac: 强化分数（0.0-1.0）
        
        Returns:
            增强后的属性字典
        """
        enhanced = {
            'hp': base_attr.hp, 'atk': base_attr.atk, 'def_': base_attr.def_, 
            'magic_str': base_attr.magic_str, 'magic_def': base_attr.magic_def,
            'physical_critical': base_attr.physical_critical, 'magic_critical': base_attr.magic_critical, 
            'dodge': base_attr.dodge, 'accuracy': base_attr.accuracy,
            'energy_recovery_rate': base_attr.energy_recovery_rate, 
            'hp_recovery_rate': base_attr.hp_recovery_rate, 'energy_reduce_rate': base_attr.energy_reduce_rate
        }
        
        for dst, mx in self._attr_mapping:
            max_val = getattr(ex_data, mx, 0)
            default_val = getattr(ex_data, mx.replace('max_', 'default_'), 0)
            if max_val or default_val:
                delta = default_val + (max_val - default_val) * float(frac)
                base_value = enhanced[dst]
                if max_val % 100 == 0 and max_val >= 100:
                    # 百分比属性
                    enhanced[dst] = float(base_value) * (1 + delta / 10000)
                else:
                    # 固定数值属性
                    enhanced[dst] = float(base_value) + delta
        
        return enhanced
    
    def _calc_power_from_attributes(self, attributes: dict) -> float:
        """
        根据属性字典计算总战力
        
        Args:
            attributes: 属性字典，包含所有角色属性
        
        Returns:
            总战力值
        """
        return (
            float(attributes['hp']) * float(self.coeff.hp_coefficient) +
            float(attributes['atk']) * float(self.coeff.atk_coefficient) +
            float(attributes['def_']) * float(self.coeff.def_coefficient) +
            float(attributes['magic_str']) * float(self.coeff.magic_str_coefficient) +
            float(attributes['magic_def']) * float(self.coeff.magic_def_coefficient) +
            float(attributes['physical_critical']) * float(self.coeff.physical_critical_coefficient) +
            float(attributes['magic_critical']) * float(self.coeff.magic_critical_coefficient) +
            float(attributes['dodge']) * float(self.coeff.dodge_coefficient) +
            float(attributes['accuracy']) * float(self.coeff.accuracy_coefficient) +
            float(attributes['energy_recovery_rate']) * float(self.coeff.energy_recovery_rate_coefficient) +
            float(attributes['hp_recovery_rate']) * float(self.coeff.hp_recovery_rate_coefficient) +
            float(attributes['energy_reduce_rate']) * float(self.coeff.energy_reduce_rate_coefficient)
        )


class ExEquipStrategyChecker:
    """策略检查器：负责检查特殊装备策略（5B>4A等）"""
    
    def __init__(self, power_calculator: ExEquipPowerCalculator):
        self.calculator = power_calculator
    
    def check_alternative_better(self, ref_uid: int, slot: int, best_ex_id: int, alt_ex_id: int, 
                                  best_level: int, alt_level: int) -> bool:
        """
        通用的XB>YA检查：检查alt装备在alt_level强化是否优于best装备在best_level强化
        
        Args:
            ref_uid: 参考角色ID
            slot: 槽位
            best_ex_id: 最优装备ID
            alt_ex_id: 备选装备ID
            best_level: 最优装备的比较强化等级
            alt_level: 备选装备的比较强化等级
        
        Returns:
            True if alt_level强的alt装备 > best_level强的best装备
        """
        try:
            best_power = self.calculator.calculate_power_increase_at_level(ref_uid, slot, best_ex_id, best_level)
            alt_power = self.calculator.calculate_power_increase_at_level(ref_uid, slot, alt_ex_id, alt_level)
            return alt_power > best_power
        except Exception:
            return False
    
    def check_5b4a(self, ref_uid: int, slot: int, best_ex_id: int, alt_ex_id: int) -> bool:
        """检查5B>4A：5强非最优 > 4强最优"""
        return self.check_alternative_better(ref_uid, slot, best_ex_id, alt_ex_id, 4, 5)
    
    def check_5b3a(self, ref_uid: int, slot: int, best_ex_id: int, alt_ex_id: int) -> bool:
        """检查5B>3A：5强非最优 > 3强最优（未来扩展）"""
        return self.check_alternative_better(ref_uid, slot, best_ex_id, alt_ex_id, 3, 5)


class ExEquipInventoryManager:
    """库存管理器：负责库存池的构建和管理"""

    def __init__(self, client):
        self.client = client
        self.ex_pool_by_id = {}  # 保留兼容旧代码
        self.cat_name_pool = {}  # 保留兼容旧代码
        self.slot_category_pool = {}  # 新结构：{(slot, cat): {ex_id: {'rank2': [...], 'rank1': [...], 'rank0': [...]}}}
        self.locked_slots = {}  # 锁定的槽位：{(unit_id, slot): serial_id}

    @staticmethod
    def is_full_enhanced(inst) -> bool:
        """
        判断装备是否满强化（rank≥2且强化点≥6000）

        Args:
            inst: 装备实例

        Returns:
            是否满强化
        """
        return inst.rank >= ExEquipConstants.RANK_MAX and inst.enhancement_pt >= ExEquipConstants.ENHANCEMENT_PT_MAX

    @staticmethod
    def is_not_full_enhanced(inst) -> bool:
        """
        判断装备是否未满强化

        Args:
            inst: 装备实例

        Returns:
            是否未满强化
        """
        return inst.enhancement_pt < ExEquipConstants.ENHANCEMENT_PT_MAX or inst.rank < ExEquipConstants.RANK_MAX

    def build_slot_category_pools(self, unit_slot_recommendations: dict):
        """
        构建按槽位/类别组织的库存池（新方法）

        Args:
            unit_slot_recommendations: 推荐结果，用于确定需要哪些槽位/类别

        Returns:
            None (结果存储在 self.slot_category_pool 和 self.locked_slots 中)
        """
        # 步骤1: 统计锁定的槽位
        self.locked_slots = {}
        for unit_id, unit in self.client.data.unit.items():
            for slot_idx in range(3):
                ex_slot = unit.ex_equip_slot[slot_idx]
                if ex_slot.serial_id != 0:
                    ex = self.client.data.ex_equips.get(ex_slot.serial_id)
                    # if ex and ex.protection_flag == ExEquipConstants.PROTECTION_LOCKED:
                    rarity = db.get_ex_equip_rarity(ex.ex_equipment_id)
                    if rarity == ExEquipConstants.RARITY_PINK:
                        # 锁定的装备
                        self.locked_slots[(unit_id, slot_idx + 1)] = ex_slot.serial_id

        # 步骤2: 收集需要的(slot, category)组合
        slot_categories = set()
        for (unit_id, slot), recommendation in unit_slot_recommendations.items():
            slot_data = db.unit_ex_equipment_slot.get(unit_id)
            if slot_data:
                category = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot - 1]
                if category is not None:
                    slot_categories.add((slot, category))

        # 步骤3: 按(slot, category)组织库存
        self.slot_category_pool = {}

        for slot, category in slot_categories:
            pool = {}

            # 找出所有该类别的3星装备
            for ex in self.client.data.ex_equips.values():
                ex_id = ex.ex_equipment_id
                if db.get_ex_equip_rarity(ex_id) != ExEquipConstants.RARITY_GOLD:
                    continue

                ex_data = db.ex_equipment_data[ex_id]
                if ex_data.category != category:
                    continue

                # 排除锁定的装备（已装备在某个角色的locked槽位上）
                if any(serial_id == ex.serial_id for serial_id in self.locked_slots.values()):
                    continue

                # 按rank分类
                rank_key = f'rank{ex.rank}'
                if ex_id not in pool:
                    pool[ex_id] = {'rank2': [], 'rank1': [], 'rank0': []}

                pool[ex_id][rank_key].append({
                    'pt': ex.enhancement_pt,
                    'serial_id': ex.serial_id,
                    'is_clan': db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex_id))
                })

            # 对每个装备的每个rank列表按pt降序、非会战优先排序
            for ex_id in pool:
                for rank_key in ['rank2', 'rank1', 'rank0']:
                    pool[ex_id][rank_key].sort(key=lambda x: (-x['pt'], x['is_clan']))

            self.slot_category_pool[(slot, category)] = pool

    def build_inventory_pools(self, to_remove_serials=None):
        """
        构建库存池
        
        Args:
            to_remove_serials: 将被撤下的serial_id集合（模拟模式下会被视为可用）
        
        Returns:
            None (结果存储在 self.ex_pool_by_id 和 self.cat_name_pool 中)
        """
        if to_remove_serials is None:
            to_remove_serials = set()
        
        # 获取已装备的EX
        use_ex_equip = set(
            ex_slot.serial_id
            for u in self.client.data.unit.values()
            for ex_slot in u.ex_equip_slot
            if ex_slot.serial_id != 0
        )

        # 按ex_id分组
        self.ex_pool_by_id = {}
        for ex in self.client.data.ex_equips.values():
            # 模拟时：把将被撤下的也当作可用
            if ex.serial_id in use_ex_equip and ex.serial_id not in to_remove_serials:
                continue
            if db.get_ex_equip_rarity(ex.ex_equipment_id) != ExEquipConstants.RARITY_GOLD:
                continue
            self.ex_pool_by_id.setdefault(ex.ex_equipment_id, []).append(ex)

        # 排序：优先满强，再按是否会战【行会】，最后按rank/pt降序
        for ex_id in self.ex_pool_by_id:
            self.ex_pool_by_id[ex_id].sort(key=lambda e: (
                -(1 if self.is_full_enhanced(e) else 0),
                0 if db.is_clan_ex_equip((eInventoryType.ExtraEquip, e.ex_equipment_id)) else -1,
                -e.rank,
                -e.enhancement_pt
            ))
        
        # 按类别-名称汇总
        self.cat_name_pool = {}
        for ex_id, lst in self.ex_pool_by_id.items():
            cat = db.ex_equipment_data[ex_id].category
            name = db.ex_equipment_data[ex_id].name
            self.cat_name_pool.setdefault(cat, {}).setdefault(name, []).extend(lst)
    
    def pop_best_instance(self, cat: int, name: str, prefer_full=True, prefer_non_guild_first=True):
        """
        从池中取出最优实例

        Args:
            cat: 类别
            name: 装备名称
            prefer_full: 是否优先满强
            prefer_non_guild_first: 是否优先非【行会】

        Returns:
            EX装备实例，如果没有则返回None
        """
        lst = self.cat_name_pool.get(cat, {}).get(name, [])
        if not lst:
            return None

        def key(inst):
            full = 1 if inst.enhancement_pt >= ExEquipConstants.ENHANCEMENT_PT_MAX else 0
            guild = 1 if db.is_clan_ex_equip((eInventoryType.ExtraEquip, inst.ex_equipment_id)) else 0
            return (-full, guild if prefer_non_guild_first else 0, -inst.rank, -inst.enhancement_pt)

        lst.sort(key=key)
        return lst.pop(0)
    
    def remove_instance(self, inst):
        """从池中移除指定实例"""
        try:
            cat = db.ex_equipment_data[inst.ex_equipment_id].category
            name = db.ex_equipment_data[inst.ex_equipment_id].name
            self.cat_name_pool[cat][name].remove(inst)
        except (KeyError, ValueError):
            pass
    
    def get_snapshot(self, cat: int) -> dict:
        """
        获取类别库存快照

        Args:
            cat: 类别

        Returns:
            {name: (full_cnt, r2_cnt, r1_cnt, r0_cnt, ex_id_example)}
        """
        snap = {}
        for nm, inst_list in self.cat_name_pool.get(cat, {}).items():
            full_cnt = sum(1 for inst in inst_list if self.is_full_enhanced(inst))
            r2_cnt = sum(1 for inst in inst_list if (inst.rank >= ExEquipConstants.RANK_MAX and inst.enhancement_pt < ExEquipConstants.ENHANCEMENT_PT_MAX))
            r1_cnt = sum(1 for inst in inst_list if inst.rank == ExEquipConstants.RANK_ONE)
            r0_cnt = sum(1 for inst in inst_list if inst.rank == ExEquipConstants.RANK_ZERO)
            ex_id_example = inst_list[0].ex_equipment_id if inst_list else None
            snap[nm] = (full_cnt, r2_cnt, r1_cnt, r0_cnt, ex_id_example)
        return snap
    
    @staticmethod
    def instance_points(inst) -> int:
        """计算装备实例的点数：rank 0→1, 1→2, 2→3"""
        return min(int(inst.rank) + 1, ExEquipConstants.POINTS_PER_EQUIP)


class ExEquipRecommender:
    """推荐器：负责计算每个角色的最佳装备推荐"""

    def __init__(self, client, power_calculator: ExEquipPowerCalculator):
        self.client = client
        self.calculator = power_calculator

    def calculate_recommendations(self):
        """
        计算所有角色的装备推荐

        Returns:
            {
                'unit_slot_recommendations': {
                    (unit_id, slot): {
                        'best_equips': [(ex_id, power_5rank2)],
                        'chosen_best': ex_id,
                        'best_power_5rank2': int,
                        'best_power_4rank1': int,
                        'best_power_3rank0': int,
                        'alt_level1': [(ex_id, power_5rank2)],
                        'alt_level2': [(ex_id, power_5rank2)],
                    }
                }
            }
        """
        unit_slot_recommendations = {}
        total_best_power_increase = 0
        
        for unit_id in self.client.data.unit:
            # 计算每个槽位的最佳EX装备（只处理普通EX槽位1-3）
            for slot in range(3):
                slot_index = slot + 1
                recommendation = self._calculate_slot_recommendation(unit_id, slot_index)

                if not recommendation:
                    continue

                unit_slot_recommendations[(unit_id, slot_index)] = recommendation
                total_best_power_increase += recommendation['best_power_5rank2']

        return {
            'unit_slot_recommendations': unit_slot_recommendations,
            'total_best_power_increase': total_best_power_increase
        }
    
    def _calculate_slot_recommendation(self, unit_id: int, slot_index: int):
        """
        计算单个槽位的装备推荐，包括最佳装备和替代装备

        Args:
            unit_id: 角色ID
            slot_index: 槽位索引（1-3）

        Returns:
            推荐结果字典，如果无可用装备则返回None
        """
        # 获取该槽位可用的EX装备
        available_ex_equips = []
        if unit_id in db.unit_ex_equipment_slot:
            slot_data = db.unit_ex_equipment_slot[unit_id]
            category = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot_index - 1]

            if category is not None:
                available_ex_equips = [
                    ex_id for ex_id, ex_data in db.ex_equipment_data.items()
                    if ex_data.category == category and db.get_ex_equip_rarity(ex_id) == ExEquipConstants.RARITY_GOLD
                ]

        if not available_ex_equips:
            return None

        # 计算所有装备在5强/rank2的战力
        equip_powers_5rank2 = []
        for ex_id in available_ex_equips:
            power = self.calculator.calculate_power_increase_at_level(unit_id, slot_index, ex_id, ExEquipConstants.ENHANCE_LEVEL_MAX)
            equip_powers_5rank2.append((ex_id, int(power)))

        # 按战力降序排序
        equip_powers_5rank2.sort(key=lambda x: x[1], reverse=True)

        # 找出最佳装备（可能有多个并列）
        max_power = equip_powers_5rank2[0][1]
        best_equips = [(ex_id, power) for ex_id, power in equip_powers_5rank2 if power == max_power]
        chosen_best = best_equips[0][0]  # 任选一个

        # 计算最佳装备在不同强化等级的战力
        best_power_5rank2 = max_power
        # 4强/rank1满强：level=4, 对应frac=0.8
        best_power_4rank1 = int(self.calculator.calculate_power_increase_at_fraction(unit_id, slot_index, chosen_best, 0.8))
        # 3强/rank0满强：level=3, 对应frac=0.6
        best_power_3rank0 = int(self.calculator.calculate_power_increase_at_fraction(unit_id, slot_index, chosen_best, 0.6))

        # 计算替代装备
        alt_level1 = []  # 1级替代：满强非最佳 > 4强最佳
        alt_level2 = []  # 2级替代：4强最佳 > 满强非最佳 > 3强最佳

        for ex_id, power_5rank2 in equip_powers_5rank2:
            if power_5rank2 == max_power:  # 跳过最佳装备
                continue

            if power_5rank2 > best_power_4rank1:
                # 1级替代
                alt_level1.append((ex_id, power_5rank2))
            elif power_5rank2 > best_power_3rank0:
                # 2级替代
                alt_level2.append((ex_id, power_5rank2))

        return {
            'best_equips': best_equips,
            'chosen_best': chosen_best,
            'best_power_5rank2': best_power_5rank2,
            'best_power_4rank1': best_power_4rank1,
            'best_power_3rank0': best_power_3rank0,
            'alt_level1': alt_level1,
            'alt_level2': alt_level2,
        }
    
