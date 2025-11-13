from ...util.linq import flow
from ...model.common import ExtraEquipChangeSlot, ExtraEquipChangeUnit, InventoryInfoPost
from ..modulebase import *
from ..config import *
from ...core.pcrclient import pcrclient
from ...model.error import *
from ...db.database import db
from ...model.enums import *
from collections import Counter

@name('强化EX装')
@default(True)
@booltype('ex_equip_enhance_view', '看消耗', True)
@inttype('ex_equip_enhance_max_num', '满强化个数', 5, list(range(-1, 51)))
@MultiChoiceConfig('ex_equip_enhance_up_kind', '强化种类', [], ['粉', '会战金', '普通金', '会战银'])
@description('仅使用强化PT强化至当前突破满星，不考虑突破。强化个数指同类满强化EX装超过阈值则不强化，-1表示不限制，看消耗指观察消耗资源情况，实际不执行强化')
class ex_equip_enhance_up(Module):
    async def do_task(self, client: pcrclient):
        use_ex_equip = {ex_slot.serial_id: [unit.id, frame, ex_slot.slot]
                        for unit in client.data.unit.values() 
                        for frame, ex_slots in enumerate([unit.ex_equip_slot, unit.cb_ex_equip_slot], start=1)
                        for ex_slot in ex_slots if ex_slot.serial_id != 0}

        ex_equip_enhance_up_kind = self.get_config('ex_equip_enhance_up_kind')
        ex_equip_enhance_max_nun = self.get_config('ex_equip_enhance_max_num')
        ex_equip_enhance_view = self.get_config('ex_equip_enhance_view')

        consider_ex_equips = flow(client.data.ex_equips.values()) \
                .where(lambda ex: 
                       '粉' in ex_equip_enhance_up_kind and db.get_ex_equip_rarity(ex.ex_equipment_id) == 4 \
                or '会战金' in ex_equip_enhance_up_kind and db.get_ex_equip_rarity(ex.ex_equipment_id) == 3 and db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex.ex_equipment_id)) \
                or '普通金' in ex_equip_enhance_up_kind and db.get_ex_equip_rarity(ex.ex_equipment_id) == 3 and not db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex.ex_equipment_id)) \
                or '会战银' in ex_equip_enhance_up_kind and db.get_ex_equip_rarity(ex.ex_equipment_id) == 2 and db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex.ex_equipment_id))) \
                .to_list()

        consider_ex_equips.sort(key=lambda ex: (ex.rank, ex.ex_equipment_id), reverse=True)

        ex_equips_max_star_cnt = Counter()

        enhanceup_equip_cnt = 0
        consume_equip_pt = 0
        cost_mana = 0
        for equip in consider_ex_equips:
            max_star = db.get_ex_equip_max_star(equip.ex_equipment_id, equip.rank)
            cur_star = db.get_ex_equip_star_from_pt(equip.ex_equipment_id, equip.enhancement_pt)
            if cur_star >= max_star:
                ex_equips_max_star_cnt[equip.ex_equipment_id] += 1
                continue
            if ex_equip_enhance_max_nun != -1 and ex_equips_max_star_cnt[equip.ex_equipment_id] >= ex_equip_enhance_max_nun:
                continue
            demand_pt = db.get_ex_equip_enhance_pt(equip.ex_equipment_id, equip.enhancement_pt, max_star)
            if not ex_equip_enhance_view and client.data.get_inventory(db.ex_pt) < demand_pt:
                self._log(f"强化PT不足{demand_pt}，无法强化EX装")
                break

            demand_mana = db.get_ex_equip_enhance_mana(equip.ex_equipment_id, equip.enhancement_pt, max_star)
            if not ex_equip_enhance_view and not await client.prepare_mana(demand_mana):
                self._log(f"mana数不足{demand_mana}，无法强化EX装")
                break

            enhanceup_equip_cnt += 1
            consume_equip_pt += demand_pt
            cost_mana += demand_mana

            ex_equips_max_star_cnt[equip.ex_equipment_id] += 1

            unit_id, frame, slot = use_ex_equip.get(equip.serial_id, [0, 0, 0])
            if ex_equip_enhance_view:
                continue
            await client.equipment_enhance_ex(
                unit_id=unit_id,
                serial_id=equip.serial_id,
                frame=frame,
                slot=slot,
                before_enhancement_pt=equip.enhancement_pt,
                after_enhancement_pt=equip.enhancement_pt + demand_pt,
                consume_gold=demand_mana,
                from_view=2,
                item_list=[
                    InventoryInfoPost(type = db.ex_pt[0], id = db.ex_pt[1], count = demand_pt)
                ],
                consume_ex_serial_id_list=[],
            )

        if enhanceup_equip_cnt:
            if ex_equip_enhance_view:
                self._log(f"需消耗{consume_equip_pt}强化PT和{cost_mana} mana来强化{enhanceup_equip_cnt}个EX装")
                self._log(f"当前强化PT {client.data.get_inventory(db.ex_pt)}, mana {client.data.get_inventory(db.mana)}")
            else:
                self._log(f"消耗了{consume_equip_pt}强化PT和{cost_mana}mana，强化了{enhanceup_equip_cnt}个EX装")
        else:
            raise SkipError("没有可强化的EX装")

@name('合成EX装')
@default(True)
@inttype('ex_equip_rank_max_num', '满突破个数', 5, list(range(-1, 51)))
@MultiChoiceConfig('ex_equip_rank_up_kind', '合成种类', [], ['粉', '会战金', '普通金', '会战银'])
@description('合成忽略已装备和锁定的EX装，满突破个数指同类满突破EX装超过阈值则不突破，-1表示不限制')
class ex_equip_rank_up(Module):
    async def do_task(self, client: pcrclient):
        use_ex_equip = {ex_slot.serial_id: [unit.id, frame, ex_slot.slot]
                        for unit in client.data.unit.values() 
                        for frame, ex_slots in enumerate([unit.ex_equip_slot, unit.cb_ex_equip_slot], start=1)
                        for ex_slot in ex_slots if ex_slot.serial_id != 0}

        ex_equip_rank_up_kind = self.get_config('ex_equip_rank_up_kind')
        ex_equip_rank_max_num = self.get_config('ex_equip_rank_max_num')

        consider_ex_equips = flow(client.data.ex_equips.values()) \
                .where(lambda ex: ex.protection_flag != 2) \
                .where(lambda ex: 
                       '粉' in ex_equip_rank_up_kind and db.get_ex_equip_rarity(ex.ex_equipment_id) == 4 \
                or '会战金' in ex_equip_rank_up_kind and db.get_ex_equip_rarity(ex.ex_equipment_id) == 3 and db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex.ex_equipment_id)) \
                or '普通金' in ex_equip_rank_up_kind and db.get_ex_equip_rarity(ex.ex_equipment_id) == 3 and not db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex.ex_equipment_id)) \
                or '会战银' in ex_equip_rank_up_kind and db.get_ex_equip_rarity(ex.ex_equipment_id) == 2 and db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex.ex_equipment_id))) \
                .to_list()

        zero_rank_ex = flow(consider_ex_equips) \
                    .where(lambda ex: ex.serial_id not in use_ex_equip and ex.rank == 0) \
                    .group_by(lambda ex: ex.ex_equipment_id) \
                    .to_dict(lambda ex: ex.key, lambda ex: ex.to_list())

        ex_equips_max_rank_cnt = Counter(flow(client.data.ex_equips.values()) \
                .where(lambda ex: ex.rank == db.get_ex_equip_max_rank(ex.ex_equipment_id)) \
                .group_by(lambda ex: ex.ex_equipment_id) \
                .to_dict(lambda ex: ex.key, lambda ex: ex.count()))

        rankup_equip_cnt = 0
        consume_equip_cnt = 0
        cost_mana = 0
        for equip in consider_ex_equips:
            if equip.serial_id not in client.data.ex_equips:
                continue
            max_rank = db.get_ex_equip_max_rank(equip.ex_equipment_id)
            if equip.rank >= max_rank:
                continue
            if ex_equip_rank_max_num != -1 and ex_equips_max_rank_cnt[equip.ex_equipment_id] >= ex_equip_rank_max_num:
                continue
            demand = max_rank - equip.rank
            use_series = flow(zero_rank_ex.get(equip.ex_equipment_id, [])) \
                .where(lambda ex: ex.serial_id != equip.serial_id 
                       and ex.serial_id in client.data.ex_equips 
                       and client.data.ex_equips[ex.serial_id].rank == 0) \
                .take(demand) \
                .select(lambda ex: ex.serial_id) \
                .to_list()

            if use_series:
                rankup_equip_cnt += 1
                consume_equip_cnt += len(use_series)
                final_rank = equip.rank + len(use_series)
                mana = db.get_ex_equip_rankup_cost(equip.ex_equipment_id, equip.rank, final_rank)
                cost_mana += mana
                if not await client.prepare_mana(mana):
                    self._log(f"mana数不足{mana}，无法合成EX装")
                    break
                unit_id, frame, slot = use_ex_equip.get(equip.serial_id, [0, 0, 0])
                await client.equipment_rankup_ex(
                    serial_id=equip.serial_id,
                    unit_id=unit_id,
                    frame=frame,
                    slot=slot,
                    before_rank=equip.rank,
                    after_rank=final_rank,
                    consume_gold=mana,
                    from_view=2,
                    item_list=[],
                    consume_ex_serial_id_list=use_series
                )
                if final_rank == max_rank:
                    ex_equips_max_rank_cnt[equip.ex_equipment_id] += 1


        if rankup_equip_cnt:
            self._log(f"消耗了{consume_equip_cnt}个零星EX装和{cost_mana}mana，合成了{rankup_equip_cnt}个EX装")
        else:
            raise SkipError("没有可合成的EX装")


@name('撤下会战EX装')
@default(True)
@description('')
class remove_cb_ex_equip(Module):
    async def do_task(self, client: pcrclient):
        ex_cnt = 0
        unit_cnt = 0
        forbidden = set(client.data.user_clan_battle_ex_equip_restriction.keys())
        no_remove_for_forbidden = 0
        for unit_id in client.data.unit:
            unit = client.data.unit[unit_id]
            exchange_list = []
            for ex_equip in unit.cb_ex_equip_slot:
                if ex_equip.serial_id != 0 and ex_equip.serial_id not in forbidden:
                    exchange_list.append(ExtraEquipChangeSlot(slot=ex_equip.slot, serial_id=0))
                    ex_cnt += 1
                elif ex_equip.serial_id in forbidden:
                    no_remove_for_forbidden += 1

            if exchange_list:
                unit_cnt += 1
                await client.unit_equip_ex([ExtraEquipChangeUnit(
                        unit_id=unit_id, 
                        ex_equip_slot = None,
                        cb_ex_equip_slot=exchange_list)])
        if ex_cnt:
            msg = f"（{no_remove_for_forbidden}个因会战CD未撤下）" if no_remove_for_forbidden else ""
            self._log(f"撤下了{unit_cnt}个角色的{ex_cnt}个会战EX装备{msg}")
        else:
            raise SkipError("所有会战EX装备均已撤下")

## add: 撤下普通ex装
@name('撤下普通EX装')
@default(True)
@description('')
class remove_normal_ex_equip(Module):
    async def do_task(self, client: pcrclient):
        ex_cnt = 0
        unit_cnt = 0
        for unit_id in client.data.unit:
            unit = client.data.unit[unit_id]
            exchange_list = []
            for ex_equip in unit.ex_equip_slot:
                if ex_equip.serial_id != 0:
                    exchange_list.append(ExtraEquipChangeSlot(slot=ex_equip.slot, serial_id=0))
                    ex_cnt += 1

            if exchange_list:
                unit_cnt += 1
                await client.unit_equip_ex([ExtraEquipChangeUnit(
                        unit_id=unit_id, 
                        ex_equip_slot=exchange_list,
                        cb_ex_equip_slot=None)])
        if ex_cnt:
            self._log(f"撤下了{unit_cnt}个角色的{ex_cnt}个普通EX装备")
        else:
            raise SkipError("所有普通EX装备均已撤下")

# add: 计算已有角色最佳3星ex装备
# 每个角色有三个槽，通过unit_ex_equipment_slot、ex_equipment_data可以找到每个槽能够使用的多个装备，通过calc_unit_power计算装上ex装备增加的战力倒序，获取最佳3星ex，以及次佳与最佳的差值。目前的calc_unit_attribute只能获取到未添加ex装备的战力，如果装上ex装备，需要在白板unit_attribute上加上"ex_equipment_data"的对应max_xxx值（如果是100的倍数需要加上百分之几，如200就加2%，如果不是就直接加上原数字）。装备显示成名称。

@name('计算最佳3星EX装备')
@default(False)
@description('计算已有角色每个槽位的最佳3星EX装备推荐')
@booltype('simulate_auto_equip', '模拟装备', True)
@booltype('auto_equip', '自动装备', False)
class calc_best_3star_ex_equip(Module):
    async def do_task(self, client: pcrclient):
        simulate_auto_equip: bool = self.get_config('simulate_auto_equip')
        auto_equip: bool = self.get_config('auto_equip')
        results = []
        total_best_power_increase = 0  # 统计所有最佳装备的战力提升总和
        all_slot_data = {}  # 存储每个角色每个槽位的所有装备选择: {(unit_id, slot): [(ex_id, power), ...]}
        
        # 汇总结构（新的）：
        # 1) summary_counts[slot][category][equip_name] = count(有多少角色该装备为绝对最佳，可重复计数用于并列最佳)
        # 2) summary_roles[slot][category] = set(该槽位/类别下出现的角色，用于统计总人数)
        # 3) role_best_names[slot][category][unit_name] = set(该角色在该槽位/类别下的绝对最佳装备名集合)
        summary_counts = {1: {}, 2: {}, 3: {}}
        summary_roles = {1: {}, 2: {}, 3: {}}
        role_best_names = {1: {}, 2: {}, 3: {}}
        
        for unit_id in client.data.unit:
            unit = client.data.unit[unit_id]
            unit_name = db.get_unit_name(unit_id)
            
            # 获取角色基础属性（用于计算增强后的战力）
            base_attr = db.calc_unit_attribute(unit, client.data.read_story_ids)
            
            # 计算每个槽位的最佳EX装备（只处理普通EX槽位1-3）
            for slot in range(3):  # 只处理槽位1-3，槽位4-6是会战EX
                slot_recommendations = []
                
                # 获取该槽位可用的EX装备
                available_ex_equips = []
                if unit_id in db.unit_ex_equipment_slot:
                    slot_data = db.unit_ex_equipment_slot[unit_id]
                    # 获取槽位的种类号
                    category = None
                    if slot == 0:  # 槽位1
                        category = slot_data.slot_category_1
                    elif slot == 1:  # 槽位2
                        category = slot_data.slot_category_2
                    elif slot == 2:  # 槽位3
                        category = slot_data.slot_category_3
                    
                    # 在ex_equipment_data中查找匹配该种类的所有装备ID
                    if category is not None:
                        available_ex_equips = [ex_id for ex_id, ex_data in db.ex_equipment_data.items() 
                                             if ex_data.category == category]
                
                if not available_ex_equips:
                    continue
                    
                for ex_equip_id in available_ex_equips:
                    # 只考虑3星品质的EX装备
                    if db.get_ex_equip_rarity(ex_equip_id) != 3:
                        continue
                        
                    # 获取EX装备数据
                    ex_data = db.ex_equipment_data[ex_equip_id]
                    
                    # 计算装备该EX装备后的属性
                    enhanced_attr = {
                        'hp': base_attr.hp,
                        'atk': base_attr.atk,
                        'def_': base_attr.def_,
                        'magic_str': base_attr.magic_str,
                        'magic_def': base_attr.magic_def,
                        'physical_critical': base_attr.physical_critical,
                        'magic_critical': base_attr.magic_critical,
                        'dodge': base_attr.dodge,
                        'accuracy': base_attr.accuracy,
                        'energy_recovery_rate': base_attr.energy_recovery_rate,
                        'hp_recovery_rate': base_attr.hp_recovery_rate,
                        'energy_reduce_rate': base_attr.energy_reduce_rate
                    }
                    
                    # 根据注释说明，添加EX装备属性
                    for attr_name in ['max_hp', 'max_atk', 'max_def', 'max_magic_str', 
                                     'max_magic_def', 'max_physical_critical', 'max_magic_critical',
                                     'max_dodge', 'max_accuracy', 'max_energy_recovery_rate',
                                     'max_hp_recovery_rate', 'max_energy_reduce_rate']:
                        ex_value = getattr(ex_data, attr_name, 0)
                        if ex_value > 0:
                            # 转换属性名：max_hp -> hp, max_def -> def_
                            base_attr_name = attr_name.replace('max_', '')
                            if base_attr_name == 'def':
                                base_attr_name = 'def_'
                            
                            base_value = enhanced_attr.get(base_attr_name, 0)
                            if ex_value % 100 == 0 and ex_value >= 100:
                                # 百分比属性
                                enhanced_attr[base_attr_name] = float(base_value) * (1 + ex_value / 10000)
                            else:
                                # 固定数值属性
                                enhanced_attr[base_attr_name] = float(base_value) + ex_value
                    
                    # 计算装备后的战力 (属性值与系数相乘求和)
                    coefficient = db.unit_status_coefficient[1]
                    
                    # 使用_coefficient形式的属性名，并转换为float避免Decimal类型冲突
                    coeff_hp = float(coefficient.hp_coefficient)
                    coeff_atk = float(coefficient.atk_coefficient)
                    coeff_def = float(coefficient.def_coefficient)
                    coeff_magic_str = float(coefficient.magic_str_coefficient)
                    coeff_magic_def = float(coefficient.magic_def_coefficient)
                    coeff_physical_critical = float(coefficient.physical_critical_coefficient)
                    coeff_magic_critical = float(coefficient.magic_critical_coefficient)
                    coeff_dodge = float(coefficient.dodge_coefficient)
                    coeff_accuracy = float(coefficient.accuracy_coefficient)
                    coeff_energy_recovery_rate = float(coefficient.energy_recovery_rate_coefficient)
                    coeff_hp_recovery_rate = float(coefficient.hp_recovery_rate_coefficient)
                    coeff_energy_reduce_rate = float(coefficient.energy_reduce_rate_coefficient)
                    # 计算基础战力
                    base_attr_power = (
                        float(base_attr.hp) * coeff_hp +
                        float(base_attr.atk) * coeff_atk +
                        float(base_attr.def_) * coeff_def +
                        float(base_attr.magic_str) * coeff_magic_str +
                        float(base_attr.magic_def) * coeff_magic_def +
                        float(base_attr.physical_critical) * coeff_physical_critical +
                        float(base_attr.magic_critical) * coeff_magic_critical +
                        float(base_attr.dodge) * coeff_dodge +
                        float(base_attr.accuracy) * coeff_accuracy +
                        float(base_attr.energy_recovery_rate) * coeff_energy_recovery_rate +
                        float(base_attr.hp_recovery_rate) * coeff_hp_recovery_rate +
                        float(base_attr.energy_reduce_rate) * coeff_energy_reduce_rate
                    )
                    
                    # 计算增强后战力
                    power_with_ex = (
                        float(enhanced_attr['hp']) * coeff_hp +
                        float(enhanced_attr['atk']) * coeff_atk +
                        float(enhanced_attr['def_']) * coeff_def +
                        float(enhanced_attr['magic_str']) * coeff_magic_str +
                        float(enhanced_attr['magic_def']) * coeff_magic_def +
                        float(enhanced_attr['physical_critical']) * coeff_physical_critical +
                        float(enhanced_attr['magic_critical']) * coeff_magic_critical +
                        float(enhanced_attr['dodge']) * coeff_dodge +
                        float(enhanced_attr['accuracy']) * coeff_accuracy +
                        float(enhanced_attr['energy_recovery_rate']) * coeff_energy_recovery_rate +
                        float(enhanced_attr['hp_recovery_rate']) * coeff_hp_recovery_rate +
                        float(enhanced_attr['energy_reduce_rate']) * coeff_energy_reduce_rate
                    )
                    
                    power_increase = power_with_ex - base_attr_power
                    
                    ex_name = ex_data.name
                    slot_recommendations.append((ex_equip_id, ex_name, int(power_increase)))
                
                # 按战力增加值排序
                slot_recommendations.sort(key=lambda x: x[2], reverse=True)
                
                # 存储该槽位的所有装备选择
                if slot_recommendations:
                    all_slot_data[(unit_id, slot + 1)] = [(ex_id, power) for ex_id, ex_name, power in slot_recommendations]
                
                # 新增：汇总每个槽位下“被视为最佳(等效最佳)”的EX类别
                if slot_recommendations:
                    # 仅纳入绝对最佳（与最佳相等的装备）。并按 槽位/类别/装备名 进行计数。
                    best_power = slot_recommendations[0][2]
                    slot_index = slot + 1
                    for ex_id, ex_name, power in slot_recommendations:
                        if power != best_power:
                            continue
                        ex_cat = db.ex_equipment_data[ex_id].category
                        # 初始化计数结构
                        summary_roles[slot_index].setdefault(ex_cat, set()).add(unit_name)
                        summary_counts[slot_index].setdefault(ex_cat, {})
                        summary_counts[slot_index][ex_cat].setdefault(ex_name, 0)
                        summary_counts[slot_index][ex_cat][ex_name] += 1  # 并列最佳时，会对不同装备分别+1
                        # 记录角色在该槽位/类别下的绝对最优名称集合
                        role_best_names[slot_index].setdefault(ex_cat, {})
                        role_best_names[slot_index][ex_cat].setdefault(unit_name, set()).add(ex_name)
                
                if len(slot_recommendations) >= 3:
                    top3 = slot_recommendations[:3]
                    best_power = top3[0][2]  # 最佳装备的战力
                    
                    result_text = f"{unit_name} 槽位{slot+1}: "
                    for i, (ex_id, ex_name, power) in enumerate(top3):
                        rank_name = ["最佳", "次佳", "第三"][i]
                        power_diff = power - best_power  # 相对于最佳的差值
                        if power_diff == 0:
                            result_text += f"{rank_name}={ex_name}(0)"
                        else:
                            result_text += f"{rank_name}={ex_name}({power_diff})"
                        if i < len(top3) - 1:
                            result_text += ", "
                    results.append(result_text)
                    # 累加最佳装备的战力提升
                    total_best_power_increase += best_power
                elif len(slot_recommendations) == 2:
                    best, second = slot_recommendations[:2]
                    power_diff = second[2] - best[2]
                    results.append(f"{unit_name} 槽位{slot+1}: 最佳={best[1]}(0), 次佳={second[1]}({power_diff})")
                    # 累加最佳装备的战力提升
                    total_best_power_increase += best[2]
                elif len(slot_recommendations) == 1:
                    best = slot_recommendations[0]
                    results.append(f"{unit_name} 槽位{slot+1}: 最佳={best[1]}(0)")
                    # 累加最佳装备的战力提升
                    total_best_power_increase += best[2]
        
        if results:
            if auto_equip or simulate_auto_equip:
                # 执行自动/模拟装备操作（模拟时不发包，但进入流程）
                await self._auto_equip_best_ex(client, all_slot_data)
            else:
                # 新的汇总输出：按 槽位/类别 统计“总角色数”和“每种绝对最佳装备名的计数”，并追加本类别点数统计
                self._log("3星EX装备汇总（按槽位/装备种类）:")

                # 预计算：按类别统计已拥有的该类别所有3星EX装备的“点数”总和（满强化=3点），不区分是否装备
                # 以及可用于“最优”名称集合的点数总和（用于计算缺口）
                category_total_points = {}
                category_best_points_by_slot = {1: {}, 2: {}, 3: {}}
                # 新增：按类别-名称统计点数（包含非最优装备）
                category_points_by_name = {}

                # 先收集所有类别的总点数（仅3星），使用rank换算点数：0→1点，1→2点，2→3点
                for ex in client.data.ex_equips.values():
                    ex_id = ex.ex_equipment_id
                    # 仅统计3星EX
                    if db.get_ex_equip_rarity(ex_id) != 3:
                        continue
                    ex_data = db.ex_equipment_data[ex_id]
                    cat = ex_data.category
                    name = ex_data.name
                    # 使用 rank → 点数 映射（0→1，1→2，2→3）
                    pts = min(int(ex.rank) + 1, 3)
                    category_total_points[cat] = category_total_points.get(cat, 0) + pts
                    if cat not in category_points_by_name:
                        category_points_by_name[cat] = {}
                    category_points_by_name[cat][name] = category_points_by_name[cat].get(name, 0) + pts

                for slot_index in [1, 2, 3]:
                    slot_counts = summary_counts.get(slot_index, {})
                    slot_roles = summary_roles.get(slot_index, {})
                    if not slot_counts:
                        continue
                    # 槽位级别累计器：统计该槽位所有类别的总点数、最优可用点、缺口
                    slot_total_points = 0
                    slot_total_best_points = 0
                    slot_total_missing = 0

                    self._log(f"槽位{slot_index}:")
                    items = list(slot_counts.items())
                    for idx, (ex_cat, equip_map) in enumerate(items):
                        total_roles = len(slot_roles.get(ex_cat, set()))

                        # 该槽位/类别下的“绝对最优名称”集合
                        best_names = set(equip_map.keys())

                        # 计算该类别下属于“最优名称”的现有点数（基于rank：0→1点，1→2点，2→3点）
                        best_points = 0
                        # 同时统计每种“名称”的点数（包含非最优），供后续分配与打印
                        all_points_per_name = {}
                        for ex in client.data.ex_equips.values():
                            ex_id = ex.ex_equipment_id
                            if db.get_ex_equip_rarity(ex_id) != 3:
                                continue
                            ex_data = db.ex_equipment_data[ex_id]
                            if ex_data.category != ex_cat:
                                continue
                            pts = min(int(ex.rank) + 1, 3)
                            all_points_per_name[ex_data.name] = all_points_per_name.get(ex_data.name, 0) + pts
                            if ex_data.name in best_names:
                                best_points += pts
                        category_best_points_by_slot[slot_index][ex_cat] = (best_points, all_points_per_name)

                        # 构造“X个角色: 名称A, 名称B (A计数, B计数)”的样式
                        names_sorted = sorted(equip_map.keys(), key=lambda n: equip_map[n], reverse=True)
                        names_join = '，'.join(names_sorted)
                        # 计数详情（每种装备名有多少角色为其绝对最优）
                        details = '， '.join(f"{equip_map[name]}个角色: {name}" for name in names_sorted)

                        # 本类别总点（不区分最优与否）
                        total_points = category_total_points.get(ex_cat, 0)
                        slot_total_points += total_points

                        # 贪心分配：考虑某角色可能有多个“并列最优名称”，从这些名称的点数池中为该角色分配最多3点，直到该角色满3或没有可用点
                        # 点数池：该类别下每个名称拥有的点数（包含非最优）
                        _, all_points_per_name = category_best_points_by_slot[slot_index].get(ex_cat, (0, {}))
                        pool = dict(all_points_per_name)

                        allocated = 0
                        # 为分配顺序选择一个稳定策略：按最佳名称计数从高到低，再按名称字典序
                        role_to_bestnames = role_best_names.get(slot_index, {}).get(ex_cat, {})
                        # 为了公平，遍历角色时不固定顺序对结果影响小，但我们按角色名排序以稳定输出
                        for rname in sorted(role_to_bestnames.keys()):
                            need = 3
                            best_set = sorted(role_to_bestnames[rname], key=lambda nm: (-equip_map.get(nm, 0), nm))
                            # 轮询该角色的最优名称，从池中扣除
                            for nm in best_set:
                                if need <= 0:
                                    break
                                avail = pool.get(nm, 0)
                                if avail <= 0:
                                    continue
                                take = min(avail, need)
                                pool[nm] = avail - take
                                need -= take
                            allocated += (3 - need)
                        # 分配完成后的“最优可用点”
                        best_points_alloc = allocated
                        # 缺口 = 角色数*3 - 已分配
                        category_missing_points = max(0, total_roles * 3 - best_points_alloc)
                        slot_total_best_points += best_points_alloc

                        # 打印各自点数：包含非最优装备的点数（同类别下所有名称）
                        all_name_pts = category_points_by_name.get(ex_cat, {})
                        per_name_pts = '， '.join(f"{name}:{all_name_pts.get(name, 0)}点" for name in sorted(all_name_pts.keys(), key=lambda n: all_name_pts[n], reverse=True))

                        self._log(
                            f"  {total_roles}个角色: {names_join} ({details}) | "
                            f"本类别总点: {total_points} | 最优可用点: {best_points_alloc}，缺口: {category_missing_points} | 各自点数: {per_name_pts}"
                        )
                        slot_total_missing += category_missing_points

                    # 槽位级别汇总输出
                    self._log(
                        f"  槽位{slot_index}汇总 | 总点数: {slot_total_points} | 总最优可用: {slot_total_best_points} | 总缺口: {slot_total_missing}"
                    )
                
                # 再输出逐角色Top3详情
                self._log("3星EX装备推荐结果:")
                for result in results:
                    self._log(result)
                self._log(f"所有最佳装备战力提升总计: +{total_best_power_increase}")
        else:
            raise SkipError("没有找到可推荐的3星EX装备")
    
    async def _auto_equip_best_ex(self, client: pcrclient, all_slot_data):
        """自动装备最佳3星EX装备（支持模拟/实际装备，贪心使用最优点数分配，按类别处理特殊/大众角色）"""
        simulate_auto_equip: bool = self.get_config('simulate_auto_equip')
        auto_equip: bool = self.get_config('auto_equip')
        include_pink_unequip: bool = getattr(self, 'include_pink_unequip', False)

        # 1) 开始前先撤下所有未锁定(protection_flag=0)的非粉装（4星以下）。如果允许，可选把4星也撤下。
        removed_cnt = 0
        removed_units = 0
        sim_logs = []  # 模拟模式下延后统一打印的日志
        to_remove_serials = set()  # 记录将被撤下的serial，用于模拟时视为可用库存
        for unit_id, unit in client.data.unit.items():
            exchange = []
            for slot in range(3):  # 只处理普通槽位1-3
                ex_slot = unit.ex_equip_slot[slot]
                if ex_slot.serial_id == 0:
                    continue
                ex = client.data.ex_equips.get(ex_slot.serial_id)
                if not ex:
                    continue
                rarity = db.get_ex_equip_rarity(ex.ex_equipment_id)
                if ex.protection_flag == 1 and (rarity < 4 or include_pink_unequip):
                    exchange.append(ExtraEquipChangeSlot(slot=slot+1, serial_id=0))
                    to_remove_serials.add(ex_slot.serial_id)
            if exchange:
                removed_units += 1
                removed_cnt += len(exchange)
                if auto_equip:
                    await client.unit_equip_ex([ExtraEquipChangeUnit(
                        unit_id=unit_id, ex_equip_slot=exchange, cb_ex_equip_slot=None
                    )])
                else:
                    pass  # 模拟模式下不逐人打印撤装明细，仅打印总计
        if removed_cnt:
            if auto_equip:
                self._log(f"已撤下未锁定的非粉装EX：{removed_units}个角色，共{removed_cnt}件")
            else:
                sim_logs.append(f"将撤下未锁定的非粉装EX：{removed_units}个角色，共{removed_cnt}件")
        # 模拟模式下，把将撤下的实例加入库存：锁定(protection_flag==1)除外
        if not auto_equip and to_remove_serials:
            for sid in list(to_remove_serials):
                ex = client.data.ex_equips.get(sid)
                if not ex:
                    continue
                if ex.protection_flag == 1:
                    continue
                ex_id = ex.ex_equipment_id
                ex_pool_by_id.setdefault(ex_id, []).append(ex)
                # 同时更新 cat_name_pool
                cat = db.ex_equipment_data[ex_id].category
                name = db.ex_equipment_data[ex_id].name
                cat_name_pool.setdefault(cat, {}).setdefault(name, []).append(ex)

        # 2) 准备库存池：按ex_id分组，记录rank/pt，便于按优先级取用
        use_ex_equip = set(ex_slot.serial_id for u in client.data.unit.values() for ex_slot in u.ex_equip_slot if ex_slot.serial_id != 0)
        ex_pool_by_id = {}
        for ex in client.data.ex_equips.values():
            # 模拟时：把将被撤下的也当作可用
            if ex.serial_id in use_ex_equip and ex.serial_id not in to_remove_serials:
                continue
            if db.get_ex_equip_rarity(ex.ex_equipment_id) != 3:
                continue
            ex_pool_by_id.setdefault(ex.ex_equipment_id, []).append(ex)
        for ex_id in ex_pool_by_id:
            # 排序：优先满强(enhancement_pt=6000)，再按是否会战【行会】放后，最后按rank/pt降序
            # 满强优先（rank2+pt6000），其后在同满强/同非满强之间非【行会】优先，其次rank与pt
            ex_pool_by_id[ex_id].sort(key=lambda e: (
                -(1 if (e.rank >= 2 and e.enhancement_pt >= 6000) else 0),
                0 if db.is_clan_ex_equip((eInventoryType.ExtraEquip, e.ex_equipment_id)) else -1,
                -e.rank,
                -e.enhancement_pt
            ))

        # 工具函数：根据增强等级估算某ex_id对某角色的战力增加
        def estimate_power_increase_at_level(unit_id: int, slot_index: int, ex_id: int, level: int) -> int:
            # 兼容旧接口：按点数比例推算
            frac = 0 if level <= 0 else (1.0 if level >= 5 else level / 5.0)
            return estimate_power_increase_at_fraction(unit_id, slot_index, ex_id, frac)

        def estimate_power_increase_at_fraction(unit_id: int, slot_index: int, ex_id: int, frac: float) -> int:
            # 按“default + frac*(max-default)”线性插值进行属性叠加后计算战力提升
            # frac in [0,1], 例如 4/5 表示 4 强的近似
            unit = client.data.unit[unit_id]
            base_attr = db.calc_unit_attribute(unit, client.data.read_story_ids)
            ex_data = db.ex_equipment_data[ex_id]
            attrs = [
                ('hp', 'max_hp'), ('atk', 'max_atk'), ('def_', 'max_def'), ('magic_str', 'max_magic_str'),
                ('magic_def', 'max_magic_def'), ('physical_critical', 'max_physical_critical'), ('magic_critical', 'max_magic_critical'),
                ('dodge', 'max_dodge'), ('accuracy', 'max_accuracy'), ('energy_recovery_rate', 'max_energy_recovery_rate'),
                ('hp_recovery_rate', 'max_hp_recovery_rate'), ('energy_reduce_rate', 'max_energy_reduce_rate')
            ]
            coeff = db.unit_status_coefficient[1]
            def calc_power(hp, atk, deff, mstr, mdef, pcri, mcri, dodge, acc, err, hprr, eer):
                return (
                    float(hp) * float(coeff.hp_coefficient) +
                    float(atk) * float(coeff.atk_coefficient) +
                    float(deff) * float(coeff.def_coefficient) +
                    float(mstr) * float(coeff.magic_str_coefficient) +
                    float(mdef) * float(coeff.magic_def_coefficient) +
                    float(pcri) * float(coeff.physical_critical_coefficient) +
                    float(mcri) * float(coeff.magic_critical_coefficient) +
                    float(dodge) * float(coeff.dodge_coefficient) +
                    float(acc) * float(coeff.accuracy_coefficient) +
                    float(err) * float(coeff.energy_recovery_rate_coefficient) +
                    float(hprr) * float(coeff.hp_recovery_rate_coefficient) +
                    float(eer) * float(coeff.energy_reduce_rate_coefficient)
                )
            base_power = calc_power(
                base_attr.hp, base_attr.atk, base_attr.def_, base_attr.magic_str, base_attr.magic_def,
                base_attr.physical_critical, base_attr.magic_critical, base_attr.dodge, base_attr.accuracy,
                base_attr.energy_recovery_rate, base_attr.hp_recovery_rate, base_attr.energy_reduce_rate
            )
            enhanced = {
                'hp': base_attr.hp, 'atk': base_attr.atk, 'def_': base_attr.def_, 'magic_str': base_attr.magic_str,
                'magic_def': base_attr.magic_def, 'physical_critical': base_attr.physical_critical,
                'magic_critical': base_attr.magic_critical, 'dodge': base_attr.dodge, 'accuracy': base_attr.accuracy,
                'energy_recovery_rate': base_attr.energy_recovery_rate, 'hp_recovery_rate': base_attr.hp_recovery_rate,
                'energy_reduce_rate': base_attr.energy_reduce_rate
            }
            for dst, mx in attrs:
                max_val = getattr(ex_data, mx, 0)
                default_val = getattr(ex_data, mx.replace('max_', 'default_'), 0)
                if max_val or default_val:
                    delta = default_val + (max_val - default_val) * float(frac)
                    if (mx.endswith('rate')) and max_val >= 100 and max_val % 100 == 0:
                        enhanced[dst] = float(enhanced[dst]) * (1 + delta / 10000)
                    else:
                        enhanced[dst] = float(enhanced[dst]) + delta
            enhanced_power = calc_power(
                enhanced['hp'], enhanced['atk'], enhanced['def_'], enhanced['magic_str'], enhanced['magic_def'],
                enhanced['physical_critical'], enhanced['magic_critical'], enhanced['dodge'], enhanced['accuracy'],
                enhanced['energy_recovery_rate'], enhanced['hp_recovery_rate'], enhanced['energy_reduce_rate']
            )
            return int(enhanced_power - base_power)

        # 3) 构建每个槽位/类别的“最佳名称统计”与角色集合，用于识别特殊/大众
        # 以及预处理每个角色在该槽位的最高战力增益（用于大众排序）
        def get_slot_category(unit_id: int, slot_idx: int):
            slot_data = db.unit_ex_equipment_slot.get(unit_id)
            if not slot_data:
                return None
            return [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3][slot_idx-1]

        # ex_id→名称、是否会战
        def ex_name(ex_id: int):
            return db.ex_equipment_data[ex_id].name
        def ex_is_guild(ex_id: int):
            return db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex_id))

        # 收集 per (slot, category)
        per_sc_roles = {}  # (slot, cat) -> set(unit_id)
        per_sc_bestname_counts = {}  # (slot, cat) -> {name: count}
        per_sc_role_bestnames = {}  # (slot, cat) -> {unit_id: set(names)}
        role_best_power = {}  # (unit_id, slot) -> max_power
        for (unit_id, slot), options in all_slot_data.items():
            if slot > 3:
                continue
            cat = get_slot_category(unit_id, slot)
            if cat is None:
                continue
            if not options:
                continue
            max_power = max(p for _, p in options)
            role_best_power[(unit_id, slot)] = max_power
            best_names = set()
            for ex_id, p in options:
                if p == max_power:
                    best_names.add(ex_name(ex_id))
            key = (slot, cat)
            per_sc_roles.setdefault(key, set()).add(unit_id)
            per_sc_bestname_counts.setdefault(key, {})
            for nm in best_names:
                per_sc_bestname_counts[key][nm] = per_sc_bestname_counts[key].get(nm, 0) + 1
            per_sc_role_bestnames.setdefault(key, {})
            per_sc_role_bestnames[key].setdefault(unit_id, set()).update(best_names)

        # 4) 为每个(slot,category)识别特殊/大众，并为特殊尽量满强穿上
        equipped_units_set = set()
        equip_actions = []  # (unit_id, slot, inst)
        logs_non_full = []
        # 槽位/类别维度的缺口累加
        category_missing_points = {}
        ref_role_by_sc = {}  # (slot, cat) -> unit_id 作为“战力参考角色”

        # 库存：按类别汇总点数与可用实例
        cat_name_pool = {}  # cat -> name -> list of ex instances
        for ex_id, lst in ex_pool_by_id.items():
            cat = db.ex_equipment_data[ex_id].category
            name = db.ex_equipment_data[ex_id].name
            cat_name_pool.setdefault(cat, {}).setdefault(name, []).extend(lst)

        def instance_points(inst):
            # rank 0→1, 1→2, 2→3
            return min(int(inst.rank) + 1, 3)

        # 实际装备或模拟记录
        async def equip_to(unit_id: int, slot: int, inst) -> bool:
            if not auto_equip:
                equip_actions.append((unit_id, slot, inst))
                return True
            try:
                await client.unit_equip_ex([ExtraEquipChangeUnit(
                    unit_id=unit_id,
                    ex_equip_slot=[ExtraEquipChangeSlot(slot=slot, serial_id=inst.serial_id)],
                    cb_ex_equip_slot=None
                )])
                return True
            except Exception as e:
                self._warn(f"为{db.get_unit_name(unit_id)}装备{db.ex_equipment_data[inst.ex_equipment_id].name}失败: {e}")
                return False

        # 分配函数：从某名称池按优先级取一个实例
        def pop_best_instance(cat: int, name: str, prefer_full=True, prefer_non_guild_first=True):
            lst = cat_name_pool.get(cat, {}).get(name, [])
            if not lst:
                return None
            # 优先策略：满强非【行会】 > 满强【行会】 > 非满强非【行会】 > 非满强【行会】
            def key(inst):
                full = 1 if inst.enhancement_pt >= 6000 else 0
                guild = 1 if ex_is_guild(inst.ex_equipment_id) else 0
                return (-full, guild if prefer_non_guild_first else 0, -inst.rank, -inst.enhancement_pt)
            lst.sort(key=key)
            return lst.pop(0)

        # 主循环：逐(slot,cat)处理
        snapshot_by_sc = {}
        for (slot, cat), roles in per_sc_roles.items():
            # 模拟模式下，记录该类别库存快照，便于汇总输出（含一个示例 ex_id）
            if not auto_equip:
                snap = {}
                for nm, inst_list in cat_name_pool.get(cat, {}).items():
                    full_cnt = sum(1 for inst in inst_list if (inst.rank >= 2 and inst.enhancement_pt >= 6000))
                    r2_cnt = sum(1 for inst in inst_list if (inst.rank >= 2 and inst.enhancement_pt < 6000))
                    r1_cnt = sum(1 for inst in inst_list if inst.rank == 1)
                    r0_cnt = sum(1 for inst in inst_list if inst.rank == 0)
                    ex_id_example = inst_list[0].ex_equipment_id if inst_list else None
                    snap[nm] = (full_cnt, r2_cnt, r1_cnt, r0_cnt, ex_id_example)
                snapshot_by_sc[(slot, cat)] = snap
            counts = per_sc_bestname_counts.get((slot, cat), {})
            if not counts:
                continue
            # 识别大众与特殊：以计数最多的名称为大众，其余集合大小总和<10则视为特殊
            majority_name = max(counts.items(), key=lambda x: x[1])[0]
            special_role_ids = [rid for rid in roles if majority_name not in per_sc_role_bestnames[(slot, cat)][rid]]
            majority_role_ids = [rid for rid in roles if rid not in special_role_ids]

            if len(special_role_ids) < 10 and special_role_ids:
                # 特殊优先：尝试为每个特殊角色满强穿上其最优之一
                for rid in special_role_ids:
                    best_names = sorted(per_sc_role_bestnames[(slot, cat)][rid])
                    chosen_inst = None
                    for nm in best_names:
                        inst = pop_best_instance(cat, nm)
                        if inst:
                            chosen_inst = inst
                            break
                    if chosen_inst:
                        ok = await equip_to(rid, slot, chosen_inst)
                        if ok:
                            equipped_units_set.add(rid)
                            if chosen_inst.enhancement_pt < 6000 or chosen_inst.rank < 2:
                                logs_non_full.append((db.get_unit_name(rid), slot, db.ex_equipment_data[chosen_inst.ex_equipment_id].name, chosen_inst.rank, chosen_inst.enhancement_pt))
                            # 从池中扣除该实例
                            try:
                                cat_name_pool[cat][db.ex_equipment_data[chosen_inst.ex_equipment_id].name].remove(chosen_inst)
                            except Exception:
                                pass

            # 大众：按该槽位的战力增加倒序
            majority_role_ids.sort(key=lambda uid: role_best_power.get((uid, slot), 0), reverse=True)
            if majority_role_ids:
                ref_role_by_sc[(slot, cat)] = majority_role_ids[0]

            # 单件选择优先级函数：满强rank2 pt6000 > rank2(其他pt) > rank1 > rank0；非【行会】优先
            def best_inst_from_names(cat, names):
                cand = []
                for nm in names:
                    cand.extend(cat_name_pool.get(cat, {}).get(nm, []))
                if not cand:
                    return None
                def key(inst):
                    # 满强优先（rank2+pt6000），其后非【行会】优先，其次rank与pt
                    full = 1 if (inst.rank >= 2 and inst.enhancement_pt >= 6000) else 0
                    nguild = 1 if not ex_is_guild(inst.ex_equipment_id) else 0
                    return (-full, -nguild, -inst.rank, -inst.enhancement_pt)
                cand.sort(key=key)
                return cand[0]

            # 为大众逐个选择“单件”装备
            for rid in majority_role_ids:
                chosen = None
                role_best_set = per_sc_role_bestnames[(slot, cat)][rid]
                # 1) 先从该角色的“并列最优名称”里挑一件
                chosen = best_inst_from_names(cat, role_best_set)
                # 2) 若完全没有，才考虑其它名称：用参考角色比较“最优4强 vs 其它5强”，若其它5强明显更优则选其它满强，否则仍选最优里可得的最高rank/pt单件
                if not chosen:
                    ref_uid = ref_role_by_sc.get((slot, cat), rid)
                    any_best_ex_id = next((eid for eid, _ in all_slot_data[(ref_uid, slot)] if ex_name(eid) in role_best_set), None)
                    if any_best_ex_id is not None:
                        best_lv4 = estimate_power_increase_at_level(ref_uid, slot, any_best_ex_id, 4)
                        # 找到任一其它名称的可用实例，若其5强更优，则选该名称的最优实例（尽量满强）
                        win_alt_name = None
                        for nm, inst_list in cat_name_pool.get(cat, {}).items():
                            if nm in role_best_set or not inst_list:
                                continue
                            any_alt_ex_id = db.ex_equipment_data[inst_list[0].ex_equipment_id].ex_equipment_id
                            alt_lv5 = estimate_power_increase_at_level(ref_uid, slot, any_alt_ex_id, 5)
                            if alt_lv5 > best_lv4:
                                win_alt_name = nm
                                break
                        if win_alt_name:
                            chosen = best_inst_from_names(cat, [win_alt_name])
                # 3) 若依然没有，就从最优集合里拿到rank/pt最高的单件（可能是rank2 pt0，或rank1/0）
                if not chosen and role_best_set:
                    chosen = best_inst_from_names(cat, role_best_set)

                # 计算缺口并记录
                if chosen:
                    ok = await equip_to(rid, slot, chosen)
                    if ok:
                        pts = instance_points(chosen)
                        if pts < 3:
                            category_missing_points[(slot, cat)] = category_missing_points.get((slot, cat), 0) + (3 - pts)
                        if chosen.enhancement_pt < 6000 or chosen.rank < 2:
                            logs_non_full.append((db.get_unit_name(rid), slot, db.ex_equipment_data[chosen.ex_equipment_id].name, chosen.rank, chosen.enhancement_pt))
                        # 从池中扣除
                        try:
                            cat_name_pool[cat][db.ex_equipment_data[chosen.ex_equipment_id].name].remove(chosen)
                        except Exception:
                            pass
                else:
                    # 完全无件可穿：缺口3点
                    category_missing_points[(slot, cat)] = category_missing_points.get((slot, cat), 0) + 3

        # 顶部打印模拟结果汇总（当 auto_equip=False）
        if not auto_equip:
            self._log("===== 模拟装备结果（槽位/类别总览） =====")
            # 打印撤装汇总
            for line in sim_logs:
                self._log(line)

            # 统计：按 (slot, category) 聚合，打印库存、分配与待强化列表
            from collections import defaultdict
            per_sc_assigned_total = defaultdict(lambda: defaultdict(int))  # (slot,cat)->name->count 总件数
            per_sc_assigned_by_level = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # (slot,cat)->level->name->count
            per_sc_nonfull = defaultdict(int)  # (slot,cat)->非满强件数
            per_sc_roles = defaultdict(set)  # (slot,cat)->涉及角色集合
            per_sc_to_strengthen = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (slot,cat)-> level -> name -> [unit_name]

            def get_cat(unit_id, slot):
                sc = db.unit_ex_equipment_slot.get(unit_id)
                if not sc:
                    return None
                return [sc.slot_category_1, sc.slot_category_2, sc.slot_category_3][slot-1]

            def bucket_level(inst):
                if inst.rank >= 2 and inst.enhancement_pt >= 6000:
                    return 'full'
                if inst.rank >= 2 and inst.enhancement_pt < 6000:
                    return 'r2'
                if inst.rank == 1:
                    return 'r1'
                return 'r0'

            for uid, slot, inst in equip_actions:
                cat = get_cat(uid, slot)
                if cat is None:
                    continue
                name = db.ex_equipment_data[inst.ex_equipment_id].name
                level = bucket_level(inst)
                per_sc_assigned_total[(slot, cat)][name] += 1
                per_sc_assigned_by_level[(slot, cat)][level][name] += 1
                per_sc_roles[(slot, cat)].add(uid)
                if level != 'full':
                    per_sc_nonfull[(slot, cat)] += 1
                    per_sc_to_strengthen[(slot, cat)][level][name].append(db.get_unit_name(uid))

            # 打印
            for (slot, cat), name_map in sorted(per_sc_assigned_total.items()):
                total_roles = len(per_sc_roles[(slot, cat)])
                # 库存（快照为分配前，当前 cat_name_pool 为分配后剩余）
                snap = snapshot_by_sc.get((slot, cat), {})
                def snap_fmt_item(nm, tpl):
                    a,b,c,d,exid = tpl if len(tpl)==5 else (*tpl, None)
                    return f"{nm}: {a}-{b}-{c}-{d}"
                snap_str = '； '.join(snap_fmt_item(nm, tpl) for nm, tpl in snap.items()) if snap else '无可用实例'

                # 分配：拆分为满强/待强化r2/待强化r1/待强化r0
                def fmt_assign(level):
                    m = per_sc_assigned_by_level[(slot, cat)].get(level, {})
                    if not m:
                        return ''
                    names_sorted = sorted(m.keys(), key=lambda n: m[n], reverse=True)
                    return '， '.join(f"{m[name]}件: {name}" for name in names_sorted)

                assign_full = fmt_assign('full')
                assign_r2 = fmt_assign('r2')
                assign_r1 = fmt_assign('r1')
                assign_r0 = fmt_assign('r0')

                # 预判：若最优可用点 >= 角色数*3，则分配应达到“满强或待强化r2”（无r0/r1残留），缺口记为0
                # 计算最优可用点（基于分配前的库存快照、仅统计该槽位/类别下的“最佳名称集合”）
                best_names_set = set(per_sc_bestname_counts.get((slot, cat), {}).keys())
                best_pts_total = 0
                for nm, tpl in snap.items():
                    if nm not in best_names_set:
                        continue
                    f = tpl[0] if len(tpl) > 0 else 0
                    r2u = tpl[1] if len(tpl) > 1 else 0
                    r1c = tpl[2] if len(tpl) > 2 else 0
                    r0c = tpl[3] if len(tpl) > 3 else 0
                    best_pts_total += f * 3 + r2u * 2 + r1c * 2 + r0c * 1

                # 未满强化与缺口（默认按实际分配统计）
                nonfull_cnt = per_sc_nonfull.get((slot, cat), 0)
                gap = category_missing_points.get((slot, cat), 0)

                # 根据预判修正展示：若 best_pts_total >= total_roles*3，则将r0尽量“提升”为r2待合成，gap置0
                r2_make = {}
                if best_pts_total >= total_roles * 3:
                    # 需要达到的非满目标数量 = 角色总数 - 满强件数
                    full_cnt_total = sum(per_sc_assigned_by_level[(slot, cat)].get('full', {}).values())
                    nonfull_target = max(0, total_roles - full_cnt_total)
                    # 现有r2件数
                    curr_r2_cnt = sum(per_sc_assigned_by_level[(slot, cat)].get('r2', {}).values())
                    need_promote = max(0, nonfull_target - curr_r2_cnt)

                    # 构建r0池（剩余库存r0 + 已分配r0）
                    from collections import defaultdict
                    r0_pool = defaultdict(int)
                    remaining = cat_name_pool.get(cat, {})
                    for nm2, insts in remaining.items():
                        r0_pool[nm2] += sum(1 for inst in insts if inst.rank == 0)
                    assigned_r0_map = per_sc_assigned_by_level[(slot, cat)].get('r0', {})
                    for nm2, cnt in assigned_r0_map.items():
                        r0_pool[nm2] += cnt

                    # 优先提升 r1 -> r2：每 1 个 r0 可把 1 个 r1 提升为 r2（优先同名消耗 r0）
                    assigned_r1_map = per_sc_assigned_by_level[(slot, cat)].get('r1', {}).copy()
                    r1_users = {nm2: list(users) for nm2, users in per_sc_to_strengthen.get((slot, cat), {}).get('r1', {}).items()}
                    r2_make = defaultdict(list)
                    assigned_r2_map = per_sc_assigned_by_level[(slot, cat)].get('r2', {}).copy()

                    # 先同名消耗
                    for nm2, users in list(r1_users.items()):
                        while users and r0_pool.get(nm2, 0) >= 1:
                            u = users.pop(0)
                            r0_pool[nm2] -= 1
                            # r1 -> r2
                            if assigned_r1_map.get(nm2, 0) > 0:
                                assigned_r1_map[nm2] -= 1
                            assigned_r2_map[nm2] = assigned_r2_map.get(nm2, 0) + 1
                            r2_make[nm2].append(u)
                    # 再跨名称消耗（选择 r0 库充足的名称扣 1 个）
                    leftover_r1 = []
                    for nm2, users in r1_users.items():
                        leftover_r1 += [(nm2, u) for u in users]
                    if leftover_r1:
                        for orig_name, u in leftover_r1:
                            pick = None
                            for name_try, cntv in sorted(r0_pool.items(), key=lambda kv: -kv[1]):
                                if cntv >= 1:
                                    pick = name_try
                                    break
                            if not pick:
                                break
                            r0_pool[pick] -= 1
                            if assigned_r1_map.get(orig_name, 0) > 0:
                                assigned_r1_map[orig_name] -= 1
                            assigned_r2_map[pick] = assigned_r2_map.get(pick, 0) + 1
                            r2_make[pick].append(u)

                    # 重新计算仍需提升的数量
                    full_cnt_total = sum(per_sc_assigned_by_level[(slot, cat)].get('full', {}).values())
                    curr_r2_cnt = sum(assigned_r2_map.values())
                    nonfull_target = max(0, total_roles - full_cnt_total)
                    need_promote = max(0, nonfull_target - curr_r2_cnt)

                    # 可推进的r2数量（r0 -> r2：3个r0合成1个r2）
                    promotable = sum(v // 3 for v in r0_pool.values())
                    promote_cnt = min(need_promote, promotable)

                    # 具体分配到名称与角色：从已分配r0的角色名单中挑选，严格按“3个r0合成1个r2”
                    r0_users = {nm2: list(users) for nm2, users in per_sc_to_strengthen.get((slot, cat), {}).get('r0', {}).items()}
                    assigned_r0_map = per_sc_assigned_by_level[(slot, cat)].get('r0', {}).copy()

                    # 第一阶段：优先用用户当前名称的 r0 进行合成（同名）
                    for nm2, users in list(r0_users.items()):
                        while promote_cnt > 0 and assigned_r0_map.get(nm2, 0) > 0 and r0_pool.get(nm2, 0) >= 2 and users:
                            u = users.pop(0)
                            # 消耗3个r0（其中1个为用户当前的r0，另2个来自池）
                            r0_pool[nm2] -= 3
                            assigned_r0_map[nm2] -= 1
                            assigned_r2_map[nm2] = assigned_r2_map.get(nm2, 0) + 1
                            r2_make[nm2].append(u)
                            promote_cnt -= 1

                    # 第二阶段：跨名称合成。为剩余用户选择任一名称的 r0 池（>=2）
                    # 若选择不同名称，则用户原 r0 返还到池（+1）
                    leftover = []
                    for nm2, users in r0_users.items():
                        for u in users:
                            leftover.append((nm2, u))
                    # 高 r0 库存名称优先
                    leftover.sort(key=lambda x: -max(r0_pool.get(x[0], 0), 0))
                    for orig_name, u in leftover:
                        if promote_cnt <= 0:
                            break
                        # 选择一个有资源的名称
                        pick = None
                        for name_try, cntv in sorted(r0_pool.items(), key=lambda kv: -kv[1]):
                            if cntv >= 2:
                                pick = name_try
                                break
                        if not pick:
                            break
                        # 应用跨名称合成：消耗 pick 的 r0×3，返还 orig_name 的 r0×1
                        r0_pool[pick] -= 3
                        r0_pool[orig_name] = r0_pool.get(orig_name, 0) + 1
                        # 更新已分配计数
                        if assigned_r0_map.get(orig_name, 0) > 0:
                            assigned_r0_map[orig_name] -= 1
                        assigned_r2_map[pick] = assigned_r2_map.get(pick, 0) + 1
                        r2_make[pick].append(u)
                        promote_cnt -= 1

                    # 修正 per_sc_to_strengthen：被提升的用户从 r0 名单删除
                    if r2_make:
                        for name2, users in r2_make.items():
                            for nm2, user_list in list(per_sc_to_strengthen.get((slot, cat), {}).get('r0', {}).items()):
                                per_sc_to_strengthen[(slot, cat)]['r0'][nm2] = [x for x in user_list if x not in users]

                    # 修正“分配”展示：将 r0 调减并计入 r2
                    def fmt_assign_with_override_from_maps(r2_map, r0_map):
                        # 基于调整后的计数构造字符串
                        def fmt_level(m):
                            if not m:
                                return ''
                            names_sorted = sorted(m.keys(), key=lambda n: m[n], reverse=True)
                            return '， '.join(f"{m[name]}件: {name}" for name in names_sorted if m[name] > 0)
                        r2_str = fmt_level(assigned_r2_map)
                        r0_str = fmt_level(assigned_r0_map)
                        return r2_str, r0_str

                    assign_r2, assign_r0 = fmt_assign_with_override_from_maps(assigned_r2_map, assigned_r0_map)
                    # 未满强化件数 = r2 + r1（不含 r0，因为已按 r2待合成处理）
                    nonfull_cnt = sum(assigned_r2_map.values()) + sum(per_sc_assigned_by_level[(slot, cat)].get('r1', {}).values())
                    gap = 0

                # 若存在“5强大于最优4强”的扩展最优情况，标注(5B>4A)
                mark_5b4a = ''
                try:
                    # 选一个原始最优 ex_id（A）与一个扩展非最优 ex_id（B），比较 5B 与 4A
                    best_names = set(per_sc_bestname_counts.get((slot, cat), {}).keys())
                    # 代表角色
                    ref_uid = ref_role_by_sc.get((slot, cat), next(iter(per_sc_roles[(slot, cat)])))
                    # 任取一个原始最优 ex_id
                    any_best_ex_id = None
                    for nm, tpl in snap.items():
                        _,_,_,_, exid = tpl if len(tpl)==5 else (*tpl, None)
                        if exid and db.ex_equipment_data[exid].name in best_names:
                            any_best_ex_id = exid
                            break
                    # 找一个非最优的 ex_id
                    any_alt_ex_id = None
                    for nm, tpl in snap.items():
                        _,_,_,_, exid = tpl if len(tpl)==5 else (*tpl, None)
                        if exid and db.ex_equipment_data[exid].name not in best_names:
                            any_alt_ex_id = exid
                            break
                    if any_best_ex_id and any_alt_ex_id:
                        best_lv4 = estimate_power_increase_at_level(ref_uid, slot, any_best_ex_id, 4)
                        alt_lv5 = estimate_power_increase_at_level(ref_uid, slot, any_alt_ex_id, 5)
                        if alt_lv5 > best_lv4:
                            mark_5b4a = ' (5B>4A)'
                except Exception:
                    pass

                self._log(f"槽位{slot}/类别{cat}: {total_roles}个角色{mark_5b4a}")
                self._log(f"**库存**：{snap_str}")
                parts = []
                if assign_full:
                    parts.append(f"满强: {assign_full}")
                if assign_r2:
                    parts.append(f"待强化r2: {assign_r2}")
                if assign_r1:
                    parts.append(f"待强化r1: {assign_r1}")
                if assign_r0:
                    parts.append(f"待强化r0: {assign_r0}")
                assign_line = '； '.join(parts) if parts else '无分配'
                self._log(f"**分配**: {assign_line} | 未满强化: {nonfull_cnt} | 缺口点数: {gap}")

                # 待强化列表：分别按 r2待合成 / r2/r1/r0，超过10人仅展示人数
                strengthen = per_sc_to_strengthen.get((slot, cat), {})
                strengthen_parts = []
                if r2_make:
                    segs = []
                    for nm2, users in r2_make.items():
                        if len(users) > 10:
                            segs.append(f"{nm2}：{len(users)}人")
                        else:
                            segs.append(f"{nm2}：" + '、'.join(users))
                    strengthen_parts.append("r2待合成：" + '； '.join(segs))
                for level in ['r2','r1','r0']:
                    level_map = strengthen.get(level, {})
                    if not level_map:
                        continue
                    segs = []
                    for nm2, users in level_map.items():
                        if not users:
                            continue
                        if len(users) > 10:
                            segs.append(f"{nm2}：{len(users)}人")
                        else:
                            segs.append(f"{nm2}：" + '、'.join(users))
                    if segs:
                        strengthen_parts.append(f"{level}：" + '； '.join(segs))
                if strengthen_parts:
                    self._log("**待强化**：" + ' | '.join(strengthen_parts))
            self._log("===== 模拟装备结果结束 =====")

        # 打印非满强记录：仅真实装备时打印详细，模拟模式下不打印以避免过长
        if auto_equip and logs_non_full:
            for uname, slot, nm, rk, pt in logs_non_full:
                self._log(f"非满强装备：{uname} 槽位{slot} 装备{nm} rank={rk} pt={pt}")

        self._log("自动装备（或模拟）完成")
