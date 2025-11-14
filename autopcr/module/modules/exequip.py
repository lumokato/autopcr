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
        from .exequip_helpers import (
            ExEquipPowerCalculator,
            ExEquipRecommender,
            ExEquipInventoryManager,
            ExEquipConstants
        )
        from .exequip_auto_equipper import ExEquipAutoEquipper

        simulate_auto_equip: bool = self.get_config('simulate_auto_equip')
        auto_equip: bool = self.get_config('auto_equip')

        # 步骤1: 创建辅助对象
        calculator = ExEquipPowerCalculator(client)
        recommender = ExEquipRecommender(client, calculator)

        # 步骤2: 计算推荐
        recommendations = recommender.calculate_recommendations()

        if not recommendations['unit_slot_recommendations']:
            raise SkipError("没有找到可推荐的3星EX装备")

        # 步骤3: 执行装备（新需求总是执行装备，不再有纯输出汇总的模式）
        if auto_equip or simulate_auto_equip:
            auto_equipper = ExEquipAutoEquipper(client, calculator, self)
            await auto_equipper.execute(recommendations, auto_equip)
        else:
            # 如果两个都是False，仍然执行模拟模式
            auto_equipper = ExEquipAutoEquipper(client, calculator, self)
            await auto_equipper.execute(recommendations, False)
    
