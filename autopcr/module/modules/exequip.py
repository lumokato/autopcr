from ...util.linq import flow
from ...util.ilp_solver import ex_equip_power_max_cost_flow
from ...model.common import ExtraEquipChangeSlot, ExtraEquipChangeUnit, InventoryInfoPost, AlcesData, ExtraEquipSubStatus
from ..modulebase import *
from ..config import *
from ...core.pcrclient import pcrclient
from ...model.error import *
from ...db.database import db
from ...model.enums import *
from collections import Counter
from ...model.custom import UnitAttribute


def _normal_ex_equip_state(client: pcrclient):
    return {
        str(unit_id): {str(ex_slot.slot): ex_slot.serial_id for ex_slot in unit.ex_equip_slot}
        for unit_id, unit in client.data.unit.items()
    }

def _group_ex_equip_changes(changes):
    grouped = {}
    for unit_id, slot, serial_id in changes:
        grouped.setdefault(unit_id, []).append(ExtraEquipChangeSlot(slot=slot, serial_id=serial_id))
    return [ExtraEquipChangeUnit(unit_id=unit_id, ex_equip_slot=slots, cb_ex_equip_slot=None) for unit_id, slots in grouped.items()]

def _ex_equip_state_cache_path(module: Module):
    from os.path import join
    return join(MODULE_STATE_DIR, "ex_equip_state", module._parent.id + ".json")

def _save_ex_equip_state_cache(module: Module, state):
    from os import makedirs
    from os.path import dirname
    import json
    cache_path = _ex_equip_state_cache_path(module)
    makedirs(dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(state, f)

def _load_ex_equip_state_cache(module: Module):
    from os.path import exists
    import json
    cache_path = _ex_equip_state_cache_path(module)
    if not exists(cache_path):
        return None
    with open(cache_path, "r") as f:
        return json.load(f)


@name('彩装究极炼成')
@default(True)
@inttype('ex_equip_rainbow_enhance_pt_hold', '保留pt数(w)', 10, list(range(0, 10000)))
@ExEquipSubStatusRankConfig('ex_equip_rainbow_enhance_rank', '属性优先级')
@inttype('ex_equip_rainbow_enhance_no_max_num', '非满属性个数', 1, [0, 1, 2, 3, 4])
@ExEquipSubStatusConfig('ex_equip_rainbow_enchance_sub_status_4', '炼成属性4')
@ExEquipSubStatusConfig('ex_equip_rainbow_enchance_sub_status_3', '炼成属性3')
@ExEquipSubStatusConfig('ex_equip_rainbow_enchance_sub_status_2', '炼成属性2')
@ExEquipSubStatusConfig('ex_equip_rainbow_enchance_sub_status_1', '炼成属性1')
@texttype('ex_equip_rainbow_enchance_id', '彩装id', 0)
@singlechoice('ex_equip_rainbow_enchance_action', '做什么', '看属性', ['看属性', '炼成', '看概率'])
@description('看属性指获取彩装id和炼成属性,炼成则进行究极炼成,看概率指根据炼成记录统计各属性概率,非满属性指属性值不必最大,以便手动用光球强化.属性优先级指目标属性值一样时,比较其他属性决定保留或放弃,优先级是按顺序从高到低,目标属性的优先级最高,不受属性优先级影响.满强目标属性会自动锁住.')
class ex_equip_rainbow_enchance(Module):

    async def do_task(self, client: pcrclient):
        ex_equip_rainbow_enchance_action = self.get_config('ex_equip_rainbow_enchance_action')
        if ex_equip_rainbow_enchance_action == '看属性':
            items = flow(client.data.ex_equips.values()) \
                .where(lambda ex: db.get_ex_equip_rarity(ex.ex_equipment_id) == 5) \
                .to_list()
            # 装备 ID 编码了类型和元素/名称，直接按 ID 排序即可同时实现分组+元素排序，无需查表
            items.sort(key=lambda ex: ex.ex_equipment_id)
            msg = flow(items) \
                .select(lambda ex: f"{ex.serial_id}: {db.get_ex_equip_name(ex.ex_equipment_id)} "
                                  f"{db.get_ex_equip_sub_status_str(ex.ex_equipment_id, ex.sub_status or [])}") \
                .to_list()
            cnt = len(msg)
            if not cnt:
                raise SkipError("无彩装")
            msg = '\n'.join(msg)
            self._log(f"{cnt}件彩装:\n{msg}")
        elif ex_equip_rainbow_enchance_action == '看概率':
            for equip, data in self.iter_cache():
                total = sum(data.values())
                if total == 0:
                    continue
                equip = int(equip)
                self._log(f"{db.get_ex_equip_name(equip)}({total}次词条刷新)")
                info = flow(data.items()) \
                    .where(lambda kv: kv[0] != 'total') \
                    .select(lambda kv: (list(map(int, kv[0].split('-'))), kv[1])) \
                    .group_by(lambda kv: kv[0][0]) \
                    .to_dict(lambda g: g.key, lambda g: g.to_list())
                for status in sorted(info):
                    status_info = info[status]
                    status_info.sort(key=lambda x: x[0][1])
                    msg = '/'.join([f"{db.get_ex_equip_sub_status_str(equip, [ExtraEquipSubStatus(status=k, step=s)])}: {v/total * 100:.2f}%" for (k, s), v in status_info])
                    self._log(f"  {msg}")
                self._log("")

        elif ex_equip_rainbow_enchance_action == '炼成':
            serial_id = self.get_config('ex_equip_rainbow_enchance_id')
            if not serial_id.isdigit():
                raise AbortError("彩装id非法")
            serial_id = int(serial_id)
            if serial_id not in client.data.ex_equips or db.get_ex_equip_rarity(client.data.ex_equips[serial_id].ex_equipment_id) != 5:
                raise AbortError("彩装id不存在")
            
            group = db.ex_equipment_sub_status_group[client.data.ex_equips[serial_id].ex_equipment_id]
            sub_status_data = db.ex_equipment_sub_status[group.group_id]
            target_sub_status = Counter()
            for i in range(1, 5):
                val = self.get_config(f'ex_equip_rainbow_enchance_sub_status_{i}')
                if val:
                    target_sub_status[val] += 1
            
            invalid_status = [status for status in target_sub_status if status not in sub_status_data]
            if invalid_status:
                raise AbortError(f"炼成属性包含该装备不支持的属性: {', '.join(UnitAttribute.index2ch[eParamType(s)] for s in invalid_status)}")

            self.cache_info = self.find_cache(str(client.data.ex_equips[serial_id].ex_equipment_id))
            if not self.cache_info:
                self.cache_info = Counter()
            else:
                self.cache_info = Counter(self.cache_info)

            base = 1
            self.weight = Counter()
            rank_order = self.get_config('ex_equip_rainbow_enhance_rank')
            for key in rank_order[::-1]:
                if key not in target_sub_status:
                    self.weight[key] += base
                    base *= 30
            for key in target_sub_status:
                self.weight[key] += base

            # self._log(f"各属性加权值: " + ', '.join(f"{UnitAttribute.index2ch[eParamType(k)]}: {v}" for k, v in self.weight.items()))

            top = await client.alces_top()
            if top.pending_alces_data:
                if top.pending_alces_data.serial_id != serial_id:
                    raise AbortError(f"{top.pending_alces_data.serial_id}炼成属性待决定,请先自行决定")
                await self.decide_alces(client, top.pending_alces_data, target_sub_status)

            no_max_num = self.get_config('ex_equip_rainbow_enhance_no_max_num')
            target_cnt = sum(target_sub_status.values())

            if no_max_num > target_cnt:
                raise AbortError(f"非满属性个数{no_max_num}不能大于非任意的目标属性个数{target_cnt}")

            consume_cnt = Counter()
            alces_exec_cnt = 0
            last_lock_cnt = 0
            stop = False

            self._log(f"当前彩装属性 " +
                      f"{serial_id}: {db.get_ex_equip_name(client.data.ex_equips[serial_id].ex_equipment_id)} "
                      f"{db.get_ex_equip_sub_status_str(client.data.ex_equips[serial_id].ex_equipment_id, client.data.ex_equips[serial_id].sub_status or [])}")

            pt_hold = self.get_config('ex_equip_rainbow_enhance_pt_hold')
                
            while not stop:
                achived_max_cnt, achived_cnt = await self.get_achived_sub_status_cnt(client, serial_id, target_sub_status)
                if achived_max_cnt >= target_cnt - no_max_num and achived_cnt >= target_cnt:
                    self._log("彩装炼成属性已达成目标")
                    break

                lock_cnt = await self.do_lock(client, serial_id, target_sub_status)

                if last_lock_cnt != lock_cnt:
                    self._log(f"L 锁定属性个数{last_lock_cnt} -> {lock_cnt}")
                    last_lock_cnt = lock_cnt

                if client.data.get_inventory(db.ex_rainbow_enhance_pt) <= pt_hold * 10000:
                    self._warn(f"彩装究极炼成PT{client.data.get_inventory(db.ex_rainbow_enhance_pt)}<={pt_hold * 10000}，停止炼成")
                    break

                to_consume = Counter()
                for consume, item in db.alces_cost.items():
                    cost = item.count
                    cur = client.data.get_inventory(consume)
                    cost *= lock_cnt + 1
                    to_consume[consume] = cost
                    if cur < cost:
                        self._warn(f"E {db.get_inventory_name_san(consume)}数量{cur}<{cost}，无法进行究极炼成")
                        stop = True
                
                if stop:
                    break
                
                consume_cnt += to_consume

                resp = await client.alces_exec(serial_id)
                accept = await self.decide_alces(client, resp.pending_alces_data, target_sub_status)
                alces_exec_cnt += 1

                self._log(f"{'A 接受' if accept else 'R 放弃'}炼成属性: {db.get_ex_equip_sub_status_str(client.data.ex_equips[serial_id].ex_equipment_id, resp.pending_alces_data.sub_status or [])}")

            if alces_exec_cnt:
                self._log(f"共进行了{alces_exec_cnt}次究极炼成，消耗了：")
                for consume in consume_cnt:
                    self._log(f"  {db.get_inventory_name_san(consume)} x {consume_cnt[consume]}")
            self.save_cache(str(client.data.ex_equips[serial_id].ex_equipment_id), self.cache_info)
            self._log(f"最终彩装属性 " +
                      f"{serial_id}: {db.get_ex_equip_name(client.data.ex_equips[serial_id].ex_equipment_id)} "
                      f"{db.get_ex_equip_sub_status_str(client.data.ex_equips[serial_id].ex_equipment_id, client.data.ex_equips[serial_id].sub_status or [])}")
        else:
            raise AbortError(f"未知操作{ex_equip_rainbow_enchance_action}")
    
    async def do_lock(self, client: pcrclient, serial_id: int, target_sub_status: Counter):
        current_max_sub_status = Counter()
        lock_cnt = 0
        for status in client.data.ex_equips[serial_id].sub_status or []:
            to_lock = False
            if current_max_sub_status[status.status] < target_sub_status[status.status] and status.step == 5:
                current_max_sub_status[status.status] += 1
                to_lock = True
                lock_cnt += 1
            if to_lock != status.is_lock:
                await client.alces_lock_slot(serial_id, status.slot_number, to_lock)
        return lock_cnt

    async def decide_alces(self, client: pcrclient, alces_data: AlcesData, target_sub_status: Counter):
        accept = False
        current_max_sub_status = Counter()

        for status in alces_data.sub_status:
            if status.is_lock:
                current_max_sub_status[status.status] += 1
                continue
            self.cache_info[f"{status.status}-{status.step}"] += 1
            if current_max_sub_status[status.status] < target_sub_status[status.status] and status.step == 5:
                current_max_sub_status[status.status] += 1
                accept = True
        
        if not accept:
            current_score = sum(status.step * self.weight[status.status] for status in client.data.ex_equips[alces_data.serial_id].sub_status or [])
            nxt_score = sum(status.step * self.weight[status.status] for status in alces_data.sub_status)
            # self._log(f"当前目标属性加权值{current_score}，新属性加权值{nxt_score}")

            if nxt_score > current_score:
                accept = True

        if accept:
            await client.alces_fix_result(alces_data.serial_id)
        else:
            await client.alces_cancel_result(alces_data.serial_id)
        return accept

    async def get_achived_sub_status_cnt(self, client: pcrclient, serial_id: int, target_sub_status: Counter) -> Tuple[int, int]:
        ex_equip = client.data.ex_equips[serial_id]
        current_sub_status = Counter(status.status for status in ex_equip.sub_status or [])
        max_sub_status = Counter(status.status for status in ex_equip.sub_status or [] if status.step == 5)

        achived_cnt = current_sub_status & target_sub_status
        achived_cnt = sum(achived_cnt.values())
        achived_max_cnt = max_sub_status & target_sub_status
        achived_max_cnt = sum(achived_max_cnt.values())
        return achived_max_cnt, achived_cnt

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


@name('EX状态保存/恢复')
@default(False)
@singlechoice('ex_equip_state_action', '行为', '保存', ['保存', '恢复'])
@description('保存或恢复所有角色当前穿戴的普通EX装备状态。不影响账号配置，恢复时只处理有差异的部分，不会全部卸载。')
class ex_equip_state(Module):
    cache_key = 'state'

    @staticmethod
    def normal_ex_equip_state(client: pcrclient):
        return {
            str(unit_id): {str(ex_slot.slot): ex_slot.serial_id for ex_slot in unit.ex_equip_slot}
            for unit_id, unit in client.data.unit.items()
        }

    @staticmethod
    def group_ex_equip_changes(changes):
        grouped = {}
        for unit_id, slot, serial_id in changes:
            grouped.setdefault(unit_id, []).append(ExtraEquipChangeSlot(slot=slot, serial_id=serial_id))
        return [ExtraEquipChangeUnit(unit_id=unit_id, ex_equip_slot=slots, cb_ex_equip_slot=None) for unit_id, slots in grouped.items()]

    async def do_task(self, client: pcrclient):
        action = self.get_config('ex_equip_state_action')
        if action == '保存':
            await self.save_state(client)
        elif action == '恢复':
            await self.restore_state(client)
        else:
            raise AbortError(f"未知操作{action}")

    async def save_state(self, client: pcrclient):
        state = self.normal_ex_equip_state(client)
        equipped_cnt = sum(1 for slots in state.values() for serial_id in slots.values() if serial_id)
        unit_cnt = sum(1 for slots in state.values() if any(slots.values()))
        self.save_cache(self.cache_key, state)
        self._log(f"已保存{unit_cnt}个角色的{equipped_cnt}件普通EX装备状态")

    async def restore_state(self, client: pcrclient):
        state = self.find_cache(self.cache_key)
        if not state:
            raise AbortError("未找到已保存的EX装备状态，请先执行保存")

        current_state = self.normal_ex_equip_state(client)
        current_position = {
            serial_id: (int(unit_id), int(slot))
            for unit_id, slots in current_state.items()
            for slot, serial_id in slots.items()
            if serial_id
        }

        remove_changes = []
        apply_changes = []
        skipped_missing = []
        touched = set()

        for unit_id in sorted(state.keys(), key=int):
            if unit_id not in current_state:
                continue
            for slot in sorted(state[unit_id].keys(), key=int):
                if slot not in current_state[unit_id]:
                    continue
                target_serial_id = state[unit_id][slot] or 0
                current_serial_id = current_state[unit_id][slot] or 0
                slot_no = int(slot)
                if current_serial_id == target_serial_id:
                    continue
                if target_serial_id and target_serial_id not in client.data.ex_equips:
                    skipped_missing.append(target_serial_id)
                    continue
                if target_serial_id and target_serial_id in current_position:
                    occupy_unit_id, occupy_slot = current_position[target_serial_id]
                    if occupy_unit_id != int(unit_id) or occupy_slot != slot_no:
                        key = (occupy_unit_id, occupy_slot)
                        if key not in touched:
                            remove_changes.append((occupy_unit_id, occupy_slot, 0))
                            touched.add(key)
                apply_changes.append((int(unit_id), slot_no, target_serial_id))

        if skipped_missing:
            skipped = ', '.join(map(str, sorted(set(skipped_missing))))
            self._warn(f"跳过{len(set(skipped_missing))}件已不存在的EX装备(serial_id: {skipped})")

        if not remove_changes and not apply_changes:
            raise SkipError("当前普通EX装备状态与保存状态一致")

        if remove_changes:
            await client.unit_equip_ex(self.group_ex_equip_changes(remove_changes))
        if apply_changes:
            await client.unit_equip_ex(self.group_ex_equip_changes(apply_changes))

        self._log(f"恢复了{len(apply_changes)}个普通EX装备槽位")


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
    
@name('EX装战力最高搭配')
@default(True)
@description('按当前推荐逻辑执行EX装最高战力搭配。仅补空槽模式适合高频使用；重配未锁定模式会先撤下未锁定的非粉装后再重配；撤下全部模式可用于和旧算法直接对比。')
@booltype('ex_equip_power_maximun_do', '执行装备', False)
@singlechoice('ex_equip_power_maximun_mode', '装备模式', '仅补空槽', ['仅补空槽', '重配未锁定', '撤下全部'])
class ex_equip_power_maximun(Module):
    async def do_task(self, client: pcrclient):
        do_equip = self.get_config('ex_equip_power_maximun_do')
        equip_mode = self.get_config('ex_equip_power_maximun_mode')
        # BIG 用于把“普通金优先”作为次级目标，不影响“总战力最大化”这一主目标。
        BIG = 1000
        read_story = set(client.data.read_story_ids)
        coefficient = db.unit_status_coefficient[1]
        removable_serials = set()
        fixed_assignments = {}
        active_slots = []

        for unit_id, unit in client.data.unit.items():
            slot_data = db.unit_ex_equipment_slot.get(unit_id)
            if not slot_data:
                continue

            categories = [slot_data.slot_category_1, slot_data.slot_category_2, slot_data.slot_category_3]
            for slot_id, ex_category in enumerate(categories, start=1):
                if ex_category is None:
                    continue

                ex_slot = unit.ex_equip_slot[slot_id - 1]
                ex = client.data.ex_equips.get(ex_slot.serial_id) if ex_slot.serial_id else None
                is_empty = ex is None
                is_locked = bool(ex and ex.protection_flag == 2)

                if equip_mode == '仅补空槽':
                    if is_empty:
                        active_slots.append((unit_id, slot_id, ex_category))
                    elif ex:
                        fixed_assignments[(unit_id, slot_id)] = ex
                elif equip_mode == '重配未锁定':
                    if is_empty or not is_locked:
                        active_slots.append((unit_id, slot_id, ex_category))
                        if ex:
                            removable_serials.add(ex.serial_id)
                    elif ex:
                        fixed_assignments[(unit_id, slot_id)] = ex
                elif equip_mode == '撤下全部':
                    active_slots.append((unit_id, slot_id, ex_category))
                    if ex:
                        removable_serials.add(ex.serial_id)

        equipped_serials = {
            ex_slot.serial_id
            for unit in client.data.unit.values()
            for ex_slot in unit.ex_equip_slot
            if ex_slot.serial_id
        }
        available_equips = [
            ex
            for ex in client.data.ex_equips.values()
            if ex.serial_id in removable_serials or ex.serial_id not in equipped_serials
        ]
        available_serials = {ex.serial_id for ex in available_equips}

        fixed_power_total = 0
        for (unit_id, slot_id), ex in fixed_assignments.items():
            unit_attr = db.calc_unit_attribute(
                client.data.unit[unit_id],
                read_story,
                client.data.ex_equips,
                exclude_ex_equip=True,
            )
            star = db.get_ex_equip_star_from_pt(ex.ex_equipment_id, ex.enhancement_pt)
            attr = db.ex_equipment_data[ex.ex_equipment_id].get_unit_attribute(star, ex.sub_status)
            bonus = unit_attr.ex_equipment_mul(attr).ceil()
            fixed_power_total += int(bonus.get_power(coefficient) + 0.5)

        edges = []
        st = 'st'
        ed = 'ed'
        slot_choice_power = {}
        slot_choice_assignment = {}

        for unit_id, slot_id, ex_category in active_slots:
            slot_node = f"u{unit_id}k{slot_id}"
            edges.append((st, slot_node, 1, 0))
            unit_attr = db.calc_unit_attribute(
                client.data.unit[unit_id],
                read_story,
                client.data.ex_equips,
                exclude_ex_equip=True,
            )

            considered_normal = set()
            for ex in available_equips:
                ex_id = ex.ex_equipment_id
                ex_data = db.ex_equipment_data[ex_id]
                if ex_data.category != ex_category:
                    continue

                star = db.get_ex_equip_star_from_pt(ex_id, ex.enhancement_pt)
                if db.get_ex_equip_rarity(ex_id) == 5:
                    ex_node = f"r{ex.serial_id}"
                    serial_id = ex.serial_id
                    attr = ex_data.get_unit_attribute(star, ex.sub_status)
                else:
                    normal_key = (ex_id, star)
                    if normal_key in considered_normal:
                        continue
                    considered_normal.add(normal_key)
                    ex_node = f"e{ex_id}s{star}"
                    serial_id = None
                    attr = ex_data.get_unit_attribute(star)

                bonus = unit_attr.ex_equipment_mul(attr).ceil()
                power = int(bonus.get_power(coefficient) + 0.5)
                is_clan = 1 if db.is_clan_ex_equip((eInventoryType.ExtraEquip, ex_id)) else 0
                edges.append((slot_node, ex_node, 1, -(power * BIG) + is_clan))
                choice_key = (unit_id, slot_id, ex_node)
                slot_choice_power[choice_key] = power
                slot_choice_assignment[choice_key] = (ex_id, star, serial_id)

        grouped_inventory = flow(available_equips) \
            .where(lambda ex: db.get_ex_equip_rarity(ex.ex_equipment_id) != 5) \
            .group_by(lambda ex: (ex.ex_equipment_id, db.get_ex_equip_star_from_pt(ex.ex_equipment_id, ex.enhancement_pt))) \
            .to_dict(lambda grp: grp.key, lambda grp: grp.count())
        for (ex_id, star), count in grouped_inventory.items():
            edges.append((f"e{ex_id}s{star}", ed, count, 0))
        for ex in available_equips:
            if db.get_ex_equip_rarity(ex.ex_equipment_id) == 5:
                edges.append((f"r{ex.serial_id}", ed, 1, 0))

        if edges:
            min_cost, strategy = ex_equip_power_max_cost_flow(edges, st, ed)
        elif fixed_assignments:
            min_cost, strategy = 0, []
        else:
            raise SkipError("没有可用于计算的EX装备或槽位")

        active_strategy = {}
        active_power_total = 0
        for u, v, flow_num in strategy:
            if u == st or v == ed or flow_num == 0:
                continue
            if not (u.startswith('u') and 'k' in u and (v.startswith('r') or (v.startswith('e') and 's' in v))):
                continue

            unit_id = int(u[1:u.index('k')])
            slot_id = int(u[u.index('k') + 1:])
            choice_key = (unit_id, slot_id, v)
            assignment = slot_choice_assignment.get(choice_key)
            if assignment is None:
                continue
            active_strategy[(unit_id, slot_id)] = assignment
            active_power_total += slot_choice_power.get(choice_key, 0)

        final_assignments = {}
        for key, ex in fixed_assignments.items():
            final_assignments[key] = (
                ex.ex_equipment_id,
                db.get_ex_equip_star_from_pt(ex.ex_equipment_id, ex.enhancement_pt),
                ex.serial_id,
            )
        final_assignments.update(active_strategy)

        self._log(f"最大战力提升：{fixed_power_total + active_power_total}")

        if do_equip and removable_serials:
            removed_by_unit = {}
            for unit_id, unit in client.data.unit.items():
                exchange_list = [
                    ExtraEquipChangeSlot(slot=ex_slot.slot, serial_id=0)
                    for ex_slot in unit.ex_equip_slot
                    if ex_slot.serial_id in removable_serials
                ]
                if exchange_list:
                    removed_by_unit[unit_id] = exchange_list

            for unit_id, exchange_list in removed_by_unit.items():
                await client.unit_equip_ex([ExtraEquipChangeUnit(
                    unit_id=unit_id,
                    ex_equip_slot=exchange_list,
                    cb_ex_equip_slot=None,
                )])

        if do_equip:
            used_serials = set()
            available_by_key = flow(available_equips) \
                .where(lambda ex: db.get_ex_equip_rarity(ex.ex_equipment_id) != 5) \
                .group_by(lambda ex: (ex.ex_equipment_id, db.get_ex_equip_star_from_pt(ex.ex_equipment_id, ex.enhancement_pt))) \
                .to_dict(
                    lambda grp: grp.key,
                    lambda grp: sorted(grp.to_list(), key=lambda ex: ex.serial_id),
                )

            equip_by_unit = {}
            for (unit_id, slot_id), (ex_id, star, serial_id) in final_assignments.items():
                if (unit_id, slot_id) in fixed_assignments:
                    continue
                if serial_id is not None:
                    ex_to_equip = client.data.ex_equips.get(serial_id)
                    if ex_to_equip and ex_to_equip.serial_id not in available_serials:
                        ex_to_equip = None
                else:
                    candidates = available_by_key.get((ex_id, star), [])
                    ex_to_equip = next((ex for ex in candidates if ex.serial_id not in used_serials), None)
                if ex_to_equip and ex_to_equip.serial_id in used_serials:
                    ex_to_equip = None
                if not ex_to_equip:
                    self._warn(f"无{db.get_ex_equip_name(ex_id)}★{star}，无法装备")
                    continue
                used_serials.add(ex_to_equip.serial_id)
                equip_by_unit.setdefault(unit_id, []).append(
                    ExtraEquipChangeSlot(slot=slot_id, serial_id=ex_to_equip.serial_id)
                )

            for unit_id, exchange_list in equip_by_unit.items():
                await client.unit_equip_ex([ExtraEquipChangeUnit(
                    unit_id=unit_id,
                    ex_equip_slot=sorted(exchange_list, key=lambda ex_slot: ex_slot.slot),
                    cb_ex_equip_slot=None,
                )])

        if do_equip:
            self._log(f"已为{len(set(unit_id for unit_id, _ in final_assignments))}个角色装备了{len(final_assignments)}件EX装备")
        else:
            self._log(f"模拟为{len(set(unit_id for unit_id, _ in final_assignments))}个角色装备{len(final_assignments)}件EX装备")

        output_by_unit = flow(final_assignments.items()) \
            .group_by(lambda item: item[0][0]) \
            .to_dict(lambda grp: grp.key, lambda grp: sorted(grp.to_list(), key=lambda item: item[0][1]))

        for unit_id in sorted(output_by_unit):
            msg = []
            for (_, _), (ex_id, star, serial_id) in output_by_unit[unit_id]:
                ex = client.data.ex_equips.get(serial_id) if serial_id is not None else None
                if ex and db.get_ex_equip_rarity(ex_id) == 5:
                    sub_status = db.get_ex_equip_sub_status_str(ex_id, ex.sub_status or [])
                    msg.append(f"{db.get_ex_equip_name(ex_id)}{star}({sub_status})")
                else:
                    msg.append(f"{db.get_ex_equip_name(ex_id)}{star}")
            self._log(f"{db.get_unit_name(unit_id)} 装备 {','.join(msg)}")
