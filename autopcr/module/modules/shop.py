from abc import abstractmethod
from collections import Counter
from typing import List

from ...model.common import ShopInfo
from ...model.custom import ItemType
from ..modulebase import *
from ..config import *
from ...core.pcrclient import pcrclient
from ...model.error import *
from ...db.database import db
from ...model.enums import *
from .sync_growth import build_sync_growth_plan

class shop_buyer(Module):
    def _get_count(self, name: str, key: str) -> int:
        if name not in self.buy_kind():
            return -999999
        return self.get_config(key)
    def _exp_count(self):
        return self._get_count('经验药水', 'shop_buy_exp_count_limit')
    def _equip_count(self):
        return self._get_count('装备', 'shop_buy_equip_count_limit')
    def _equip_raw_ore_count(self):
        return self._get_count('原矿', 'shop_buy_equip_raw_ore_count_limit')
    def _equip_upper_count(self):
        return self._get_count('强化石', 'shop_buy_equip_upper_count_limit')
    def _unit_memory_count(self):
        return self._get_count('记忆碎片', 'shop_buy_memory_count_limit')

    @abstractmethod
    def coin_limit(self) -> int: ...
    @abstractmethod
    def system_id(self) -> eSystemId: ...
    @abstractmethod
    def reset_count(self) -> int: ...
    @abstractmethod
    def buy_kind(self) -> List[str]: ...

    def require_equip_units_fav(self) -> bool:
        return False

    def require_equip_units_rank(self) -> str:
        return '所有'

    async def _get_shop(self, client: pcrclient):
        res = await client.get_shop_item_list()
        for shop in res.shop_list:
            if shop.system_id == self.system_id().value:
                return shop
        raise SkipError("商店未开启")

    async def do_task(self, client: pcrclient):
        lmt = self.coin_limit()
        reset_cnt = self.reset_count()

        shop_content = await self._get_shop(client)

        prev = client.data.get_shop_gold(shop_content.system_id)
        old_reset_cnt = shop_content.reset_count
        result = []

        while True:
            opt: Dict[Union[int, str], int] = {
                '所有': 1,
                '最高': db.equip_max_rank,
                '次高': db.equip_max_rank - 1,
                '次次高': db.equip_max_rank - 2,
            }
            equip_demand_gap = client.data.get_equip_demand_gap(like_unit_only=self.require_equip_units_fav(), start_rank=opt[self.require_equip_units_rank()])

            memory_demand_gap = client.data.get_memory_demand_gap()

            gold = client.data.get_shop_gold(shop_content.system_id)
            if gold < lmt:
                raise SkipError(f"商店货币{gold}不足{lmt}，将不进行购买")

            target = [
                (item.slot_id, item.price.currency_num) for item in shop_content.item_list if not item.sold and
                    (
                        (db.is_exp_upper((item.type, item.item_id)) and client.data.get_inventory((item.type, item.item_id)) < self._exp_count()) or
                        (db.is_equip_upper((item.type, item.item_id)) and client.data.get_inventory((item.type, item.item_id)) < self._equip_upper_count()) or
                        (db.is_equip_raw_ore((item.type, item.item_id)) and client.data.get_inventory((item.type, item.item_id)) < self._equip_raw_ore_count()) or
                        (db.is_equip((item.type, item.item_id)) and -equip_demand_gap[(item.type, item.item_id)] < self._equip_count()) or
                        (db.is_unit_memory((item.type, item.item_id)) and -memory_demand_gap[(item.type, item.item_id)] < self._unit_memory_count())
                    )
            ]

            if len(target) == 0 and all(-it >= self._equip_count() for it in equip_demand_gap.values()):
                self._log(f'商店物品全部盈余，停止购买')
                break

            slots_to_buy = [item[0] for item in target]
            cost_gold = sum([item[1] for item in target])

            if cost_gold > gold: # 货币不足
                self._log(f"商店货币{gold}不足购买需求的{cost_gold}，停止购买")
                break
            
            if slots_to_buy:
                res = await client.shop_buy_item(shop_content.system_id, slots_to_buy)
                gold -= cost_gold
                result.extend(res.purchase_list)
            # else: # 无商品购买还需要重置吗
            #     break

            if shop_content.reset_count >= reset_cnt:
                self._log(f"商店已重置{shop_content.reset_count}次，停止购买")
                break
            
            if gold < shop_content.reset_cost:
                self._log(f"商店货币{gold}不足重置{shop_content.reset_cost}，停止购买")
                break
            await client.shop_reset(shop_content.system_id)
            shop_content = await self._get_shop(client)

        cost_gold = prev - client.data.get_shop_gold(shop_content.system_id)
        if cost_gold == 0:
            raise SkipError("无对应商品购买")
        else:
            self._log(f"花费了{cost_gold}货币，重置了{shop_content.reset_count - old_reset_cnt}次，购买了:")
            msg = await client.serialize_reward_summary(result)
            self._log(msg)

@singlechoice('shop_buy_exp_count_limit', "经验药水储备", 999000, [1000, 10000, 100000, 500000, 999000])
@singlechoice('shop_buy_equip_upper_count_limit', "强化石储备", 999000, [1000, 10000, 100000, 500000, 999000])
@singlechoice('normal_shop_buy_coin_limit', "货币阈值", 5000000, [0, 5000000, 10000000, 20000000])
@inttype('normal_shop_reset_count', "重置次数(<=20)", 0, [i for i in range(21)])
@multichoice("normal_shop_buy_kind", "购买种类", ['经验药水', '强化石'], ['经验药水', '强化石'])
@description('')
@name('通用商店购买')
@default(False)
class normal_shop(shop_buyer):
    def coin_limit(self) -> int: return self.get_config('normal_shop_buy_coin_limit')
    def system_id(self) -> eSystemId: return eSystemId.NORMAL_SHOP
    def reset_count(self) -> int: return self.get_config('normal_shop_reset_count')
    def buy_kind(self) -> List[str]: return self.get_config('normal_shop_buy_kind')

@singlechoice('limit_shop_buy_coin_limit', "货币阈值", 5000000, [0, 5000000, 10000000, 20000000])
@multichoice("limit_shop_buy_kind", "购买种类", ['经验药水', '装备', '原矿'], ['经验药水', '装备', '原矿'])
@description('此项购买不使用最大值')
@name('限定商店购买')
@default(False)
class limit_shop(shop_buyer):
    def _exp_count(self): return 999000 if "经验药水" in self.get_config('limit_shop_buy_kind') else 0
    def _equip_count(self): return 999000 if "装备" in self.get_config('limit_shop_buy_kind') else -999000
    def _equip_raw_ore_count(self): return 999000 if "原矿" in self.get_config('limit_shop_buy_kind') else 0
    def coin_limit(self) -> int: return self.get_config('limit_shop_buy_coin_limit')
    def system_id(self) -> eSystemId: return eSystemId.LIMITED_SHOP
    def reset_count(self) -> int: return 0
    def buy_kind(self) -> List[str]: return self.get_config('limit_shop_buy_kind')

@singlechoice('underground_shop_buy_memory_count_limit', "记忆碎片盈余值", 0, [0, 10, 20, 120, 270, 9900])
@singlechoice('underground_shop_buy_equip_count_limit', "装备盈余值", 0, [0, 20, 50, 100, 200, 500, 9900])
@singlechoice('underground_shop_buy_coin_limit', "货币阈值", 10000, [0, 10000, 50000, 100000, 200000])
@singlechoice("underground_shop_buy_equip_consider_unit_rank", "角色起始品级", "所有", ["所有", "最高", "次高", "次次高"])
@booltype("underground_shop_buy_equip_consider_unit_fav", "收藏角色", False) 
@inttype('underground_shop_reset_count', "重置次数(<=200)", 0, [i for i in range(201)])
@multichoice("underground_shop_buy_kind", "购买种类", ['记忆碎片', '装备'], ['记忆碎片', '装备'])
@name('地下城商店购买')
@description('根据需求购买装备和记忆碎片，可设置需求角色的品级和收藏角色')
@default(False)
class underground_shop(shop_buyer):
    def _equip_count(self):
        return self._get_count('装备', 'underground_shop_buy_equip_count_limit')
    def _unit_memory_count(self):
        return self._get_count('记忆碎片', 'underground_shop_buy_memory_count_limit')
    def coin_limit(self) -> int: return self.get_config('underground_shop_buy_coin_limit')
    def system_id(self) -> eSystemId: return eSystemId.EXPEDITION_SHOP
    def reset_count(self) -> int: return self.get_config('underground_shop_reset_count')
    def buy_kind(self) -> List[str]: return self.get_config('underground_shop_buy_kind')
    def require_equip_units_fav(self) -> bool: return self.get_config('underground_shop_buy_equip_consider_unit_fav')
    def require_equip_units_rank(self) -> str: return self.get_config('underground_shop_buy_equip_consider_unit_rank')


class sync_growth_shop_buyer(shop_buyer):
    def _unit_memory_count(self):
        return -999999

    def _get_fragment_compose_count(self, item: ItemType) -> int:
        if item[0] != eInventoryType.Equip or item[1] not in db.equip_data:
            return 1
        original = db.equip_data[item[1]].original_equipment_id
        if not original:
            return 1
        for material, count in db.equip_craft.get((eInventoryType.Equip, original), []):
            if material == item and count > 0:
                return count
        return 1

    def _get_sync_growth_equip_targets(self, client: pcrclient) -> Tuple[Counter, Counter]:
        _, _, steps = build_sync_growth_plan(client)
        demand = Counter()
        for step in steps:
            demand += step.candidate_consumption

        item_target = Counter()
        category_target = Counter()
        for item, need in demand.items():
            current = client.data.get_inventory(item)
            if current >= need:
                continue
            compose_count = self._get_fragment_compose_count(item)
            rounded_target = ((need + compose_count - 1) // compose_count) * compose_count
            item_target[item] = rounded_target
            category = db.equip_data[item[1]].equipment_category
            category_target[category] += rounded_target

        return item_target, category_target

    def _log_sync_growth_targets(self, item_target: Counter, category_target: Counter):
        if not item_target:
            return
        grouped = {}
        for item, target_count in item_target.items():
            category = db.equip_data[item[1]].equipment_category
            grouped.setdefault(category, []).append((item, target_count))

        for category, items in grouped.items():
            names = "，".join(f"{db.get_inventory_name_san(item)}->{target_count}" for item, target_count in items)
            self._log(f"同步器类型{category}待购买：{names}；总量{category_target[category]}")

    async def do_task(self, client: pcrclient):
        lmt = self.coin_limit()
        reset_cnt = self.reset_count()

        shop_content = await self._get_shop(client)

        prev = client.data.get_shop_gold(shop_content.system_id)
        old_reset_cnt = shop_content.reset_count
        result = []
        equip_item_target, equip_category_target = self._get_sync_growth_equip_targets(client)
        self._log_sync_growth_targets(equip_item_target, equip_category_target)

        while True:
            category_inventory = Counter({
                db.equip_data[item[1]].equipment_category: 0
                for item in equip_item_target
            })
            for token, count in client.data.inventory.items():
                if token[0] != eInventoryType.Equip or token[1] not in db.equip_data:
                    continue
                category = db.equip_data[token[1]].equipment_category
                if category in equip_category_target:
                    category_inventory[category] += count

            gold = client.data.get_shop_gold(shop_content.system_id)
            if gold < lmt:
                raise SkipError(f"商店货币{gold}不足{lmt}，将不进行购买")

            target = [
                (item.slot_id, item.price.currency_num)
                for item in shop_content.item_list
                if not item.sold and db.is_equip((item.type, item.item_id)) and (
                    client.data.get_inventory((item.type, item.item_id)) < equip_item_target[(item.type, item.item_id)] or
                    category_inventory[db.equip_data[item.item_id].equipment_category] < equip_category_target[db.equip_data[item.item_id].equipment_category]
                )
            ]

            if len(target) == 0 and all(
                client.data.get_inventory(item) >= target_count
                for item, target_count in equip_item_target.items()
            ) and all(
                category_inventory[category] >= target_count
                for category, target_count in equip_category_target.items()
            ):
                self._log('当前已无同步器缺口装备需求，停止购买')
                break

            slots_to_buy = [item[0] for item in target]
            cost_gold = sum(item[1] for item in target)

            if cost_gold > gold:
                self._log(f"商店货币{gold}不足购买需求的{cost_gold}，停止购买")
                break

            if slots_to_buy:
                res = await client.shop_buy_item(shop_content.system_id, slots_to_buy)
                gold -= cost_gold
                result.extend(res.purchase_list)

            if shop_content.reset_count >= reset_cnt:
                self._log(f"商店已重置{shop_content.reset_count}次，停止购买")
                break

            if gold < shop_content.reset_cost:
                self._log(f"商店货币{gold}不足重置{shop_content.reset_cost}，停止购买")
                break

            await client.shop_reset(shop_content.system_id)
            shop_content = await self._get_shop(client)

        cost_gold = prev - client.data.get_shop_gold(shop_content.system_id)
        if cost_gold == 0:
            raise SkipError("无对应商品购买")
        else:
            self._log(f"花费了{cost_gold}货币，重置了{shop_content.reset_count - old_reset_cnt}次，购买了")
            msg = await client.serialize_reward_summary(result)
            self._log(msg)


@singlechoice('sync_growth_underground_shop_buy_coin_limit', "货币阈值", 10000, [0, 10000, 50000, 100000, 200000])
@inttype('sync_growth_underground_shop_reset_count', "重置次数(<=200)", 0, [i for i in range(201)])
@name('同步器地下城购装')
@description('按同步器满强化规划购买地下城商店装备，只在需要新拉角色时手动执行')
@default(False)
class sync_growth_underground_shop(sync_growth_shop_buyer):
    def coin_limit(self) -> int: return self.get_config('sync_growth_underground_shop_buy_coin_limit')
    def system_id(self) -> eSystemId: return eSystemId.EXPEDITION_SHOP
    def reset_count(self) -> int: return self.get_config('sync_growth_underground_shop_reset_count')
    def buy_kind(self) -> List[str]: return ['装备']

@singlechoice('jjc_shop_buy_memory_count_limit', "记忆碎片盈余值", 0, [0, 10, 20, 120, 270, 9900])
@singlechoice('jjc_shop_buy_equip_count_limit', "装备盈余值", 0, [0, 20, 50, 100, 200, 500, 9900])
@singlechoice('jjc_shop_buy_coin_limit', "货币阈值", 10000, [0, 10000, 50000, 100000, 200000])
@singlechoice("jjc_shop_buy_equip_consider_unit_rank", "角色起始品级", "所有", ["所有", "最高", "次高", "次次高"])
@booltype("jjc_shop_buy_equip_consider_unit_fav", "收藏角色", False) 
@inttype('jjc_shop_reset_count', "重置次数(<=20)", 0, [i for i in range(21)])
@multichoice("jjc_shop_buy_kind", "购买种类", ['记忆碎片', '装备'], ['记忆碎片', '装备'])
@name('jjc商店购买')
@description('根据需求购买装备和记忆碎片，可设置需求角色的品级和收藏角色')
@default(False)
class jjc_shop(shop_buyer):
    def _equip_count(self):
        return self._get_count('装备', 'jjc_shop_buy_equip_count_limit')
    def _unit_memory_count(self):
        return self._get_count('记忆碎片', 'jjc_shop_buy_memory_count_limit')
    def coin_limit(self) -> int: return self.get_config('jjc_shop_buy_coin_limit')
    def system_id(self) -> eSystemId: return eSystemId.ARENA_SHOP
    def reset_count(self) -> int: return self.get_config('jjc_shop_reset_count')
    def buy_kind(self) -> List[str]: return self.get_config('jjc_shop_buy_kind')
    def require_equip_units_fav(self) -> bool: return self.get_config('jjc_shop_buy_equip_consider_unit_fav')
    def require_equip_units_rank(self) -> str: return self.get_config('jjc_shop_buy_equip_consider_unit_rank')

@singlechoice('pjjc_shop_buy_memory_count_limit', "记忆碎片盈余值", 0, [0, 10, 20, 120, 270, 9900])
@singlechoice('pjjc_shop_buy_equip_count_limit', "装备盈余值", 0, [0, 20, 50, 100, 200, 500, 9900])
@singlechoice('pjjc_shop_buy_coin_limit', "货币阈值", 10000, [0, 10000, 50000, 100000, 200000])
@singlechoice("pjjc_shop_buy_equip_consider_unit_rank", "角色起始品级", "所有", ["所有", "最高", "次高", "次次高"])
@booltype("pjjc_shop_buy_equip_consider_unit_fav", "收藏角色", False) 
@inttype('pjjc_shop_reset_count', "重置次数(<=20)", 0, [i for i in range(21)])
@multichoice("pjjc_shop_buy_kind", "购买种类", ['记忆碎片', '装备'], ['记忆碎片', '装备'])
@name('pjjc商店购买')
@description('根据需求购买装备和记忆碎片，可设置需求角色的品级和收藏角色')
@default(False)
class pjjc_shop(shop_buyer):
    def _equip_count(self):
        return self._get_count('装备', 'pjjc_shop_buy_equip_count_limit')
    def _unit_memory_count(self):
        return self._get_count('记忆碎片', 'pjjc_shop_buy_memory_count_limit')
    def coin_limit(self) -> int: return self.get_config('pjjc_shop_buy_coin_limit')
    def system_id(self) -> eSystemId: return eSystemId.GRAND_ARENA_SHOP
    def reset_count(self) -> int: return self.get_config('pjjc_shop_reset_count')
    def buy_kind(self) -> List[str]: return self.get_config('pjjc_shop_buy_kind')
    def require_equip_units_fav(self) -> bool: return self.get_config('pjjc_shop_buy_equip_consider_unit_fav')
    def require_equip_units_rank(self) -> str: return self.get_config('pjjc_shop_buy_equip_consider_unit_rank')


@singlechoice('clanbattle_shop_buy_memory_count_limit', "记忆碎片盈余值", 0, [0, 10, 20, 120, 270, 9900])
@singlechoice('clanbattle_shop_buy_equip_count_limit', "装备盈余值", 0, [0, 20, 50, 100, 200, 500, 9900])
@singlechoice('clanbattle_shop_buy_coin_limit', "货币阈值", 10000, [0, 10000, 50000, 100000, 200000])
@singlechoice("clanbattle_shop_buy_equip_consider_unit_rank", "角色起始品级", "所有", ["所有", "最高", "次高", "次次高"])
@booltype("clanbattle_shop_buy_equip_consider_unit_fav", "收藏角色", False) 
@inttype('clanbattle_shop_reset_count', "重置次数(<=20)", 0, [i for i in range(21)])
@multichoice("clanbattle_shop_buy_kind", "购买种类", ['记忆碎片'], ['记忆碎片', '装备'])
@name('会战商店购买')
@description('根据需求购买装备和记忆碎片，可设置需求角色的品级和收藏角色')
@default(False)
class clanbattle_shop(shop_buyer):
    def _equip_count(self):
        return self._get_count('装备', 'clanbattle_shop_buy_equip_count_limit')
    def _unit_memory_count(self):
        return self._get_count('记忆碎片', 'clanbattle_shop_buy_memory_count_limit')
    def coin_limit(self) -> int: return self.get_config('clanbattle_shop_buy_coin_limit')
    def system_id(self) -> eSystemId: return eSystemId.CLAN_BATTLE_SHOP
    def reset_count(self) -> int: return self.get_config('clanbattle_shop_reset_count')
    def buy_kind(self) -> List[str]: return self.get_config('clanbattle_shop_buy_kind')
    def require_equip_units_fav(self) -> bool: return self.get_config('clanbattle_shop_buy_equip_consider_unit_fav')
    def require_equip_units_rank(self) -> str: return self.get_config('clanbattle_shop_buy_equip_consider_unit_rank')


class master_shop_buyer(Module):
    @abstractmethod
    def get_buy_items(self, shop: ShopInfo, client: pcrclient) -> List: ...

    async def do_task(self, client: pcrclient):
        shop_id = eSystemId.COUNTER_STOP_SHOP
        shops = {shop.system_id: shop for shop in (await client.get_shop_item_list()).shop_list}

        master_shop = shops.get(shop_id, None)
        if not master_shop:
            raise SkipError("大师店未开启")

        items = self.get_buy_items(master_shop, client)
        cost = sum((item.stock_count - item.purchase_count) * item.price.currency_num for item in items)
        golds = client.data.get_shop_gold(shop_id)
        if cost > golds:
            raise AbortError(f"大师币不足{golds}<{cost}")

        buy = Counter({item.slot_id: item.stock_count - item.purchase_count
            for item in items
        })

        ret = await client.shop_buy_bulk(shop_id, buy)
        msg = await client.serlize_reward(ret.purchase_list)
        self._log(f"花费{cost}大师币购买了:\n{msg}")

@singlechoice('master_shop_buy_memory_count_limit', "记忆碎片盈余值", 0, [0, 10, 20, 40, 120])
@LimitUnitListConfig('master_shop_buy_memory_ids', "记忆碎片")
@description('购买大师币商店的指定记忆碎片，直到碎片盈余超过阈值')
@name('记忆碎片购买')
@default(False)
class master_shop(master_shop_buyer):
    def get_buy_items(self, shop: ShopInfo, client: pcrclient) -> List:
        master_memory_ids = self.get_config('master_shop_buy_memory_ids')
        target_memory = set(db.unit_to_memory[i] for i in master_memory_ids)
        memory_demand_gap = client.data.get_memory_demand_gap()
        master_shop_buy_memory_count_limit = self.get_config('master_shop_buy_memory_count_limit')
        target_memory = {item_id for item_id in target_memory if -memory_demand_gap
            [(eInventoryType.Item, item_id)] < master_shop_buy_memory_count_limit}
        if not target_memory:
            raise SkipError("指定记忆碎片盈余值均满足")
        items = [item for item in shop.item_list if item.item_id in target_memory and
            not item.sold]
        if not items:
            raise SkipError("需购买的碎片已售罄")
        return items

@description('购买大师币商店里的属性材料')
@name('属性材料购买')
@default(False)
class master_shop_talent(master_shop_buyer):
    def get_buy_items(self, shop: ShopInfo, client: pcrclient) -> List:
        items = [item for item in shop.item_list if db.is_talent_material((item.type, item.item_id)) and not item.sold]
        if not items:
            raise SkipError("属性材料已售罄")
        return items
