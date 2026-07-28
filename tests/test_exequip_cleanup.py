import importlib
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from autopcr.module.modules.exequip_cleanup_analyzer import compute_max_possible_r2
from autopcr.model.enums import eInventoryType, eSystemId


cleanup_module = importlib.import_module("autopcr.module.modules.exequip_cleanup")


class ExEquipCraftCapacityTest(unittest.TestCase):
    def test_counts_only_rankups_supported_by_cleanup_executor(self):
        cases = [
            ({"current_full_r2": 1, "current_r2_not_full": 2, "current_r1": 1, "current_r0": 1}, 4),
            ({"current_full_r2": 0, "current_r2_not_full": 0, "current_r1": 5, "current_r0": 1}, 1),
            ({"current_full_r2": 0, "current_r2_not_full": 0, "current_r1": 1, "current_r0": 4}, 2),
            ({"current_full_r2": 0, "current_r2_not_full": 0, "current_r1": 0, "current_r0": 5}, 1),
        ]

        for stats, expected in cases:
            with self.subTest(stats=stats):
                self.assertEqual(compute_max_possible_r2(stats), expected)


class MissingNormalBestBuyPlanningTest(unittest.TestCase):
    @staticmethod
    def row(
        ex_equipment_id: int,
        *,
        best_count: int,
        exclusive_best_count: int = 0,
        shared_best_count: int = 0,
        possible_r2: int = 0,
        is_clan_battle: bool = False,
    ):
        return SimpleNamespace(
            slot_index=1,
            category=1,
            ex_equipment_id=ex_equipment_id,
            equip_name=f"equip-{ex_equipment_id}",
            is_clan_battle=is_clan_battle,
            best_count=best_count,
            exclusive_best_count=exclusive_best_count,
            shared_best_count=shared_best_count,
            current_full_r2=possible_r2,
            current_r2_not_full=0,
            current_r1=0,
            current_r0=0,
        )

    @staticmethod
    def report(*rows):
        return SimpleNamespace(
            slot_reports=[SimpleNamespace(equip_reports=list(rows))]
        )

    def plan(self, *rows):
        with patch.object(cleanup_module.db, "get_ex_equip_rarity", return_value=3):
            return cleanup_module._plan_missing_normal_best_ex_equip_buys(
                self.report(*rows)
            )

    def test_buys_when_clan_tie_would_cover_normal_shortage(self):
        row = self.row(
            4101001,
            best_count=8,
            shared_best_count=4,
            possible_r2=6,
        )

        planned = self.plan(row)

        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["best_demand"], 8)
        self.assertEqual(planned[0]["target"], 10)
        self.assertEqual(planned[0]["shortage"], 4)
        self.assertEqual(planned[0]["buy_count"], 3)

    def test_skips_when_normal_best_itself_covers_demand_and_surplus(self):
        row = self.row(
            4101001,
            best_count=8,
            shared_best_count=4,
            possible_r2=10,
        )

        self.assertEqual(self.plan(row), [])

    def test_never_buys_clan_battle_best(self):
        row = self.row(
            4101001,
            best_count=8,
            possible_r2=0,
            is_clan_battle=True,
        )

        self.assertEqual(self.plan(row), [])


class ExEquipShopBuyTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def planned_buy(ex_equipment_id=4101001, slot_index=1, buy_count=3):
        row = SimpleNamespace(
            ex_equipment_id=ex_equipment_id,
            equip_name=f"equip-{ex_equipment_id}",
            slot_index=slot_index,
        )
        return {
            "row": row,
            "buy_count": buy_count,
            "shortage": 1,
        }

    @staticmethod
    def shop_item(
        ex_equipment_id=4101001,
        *,
        stock_count=3,
        purchase_count=0,
        price=100,
    ):
        return SimpleNamespace(
            type=eInventoryType.ExtraEquip,
            item_id=ex_equipment_id,
            slot_id=7,
            sold=0,
            stock_count=stock_count,
            purchase_count=purchase_count,
            exchange_count=0,
            available_num=None,
            is_unlimited_stock=False,
            price_group=0,
            price=SimpleNamespace(currency_num=price),
        )

    @staticmethod
    def client(shop_item, currency_values):
        shop = SimpleNamespace(
            system_id=eSystemId.EX_EQUIPMENT_WEAPON_SHOP,
            item_list=[shop_item],
        )
        client = SimpleNamespace(
            data=SimpleNamespace(
                get_shop_gold=MagicMock(side_effect=currency_values),
            ),
            get_shop_item_list=AsyncMock(
                return_value=SimpleNamespace(shop_list=[shop])
            ),
            shop_buy_bulk=AsyncMock(),
        )
        return client

    async def test_buys_remaining_monthly_stock_from_matching_ex_shop(self):
        client = self.client(
            self.shop_item(stock_count=3, purchase_count=1),
            [1000, 1000, 800],
        )

        result = await cleanup_module._buy_planned_normal_best_ex_equips(
            client,
            [self.planned_buy()],
        )

        client.shop_buy_bulk.assert_awaited_once_with(
            eSystemId.EX_EQUIPMENT_WEAPON_SHOP,
            Counter({7: 2}),
        )
        self.assertEqual(result["items"][0]["buy_count"], 2)
        self.assertEqual(
            result["cost_by_shop"][eSystemId.EX_EQUIPMENT_WEAPON_SHOP],
            200,
        )

    async def test_buys_affordable_partial_count_and_reports_remainder(self):
        client = self.client(
            self.shop_item(price=100),
            [150, 150, 50],
        )

        result = await cleanup_module._buy_planned_normal_best_ex_equips(
            client,
            [self.planned_buy()],
        )

        client.shop_buy_bulk.assert_awaited_once_with(
            eSystemId.EX_EQUIPMENT_WEAPON_SHOP,
            Counter({7: 1}),
        )
        self.assertEqual(result["items"][0]["buy_count"], 1)
        self.assertIn("仅购买1/3", result["skipped"][0])


class ExEquipRecycleTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def ex(serial_id, rank, enhancement_pt):
        return SimpleNamespace(
            serial_id=serial_id,
            ex_equipment_id=4101001,
            rank=rank,
            enhancement_pt=enhancement_pt,
            protection_flag=1,
        )

    @staticmethod
    def client(*equips):
        return SimpleNamespace(
            data=SimpleNamespace(
                ex_equips={item.serial_id: item for item in equips},
                unit={},
                user_clan_battle_ex_equip_restriction={},
                settings=SimpleNamespace(
                    ex_equip=SimpleNamespace(ex_equip_limit_consume_num=100)
                ),
            ),
            item_recycle_ex=AsyncMock(),
        )

    async def test_preserves_rank_zero_progress_while_target_is_unmet(self):
        client = self.client(self.ex(1, 0, 0))

        with patch.object(cleanup_module.db, "get_ex_equip_rarity", return_value=3):
            actions = await cleanup_module._recycle_excess(
                client,
                4101001,
                keep_total=1,
            )

        self.assertEqual(actions, 0)
        client.item_recycle_ex.assert_not_awaited()

    async def test_recycles_lowest_enhanced_surplus_rank_two_first(self):
        client = self.client(
            self.ex(1, 2, 100),
            self.ex(2, 2, 500),
            self.ex(3, 2, 6000),
        )

        with patch.object(cleanup_module.db, "get_ex_equip_rarity", return_value=3):
            actions = await cleanup_module._recycle_excess(
                client,
                4101001,
                keep_total=2,
            )

        self.assertEqual(actions, 1)
        client.item_recycle_ex.assert_awaited_once_with([1])


class ExEquipCleanupExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_mode_buys_and_keeps_to_purchase_target(self):
        parent = SimpleNamespace(id="test", legacy_id=None, alias="test")
        module = cleanup_module.ex_equip_cleanup_execute(parent)
        configs = {
            "ex_equip_cleanup_execute_apply": True,
            "ex_equip_cleanup_execute_prepare": False,
            "ex_equip_cleanup_normal_floor_total": 5,
            "ex_equip_cleanup_clan_floor_total": 10,
            "ex_equip_cleanup_enhance_mode": "强化一半",
            "ex_equip_cleanup_clan_full_cap": 20,
            "ex_equip_cleanup_buy_missing_normal_best": True,
        }
        module.get_config = MagicMock(side_effect=configs.__getitem__)
        client = SimpleNamespace(
            data=SimpleNamespace(ex_equips={}, unit={}),
        )
        row = SimpleNamespace(
            ex_equipment_id=4101001,
            is_clan_battle=False,
            best_count=8,
            keep_target_min=5,
            full_target=0,
            current_total=0,
            current_full_r2=0,
        )
        report = SimpleNamespace(
            slot_reports=[SimpleNamespace(slot_index=1, equip_reports=[row])]
        )
        planned = [{"row": row, "target": 10}]
        shop_result = cleanup_module._empty_ex_equip_shop_buy_result()

        with patch.object(
            cleanup_module,
            "ExEquipCleanupAnalyzer",
        ) as analyzer, patch.object(
            cleanup_module,
            "_plan_missing_normal_best_ex_equip_buys",
            return_value=planned,
        ), patch.object(
            cleanup_module,
            "_buy_planned_normal_best_ex_equips",
            new_callable=AsyncMock,
            return_value=shop_result,
        ) as buyer, patch.object(
            cleanup_module,
            "_rankup_to_target",
            new_callable=AsyncMock,
            return_value=0,
        ) as rankup, patch.object(
            cleanup_module,
            "_enhance_to_target",
            new_callable=AsyncMock,
            return_value=0,
        ), patch.object(
            cleanup_module,
            "_recycle_excess",
            new_callable=AsyncMock,
            return_value=0,
        ) as recycle, patch.object(
            cleanup_module.db,
            "get_ex_equip_rarity",
            return_value=3,
        ):
            analyzer.return_value.analyze.return_value = report
            await module.do_task(client)

        buyer.assert_awaited_once_with(client, planned)
        rankup.assert_awaited_once_with(client, 4101001, 10)
        recycle.assert_awaited_once_with(client, 4101001, 10)


if __name__ == "__main__":
    unittest.main()
