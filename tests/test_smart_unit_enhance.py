import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from autopcr.module.modules.smart_unit_enhance import (
    Unique1GlowBallAssignment,
    Unique1GlowBallPlanningResult,
    Unique1GlowBallStock,
    Unique1GlowBallUnitInput,
    plan_unique1_glow_balls,
    smart_unit_enhance,
)


smart_unit_enhance_module = importlib.import_module(
    "autopcr.module.modules.smart_unit_enhance"
)
unit_module = importlib.import_module("autopcr.module.modules.unit")


class Unique1GlowBallPlanningTest(unittest.TestCase):
    @staticmethod
    def unit(
        unit_id: int,
        memory: int,
        *,
        eligible: bool = True,
        equipped: bool = False,
        crafted: bool = False,
        has_ball: bool = False,
    ) -> Unique1GlowBallUnitInput:
        return Unique1GlowBallUnitInput(
            unit_id=unit_id,
            eligible=eligible,
            unique1_equipped=equipped,
            unique1_crafted=crafted,
            has_growth_ball=has_ball,
            memory_inventory=memory,
        )

    def test_reserves_total_inventory_and_uses_lowest_levels_first(self):
        units = [
            self.unit(1001, 40),
            self.unit(1002, 0),
            self.unit(1003, 49),
            self.unit(1004, 50),
            self.unit(1005, 10, equipped=True),
            self.unit(1006, 10, crafted=True),
            self.unit(1007, 10, has_ball=True),
            self.unit(1008, 10, eligible=False),
        ]
        stocks = [
            Unique1GlowBallStock(21954, 10004, 340, 2),
            Unique1GlowBallStock(21951, 10001, 240, 2),
            Unique1GlowBallStock(21952, 10002, 270, 2),
        ]

        result = plan_unique1_glow_balls(units, stocks, keep=3)

        self.assertEqual(result.inventory, 6)
        self.assertEqual(result.keep, 3)
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.consumption, 3)
        self.assertEqual(
            [(item.unit_id, item.target_level) for item in result.assignments],
            [(1002, 240), (1001, 240), (1003, 270)],
        )

    def test_keep_larger_than_inventory_blocks_all_assignments(self):
        result = plan_unique1_glow_balls(
            [self.unit(1001, 0)],
            [Unique1GlowBallStock(21951, 10001, 240, 2)],
            keep=5,
        )

        self.assertEqual(result.inventory, 2)
        self.assertEqual(result.consumption, 0)

    def test_ball_priority_uses_target_level_instead_of_item_id(self):
        result = plan_unique1_glow_balls(
            [self.unit(1001, 0), self.unit(1002, 0)],
            [
                Unique1GlowBallStock(100, 200, 340, 1),
                Unique1GlowBallStock(900, 300, 240, 1),
            ],
            keep=0,
        )

        self.assertEqual(
            [item.target_level for item in result.assignments],
            [240, 340],
        )


class SmartUnitEnhanceConfigTest(unittest.TestCase):
    def test_glow_ball_keep_options_and_default(self):
        module = smart_unit_enhance(
            SimpleNamespace(id="test", legacy_id=None)
        )
        config = module.config["smart_unit_enhance_ball_keep"]

        self.assertEqual(config.default, 3)
        self.assertEqual(config.candidates, [0, 1, 3, 5])


class Unique1GlowBallExecutionTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def result() -> Unique1GlowBallPlanningResult:
        return Unique1GlowBallPlanningResult(
            assignments=[
                Unique1GlowBallAssignment(1001, 21951, 10001, 240)
            ],
            inventory=1,
            keep=0,
            candidate_count=1,
        )

    @staticmethod
    def fake_db(limit):
        return SimpleNamespace(
            growth_parameter_unique={10001: limit},
            unit_unique_equip={1: {1001: SimpleNamespace(equip_id=140001)}},
            get_item_name=lambda item_id: f"ball-{item_id}",
            get_unit_name=lambda unit_id: f"unit-{unit_id}",
        )

    @staticmethod
    def module(unit, unique1_inventory=0):
        module = object.__new__(smart_unit_enhance)
        data = SimpleNamespace(
            unit={1001: unit},
            get_inventory=MagicMock(return_value=unique1_inventory),
        )
        module.client = SimpleNamespace(
            data=data,
            set_growth_item_unique=AsyncMock(),
        )
        module.log = []
        module.warn = []
        module.is_warn = False
        return module

    async def test_assigns_ball_after_live_checks(self):
        limit = SimpleNamespace(
            unique_equip_rank_1=17,
            unique_equip_strength_point_1=39990,
        )
        unit = SimpleNamespace(
            unit_rarity=5,
            exceed_stage=1,
            unique_equip_slot=[SimpleNamespace(is_slot=0)],
        )
        module = self.module(unit)
        module.is_unique_growth_unit = AsyncMock(side_effect=[None, limit])
        fake_db = self.fake_db(limit)

        with patch.object(smart_unit_enhance_module, "db", fake_db), patch.object(
            unit_module, "db", fake_db
        ):
            blocked = await module._execute_unique1_glow_balls(self.result())

        self.assertEqual(blocked, set())
        module.client.set_growth_item_unique.assert_awaited_once_with(1001, 21951)

    async def test_skips_ball_when_star_exceed_did_not_finish(self):
        limit = SimpleNamespace(
            unique_equip_rank_1=17,
            unique_equip_strength_point_1=39990,
        )
        unit = SimpleNamespace(
            unit_rarity=5,
            exceed_stage=0,
            unique_equip_slot=[SimpleNamespace(is_slot=0)],
        )
        module = self.module(unit)
        module.is_unique_growth_unit = AsyncMock()
        fake_db = self.fake_db(limit)

        with patch.object(smart_unit_enhance_module, "db", fake_db), patch.object(
            unit_module, "db", fake_db
        ):
            blocked = await module._execute_unique1_glow_balls(self.result())

        self.assertEqual(blocked, {1001})
        module.client.set_growth_item_unique.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
