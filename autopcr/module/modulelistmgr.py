from typing import Dict, List, Callable, Any
from .modules import (
    Module,
    ModuleList,
    CronModule,
    clan_modules,
    cron_modules,
    daily_modules,
    danger_modules,
    growth_modules,
    planning_modules,
    routine_modules,
    shop_modules,
    strategy_modules,
    story_modules,
    sweep_modules,
    table_modules,
    tool_modules,
    unit_modules,
)
from .modulemgr import ModuleManager

class ModuleListManager:

    modules: Dict[str, ModuleList] = {
        cron_modules.key: cron_modules,
        routine_modules.key: routine_modules,
        sweep_modules.key: sweep_modules,
        shop_modules.key: shop_modules,
        strategy_modules.key: strategy_modules,
        story_modules.key: story_modules,
        growth_modules.key: growth_modules,
        planning_modules.key: planning_modules,
        tool_modules.key: tool_modules,
        daily_modules.key: daily_modules,
        unit_modules.key: unit_modules,
        table_modules.key: table_modules,
        clan_modules.key: clan_modules,
        danger_modules.key: danger_modules,
    }
    name_to_modules: Dict[str, Callable] = {m.__name__: m for ml in modules.values() for m in ml.modules}

    def __init__(self, modulemgr: ModuleManager):
        self.modulemgr = modulemgr

    @property
    def daily_modules(self) -> List[Module]:
        return [module for module in self.get_modules_list('daily') if not module.hidden]

    @property
    def cron_modules(self) -> List[CronModule]:
        return self.get_modules_list('cron')

    def get_module_from_key(self, key: str) -> Module:
        if key not in self.name_to_modules:
            raise ValueError(f"模块{key}未找到")
        return self.name_to_modules[key](self.modulemgr)

    def get_modules_list(self, key: str) -> List[Any]:
        modules = self.modules.get(key, ModuleList()).modules
        return [m(self.modulemgr) for m in modules]
    
    def generate_info(self, key: str):
        module_list = self.modules.get(key, ModuleList())
        modules = [module for module in self.get_modules_list(key) if not module.hidden]
        return {
            'config': {**{key: m.get_config(key) for m in modules for key in m.config}, **{m.key: m.get_config(m.key) for m in modules}},
            'order': [m.key for m in modules],
            'info': {m.key: m.generate_info() for m in modules},
            'execution_mode': module_list.execution_mode,
        }

    def generate_tab(self, clan: bool = False, batch: bool = False):
        modules = []
        for ml in self.modules.values():
            if clan and ml.hidden_in_clan:
                continue
            if batch and ml.hidden_in_batch:
                continue
            if not clan and not batch and ml.hidden:
                continue
            if clan and ml.visible_in_clan or batch and ml.visible_in_batch or not ml.hidden:
                modules.append(ml)
        modules.sort(key=lambda module: module.key == danger_modules.key)
        return [{'key': m.key, 'name': m.name} for m in modules]
