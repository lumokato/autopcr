from dataclasses import field
from typing import Any
from .abyss import *
from .autosweep import *
from .box import *
from .nologin import *
from .caravan import *
from .clan import *
from .cron import *
from .daily import *
from .exequip import *
from .exequip_cleanup import *
from .gacha import *
from .hatsune import *
from .labyrinth import *
from .room import *
from .shiori import *
from .shop import *
from .story import *
from .sweep import *
from .tower import *
from .tools import *
from .travel import *
from .unit import *
from .smart_unit_enhance import *
from .sync_growth import *
from .talent import *
from .mirage import *

@dataclass
class ModuleList:
    name: str = ""
    key: str = ""
    modules: List[Any] = field(default_factory=list)
    hidden_in_batch: bool = False
    hidden_in_clan: bool = False
    hidden: bool = False
    visible_in_clan: bool = False
    visible_in_batch: bool = False
    execution_mode: str = "manual"

cron_modules = ModuleList(
    '定时',
    'cron',
    [
        cron1,
        cron2,
        cron3,
        cron4,
        cron5,
        cron6,
    ],
    hidden_in_batch=True,
    hidden_in_clan=True,
    execution_mode="cron",
)

daily_modules = ModuleList(
    '日常',
    'daily',
    [
        global_config,
        chara_fortune,
        mission_receive_first,
        clan_like,
        free_gacha,
        normal_gacha,
        monthly_gacha,
        room_accept_all,
        travel_round,
        travel_quest_sweep,
        ex_equip_recycle,
        explore_exp,
        explore_mana,
        underground_skip,
        # underground_donate,
        special_underground_skip,
        mirage_floor_receive,
        mirage_nemesis_sweep,
        tower_cloister_sweep,
        labyrinth_sweep,
        jjc_reward,
        abyss_quest_sweep,
        abyss_boss_sweep,
        talent_sweep,
        present_receive,
        talent_sweep2,
        smart_very_hard_sweep,
        xinsui_sweep,
        starcup_sweep,
        hatsune_h_sweep,
        hatsune_dear_reading,
        smart_sweep,
        mirai_very_hard_sweep,
        smart_hard_sweep,
        smart_shiori_sweep,
        mirai_sp1_h_sweep,
        mirai_sp1_shiori_sweep,
        last_normal_quest_sweep,
        lazy_normal_sweep,

        last_hard_quest_sweep,
        last_unlock_normal_quest_sweep,

        all_in_hatsune,
        
        hatsune_vhboss_sweep,
        hatsune_hboss_sweep,
        hatsune_mission_accept1,
        hatsune_gacha_exchange,
        hatsune_mission_accept2,

        # unit_equip_enhance_up,
        # unit_skill_level_up,

        mission_receive_last,
        seasonpass_accept,
        seasonpass_reward,
        role_gacha,

        normal_shop,
        limit_shop,
        underground_shop,
        jjc_shop,
        pjjc_shop,
        clanbattle_shop,
        master_shop_talent,
        master_shop,

        clan_equip_request,
        love_up,
        shiori_mission_check,
        alces_story_reading,
        main_story_reading,
        tower_story_reading,
        hatsune_story_reading,
        seven_obtent_reading,
        hatsune_sub_story_reading,
        guild_story_reading,
        unit_story_reading,
        birthday_story_reading,
        room_upper_all,
        user_info,
    ],
    hidden=True,
    execution_mode="daily",
)

routine_modules = ModuleList(
    '日常',
    'routine',
    [
        chara_fortune,
        mission_receive_first,
        clan_like,
        normal_gacha,
        monthly_gacha,
        room_accept_all,
        explore_exp,
        explore_mana,
        underground_skip,
        special_underground_skip,
        tower_cloister_sweep,
        mirage_floor_receive,
        mirage_nemesis_sweep,
        jjc_reward,
        abyss_quest_sweep,
        abyss_boss_sweep,
        hatsune_dear_reading,
        hatsune_mission_accept1,
        hatsune_gacha_exchange,
        hatsune_mission_accept2,
        mission_receive_last,
        seasonpass_accept,
        role_gacha,
        love_up,
        shiori_mission_check,
        alces_story_reading,
        main_story_reading,
        tower_story_reading,
        hatsune_story_reading,
        seven_obtent_reading,
        hatsune_sub_story_reading,
        guild_story_reading,
        unit_story_reading,
        birthday_story_reading,
        room_upper_all,
    ],
    execution_mode="daily",
)

sweep_modules = ModuleList(
    '刷取',
    'sweep',
    [
        global_config,
        talent_sweep,
        talent_sweep2,
        smart_very_hard_sweep,
        xinsui_sweep,
        starcup_sweep,
        hatsune_h_sweep,
        smart_sweep,
        mirai_very_hard_sweep,
        smart_hard_sweep,
        smart_shiori_sweep,
        mirai_sp1_h_sweep,
        mirai_sp1_shiori_sweep,
        last_normal_quest_sweep,
        lazy_normal_sweep,
        last_hard_quest_sweep,
        last_unlock_normal_quest_sweep,
        all_in_hatsune,
        hatsune_vhboss_sweep,
        hatsune_hboss_sweep,
    ],
    execution_mode="daily",
)

shop_modules = ModuleList(
    '商店',
    'shop',
    [
        normal_shop,
        limit_shop,
        underground_shop,
        jjc_shop,
        pjjc_shop,
        clanbattle_shop,
        master_shop_talent,
        master_shop,
    ],
    execution_mode="daily",
)

story_modules = ModuleList(
    '剧情',
    'story',
    [
        hatsune_dear_reading,
        shiori_mission_check,
        alces_story_reading,
        main_story_reading,
        tower_story_reading,
        hatsune_story_reading,
        seven_obtent_reading,
        hatsune_sub_story_reading,
        guild_story_reading,
        unit_story_reading,
        birthday_story_reading,
    ],
    hidden=True,
    execution_mode="daily",
)

strategy_modules = ModuleList(
    '策略',
    'strategy',
    [
        user_info,
        free_gacha,
        travel_quest_sweep,
        travel_round,
        labyrinth_sweep,
        ex_equip_recycle,
        present_receive,
        seasonpass_reward,
        clan_equip_request,
    ],
    execution_mode="daily",
)

planning_modules = ModuleList(
    '规划',
    'planning',
    [
        get_need_memory,
        get_need_pure_memory,
        get_need_sp_memory,
        get_need_xinsui,
        search_unit,
        missing_unit,
        missing_emblem,
        find_talent_quest,
        find_clan_talent_quest,
        get_library_import_data,
        get_need_equip,
        get_normal_quest_recommand,
    ],
    hidden_in_batch=True,
)

table_modules = ModuleList(
    '表格',
    'table',
    [
    ],
    hidden=True,
    visible_in_batch=True,
)


unit_modules = ModuleList(
    '角色',
    'unit',
    [
        search_unit,
        missing_unit,
        refresh_box,
        unit_promote,
        sync_growth,
        sync_growth_underground_shop,
        unit_memory_buy,
        smart_unit_enhance,
        unit_exceed,
        unit_evolution,
    ],
    hidden=True,
)

growth_modules = ModuleList(
    '养成',
    'growth',
    [
        sync_growth,
        sync_growth_underground_shop,
        smart_unit_enhance,
        ex_equip_rainbow_enchance,
        ex_equip_power_maximun,
        ex_equip_info,
        ex_equip_rank_up,
        ex_equip_enhance_up,
        ex_equip_cleanup_execute,
        remove_cb_ex_equip,
        ex_equip_state,
        unit_memory_buy,
        unit_promote,
        unit_exceed,
        unit_evolution,
        refresh_box,
    ],
)

clan_modules = ModuleList(
    '公会',
    'clan',
    [
        unit_promote_batch,
        unit_memory_buy_batch,
        set_my_party,
        get_box_table,
    ],
    hidden=True,
    visible_in_clan=True,
)

danger_modules = ModuleList(
    '危险',
    'danger',
    [
        gacha_start,
        gacha_exchange_chara,
    ],
    hidden_in_clan=True,
)

tool_modules = ModuleList(
    '工具',
    'tool',
    [
        labyrinth_start_reroll,
        travel_team_view,
        caravan_play,
        caravan_shop_buy,
        set_my_party2,
        half_schedule,
        clan_battle_knive,
        redeem_unit_swap,
        get_clan_support_unit,
        jjc_back,
        pjjc_back,
        jjc_info,
        pjjc_info,
        pjjc_def_shuffle_team,
        set_my_party,
        clear_my_party,
        remove_cb_support,
        pjjc_atk_shuffle_team,
    ]
)
