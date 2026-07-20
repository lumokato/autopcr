import asyncio
import glob, os

from ..sdk.sdkclients import bsdkclient
from ..core.sdkclient import account, platform
from ..core.apiclient import apiclient
from ..model.sdkrequests import SourceIniGetMaintenanceStatusRequest
from ..constants import CACHE_DIR
from ..core.datamgr import datamgr
from ..util import aiorequests
from ..util.cache_cleanup import atomic_write_bytes, cleanup_runtime_cache
from ..util.logger import instance as logger
import brotli

async def db_start():
    os.makedirs(os.path.join(CACHE_DIR, 'db'), exist_ok=True)
    dbs = [
        path for path in glob.glob(os.path.join(CACHE_DIR, "db", "*.db"))
        if os.path.basename(path).removesuffix(".db").isdigit()
    ]
    if dbs:
        db = max(dbs, key=lambda path: int(os.path.basename(path).removesuffix(".db")))
        version = int(os.path.basename(db).split('.')[0])
    else:
        version = int(
                (await apiclient(bsdkclient(account("autopcr", "autopcr", platform.Android)))
                .request(SourceIniGetMaintenanceStatusRequest()))
                .manifest_ver
        )
    await datamgr.try_update_database(version)
    try:
        await asyncio.to_thread(cleanup_runtime_cache, compact_cron_log=True)
    except Exception:
        logger.exception("Startup cache cleanup failed")

async def do_update_database() -> int:
    info = f'https://redive.estertion.win/last_version_cn.json'

    rsp = await aiorequests.get(info, stream=True, timeout=20)
    version = (await rsp.json())['TruthVersion']

    url = f'https://redive.estertion.win/db/redive_cn.db.br'

    save_path = os.path.join(CACHE_DIR, "db", f"{version}.db")
    try:
        rsp = await aiorequests.get(url, headers={'Accept-Encoding': 'br'}, stream=True, timeout=20)
        if 200 == rsp.status_code:
            atomic_write_bytes(save_path, brotli.decompress((await rsp.content)))
        else:
            raise ValueError("下载失败")
    except Exception as e:
        raise e
    return int(version)
