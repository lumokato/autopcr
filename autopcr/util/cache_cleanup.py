from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

from ..constants import (
    ARENA_CACHE_MAX_AGE_SECONDS,
    CACHE_CLEANUP_DRY_RUN,
    CACHE_CLEANUP_INTERVAL_SECONDS,
    CACHE_DIR,
    CACHE_VERSION_KEEP,
    CONFIG_PATH,
    CRON_LOG_PATH,
    CRON_LOG_MAX_BYTES,
    MODULE_STATE_DIR,
    RESULT_DIR,
    RESULT_ORPHAN_GRACE_SECONDS,
)
from .logger import instance as logger


@dataclass
class CleanupReport:
    dry_run: bool = False
    selected_files: int = 0
    selected_bytes: int = 0
    migrated_files: int = 0
    compacted_bytes: int = 0
    errors: int = 0
    categories: Dict[str, int] = field(default_factory=dict)

    def record_remove(self, category: str, size: int) -> None:
        self.selected_files += 1
        self.selected_bytes += size
        self.categories[category] = self.categories.get(category, 0) + 1

    def record_migration(self) -> None:
        self.migrated_files += 1

    def record_compaction(self, reclaimed: int) -> None:
        self.compacted_bytes += max(0, reclaimed)

    @property
    def changed(self) -> bool:
        return bool(self.selected_files or self.migrated_files or self.compacted_bytes)


@dataclass
class AccountCacheReferences:
    valid: bool = True
    config_files: int = 0
    pool_ids: Set[str] = field(default_factory=set)
    token_ids: Set[str] = field(default_factory=set)
    result_files: Set[str] = field(default_factory=set)
    module_migrations: Dict[str, Set[str]] = field(default_factory=dict)


def account_cache_id(qid: str, alias: str) -> str:
    return hashlib.md5(f"{qid}\0{alias}".encode("utf-8")).hexdigest()


def legacy_account_cache_id(alias: str) -> str:
    return hashlib.md5(alias.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: os.PathLike | str, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = 0o644
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: os.PathLike | str, data: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, data.encode(encoding))


def _remove_file(path: Path, report: CleanupReport, category: str) -> bool:
    try:
        size = path.stat().st_size
        if not report.dry_run:
            path.unlink()
        report.record_remove(category, size)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        report.errors += 1
        logger.exception("Failed to remove cache file %s", path)
        return False


def _result_name(path: object) -> Optional[str]:
    if not isinstance(path, str) or not path:
        return None
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name if name.endswith(".json") else None


def _collect_result_files(data: dict) -> Set[str]:
    result_files: Set[str] = set()
    daily_results = data.get("daily_result", [])
    if not isinstance(daily_results, list):
        daily_results = []
    for item in daily_results:
        if isinstance(item, dict):
            name = _result_name(item.get("path"))
            if name:
                result_files.add(name)
    single_results = data.get("single_result", {})
    if not isinstance(single_results, dict):
        single_results = {}
    for items in single_results.values():
        if not isinstance(items, list):
            continue
        for item in items or []:
            if isinstance(item, dict):
                name = _result_name(item.get("path"))
                if name:
                    result_files.add(name)
    return result_files


def scan_account_cache_references(config_dir: os.PathLike | str = CONFIG_PATH) -> AccountCacheReferences:
    refs = AccountCacheReferences()
    root = Path(config_dir)
    if not root.exists():
        refs.valid = False
        logger.warning("Account cache cleanup skipped because config directory does not exist: %s", root)
        return refs

    for user_dir in root.iterdir():
        if not user_dir.is_dir():
            continue
        qid = user_dir.name
        for config_file in user_dir.glob("*.json"):
            refs.config_files += 1
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("account config is not an object")
            except Exception:
                refs.valid = False
                logger.exception("Failed to parse account config during cache cleanup: %s", config_file)
                continue

            alias = config_file.stem
            legacy_id = legacy_account_cache_id(alias)
            scoped_id = account_cache_id(qid, alias)
            refs.module_migrations.setdefault(legacy_id, set()).add(scoped_id)
            refs.result_files.update(_collect_result_files(data))

            username = str(data.get("username") or "")
            if not username:
                continue
            password = str(data.get("password") or "")
            refs.pool_ids.add(hashlib.md5(username.encode("utf-8")).hexdigest())
            refs.token_ids.add(hashlib.md5((username + password).encode("utf-8")).hexdigest())

    return refs


def cleanup_version_cache(
    current_version: int,
    cache_dir: os.PathLike | str = CACHE_DIR,
    keep: int = CACHE_VERSION_KEEP,
    dry_run: bool = CACHE_CLEANUP_DRY_RUN,
) -> CleanupReport:
    report = CleanupReport(dry_run=dry_run)
    root = Path(cache_dir)
    keep = max(1, keep)

    for directory_name, suffix in (("db", ".db"), ("manifest", ".json")):
        directory = root / directory_name
        if not directory.exists():
            continue
        files = sorted(
            (path for path in directory.glob(f"*{suffix}") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
            reverse=True,
        )
        protected = {path.name for path in files[:keep]}
        protected.add(f"{current_version}{suffix}")
        for path in files:
            if path.name not in protected:
                _remove_file(path, report, directory_name)

    _log_report("version cache cleanup", report)
    return report


def _cleanup_unreferenced_files(
    directory: Path,
    referenced: Set[str],
    report: CleanupReport,
    category: str,
    grace_seconds: int = 0,
    now: Optional[float] = None,
    allowed_suffixes: Optional[Set[str]] = None,
) -> None:
    if not directory.exists():
        return
    cutoff = (time.time() if now is None else now) - max(0, grace_seconds)
    for path in directory.iterdir():
        if not path.is_file() or path.name.startswith(".") or path.name in referenced:
            continue
        if allowed_suffixes is not None and path.suffix not in allowed_suffixes:
            continue
        try:
            if grace_seconds and path.stat().st_mtime > cutoff:
                continue
        except FileNotFoundError:
            continue
        _remove_file(path, report, category)


def _migrate_module_caches(modules_dir: Path, refs: AccountCacheReferences, report: CleanupReport) -> None:
    if not modules_dir.exists():
        return
    for module_dir in modules_dir.iterdir():
        if not module_dir.is_dir():
            continue
        for legacy_path in module_dir.glob("*.json"):
            scoped_ids = refs.module_migrations.get(legacy_path.stem)
            if not scoped_ids:
                continue
            targets = [module_dir / f"{scoped_id}.json" for scoped_id in scoped_ids]
            migration_ok = True
            for target in targets:
                if target == legacy_path or target.exists():
                    continue
                if report.dry_run:
                    report.record_migration()
                    continue
                try:
                    atomic_write_bytes(target, legacy_path.read_bytes())
                    report.record_migration()
                except OSError:
                    migration_ok = False
                    report.errors += 1
                    logger.exception("Failed to migrate module cache %s to %s", legacy_path, target)
            if migration_ok and all(target.exists() or report.dry_run for target in targets):
                _remove_file(legacy_path, report, "module_legacy")


def _cleanup_orphan_module_caches(
    modules_dir: Path,
    refs: AccountCacheReferences,
    report: CleanupReport,
    now: Optional[float] = None,
) -> None:
    if not modules_dir.exists():
        return
    active_ids = set(refs.module_migrations)
    for scoped_ids in refs.module_migrations.values():
        active_ids.update(scoped_ids)
    referenced = {f"{cache_id}.json" for cache_id in active_ids}
    for module_dir in modules_dir.iterdir():
        if not module_dir.is_dir():
            continue
        _cleanup_unreferenced_files(
            module_dir,
            referenced,
            report,
            "module_orphan",
            grace_seconds=RESULT_ORPHAN_GRACE_SECONDS,
            now=now,
            allowed_suffixes={".json"},
        )


def migrate_legacy_module_cache(target: os.PathLike | str, legacy: os.PathLike | str) -> bool:
    target_path = Path(target)
    legacy_path = Path(legacy)
    if target_path.exists() or target_path == legacy_path or not legacy_path.exists():
        return False
    atomic_write_bytes(target_path, legacy_path.read_bytes())
    return True


def _cleanup_arena_buffer(
    buffer_dir: Path,
    report: CleanupReport,
    max_age_seconds: int = ARENA_CACHE_MAX_AGE_SECONDS,
    now: Optional[float] = None,
) -> None:
    metadata_path = buffer_dir / "buffer.json"
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("arena buffer metadata is not an object")
    except Exception:
        report.errors += 1
        logger.exception("Failed to parse arena buffer metadata: %s", metadata_path)
        return

    current_time = time.time() if now is None else now
    retained = {}
    changed = False
    for key, timestamp in metadata.items():
        result_path = buffer_dir / f"{key}.json"
        try:
            recent = current_time - float(timestamp) < max_age_seconds
        except (TypeError, ValueError):
            recent = False
        if recent and result_path.exists():
            retained[key] = timestamp
        else:
            changed = True
            if result_path.exists():
                _remove_file(result_path, report, "arena_buffer")

    retained_names = {f"{key}.json" for key in retained}
    cutoff = current_time - max_age_seconds
    for path in buffer_dir.glob("*.json"):
        if path == metadata_path or path.name in retained_names:
            continue
        try:
            if path.stat().st_mtime <= cutoff:
                changed = _remove_file(path, report, "arena_buffer") or changed
        except FileNotFoundError:
            continue

    if changed and not report.dry_run:
        atomic_write_text(metadata_path, json.dumps(retained, ensure_ascii=False, indent=4))


def compact_file_tail(
    path: os.PathLike | str,
    max_bytes: int,
    dry_run: bool = False,
) -> int:
    target = Path(path)
    try:
        original_size = target.stat().st_size
    except FileNotFoundError:
        return 0
    if original_size <= max_bytes:
        return 0
    if dry_run:
        return original_size - max_bytes

    with target.open("rb") as fp:
        fp.seek(max(0, original_size - max_bytes))
        data = fp.read()
    if original_size > max_bytes:
        newline = data.find(b"\n")
        data = data[newline + 1:] if newline >= 0 else b""
    atomic_write_bytes(target, data)
    return original_size - len(data)


def cleanup_runtime_cache(
    cache_dir: os.PathLike | str = CACHE_DIR,
    config_dir: os.PathLike | str = CONFIG_PATH,
    module_state_dir: os.PathLike | str = MODULE_STATE_DIR,
    result_dir: os.PathLike | str = RESULT_DIR,
    cron_log_path: os.PathLike | str = CRON_LOG_PATH,
    dry_run: bool = CACHE_CLEANUP_DRY_RUN,
    now: Optional[float] = None,
    compact_cron_log: bool = False,
) -> CleanupReport:
    report = CleanupReport(dry_run=dry_run)
    cache_root = Path(cache_dir)
    refs = scan_account_cache_references(config_dir)

    if refs.valid:
        _cleanup_unreferenced_files(cache_root / "pool", refs.pool_ids, report, "pool", now=now)
        _cleanup_unreferenced_files(cache_root / "token", refs.token_ids, report, "token", now=now)
        _cleanup_unreferenced_files(
            Path(result_dir),
            refs.result_files,
            report,
            "result",
            grace_seconds=RESULT_ORPHAN_GRACE_SECONDS,
            now=now,
            allowed_suffixes={".json"},
        )
        modules_dir = Path(module_state_dir)
        _migrate_module_caches(modules_dir, refs, report)
        _cleanup_orphan_module_caches(modules_dir, refs, report, now=now)
    else:
        report.errors += 1
        logger.warning("Account cache cleanup skipped because one or more configs could not be parsed")

    _cleanup_arena_buffer(cache_root / "buffer", report, now=now)
    if compact_cron_log:
        reclaimed = compact_file_tail(
            cron_log_path,
            CRON_LOG_MAX_BYTES,
            dry_run=dry_run,
        )
        report.record_compaction(reclaimed)
    _log_report("runtime cache cleanup", report)
    return report


def cleanup_account_artifacts(
    qid: str,
    alias: str,
    config_file: os.PathLike | str,
    module_state_dir: os.PathLike | str = MODULE_STATE_DIR,
    result_dir: os.PathLike | str = RESULT_DIR,
    dry_run: bool = CACHE_CLEANUP_DRY_RUN,
) -> CleanupReport:
    report = CleanupReport(dry_run=dry_run)
    config_path = Path(config_file)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("account config is not an object")
    except FileNotFoundError:
        data = {}
    except Exception:
        report.errors += 1
        logger.exception("Failed to parse deleted account config: %s", config_path)
        data = {}

    result_root = Path(result_dir)
    for result_name in _collect_result_files(data):
        path = result_root / result_name
        if path.exists():
            _remove_file(path, report, "account_result")

    scoped_id = account_cache_id(qid, alias)
    modules_dir = Path(module_state_dir)
    if modules_dir.exists():
        for path in modules_dir.glob(f"*/{scoped_id}.json"):
            _remove_file(path, report, "account_module")

    _log_report("account artifact cleanup", report)
    return report


def _log_report(label: str, report: CleanupReport) -> None:
    if not report.changed and not report.errors:
        return
    action = "selected" if report.dry_run else "removed"
    logger.info(
        "%s: %s_files=%s bytes=%s migrated=%s compacted_bytes=%s errors=%s categories=%s",
        label,
        action,
        report.selected_files,
        report.selected_bytes,
        report.migrated_files,
        report.compacted_bytes,
        report.errors,
        report.categories,
    )


_cleanup_task: Optional[asyncio.Task] = None


async def _periodic_cleanup() -> None:
    while True:
        await asyncio.sleep(CACHE_CLEANUP_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(cleanup_runtime_cache)
        except Exception:
            logger.exception("Periodic cache cleanup failed")


def queue_cache_cleanup() -> None:
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        return
    _cleanup_task = asyncio.get_event_loop().create_task(_periodic_cleanup())
