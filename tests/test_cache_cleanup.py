import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autopcr.util.cache_cleanup import (
    account_cache_id,
    cleanup_account_artifacts,
    cleanup_runtime_cache,
    cleanup_version_cache,
    legacy_account_cache_id,
)


class CacheCleanupTest(unittest.TestCase):
    def test_version_cleanup_keeps_current_and_previous_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            for directory, suffix in (("db", ".db"), ("manifest", ".json")):
                path = cache_dir / directory
                path.mkdir()
                for version in (100, 200, 300):
                    (path / f"{version}{suffix}").write_bytes(str(version).encode())
                (path / f"invalid{suffix}").write_bytes(b"keep")

            report = cleanup_version_cache(300, cache_dir=cache_dir, keep=2, dry_run=False)

            self.assertFalse((cache_dir / "db" / "100.db").exists())
            self.assertTrue((cache_dir / "db" / "200.db").exists())
            self.assertTrue((cache_dir / "db" / "300.db").exists())
            self.assertTrue((cache_dir / "db" / "invalid.db").exists())
            self.assertFalse((cache_dir / "manifest" / "100.json").exists())
            self.assertEqual(report.selected_files, 2)

    def test_runtime_cleanup_uses_config_references_and_migrates_modules(self):
        now = 1_800_000_000.0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            config_dir = cache_dir / "http_server"
            result_dir = root / "result"
            user_dir = config_dir / "123456"
            pool_dir = cache_dir / "pool"
            token_dir = cache_dir / "token"
            module_dir = cache_dir / "modules" / "sample"
            buffer_dir = cache_dir / "buffer"
            for path in (user_dir, result_dir, pool_dir, token_dir, module_dir, buffer_dir):
                path.mkdir(parents=True, exist_ok=True)

            alias = "3-1"
            username = "active-user"
            password = "active-password"
            config = {
                "username": username,
                "password": password,
                "daily_result": [{"path": "/app/result/keep.json"}],
                "single_result": {},
            }
            (user_dir / f"{alias}.json").write_text(json.dumps(config), encoding="utf-8")

            expected_pool = hashlib.md5(username.encode()).hexdigest()
            expected_token = hashlib.md5((username + password).encode()).hexdigest()
            (pool_dir / expected_pool).write_bytes(b"active")
            (pool_dir / "orphan").write_bytes(b"orphan")
            (token_dir / expected_token).write_bytes(b"active")
            (token_dir / "orphan").write_bytes(b"orphan")

            keep_result = result_dir / "keep.json"
            old_orphan_result = result_dir / "old-orphan.json"
            recent_orphan_result = result_dir / "recent-orphan.json"
            for path in (keep_result, old_orphan_result, recent_orphan_result):
                path.write_text("{}", encoding="utf-8")
            os.utime(old_orphan_result, (now - 8 * 86400, now - 8 * 86400))
            os.utime(recent_orphan_result, (now - 86400, now - 86400))

            legacy_module = module_dir / f"{legacy_account_cache_id(alias)}.json"
            scoped_module = module_dir / f"{account_cache_id('123456', alias)}.json"
            legacy_module.write_text('{"state": 1}', encoding="utf-8")
            orphan_module = module_dir / "orphan.json"
            orphan_module.write_text("{}", encoding="utf-8")
            os.utime(orphan_module, (now - 8 * 86400, now - 8 * 86400))

            (buffer_dir / "recent.json").write_text("[]", encoding="utf-8")
            (buffer_dir / "old.json").write_text("[]", encoding="utf-8")
            stray_buffer = buffer_dir / "stray.json"
            stray_buffer.write_text("[]", encoding="utf-8")
            os.utime(stray_buffer, (now - 8 * 86400, now - 8 * 86400))
            (buffer_dir / "buffer.json").write_text(
                json.dumps({"recent": now - 86400, "old": now - 8 * 86400}),
                encoding="utf-8",
            )

            cron_log = config_dir / "cron_log.txt"
            cron_log.write_bytes(b"line\n" * 100)

            with patch("autopcr.util.cache_cleanup.CRON_LOG_MAX_BYTES", 64):
                report = cleanup_runtime_cache(
                    cache_dir=cache_dir,
                    config_dir=config_dir,
                    module_state_dir=cache_dir / "modules",
                    result_dir=result_dir,
                    cron_log_path=cron_log,
                    dry_run=False,
                    now=now,
                    compact_cron_log=True,
                )

            self.assertTrue((pool_dir / expected_pool).exists())
            self.assertFalse((pool_dir / "orphan").exists())
            self.assertTrue((token_dir / expected_token).exists())
            self.assertFalse((token_dir / "orphan").exists())
            self.assertTrue(keep_result.exists())
            self.assertFalse(old_orphan_result.exists())
            self.assertTrue(recent_orphan_result.exists())
            self.assertFalse(legacy_module.exists())
            self.assertEqual(scoped_module.read_text(encoding="utf-8"), '{"state": 1}')
            self.assertFalse(orphan_module.exists())
            self.assertTrue((buffer_dir / "recent.json").exists())
            self.assertFalse((buffer_dir / "old.json").exists())
            self.assertFalse(stray_buffer.exists())
            self.assertEqual(
                json.loads((buffer_dir / "buffer.json").read_text(encoding="utf-8")),
                {"recent": now - 86400},
            )
            self.assertLessEqual(cron_log.stat().st_size, 64)
            self.assertGreater(report.selected_files, 0)
            self.assertEqual(report.errors, 0)

    def test_invalid_config_prevents_reference_based_deletion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            config_dir = cache_dir / "http_server"
            result_dir = root / "result"
            user_dir = config_dir / "123456"
            pool_dir = cache_dir / "pool"
            token_dir = cache_dir / "token"
            for path in (user_dir, result_dir, pool_dir, token_dir):
                path.mkdir(parents=True, exist_ok=True)
            (user_dir / "broken.json").write_text("{", encoding="utf-8")
            pool_file = pool_dir / "orphan"
            token_file = token_dir / "orphan"
            result_file = result_dir / "orphan.json"
            pool_file.write_bytes(b"pool")
            token_file.write_bytes(b"token")
            result_file.write_text("{}", encoding="utf-8")

            report = cleanup_runtime_cache(
                cache_dir=cache_dir,
                config_dir=config_dir,
                module_state_dir=cache_dir / "modules",
                result_dir=result_dir,
                dry_run=False,
                now=2_000_000_000,
            )

            self.assertTrue(pool_file.exists())
            self.assertTrue(token_file.exists())
            self.assertTrue(result_file.exists())
            self.assertGreater(report.errors, 0)

    def test_account_artifact_cleanup_removes_results_and_scoped_modules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            result_dir = root / "result"
            config_file = root / "account.json"
            module_dir = cache_dir / "modules" / "sample"
            result_dir.mkdir()
            module_dir.mkdir(parents=True)

            config_file.write_text(
                json.dumps({
                    "daily_result": [{"path": "/app/result/daily.json"}],
                    "single_result": {"sample": [{"path": "C:\\result\\single.json"}]},
                }),
                encoding="utf-8",
            )
            (result_dir / "daily.json").write_text("{}", encoding="utf-8")
            (result_dir / "single.json").write_text("{}", encoding="utf-8")
            scoped = module_dir / f"{account_cache_id('123456', '3-1')}.json"
            scoped.write_text("{}", encoding="utf-8")

            cleanup_account_artifacts(
                "123456",
                "3-1",
                config_file,
                module_state_dir=cache_dir / "modules",
                result_dir=result_dir,
                dry_run=False,
            )

            self.assertFalse((result_dir / "daily.json").exists())
            self.assertFalse((result_dir / "single.json").exists())
            self.assertFalse(scoped.exists())

    def test_account_cache_id_is_scoped_by_user(self):
        self.assertNotEqual(account_cache_id("123456", "3-1"), account_cache_id("654321", "3-1"))
        self.assertEqual(legacy_account_cache_id("3-1"), legacy_account_cache_id("3-1"))

    def test_dry_run_reports_files_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            db_dir = cache_dir / "db"
            manifest_dir = cache_dir / "manifest"
            db_dir.mkdir()
            manifest_dir.mkdir()
            for version in (100, 200, 300):
                (db_dir / f"{version}.db").write_bytes(b"db")
                (manifest_dir / f"{version}.json").write_bytes(b"manifest")

            report = cleanup_version_cache(300, cache_dir=cache_dir, keep=2, dry_run=True)

            self.assertTrue((db_dir / "100.db").exists())
            self.assertTrue((manifest_dir / "100.json").exists())
            self.assertEqual(report.selected_files, 2)


if __name__ == "__main__":
    unittest.main()
