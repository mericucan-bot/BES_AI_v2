import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from src.cache_manager import is_market_hours, get_smart_ttl, DiskCache

TR_TZ = ZoneInfo("Europe/Istanbul")


class TestMarketHours:
    def test_weekday_open_hours(self):
        # Carsamba 14:00 -> acik
        dt = datetime(2024, 11, 13, 14, 0, tzinfo=TR_TZ)
        assert is_market_hours(dt) is True

    def test_weekday_before_open(self):
        # Carsamba 09:00 -> kapali
        dt = datetime(2024, 11, 13, 9, 0, tzinfo=TR_TZ)
        assert is_market_hours(dt) is False

    def test_weekday_after_close(self):
        # Carsamba 19:00 -> kapali
        dt = datetime(2024, 11, 13, 19, 0, tzinfo=TR_TZ)
        assert is_market_hours(dt) is False

    def test_saturday(self):
        dt = datetime(2024, 11, 16, 14, 0, tzinfo=TR_TZ)
        assert is_market_hours(dt) is False

    def test_sunday(self):
        dt = datetime(2024, 11, 17, 14, 0, tzinfo=TR_TZ)
        assert is_market_hours(dt) is False


class TestSmartTTL:
    def test_market_open_ttl_short(self):
        dt = datetime(2024, 11, 13, 14, 0, tzinfo=TR_TZ)
        assert get_smart_ttl(dt) == 30 * 60

    def test_after_close_ttl_until_morning(self):
        # Carsamba 19:00 -> Persembe 10:00 = 15 saat = 54000 sn
        dt = datetime(2024, 11, 13, 19, 0, tzinfo=TR_TZ)
        ttl = get_smart_ttl(dt)
        assert 14 * 3600 < ttl < 16 * 3600

    def test_friday_evening_until_monday(self):
        # Cuma 19:00 -> Pazartesi 10:00 = ~63 saat
        dt = datetime(2024, 11, 15, 19, 0, tzinfo=TR_TZ)
        ttl = get_smart_ttl(dt)
        assert ttl > 60 * 3600

    def test_ttl_minimum(self):
        # Hicbir durumda 60 saniyenin altinda olmamali
        dt = datetime(2024, 11, 13, 9, 59, tzinfo=TR_TZ)
        assert get_smart_ttl(dt) >= 60


class TestDiskCache:
    def test_set_and_get(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("test_key", {"value": 42})
        result = cache.get("test_key", max_age_seconds=3600)
        assert result == {"value": 42}

    def test_missing_key_returns_none(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        assert cache.get("nonexistent", 3600) is None

    def test_expired_returns_none(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("stale", "data")
        # max_age=0 -> her zaman expired
        result = cache.get("stale", max_age_seconds=0)
        assert result is None

    def test_clear(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("a", 1)
        cache.set("b", 2)
        count = cache.clear()
        assert count == 2
        assert cache.get("a", 3600) is None
