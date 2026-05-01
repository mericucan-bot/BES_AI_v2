import json
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

TR_TZ = ZoneInfo("Europe/Istanbul")
BIST_OPEN  = time(10, 0)   # 10:00 TR
BIST_CLOSE = time(18, 0)   # 18:00 TR


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """Suan BIST acik mi? Hafta ici 10:00-18:00 TR."""
    if dt is None:
        dt = datetime.now(TR_TZ)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=TR_TZ)

    if dt.weekday() >= 5:  # 5=Cumartesi, 6=Pazar
        return False

    return BIST_OPEN <= dt.time() <= BIST_CLOSE


def get_smart_ttl(now: Optional[datetime] = None) -> int:
    """
    Akilli TTL:
    - Piyasa acikken: 30 dakika (1800 sn)
    - Piyasa kapali, hafta ici: bir sonraki acilisa kadar
    - Hafta sonu: Pazartesi 10:00'a kadar
    """
    if now is None:
        now = datetime.now(TR_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=TR_TZ)

    if is_market_hours(now):
        return 30 * 60

    next_open = now.replace(hour=10, minute=0, second=0, microsecond=0)

    # Bugunun 10:00'undan onceyse ve hafta iciyse, bugun 10:00
    if now.time() < BIST_OPEN and now.weekday() < 5:
        pass
    else:
        # Yarin 10:00 (veya sonraki is gunu)
        next_open += timedelta(days=1)

    # Hafta sonu atlat
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    delta = (next_open - now).total_seconds()
    return max(int(delta), 60)


class DiskCache:
    """JSON tabanli basit disk cache. Streamlit cache'in ustunde ek katman."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str, max_age_seconds: int) -> Optional[Any]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                cached = json.load(f)
            cached_at = datetime.fromisoformat(cached["timestamp"])
            age = (datetime.now(TR_TZ) - cached_at).total_seconds()
            if age > max_age_seconds:
                logger.info(f"Cache expired: {key} (age={age:.0f}s, max={max_age_seconds}s)")
                return None
            logger.info(f"Cache HIT: {key} (age={age:.0f}s)")
            return cached["value"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache okuma hatasi ({key}): {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        try:
            with open(path, "w") as f:
                json.dump({
                    "timestamp": datetime.now(TR_TZ).isoformat(),
                    "value": value,
                }, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Cache SET: {key}")
        except (OSError, TypeError) as e:
            logger.error(f"Cache yazma hatasi ({key}): {e}")

    def clear(self) -> int:
        """Tum cache'i temizle. Silinen dosya sayisini doner."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        logger.info(f"Cache temizlendi: {count} dosya silindi")
        return count
