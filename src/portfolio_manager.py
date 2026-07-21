import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data: dict) -> None:
    """Yazimi atomic yap: temp dosyaya yaz, sonra rename ile yerine koy.

    Race condition'da bozuk dosya kalmaz — ya eski hâli ya yeni hâli olur,
    yarim yazilmis JSON olmaz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class PortfolioManager:
    """
    Çoklu portföy yöneticisi.

    Her portföy data/portfolios/<slug>.json olarak saklanır.
    Format:
    {
        "name": "Benim Portföyüm",
        "created_at": "2026-05-08",
        "holdings_tl": {"AHB": 30000, "BGL": 20000}
    }
    """

    def __init__(self, portfolios_dir: str = "data/portfolios"):
        self.portfolios_dir = Path(portfolios_dir)
        self.portfolios_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy()

    def _migrate_legacy(self):
        """
        Eski data/my_portfolio.json varsa ve portfolios/ boşsa,
        varsayılan portföy olarak taşı.
        """
        legacy_path  = Path("data/my_portfolio.json")
        default_path = self.portfolios_dir / "varsayilan.json"

        # Sadece portfolios/ klasörü tamamen boşsa taşı
        existing = list(self.portfolios_dir.glob("*.json"))
        if legacy_path.exists() and not existing:
            try:
                with open(legacy_path, encoding="utf-8") as f:
                    old_data = json.load(f)

                new_data = {
                    "name":       "Varsayılan Portföy",
                    "created_at": datetime.now().strftime("%Y-%m-%d"),
                    "holdings_tl": old_data.get("holdings_tl", {}),
                }

                _atomic_write_json(default_path, new_data)

                logger.info("Eski portföy varsayılan olarak taşındı")
            except Exception as e:
                logger.warning(f"Legacy portföy taşıma hatası: {e}")

    def list_portfolios(self) -> List[Dict]:
        """
        Tüm portföyleri listele.
        Returns: [{"slug": "benim", "name": "Benim Portföyüm", "total_tl": 100000}, ...]
        """
        portfolios = []
        for path in sorted(self.portfolios_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                holdings = data.get("holdings_tl", {})
                portfolios.append({
                    "slug":       path.stem,
                    "name":       data.get("name", path.stem),
                    "total_tl":   sum(holdings.values()),
                    "fund_count": len([v for v in holdings.values() if v > 0]),
                    "created_at": data.get("created_at", "?"),
                })
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Portföy okuma hatası ({path.name}): {e}")

        return portfolios

    def get_portfolio(self, slug: str) -> Optional[Dict]:
        """Tek bir portföyü yükle."""
        path = self.portfolios_dir / f"{slug}.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Portföy yükleme hatası ({slug}): {e}")
            return None

    def save_portfolio(
        self,
        slug: str,
        name: str,
        holdings_tl: Dict[str, int],
        notes: str = "",
        monthly_contribution_tl: float = 0,
    ) -> bool:
        """Portföy kaydet (yeni veya güncelleme)."""
        path = self.portfolios_dir / f"{slug}.json"

        existing = {}
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                pass

        data = {
            "name":        name,
            "created_at":  existing.get("created_at", datetime.now().strftime("%Y-%m-%d")),
            "updated_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "notes":       notes if notes else existing.get("notes", ""),
            "holdings_tl": holdings_tl,
            "monthly_contribution_tl": float(monthly_contribution_tl or 0),
        }

        try:
            _atomic_write_json(path, data)
            logger.info(f"Portföy kaydedildi: {name} ({slug})")
            return True
        except OSError as e:
            logger.error(f"Portföy kaydetme hatası ({slug}): {e}")
            return False

    def delete_portfolio(self, slug: str) -> bool:
        """Portföy sil."""
        path = self.portfolios_dir / f"{slug}.json"
        if not path.exists():
            return False
        try:
            path.unlink()
            logger.info(f"Portföy silindi: {slug}")
            return True
        except OSError as e:
            logger.error(f"Portföy silme hatası ({slug}): {e}")
            return False

    def create_slug(self, name: str) -> str:
        """İsimden URL-safe slug oluştur."""
        slug = name.lower().strip()
        slug = slug.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU"))
        slug = re.sub(r"[^a-z0-9]+", "_", slug)
        slug = slug.strip("_")
        return slug or "portfoy"
