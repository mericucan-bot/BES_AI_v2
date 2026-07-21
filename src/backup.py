"""Kisisel verinin zip yedegi — git'e girmeyen dosyalarin tek sigortasi."""
import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Yedeklenecek kisisel veri yollari (VAR OLANLAR alinir, olmayan sessiz atlanir)
PERSONAL_PATHS = [
    "data/my_portfolio.json",
    "data/portfolios",
    "data/history",
    "data/learning_history.json",
    "data/user_prefs.json",
    "data/notification_prefs.json",
    "data/user_class_overrides.json",
]


def _is_under_backups(path: Path, base: Path) -> bool:
    """data/backups altini zip'e ASLA alma (kendi kendini yedeklemesin)."""
    try:
        rel = path.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return False
    parts = rel.parts
    return len(parts) >= 2 and parts[0] == "data" and parts[1] == "backups"


def _collect_files(base: Path) -> List[Path]:
    files: List[Path] = []
    for rel in PERSONAL_PATHS:
        p = base / rel
        if not p.exists():
            continue
        if p.is_file():
            if not _is_under_backups(p, base):
                files.append(p)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and not _is_under_backups(f, base):
                    files.append(f)
    return files


def backup_personal_data(
    dest_dir: Optional[str] = None,
    keep: int = 6,
    base_dir: str = ".",
) -> Optional[str]:
    """Kisisel verileri data/backups/bes_backup_YYYYMMDD_HHMM.zip'e arsivle.
    dest_dir None ise once BES_BACKUP_DIR env, yoksa data/backups.
    Eski yedekleri dondurur: en yeni `keep` adet kalir.
    Returns: yazilan zip yolu | hicbir kaynak yoksa/hata None (ASLA exception)."""
    try:
        base = Path(base_dir)
        env_dest = os.environ.get("BES_BACKUP_DIR")
        dest = Path(dest_dir or env_dest or (base / "data" / "backups"))
        dest.mkdir(parents=True, exist_ok=True)

        files = _collect_files(base)
        if not files:
            logger.info("Yedek: eklenecek kisisel dosya yok")
            return None

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        zip_path = dest / f"bes_backup_{stamp}.zip"

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            added = 0
            for f in files:
                try:
                    arcname = str(f.resolve().relative_to(base.resolve()))
                except ValueError:
                    arcname = f.name
                # Guvenlik: arcname data/backups altinda olmasin
                if arcname.replace("\\", "/").startswith("data/backups"):
                    continue
                zf.write(f, arcname=arcname)
                added += 1

        if added == 0:
            try:
                zip_path.unlink()
            except OSError:
                pass
            return None

        # Rotasyon: en yeni `keep` adet kalir
        existing = sorted(dest.glob("bes_backup_*.zip"))
        if keep >= 0 and len(existing) > keep:
            for old in existing[:-keep]:
                try:
                    old.unlink()
                except OSError as e:
                    logger.warning(f"Eski yedek silinemedi ({old.name}): {e}")

        logger.info(f"Yedek alindi: {zip_path} ({added} dosya)")
        return str(zip_path)
    except Exception as e:
        logger.warning(f"Yedekleme hatasi: {e}")
        return None
