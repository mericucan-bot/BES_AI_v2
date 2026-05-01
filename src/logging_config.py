import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Renkli console output. TTY degilse renksiz."""
    COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.levelname_colored = f"{color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def configure_logging(
    log_dir: str = "logs",
    log_file: str = "bes_ai.log",
    level: str = "INFO",
    use_color: Optional[bool] = None,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 5,
    quiet_console: bool = False,
) -> None:
    """
    Merkezi logging setup. Idempotent (birden fazla cagrilabilir).

    Parametreler:
        log_dir       : Log dizini (yoksa olusturulur)
        log_file      : Ana log dosya adi
        level         : Console log seviyesi (DEBUG/INFO/WARNING/ERROR)
        use_color     : None ise auto-detect (TTY ise renkli)
        max_bytes     : Rotasyon esigi (default 5 MB)
        backup_count  : Kac eski log saklanacak (default 5)
        quiet_console : True ise console'a hic yazma (cron --json icin)

    Davranis:
        - Console : ayarlanan seviye (default INFO), stderr'e
        - Dosya   : her zaman DEBUG, rotasyonlu
        - 3. parti: WARNING'e bastirildi
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect renk: console handler stderr kullandigi icin stderr'e bakiyoruz.
    # Windows CMD'de stderr.isatty() True donse bile ANSI desteklemeyebilir,
    # bu yuzden Windows Terminal / ANSICON / ConEmu varligi kontrol ediliyor.
    if use_color is None:
        is_tty = sys.stderr.isatty()
        if sys.platform == "win32":
            import os
            has_ansi_support = bool(
                os.environ.get("WT_SESSION") or
                os.environ.get("ANSICON") or
                os.environ.get("ConEmuANSI") == "ON"
            )
            use_color = is_tty and has_ansi_support
        else:
            use_color = is_tty

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # root her seyi yakalar, handler'lar filtreler

    # Eski handler'lari temizle (idempotent)
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    # Console handler (quiet_console=False ise)
    if not quiet_console:
        console = logging.StreamHandler(sys.stderr)  # stderr: stdout JSON'u kirletmez
        console.setLevel(getattr(logging, level.upper()))

        if use_color:
            fmt = ColoredFormatter(
                "%(asctime)s | %(levelname_colored)s | %(name)-25s | %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
                datefmt="%H:%M:%S",
            )
        console.setFormatter(fmt)
        root.addHandler(console)

    # File handler (her zaman DEBUG, rotasyonlu)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s:%(lineno)-4d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)

    # 3. parti gurultusunu kis
    for noisy in ["urllib3", "yfinance", "matplotlib", "PIL", "peewee", "asyncio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging hazir: console={level if not quiet_console else 'OFF'}, "
        f"dosya=DEBUG -> {log_path / log_file}"
    )


def get_logger(name: str) -> logging.Logger:
    """Modul-bazli logger almak icin convenience function."""
    return logging.getLogger(name)
