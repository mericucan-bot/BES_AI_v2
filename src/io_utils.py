"""Kucuk, bagimsiz I/O yardimcilari (Streamlit/pandas bagimsiz)."""
import os
import tempfile
from pathlib import Path
from typing import Union


def atomic_write_text(path: Union[str, Path], content: str) -> None:
    """Atomic write: temp dosyaya yaz, sonra rename. Race condition / power
    failure'da yarim yazilmis dosya birakmaz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
