from pathlib import Path

from src.io_utils import atomic_write_text


def test_writes_content(tmp_path):
    p = tmp_path / "f.txt"
    atomic_write_text(p, "merhaba")
    assert p.read_text(encoding="utf-8") == "merhaba"


def test_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "c.json"
    atomic_write_text(p, "{}")
    assert p.exists()


def test_overwrites_existing(tmp_path):
    p = tmp_path / "f.txt"
    atomic_write_text(p, "eski")
    atomic_write_text(p, "yeni")
    assert p.read_text(encoding="utf-8") == "yeni"


def test_accepts_str_path(tmp_path):
    p = str(tmp_path / "f.txt")
    atomic_write_text(p, "x")
    assert Path(p).read_text(encoding="utf-8") == "x"


def test_no_temp_files_left(tmp_path):
    p = tmp_path / "f.txt"
    atomic_write_text(p, "x")
    leftovers = [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
    assert leftovers == []
