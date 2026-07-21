"""PLAN-24: kisisel veri zip yedegi (tmp_path, agsiz)."""
import zipfile
from pathlib import Path

from src.backup import backup_personal_data


def _seed_personal(base: Path) -> None:
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "data" / "my_portfolio.json").write_text('{"holdings_tl":{}}', encoding="utf-8")
    pf = base / "data" / "portfolios"
    pf.mkdir(parents=True, exist_ok=True)
    (pf / "x.json").write_text('{"name":"X","holdings_tl":{}}', encoding="utf-8")
    hist = base / "data" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    (hist / "a_snapshot.json").write_text('{"run_date":"2026-07-01"}', encoding="utf-8")
    # backups icinde dosya — zip'e GIRMEMELI
    bak = base / "data" / "backups"
    bak.mkdir(parents=True, exist_ok=True)
    (bak / "should_not_include.txt").write_text("secret", encoding="utf-8")


class TestBackupPersonalData:
    def test_creates_zip_with_personal_files(self, tmp_path):
        _seed_personal(tmp_path)
        dest = tmp_path / "out_backups"
        path = backup_personal_data(
            dest_dir=str(dest), keep=6, base_dir=str(tmp_path)
        )
        assert path is not None
        zp = Path(path)
        assert zp.exists()
        with zipfile.ZipFile(zp, "r") as zf:
            names = set(zf.namelist())
        assert "data/my_portfolio.json" in names
        assert "data/portfolios/x.json" in names
        assert "data/history/a_snapshot.json" in names
        # data/backups icerigi zip'te YOK
        assert not any(n.startswith("data/backups") for n in names)
        assert "should_not_include.txt" not in names

    def test_rotation_keeps_newest(self, tmp_path):
        _seed_personal(tmp_path)
        dest = tmp_path / "rot"
        dest.mkdir()
        # 3 eski yedek elle
        for stamp in ("20200101_0000", "20200102_0000", "20200103_0000"):
            (dest / f"bes_backup_{stamp}.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        path = backup_personal_data(
            dest_dir=str(dest), keep=2, base_dir=str(tmp_path)
        )
        assert path is not None
        remaining = sorted(dest.glob("bes_backup_*.zip"))
        assert len(remaining) == 2
        # en eski silinmis olmali
        names = {p.name for p in remaining}
        assert "bes_backup_20200101_0000.zip" not in names

    def test_no_sources_returns_none(self, tmp_path):
        empty = tmp_path / "empty_base"
        empty.mkdir()
        dest = tmp_path / "empty_dest"
        path = backup_personal_data(
            dest_dir=str(dest), keep=6, base_dir=str(empty)
        )
        assert path is None
        # zip artigi yok
        if dest.exists():
            assert list(dest.glob("bes_backup_*.zip")) == []

    def test_env_backup_dir(self, tmp_path, monkeypatch):
        _seed_personal(tmp_path)
        env_dest = tmp_path / "env_backups"
        monkeypatch.setenv("BES_BACKUP_DIR", str(env_dest))
        path = backup_personal_data(base_dir=str(tmp_path))
        assert path is not None
        assert Path(path).parent == env_dest

    def test_unwritable_dest_no_exception(self, tmp_path):
        _seed_personal(tmp_path)
        # Var olan bir DOSYA'yi dest_dir gibi ver — mkdir/zip patlar, None doner
        as_file = tmp_path / "not_a_dir"
        as_file.write_text("x", encoding="utf-8")
        path = backup_personal_data(
            dest_dir=str(as_file), keep=6, base_dir=str(tmp_path)
        )
        assert path is None
