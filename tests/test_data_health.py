"""PLAN-22 + PLAN-24: veri saglik/tazelik + aylik kosum bekcisi (agsiz, tmp_path)."""
import json
import os
from datetime import datetime, timedelta

import pandas as pd

from src.data_health import check_data_health, HealthThresholds, _check_monthly_run


def _write_nav(cache_dir, last_date):
    df = pd.DataFrame({
        "fund_code": ["F1", "F1"],
        "date": pd.to_datetime([last_date - timedelta(days=1), last_date]),
        "price": [100.0, 101.0],
    })
    df.to_parquet(cache_dir / "nav_history.parquet")


def _write_snapshot(cache_dir, name="snapshot_EMK_20260718.parquet"):
    pd.DataFrame({"fund_code": ["F1"], "category": ["Stock Fund"]}).to_parquet(
        cache_dir / name)


def _write_macro(macro_dir, fetched_at):
    (macro_dir / "macro_TP_TRY_MT01.json").write_text(
        json.dumps({"fetched_at": fetched_at.isoformat(), "data": []}),
        encoding="utf-8")


def _write_ml(ml_dir, run_date):
    (ml_dir / "latest_run_summary.json").write_text(
        json.dumps({"status": "SUCCESS", "run_date": run_date.isoformat()}),
        encoding="utf-8")


def _write_history_snapshot(history_dir, run_date, name="2026_07_snapshot.json",
                            subdir=None):
    target = history_dir / subdir if subdir else history_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_text(
        json.dumps({
            "run_date": run_date.isoformat(),
            "status": "SUCCESS",
            "portfolio_value": {"total_value": 100000},
        }),
        encoding="utf-8",
    )


def _dirs(tmp_path):
    cache = tmp_path / "tefas_cache"; cache.mkdir()
    macro = tmp_path / "cache"; macro.mkdir()
    ml = tmp_path / "ml"; ml.mkdir()
    history = tmp_path / "history"; history.mkdir()
    return cache, macro, ml, history


def _fresh_all(tmp_path, now=None):
    cache, macro, ml, history = _dirs(tmp_path)
    now = now or datetime.now()
    _write_nav(cache, now)
    _write_snapshot(cache)
    _write_macro(macro, now)
    _write_ml(ml, now)
    _write_history_snapshot(history, now)
    return cache, macro, ml, history


class TestDataHealth:
    def test_all_fresh_ok(self, tmp_path):
        cache, macro, ml, history = _fresh_all(tmp_path)
        res = check_data_health(str(cache), str(macro), str(ml), str(history))
        assert res["ok"] is True
        assert res["warnings"] == []
        assert {c["name"] for c in res["checks"]} == {
            "nav_history", "tefas_snapshot", "macro_cache", "ml_summary", "monthly_run"}
        assert all(c["status"] == "ok" for c in res["checks"])

    def test_stale_nav_warns(self, tmp_path):
        cache, macro, ml, history = _dirs(tmp_path)
        now = datetime.now()
        _write_nav(cache, now - timedelta(days=20))   # nav_stale_days=10 asilir
        _write_snapshot(cache)
        _write_macro(macro, now)
        _write_ml(ml, now)
        _write_history_snapshot(history, now)
        res = check_data_health(str(cache), str(macro), str(ml), str(history))
        assert res["ok"] is False
        nav = next(c for c in res["checks"] if c["name"] == "nav_history")
        assert nav["status"] == "stale"
        assert nav["age_days"] >= 20
        assert any("NAV" in w for w in res["warnings"])

    def test_missing_files(self, tmp_path):
        cache, macro, ml, history = _dirs(tmp_path)   # bos dizinler
        res = check_data_health(str(cache), str(macro), str(ml), str(history))
        assert res["ok"] is False
        assert all(c["status"] == "missing" for c in res["checks"])
        assert len(res["warnings"]) == 5

    def test_never_raises_on_corrupt(self, tmp_path):
        cache, macro, ml, history = _dirs(tmp_path)
        # bozuk parquet + bozuk json
        (cache / "nav_history.parquet").write_text("bu parquet degil", encoding="utf-8")
        (ml / "latest_run_summary.json").write_text("{bozuk json", encoding="utf-8")
        _write_snapshot(cache)
        _write_macro(macro, datetime.now())
        res = check_data_health(str(cache), str(macro), str(ml), str(history))
        nav = next(c for c in res["checks"] if c["name"] == "nav_history")
        mlc = next(c for c in res["checks"] if c["name"] == "ml_summary")
        assert nav["status"] == "missing"
        assert mlc["status"] == "missing"

    def test_custom_thresholds(self, tmp_path):
        cache, macro, ml, history = _dirs(tmp_path)
        now = datetime.now()
        _write_nav(cache, now - timedelta(days=5))
        _write_snapshot(cache)
        _write_macro(macro, now)
        _write_ml(ml, now)
        _write_history_snapshot(history, now)
        # nav_stale_days=3 -> 5 gunluk nav stale olur
        th = HealthThresholds(nav_stale_days=3)
        res = check_data_health(
            str(cache), str(macro), str(ml), str(history), thresholds=th
        )
        nav = next(c for c in res["checks"] if c["name"] == "nav_history")
        assert nav["status"] == "stale"

    def test_macro_uses_mtime_fallback(self, tmp_path):
        cache, macro, ml, history = _dirs(tmp_path)
        now = datetime.now()
        _write_nav(cache, now)
        _write_snapshot(cache)
        # fetched_at olmayan json -> mtime fallback (taze)
        (macro / "macro_x.json").write_text(json.dumps({"data": []}), encoding="utf-8")
        _write_ml(ml, now)
        _write_history_snapshot(history, now)
        res = check_data_health(str(cache), str(macro), str(ml), str(history))
        mc = next(c for c in res["checks"] if c["name"] == "macro_cache")
        assert mc["status"] == "ok"


class TestMonthlyRunCheck:
    def test_fresh_snapshot_ok(self, tmp_path):
        history = tmp_path / "history"
        history.mkdir()
        _write_history_snapshot(history, datetime.now())
        out = _check_monthly_run(str(history), max_age_days=40)
        assert out["name"] == "monthly_run"
        assert out["status"] == "ok"

    def test_stale_snapshot_warns(self, tmp_path):
        history = tmp_path / "history"
        history.mkdir()
        old = datetime.now() - timedelta(days=50)
        _write_history_snapshot(history, old)
        out = _check_monthly_run(str(history), max_age_days=40)
        assert out["status"] == "stale"
        assert out["age_days"] >= 50
        assert "çalışmamış" in out["detail"]

    def test_slug_subdir_found(self, tmp_path):
        history = tmp_path / "history"
        history.mkdir()
        _write_history_snapshot(
            history, datetime.now(), name="2026_07_snapshot.json", subdir="meric"
        )
        out = _check_monthly_run(str(history), max_age_days=40)
        assert out["status"] == "ok"

    def test_corrupt_json_mtime_fallback(self, tmp_path):
        history = tmp_path / "history"
        history.mkdir()
        bad = history / "2026_06_snapshot.json"
        bad.write_text("{not json", encoding="utf-8")
        # mtime taze (simdi yazildi) -> ok
        out = _check_monthly_run(str(history), max_age_days=40)
        assert out["status"] == "ok"
        assert out["age_days"] is not None

    def test_missing_history(self, tmp_path):
        history = tmp_path / "history"
        history.mkdir()
        out = _check_monthly_run(str(history), max_age_days=40)
        assert out["status"] == "missing"
        assert "Hiç aylık" in out["detail"] or "snapshot" in out["detail"].lower()
