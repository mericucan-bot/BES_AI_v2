"""PLAN-14: onemlilik skoru testleri (tablo gudumlu)."""
from src.significance import compute_significance, SignificanceConfig


def _regime(detected="STABLE", anomalies=None):
    r = {"detected": detected}
    if anomalies is not None:
        r["anomalies"] = anomalies
    return r


class TestSignificanceScoring:
    def test_no_signal_is_quiet(self):
        # class_weights hedefe esit, turnover yok, anomali yok, ilk ay
        cw = {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1}
        out = compute_significance(_regime(), None, cw, cw, {"turnover_pct": 0.0})
        assert out["score"] == 0
        assert out["level"] == "quiet"
        assert out["reasons"] == []

    def test_regime_change_alone(self):
        cw = {"VEF": 0.5}
        out = compute_significance(
            _regime("RATE_HIKE"),
            {"previous_regime": "STABLE"},
            cw, {"VEF": 0.5}, {"turnover_pct": 0.0},
        )
        assert out["score"] == 40
        assert out["level"] == "notable"
        assert any("Rejim degisti" in r for r in out["reasons"])
        assert any("Sakin" in r and "Faiz" in r for r in out["reasons"])

    def test_crisis_plus_high_drift_caps_at_100(self):
        cw = {"ALT": 1.0}
        tw = {"VEF": 0.3, "KTS": 0.3, "ALT": 0.3, "CASH": 0.1}  # ALT sapma 0.70
        out = compute_significance(
            _regime("CRISIS"),
            {"previous_regime": "STABLE"},   # +40 rejim degisti
            cw, tw, {"turnover_pct": 0.9},   # +40 crisis +30 drift +10 turnover
        )
        assert out["score"] == 100    # 40+40+30+10 = 120 -> kirp 100
        assert out["level"] == "action"

    def test_drift_notable_threshold(self):
        cw = {"VEF": 0.48, "ALT": 0.52}
        tw = {"VEF": 0.30, "ALT": 0.70}   # ALT sapma 0.18 -> notable (+15)
        out = compute_significance(_regime(), None, cw, tw, {"turnover_pct": 0.0})
        assert out["score"] == 15
        assert any("Hedeften sapma" in r for r in out["reasons"])

    def test_drift_action_threshold(self):
        cw = {"VEF": 0.35, "ALT": 0.65}
        tw = {"VEF": 0.30, "ALT": 0.30}   # ALT sapma 0.35 -> action drift (+30)
        out = compute_significance(_regime(), None, cw, tw, {"turnover_pct": 0.0})
        assert out["score"] == 30
        assert any("belirgin sapma" in r for r in out["reasons"])

    def test_turnover_notable(self):
        cw = {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1}
        out = compute_significance(_regime(), None, cw, cw, {"turnover_pct": 0.25})
        assert out["score"] == 10
        assert any("Onerilen degisim" in r for r in out["reasons"])

    def test_high_anomaly(self):
        cw = {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1}
        out = compute_significance(
            _regime("STABLE", anomalies=[{"severity": "high", "message": "BIST cakildi"}]),
            None, cw, cw, {"turnover_pct": 0.0},
        )
        assert out["score"] == 20
        assert "BIST cakildi" in out["reasons"]

    def test_medium_anomaly_only(self):
        cw = {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1}
        out = compute_significance(
            _regime("STABLE", anomalies=[{"severity": "medium", "message": "sert hareket"}]),
            None, cw, cw, {"turnover_pct": 0.0},
        )
        assert out["score"] == 10
        assert "sert hareket" in out["reasons"]

    def test_empty_class_weights_adds_map_missing(self):
        out = compute_significance(_regime(), None, {}, {"VEF": 0.3}, {"turnover_pct": 0.0})
        assert out["score"] == 15
        assert any("harita" in r for r in out["reasons"])

    def test_first_month_no_regime_change_signal(self):
        # evaluation None -> rejim degisimi sinyali yok, crash yok
        cw = {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1}
        out = compute_significance(_regime("CRISIS"), None, cw, cw, {"turnover_pct": 0.0})
        assert out["score"] == 40   # yalniz CRISIS
        assert not any("Rejim degisti" in r for r in out["reasons"])

    def test_none_cost_analysis_safe(self):
        cw = {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1}
        out = compute_significance(_regime(), None, cw, cw, None)
        assert out["score"] == 0

    def test_custom_config_thresholds(self):
        cfg = SignificanceConfig(level_notable=5)
        cw = {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1}
        out = compute_significance(
            _regime(), None, cw, cw, {"turnover_pct": 0.25}, config=cfg,
        )
        assert out["score"] == 10
        assert out["level"] == "notable"   # 10 >= 5
