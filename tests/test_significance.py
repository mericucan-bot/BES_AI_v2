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
        # cesitlendirilmis cw (tek sinif <%50) — yalniz rejim degisimi sinyali izole
        cw = {"VEF": 0.34, "ALT": 0.33, "KTS": 0.33}
        out = compute_significance(
            _regime("RATE_HIKE"),
            {"previous_regime": "STABLE"},
            cw, cw, {"turnover_pct": 0.0},   # cw==tw: drift 0
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
        # +25 konsantrasyon (ALT %100) da eklenir; toplam 145 -> kirp 100
        assert out["score"] == 100
        assert out["level"] == "action"

    def test_drift_notable_threshold(self):
        # cesitlendirilmis cw (max <%50) — yalniz drift sinyali izole
        cw = {"VEF": 0.42, "ALT": 0.34, "KTS": 0.24}
        tw = {"VEF": 0.24, "ALT": 0.34, "KTS": 0.42}   # max sapma 0.18 -> notable (+15)
        out = compute_significance(_regime(), None, cw, tw, {"turnover_pct": 0.0})
        assert out["score"] == 15
        assert any("Hedeften sapma" in r for r in out["reasons"])

    def test_drift_action_threshold(self):
        cw = {"VEF": 0.45, "ALT": 0.30, "KTS": 0.25}
        tw = {"VEF": 0.10, "ALT": 0.65, "KTS": 0.25}   # max sapma 0.35 -> action drift (+30)
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


class TestConcentrationGuard:
    """PLAN-20: tek sinifin asiri agirligi -> onemlilik sinyali."""

    def test_concentration_action(self):
        cw = {"ALT": 0.76, "VEF": 0.24}   # hedefe esit (drift 0), yalniz konsantrasyon
        out = compute_significance(_regime(), None, cw, cw, {"turnover_pct": 0.0})
        assert out["score"] == 25
        assert any("Konsantrasyon riski" in r and "%76" in r and "ALT" in r
                   for r in out["reasons"])

    def test_concentration_notable(self):
        cw = {"ALT": 0.55, "VEF": 0.45}
        out = compute_significance(_regime(), None, cw, cw, {"turnover_pct": 0.0})
        assert out["score"] == 12
        assert any("Yüksek yoğunlaşma" in r for r in out["reasons"])

    def test_no_concentration_below_threshold(self):
        cw = {"ALT": 0.40, "VEF": 0.35, "KTS": 0.25}
        out = compute_significance(_regime(), None, cw, cw, {"turnover_pct": 0.0})
        assert out["score"] == 0

    def test_concentration_stacks_with_crisis(self):
        cw = {"ALT": 0.76, "VEF": 0.24}
        out = compute_significance(_regime("CRISIS"), None, cw, cw, {"turnover_pct": 0.0})
        assert out["score"] == 65   # 40 crisis + 25 konsantrasyon
        assert out["level"] == "action"

    def test_empty_class_weights_no_concentration(self):
        # bos class_weights -> 'harita eksik' +15 (mevcut) ama konsantrasyon puani yok
        out = compute_significance(_regime(), None, {}, {"VEF": 0.3}, {"turnover_pct": 0.0})
        assert out["score"] == 15
        assert not any("Konsantrasyon" in r or "yoğunlaşma" in r for r in out["reasons"])
