"""PLAN-19: BES devlet katkisi (%30) tavan optimizasyonu."""
from src.state_contribution import (
    analyze_contribution,
    ContributionConfig,
    DEFAULT_MIN_WAGE_MONTHLY_2026,
)


class TestAnalyzeContribution:
    """Tablo: min_wage_monthly=30000 -> yillik 360000, tavan 108000."""

    def test_at_cap_when_monthly_equals_min_wage(self):
        # Ayda 30.000 (= asgari ucret) -> annual_match 108.000 = tavan
        out = analyze_contribution(30000.0)
        assert out["annual_match"] == 108000.0
        assert out["max_annual_match"] == 108000.0
        assert out["at_cap"] is True
        assert out["suggested_extra_monthly"] == 0.0
        assert out["match_gap"] == 0.0
        assert out["utilization_pct"] == 1.0
        assert out["annual_contribution"] == 360000.0

    def test_partial_contribution(self):
        # Ayda 9.000 -> annual 108.000, match 32.400, gap 75.600
        # suggested_extra = (360.000 - 108.000) / 12 = 21.000
        out = analyze_contribution(9000.0)
        assert out["annual_contribution"] == 108000.0
        assert out["annual_match"] == 32400.0
        assert out["match_gap"] == 75600.0
        assert out["suggested_extra_monthly"] == 21000.0
        assert out["at_cap"] is False
        assert out["max_annual_match"] == 108000.0
        assert abs(out["utilization_pct"] - 32400.0 / 108000.0) < 1e-9

    def test_none_contribution(self):
        # Katki None -> match 0, gap 108.000, suggested_extra 30.000
        out = analyze_contribution(None)
        assert out["monthly_contribution"] is None
        assert out["annual_match"] == 0.0
        assert out["match_gap"] == 108000.0
        assert out["suggested_extra_monthly"] == 30000.0
        assert out["at_cap"] is False
        assert out["utilization_pct"] == 0.0

    def test_zero_contribution(self):
        out = analyze_contribution(0.0)
        assert out["annual_match"] == 0.0
        assert out["match_gap"] == 108000.0
        assert out["suggested_extra_monthly"] == 30000.0
        assert out["at_cap"] is False

    def test_env_override_min_wage(self, monkeypatch):
        # BES_MIN_WAGE_MONTHLY env override
        monkeypatch.setenv("BES_MIN_WAGE_MONTHLY", "40000")
        out = analyze_contribution(40000.0)
        # yillik 480000, tavan 144000
        assert out["max_annual_match"] == 144000.0
        assert out["annual_match"] == 144000.0
        assert out["at_cap"] is True
        assert out["suggested_extra_monthly"] == 0.0

    def test_env_override_with_partial(self, monkeypatch):
        monkeypatch.setenv("BES_MIN_WAGE_MONTHLY", "20000")
        # tavan = 20000*12*0.3 = 72000
        out = analyze_contribution(10000.0)
        assert out["max_annual_match"] == 72000.0
        assert out["annual_match"] == 36000.0  # 120000 * 0.3
        assert out["match_gap"] == 36000.0
        assert out["suggested_extra_monthly"] == 10000.0  # (240000-120000)/12
        assert out["at_cap"] is False

    def test_default_min_wage_constant(self):
        assert DEFAULT_MIN_WAGE_MONTHLY_2026 == 30000.0

    def test_config_override(self):
        cfg = ContributionConfig(min_wage_monthly=25000.0, match_rate=0.30)
        out = analyze_contribution(25000.0, config=cfg)
        assert out["max_annual_match"] == 90000.0  # 25000*12*0.3
        assert out["at_cap"] is True
