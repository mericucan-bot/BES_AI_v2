import pytest
from datetime import datetime
from src.performance_tracker import PerformanceTracker


class TestPerformanceTracker:
    def test_total_value_calculation(self):
        tracker = PerformanceTracker()
        result = tracker.calculate_current_portfolio_value({
            "VEF": 30000, "ALT": 25000, "KTS": 20000, "KCH": 15000, "CASH": 10000,
        })
        assert result["total_value"] == 100000

    def test_weights_sum_to_one(self):
        tracker = PerformanceTracker()
        result = tracker.calculate_current_portfolio_value({
            "VEF": 30000, "ALT": 25000, "KTS": 20000, "KCH": 15000, "CASH": 10000,
        })
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_individual_weight_correct(self):
        tracker = PerformanceTracker()
        result = tracker.calculate_current_portfolio_value({"VEF": 50000, "CASH": 50000})
        assert result["weights"]["VEF"] == pytest.approx(0.5)
        assert result["weights"]["CASH"] == pytest.approx(0.5)

    def test_date_field_iso_format(self):
        tracker = PerformanceTracker()
        result = tracker.calculate_current_portfolio_value({"VEF": 100})
        datetime.fromisoformat(result["date"])


class TestRealReturn:
    def setup_method(self):
        self.tracker = PerformanceTracker()

    def test_basic_real_return(self):
        """Nominal %5, yillik enflasyon %30 → aylik reel nominal'den kucuk olmali"""
        result = self.tracker.calculate_real_return(
            nominal_return=0.05,
            cpi_yoy=0.30,
            period_months=1,
        )
        assert result["real_return"] is not None
        assert result["inflation_period"] is not None
        assert result["real_return"] < result["nominal_return"]
        assert result["inflation_drag"] < 0

    def test_zero_nominal_negative_real(self):
        """Nominal %0 getiri + enflasyon → reel negatif"""
        result = self.tracker.calculate_real_return(0.0, 0.30, 1)
        assert result["real_return"] < 0

    def test_no_cpi_returns_none(self):
        """CPI yoksa reel hesaplanamaz"""
        result = self.tracker.calculate_real_return(0.05, None, 1)
        assert result["real_return"] is None
        assert result["inflation_period"] is None

    def test_zero_inflation(self):
        """Enflasyon %0 → reel = nominal"""
        result = self.tracker.calculate_real_return(0.05, 0.0, 1)
        assert result["real_return"] == pytest.approx(0.05, abs=1e-6)

    def test_high_inflation_scenario(self):
        """Turkiye senaryosu: %80 yillik enflasyon, %3 aylik nominal"""
        result = self.tracker.calculate_real_return(0.03, 0.80, 1)
        # Aylik enflasyon ~%5, nominal %3 → reel negatif
        assert result["real_return"] < 0

    def test_annual_period(self):
        """12 aylik donem: yillik enflasyon direkt uygulanir"""
        result = self.tracker.calculate_real_return(0.35, 0.30, 12)
        # Nominal %35, yillik enflasyon %30 → reel ~%3.8
        assert result["real_return"] > 0
        assert result["real_return"] == pytest.approx(0.0385, abs=0.005)

    def test_fisher_equation_identity(self):
        """Fisher denklemi: (1+r) = (1+n)/(1+i)"""
        result = self.tracker.calculate_real_return(0.10, 0.05, 12)
        expected = (1.10 / 1.05) - 1
        assert result["real_return"] == pytest.approx(expected, abs=1e-4)


class TestRealPortfolioValue:
    def setup_method(self):
        self.tracker = PerformanceTracker()

    def test_basic_real_value(self):
        result = self.tracker.calculate_real_portfolio_value(
            current_value=120000,
            initial_value=100000,
            initial_date="2025-10-01T00:00:00",
            current_date="2026-04-01T00:00:00",
            cpi_yoy=0.30,
        )
        assert result["nominal_total_return"] == pytest.approx(0.20)
        assert result["months_elapsed"] == 6
        assert result["real_total_return"] is not None
        assert result["real_total_return"] < result["nominal_total_return"]
        assert result["real_value"] is not None
        assert result["real_value"] < result["nominal_value"]

    def test_no_cpi_nominal_only(self):
        result = self.tracker.calculate_real_portfolio_value(
            current_value=110000,
            initial_value=100000,
            initial_date="2025-01-01",
            current_date="2026-01-01",
            cpi_yoy=None,
        )
        assert result["nominal_total_return"] == pytest.approx(0.10)
        assert result["real_total_return"] is None
        assert result["real_value"] is None

    def test_zero_initial_value(self):
        result = self.tracker.calculate_real_portfolio_value(
            current_value=100000,
            initial_value=0,
            initial_date="2025-01-01",
            current_date="2026-01-01",
            cpi_yoy=0.30,
        )
        assert result["nominal_total_return"] is None
