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

    def test_zero_holdings_no_crash(self):
        """Tum bakiyeler 0 ise ZeroDivisionError olmamali; agirliklar 0.0 olmali."""
        tracker = PerformanceTracker()
        result = tracker.calculate_current_portfolio_value({"A": 0, "B": 0})
        assert result["total_value"] == 0
        assert result["weights"] == {"A": 0.0, "B": 0.0}


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


class TestPortfolioHistory:
    def test_empty_history(self, tmp_path):
        tracker = PerformanceTracker()
        df = tracker.get_portfolio_history(str(tmp_path))
        assert df.empty

    def test_reads_snapshots(self, tmp_path):
        import json

        snap1 = {
            "run_date": "2026-03-01T10:00:00",
            "portfolio_value": {"total_value": 100000},
            "regime": {"detected": "STABLE", "confidence": 0.85},
        }
        snap2 = {
            "run_date": "2026-04-01T10:00:00",
            "portfolio_value": {"total_value": 103000},
            "regime": {"detected": "RISK_ON", "confidence": 0.7},
            "previous_evaluation": {"monthly_return": 0.03},
        }
        (tmp_path / "2026_03_snapshot.json").write_text(json.dumps(snap1))
        (tmp_path / "2026_04_snapshot.json").write_text(json.dumps(snap2))

        df = PerformanceTracker().get_portfolio_history(str(tmp_path))

        assert len(df) == 2
        assert df.iloc[0]["total_value"] == 100000
        assert df.iloc[1]["total_value"] == 103000
        assert df.iloc[1]["regime"] == "RISK_ON"
        assert df.iloc[1]["monthly_return"] == pytest.approx(0.03)

    def test_sorted_by_date(self, tmp_path):
        import json

        (tmp_path / "2026_04_snapshot.json").write_text(json.dumps({
            "run_date": "2026-04-01", "portfolio_value": {"total_value": 110000},
            "regime": {"detected": "STABLE"},
        }))
        (tmp_path / "2026_02_snapshot.json").write_text(json.dumps({
            "run_date": "2026-02-01", "portfolio_value": {"total_value": 100000},
            "regime": {"detected": "CRISIS"},
        }))

        df = PerformanceTracker().get_portfolio_history(str(tmp_path))

        assert df.iloc[0]["date"] < df.iloc[1]["date"]

    def test_handles_corrupt_file(self, tmp_path):
        import json

        (tmp_path / "2026_03_snapshot.json").write_text("corrupt json{{{")
        (tmp_path / "2026_04_snapshot.json").write_text(json.dumps({
            "run_date": "2026-04-01", "portfolio_value": {"total_value": 100000},
            "regime": {"detected": "STABLE"},
        }))

        df = PerformanceTracker().get_portfolio_history(str(tmp_path))
        assert len(df) == 1


class TestRevalueHoldings:
    def _tracker(self):
        return PerformanceTracker()

    def test_market_return_from_fund_returns(self):
        t = self._tracker()
        prev = {"AHS": 100.0, "BGL": 100.0}  # esit agirlik
        # AHS +%10, BGL -%2 -> piyasa getirisi = +%4
        res = t.revalue_holdings(prev, {"AHS": 0.10, "BGL": -0.02})
        assert res is not None
        assert abs(res["market_return"] - 0.04) < 1e-9
        assert res["coverage"] == 1.0
        assert set(res["applied_codes"]) == {"AHS", "BGL"}

    def test_case_insensitive_codes(self):
        t = self._tracker()
        res = t.revalue_holdings({"ahs": 200.0}, {"AHS": 0.05})
        assert res is not None
        assert abs(res["market_return"] - 0.05) < 1e-9

    def test_low_coverage_returns_none(self):
        t = self._tracker()
        # Bilinen getiri yalniz %20'lik TL -> kapsam < 0.5 -> None (fallback)
        prev = {"A": 20.0, "B": 80.0}
        res = t.revalue_holdings(prev, {"A": 0.10})
        assert res is None

    def test_missing_codes_held_flat(self):
        t = self._tracker()
        prev = {"A": 60.0, "B": 40.0}
        res = t.revalue_holdings(prev, {"A": 0.10})  # B bilinmiyor
        assert res is not None
        # A: 60*1.1=66, B: 40 (degismez) -> toplam 106 -> %6
        assert abs(res["market_return"] - 0.06) < 1e-9
        assert res["missing_codes"] == ["B"]
        assert res["coverage"] == 0.6

    def test_empty_or_zero_returns_none(self):
        t = self._tracker()
        assert t.revalue_holdings({}, {"A": 0.1}) is None
        assert t.revalue_holdings({"A": 0.0}, {"A": 0.1}) is None
