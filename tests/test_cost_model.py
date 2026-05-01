import pytest
from src.cost_model import TransactionCostModel, CostConfig


class TestCostConfig:
    def test_default_values(self):
        config = CostConfig()
        assert config.slippage_pct == 0.002
        assert config.max_monthly_switches == 6
        assert config.switch_fee_pct == 0.0

    def test_custom_config(self):
        config = CostConfig(slippage_pct=0.005, max_monthly_switches=4)
        assert config.slippage_pct == 0.005
        assert config.max_monthly_switches == 4


class TestRebalanceCost:
    def setup_method(self):
        self.model = TransactionCostModel()

    def test_no_action_zero_cost(self):
        recs = [{"asset": "VEF", "action": "HOLD", "diff_tl": 0}]
        result = self.model.calculate_rebalance_cost(recs, 100000)
        assert result["total_cost_tl"] == 0
        assert result["switch_count"] == 0

    def test_buy_cost_is_slippage_only(self):
        """BES'te alim komisyonu yok, sadece slippage"""
        recs = [{"asset": "VEF", "action": "BUY", "diff_tl": 10000}]
        result = self.model.calculate_rebalance_cost(recs, 100000)
        assert result["total_cost_tl"] == pytest.approx(20.0)
        assert result["switch_count"] == 1

    def test_sell_cost_includes_exit_load(self):
        """Exit load sadece satista"""
        config = CostConfig(exit_load_pct=0.05, slippage_pct=0.002)
        model = TransactionCostModel(config)
        recs = [{"asset": "ALT", "action": "SELL", "diff_tl": -10000}]
        result = model.calculate_rebalance_cost(recs, 100000)
        assert result["total_cost_tl"] == pytest.approx(520.0)

    def test_multiple_trades_summed(self):
        recs = [
            {"asset": "VEF", "action": "BUY", "diff_tl": 5000},
            {"asset": "ALT", "action": "SELL", "diff_tl": -5000},
        ]
        result = self.model.calculate_rebalance_cost(recs, 100000)
        assert result["total_cost_tl"] == pytest.approx(20.0)
        assert result["turnover_tl"] == pytest.approx(10000)
        assert result["switch_count"] == 2

    def test_below_min_amount_skipped(self):
        """min_switch_amount_tl alti islemler maliyet hesabina dahil edilmez"""
        recs = [{"asset": "CASH", "action": "BUY", "diff_tl": 50}]
        result = self.model.calculate_rebalance_cost(recs, 100000)
        assert result["total_cost_tl"] == 0
        assert result["switch_count"] == 0

    def test_turnover_percentage(self):
        recs = [
            {"asset": "VEF", "action": "BUY", "diff_tl": 20000},
            {"asset": "KTS", "action": "SELL", "diff_tl": -20000},
        ]
        result = self.model.calculate_rebalance_cost(recs, 100000)
        assert result["turnover_pct"] == pytest.approx(0.40)

    def test_exceeds_monthly_limit_flag(self):
        recs = [{"asset": f"FUND{i}", "action": "BUY", "diff_tl": 1000} for i in range(8)]
        result = self.model.calculate_rebalance_cost(recs, 100000)
        assert result["exceeds_monthly_limit"] is True
        assert result["switch_count"] == 8


class TestNetAlpha:
    def setup_method(self):
        self.model = TransactionCostModel()

    def test_positive_net_alpha(self):
        result = self.model.calculate_net_alpha(0.012, 0.0004)
        assert result["net_alpha"] == pytest.approx(0.0116)
        assert result["cost_effective"] is True

    def test_cost_exceeds_alpha(self):
        """Maliyet alpha'dan buyuk → negatif net alpha"""
        result = self.model.calculate_net_alpha(0.001, 0.002)
        assert result["net_alpha"] < 0

    def test_cost_ratio_high_not_effective(self):
        """Maliyet alpha'nin %50'sinden fazla → cost_effective=False"""
        result = self.model.calculate_net_alpha(0.01, 0.006)
        assert result["cost_effective"] is False
        assert result["cost_ratio"] == pytest.approx(0.6)

    def test_zero_alpha_zero_cost(self):
        result = self.model.calculate_net_alpha(0.0, 0.0)
        assert result["net_alpha"] == 0
        assert result["cost_effective"] is True


class TestFilterByLimit:
    def test_under_limit_no_change(self):
        model = TransactionCostModel(CostConfig(max_monthly_switches=6))
        recs = [
            {"asset": "A", "action": "BUY", "diff_tl": 5000},
            {"asset": "B", "action": "SELL", "diff_tl": -3000},
        ]
        filtered = model.filter_recommendations_by_limit(recs)
        actionable = [r for r in filtered if r["action"] != "HOLD"]
        assert len(actionable) == 2

    def test_over_limit_defers_smallest(self):
        """Limit asiminda en kucuk diff'li islemler ertelenir"""
        model = TransactionCostModel(CostConfig(max_monthly_switches=2))
        recs = [
            {"asset": "A", "action": "BUY", "diff_tl": 10000},
            {"asset": "B", "action": "SELL", "diff_tl": -8000},
            {"asset": "C", "action": "BUY", "diff_tl": 3000},
            {"asset": "D", "action": "SELL", "diff_tl": -1000},
        ]
        filtered = model.filter_recommendations_by_limit(recs)
        kept = [r for r in filtered if r["action"] != "HOLD"]
        deferred = [r for r in filtered if r.get("deferred")]

        assert len(kept) == 2
        assert len(deferred) == 2
        kept_assets = {r["asset"] for r in kept}
        assert "A" in kept_assets
        assert "B" in kept_assets

    def test_holds_preserved(self):
        model = TransactionCostModel(CostConfig(max_monthly_switches=1))
        recs = [
            {"asset": "A", "action": "BUY", "diff_tl": 5000},
            {"asset": "B", "action": "BUY", "diff_tl": 3000},
            {"asset": "C", "action": "HOLD", "diff_tl": 0},
        ]
        filtered = model.filter_recommendations_by_limit(recs)
        holds = [r for r in filtered if r["action"] == "HOLD" and not r.get("deferred")]
        assert len(holds) == 1
