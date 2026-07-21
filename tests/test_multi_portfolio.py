"""PLAN-18: coklu portfoy otomasyonu (--all-portfolios)."""
from unittest.mock import MagicMock, patch, call

from main import run_all_portfolios


def _fake_portfolio(slug, name):
    return {
        "slug": slug,
        "name": name,
        "total_tl": 100000,
        "fund_count": 3,
        "created_at": "2026-01-01",
    }


class TestRunAllPortfolios:
    def test_runs_each_with_separate_history(self):
        portfolios = [
            _fake_portfolio("meric", "Meric"),
            _fake_portfolio("annem_1", "Annem 1"),
        ]
        mock_pm = MagicMock()
        mock_pm.list_portfolios.return_value = portfolios

        mock_pipe_inst = MagicMock()
        mock_pipe_inst.run.return_value = {
            "status": "SUCCESS",
            "portfolio_value": {"total_value": 100000},
            "regime": {"detected": "STABLE"},
            "recommendation": {"actions": []},
        }

        logger = MagicMock()
        with patch("src.portfolio_manager.PortfolioManager", return_value=mock_pm), \
             patch("src.pipeline.MonthlyPipeline", return_value=mock_pipe_inst) as MockPipe:
            results = run_all_portfolios(args=MagicMock(), logger=logger)

        assert len(results) == 2
        assert results[0]["slug"] == "meric"
        assert results[0]["name"] == "Meric"
        assert results[0]["result"]["status"] == "SUCCESS"
        assert results[1]["slug"] == "annem_1"

        # Her portfoy ayri history_dir / learning_path ile cagrildi
        assert MockPipe.call_count == 2
        kwargs0 = MockPipe.call_args_list[0].kwargs
        assert kwargs0["portfolio_path"] == "data/portfolios/meric.json"
        assert kwargs0["history_dir"] == "data/history/meric"
        assert kwargs0["learning_path"] == "data/history/meric/learning_history.json"

        kwargs1 = MockPipe.call_args_list[1].kwargs
        assert kwargs1["history_dir"] == "data/history/annem_1"
        assert kwargs1["learning_path"] == "data/history/annem_1/learning_history.json"

    def test_one_failure_continues_others(self):
        portfolios = [
            _fake_portfolio("ok_pf", "OK"),
            _fake_portfolio("bad_pf", "BAD"),
            _fake_portfolio("ok2_pf", "OK2"),
        ]
        mock_pm = MagicMock()
        mock_pm.list_portfolios.return_value = portfolios

        ok_result = {"status": "SUCCESS", "portfolio_value": {"total_value": 1}}
        pipe_ok = MagicMock()
        pipe_ok.run.return_value = ok_result
        pipe_bad = MagicMock()
        pipe_bad.run.side_effect = RuntimeError("boom")

        # call order: ok, bad, ok2
        instances = [pipe_ok, pipe_bad, pipe_ok]

        logger = MagicMock()
        with patch("src.portfolio_manager.PortfolioManager", return_value=mock_pm), \
             patch("src.pipeline.MonthlyPipeline", side_effect=instances):
            results = run_all_portfolios(args=MagicMock(), logger=logger)

        assert len(results) == 3
        assert results[0]["result"]["status"] == "SUCCESS"
        assert results[1]["result"]["status"] == "ERROR"
        assert "boom" in results[1]["result"]["message"]
        assert results[2]["result"]["status"] == "SUCCESS"

    def test_empty_portfolio_list(self):
        mock_pm = MagicMock()
        mock_pm.list_portfolios.return_value = []
        logger = MagicMock()
        with patch("src.portfolio_manager.PortfolioManager", return_value=mock_pm), \
             patch("src.pipeline.MonthlyPipeline") as MockPipe:
            results = run_all_portfolios(args=MagicMock(), logger=logger)
        assert results == []
        MockPipe.assert_not_called()
