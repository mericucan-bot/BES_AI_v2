import pytest
from pathlib import Path
from src.report_generator import ReportGenerator


@pytest.fixture
def sample_pipeline_result():
    return {
        "status": "SUCCESS",
        "run_date": "2026-05-08T10:00:00",
        "portfolio_value": {"total_value": 100000, "weights": {"VEF": 0.3, "ALT": 0.25}},
        "regime": {
            "detected": "STABLE",
            "confidence": 0.87,
            "metrics": {"dd": -0.02, "vol": 0.19, "usd_mom": 0.017},
            "macro": {"cpi_yoy": 0.306, "usdtry_official": 45.16},
        },
        "recommendation": {
            "target_weights": {"VEF": 0.3, "ALT": 0.3, "KTS": 0.3, "CASH": 0.1},
            "actions": [
                {"asset": "KTS",  "action": "BUY",  "diff_tl":  10000, "current_weight": 0.2,  "target_weight": 0.3},
                {"asset": "KCH",  "action": "SELL", "diff_tl": -15000, "current_weight": 0.15, "target_weight": 0.0},
                {"asset": "CASH", "action": "HOLD", "diff_tl":     50, "current_weight": 0.1,  "target_weight": 0.1},
            ],
            "cost_analysis": {"total_cost_tl": 60, "total_cost_pct": 0.0006, "switch_count": 2},
        },
        "real_portfolio": {
            "nominal_total_return": 0.05,
            "real_total_return": -0.02,
            "real_value": 98000,
            "months_elapsed": 3,
        },
    }


@pytest.fixture
def sample_ml_summary():
    return {
        "status": "SUCCESS",
        "best_model": "xgboost",
        "best_ic": 0.797,
        "best_dir_acc": 1.0,
        "fund_count": 390,
        "top_features": {
            "return_1m": "0.6975",
            "return_3m": "0.2482",
            "return_6m": "0.0252",
        },
    }


class TestReportGenerator:
    def test_generate_creates_pdf(self, tmp_path, sample_pipeline_result, sample_ml_summary):
        gen  = ReportGenerator()
        path = gen.generate(
            pipeline_result=sample_pipeline_result,
            ml_summary=sample_ml_summary,
            output_path=str(tmp_path),
        )
        assert path is not None
        assert Path(path).exists()
        assert Path(path).suffix == ".pdf"
        assert Path(path).stat().st_size > 1000

    def test_generate_without_ml(self, tmp_path, sample_pipeline_result):
        gen  = ReportGenerator()
        path = gen.generate(
            pipeline_result=sample_pipeline_result,
            ml_summary=None,
            output_path=str(tmp_path),
        )
        assert path is not None
        assert Path(path).exists()

    def test_generate_without_pipeline(self, tmp_path, sample_ml_summary):
        gen  = ReportGenerator()
        path = gen.generate(
            pipeline_result=None,
            ml_summary=sample_ml_summary,
            output_path=str(tmp_path),
        )
        assert path is not None

    def test_generate_empty_actions(self, tmp_path, sample_pipeline_result):
        sample_pipeline_result["recommendation"]["actions"] = [
            {"asset": "VEF", "action": "HOLD", "diff_tl": 10, "current_weight": 0.3, "target_weight": 0.3}
        ]
        gen  = ReportGenerator()
        path = gen.generate(
            pipeline_result=sample_pipeline_result,
            output_path=str(tmp_path),
        )
        assert path is not None

    def test_pdf_filename_format(self, tmp_path, sample_pipeline_result):
        gen      = ReportGenerator()
        path     = gen.generate(
            pipeline_result=sample_pipeline_result,
            output_path=str(tmp_path),
        )
        filename = Path(path).name
        assert filename.startswith("BES_AI_Rapor_")
        assert filename.endswith(".pdf")
