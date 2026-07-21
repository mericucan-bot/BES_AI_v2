"""PLAN-21: LLM anlati ozeti testleri (gercek API cagrisi YOK — mock/fallback)."""
from unittest.mock import patch

from src.narrative import generate_narrative, _template_summary


def _result(regime="STABLE", actions=None, sig=None, status="SUCCESS"):
    return {
        "status": status,
        "regime": {"detected": regime, "confidence": 0.7},
        "portfolio_value": {"total_value": 574426},
        "recommendation": {"actions": actions or []},
        "significance": sig or {"level": "quiet", "score": 0, "reasons": []},
    }


class TestTemplateSummary:
    def test_no_action(self):
        txt = _template_summary(_result())
        assert "Sakin Piyasa" in txt
        assert "574.426" in txt
        assert "degisiklik gerekmiyor" in txt

    def test_with_actions(self):
        acts = [{"asset": "KTS", "action": "BUY", "diff_tl": 5000},
                {"asset": "ALT", "action": "SELL", "diff_tl": -3000}]
        txt = _template_summary(_result(actions=acts))
        assert "2 degisiklik onerisi" in txt

    def test_notable_reason_included(self):
        sig = {"level": "action", "score": 65,
               "reasons": ["Konsantrasyon riski: %76 ALT"]}
        txt = _template_summary(_result(sig=sig))
        assert "Konsantrasyon riski" in txt

    def test_error_result(self):
        txt = _template_summary({"status": "ERROR"})
        assert "tamamlanamadi" in txt


class TestGenerateNarrative:
    def test_no_api_key_uses_template(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("src.narrative._call_claude") as mock_call:
            txt = generate_narrative(_result())
            mock_call.assert_not_called()   # ag cagrisi YOK
        assert "Sakin Piyasa" in txt   # sablon sonucu

    def test_api_key_calls_claude(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("src.narrative._call_claude", return_value="LLM ozeti burada.") as mock_call:
            txt = generate_narrative(_result())
            mock_call.assert_called_once()
        assert txt == "LLM ozeti burada."

    def test_api_error_falls_back(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("src.narrative._call_claude", side_effect=RuntimeError("ag hatasi")):
            txt = generate_narrative(_result())
        assert "Sakin Piyasa" in txt   # fallback sablon

    def test_empty_llm_response_falls_back(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("src.narrative._call_claude", return_value="   "):
            txt = generate_narrative(_result())
        assert "Sakin Piyasa" in txt
