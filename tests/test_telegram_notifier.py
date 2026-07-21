"""PLAN-23: Telegram bildirimi — agsiz (requests.post her zaman mock)."""
from unittest.mock import MagicMock, patch

import pytest

from src.telegram_notifier import (
    TelegramNotifier,
    build_alert_message,
    build_multi_message,
    _MAX_LEN,
)


class TestTelegramNotifierSend:
    def test_not_configured_no_post(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        n = TelegramNotifier()
        assert n.is_configured is False
        with patch("requests.post") as mock_post:
            assert n.send("hello") is False
            mock_post.assert_not_called()

    def test_send_200_true(self):
        n = TelegramNotifier(bot_token="tok", chat_id="42")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        with patch("requests.post", return_value=mock_resp) as mock_post:
            assert n.send("merhaba") is True
            mock_post.assert_called_once()
            kwargs = mock_post.call_args.kwargs
            assert kwargs["json"]["chat_id"] == "42"
            assert kwargs["json"]["text"] == "merhaba"
            assert kwargs["timeout"] == 15

    def test_send_exception_returns_false(self):
        n = TelegramNotifier(bot_token="tok", chat_id="42")
        with patch("requests.post", side_effect=RuntimeError("network down")):
            assert n.send("x") is False  # raise yok

    def test_long_text_truncated(self):
        n = TelegramNotifier(bot_token="tok", chat_id="42")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        long_text = "A" * 5000
        with patch("requests.post", return_value=mock_resp) as mock_post:
            assert n.send(long_text) is True
            sent = mock_post.call_args.kwargs["json"]["text"]
            assert len(sent) <= _MAX_LEN
            assert len(sent) == _MAX_LEN


class TestBuildAlertMessage:
    def _result(self, level="action", score=80, reasons=None, total=100000):
        return {
            "status": "SUCCESS",
            "significance": {
                "level": level,
                "score": score,
                "reasons": reasons or [],
            },
            "portfolio_value": {"total_value": total},
            "regime": {"detected": "CRISIS"},
            "recommendation": {
                "actions": [
                    {"asset": "ALT", "action": "BUY", "diff_tl": 5000},
                    {"asset": "VEF", "action": "SELL", "diff_tl": -3000},
                ]
            },
        }

    def test_quiet_returns_none(self):
        assert build_alert_message(self._result(level="quiet")) is None

    def test_no_significance_returns_none(self):
        r = {"status": "SUCCESS", "portfolio_value": {"total_value": 1}}
        assert build_alert_message(r) is None

    def test_action_includes_score_reasons_portfolio(self):
        msg = build_alert_message(self._result(
            level="action",
            score=85,
            reasons=["Kriz rejimi", "Yüksek turnover"],
            total=250000,
        ))
        assert msg is not None
        assert "85/100" in msg
        assert "action" in msg
        assert "Kriz rejimi" in msg
        assert "Yüksek turnover" in msg
        assert "250,000" in msg or "250.000" in msg
        assert "BUY ALT" in msg
        assert "Detay:" in msg

    @patch("src.data_health.check_data_health")
    def test_health_warning_appended(self, mock_health):
        mock_health.return_value = {
            "ok": False,
            "warnings": ["⚠️ NAV verisi 20 gündür güncellenmedi"],
        }
        msg = build_alert_message(self._result(level="notable", score=40, reasons=["X"]))
        assert msg is not None
        assert "⚙️" in msg
        assert "NAV verisi" in msg

    @patch("src.data_health.check_data_health", side_effect=RuntimeError("boom"))
    def test_health_exception_still_builds(self, _mock_health):
        msg = build_alert_message(self._result(
            level="action", score=70, reasons=["R1"],
        ))
        assert msg is not None
        assert "70/100" in msg
        assert "R1" in msg


class TestBuildMultiMessage:
    def test_all_quiet_returns_none(self):
        all_results = [
            {
                "slug": "a", "name": "A",
                "result": {
                    "status": "SUCCESS",
                    "significance": {"level": "quiet", "score": 5, "reasons": []},
                    "portfolio_value": {"total_value": 10000},
                },
            },
            {
                "slug": "b", "name": "B",
                "result": {
                    "status": "SUCCESS",
                    "significance": {"level": "quiet", "score": 0, "reasons": []},
                    "portfolio_value": {"total_value": 20000},
                },
            },
        ]
        assert build_multi_message(all_results) is None

    def test_one_action_includes_name_and_reason(self):
        all_results = [
            {
                "slug": "meric", "name": "Meric",
                "result": {
                    "status": "SUCCESS",
                    "significance": {
                        "level": "action",
                        "score": 80,
                        "reasons": ["Kriz rejimi tespit edildi"],
                    },
                    "portfolio_value": {"total_value": 150000},
                },
            },
            {
                "slug": "annem", "name": "Annem",
                "result": {
                    "status": "SUCCESS",
                    "significance": {"level": "quiet", "score": 10, "reasons": []},
                    "portfolio_value": {"total_value": 80000},
                },
            },
        ]
        msg = build_multi_message(all_results)
        assert msg is not None
        assert "Meric" in msg
        assert "80/100" in msg
        assert "Kriz rejimi tespit edildi" in msg
        assert "Annem" in msg
