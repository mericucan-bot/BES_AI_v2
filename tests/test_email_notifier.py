import pytest
import smtplib
from unittest.mock import patch, MagicMock
from src.email_notifier import EmailNotifier


class TestEmailNotifierInit:
    def test_not_configured_without_env(self, monkeypatch):
        monkeypatch.delenv("EMAIL_SENDER",     raising=False)
        monkeypatch.delenv("EMAIL_PASSWORD",   raising=False)
        monkeypatch.delenv("EMAIL_RECIPIENTS", raising=False)
        notifier = EmailNotifier()
        assert notifier.is_configured is False

    def test_configured_with_params(self):
        notifier = EmailNotifier(
            sender="test@gmail.com",
            password="testpass",
            recipients=["alici@email.com"],
        )
        assert notifier.is_configured is True
        assert notifier.sender == "test@gmail.com"


class TestBuildSubject:
    def test_subject_with_result(self):
        notifier = EmailNotifier(sender="a", password="b", recipients=["c"])
        result = {
            "status": "SUCCESS",
            "regime": {"detected": "STABLE"},
            "portfolio_value": {"total_value": 100000},
        }
        subject = notifier._build_subject(result)
        assert "Sakin Piyasa" in subject
        assert "100,000" in subject or "100.000" in subject

    def test_subject_without_result(self):
        notifier = EmailNotifier(sender="a", password="b", recipients=["c"])
        subject = notifier._build_subject(None)
        assert "BES AI" in subject


class TestBuildHtmlBody:
    def test_html_contains_sections(self):
        notifier = EmailNotifier(sender="a", password="b", recipients=["c"])
        result = {
            "status": "SUCCESS",
            "regime": {"detected": "STABLE", "confidence": 0.85},
            "portfolio_value": {"total_value": 100000},
            "recommendation": {"actions": [
                {"asset": "KTS", "action": "BUY", "diff_tl": 10000},
            ]},
        }
        html = notifier._build_html_body(result, None)
        assert "Piyasa Durumu" in html
        assert "BES Akıllı Fon Danışmanı" in html
        assert "EKLE" in html

    def test_html_with_ml_summary(self):
        notifier = EmailNotifier(sender="a", password="b", recipients=["c"])
        ml = {"status": "SUCCESS", "best_model": "xgboost", "best_ic": 0.8, "fund_count": 390}
        html = notifier._build_html_body(None, ml)
        assert "XGBOOST" in html
        assert "0.80" in html


class TestSendEmail:
    def test_send_when_not_configured(self):
        notifier = EmailNotifier()
        assert notifier.send_monthly_report({}) is False

    @patch("src.email_notifier.smtplib.SMTP")
    def test_send_success(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        notifier = EmailNotifier(
            sender="test@gmail.com",
            password="testpass",
            recipients=["alici@email.com"],
        )
        result = {
            "status": "SUCCESS",
            "regime": {"detected": "STABLE", "confidence": 0.8},
            "portfolio_value": {"total_value": 100000},
            "recommendation": {"actions": []},
        }

        success = notifier.send_monthly_report(result)
        assert success is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()

    @patch("src.email_notifier.smtplib.SMTP")
    def test_send_auth_failure(self, mock_smtp):
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        notifier = EmailNotifier(
            sender="test@gmail.com",
            password="wrongpass",
            recipients=["alici@email.com"],
        )

        success = notifier.send_monthly_report({})
        assert success is False


class TestSignificanceMode:
    _RESULT = {
        "status": "SUCCESS",
        "regime": {"detected": "STABLE", "confidence": 0.8},
        "portfolio_value": {"total_value": 100000},
        "recommendation": {"actions": [
            {"asset": "KTS", "action": "BUY", "diff_tl": 5000}
        ]},
    }

    def _notifier(self):
        return EmailNotifier(sender="t@gmail.com", password="p",
                             recipients=["a@b.com"])

    def _captured_msg(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        return mock_server

    @patch("src.email_notifier.smtplib.SMTP")
    def test_quiet_sends_short_body(self, mock_smtp):
        server = self._captured_msg(mock_smtp)
        n = self._notifier()
        ok = n.send_monthly_report(
            self._RESULT, significance={"level": "quiet", "score": 0, "reasons": []},
        )
        assert ok is True
        sent = server.send_message.call_args[0][0]
        body = sent.get_payload()[0].get_payload(decode=True).decode("utf-8")
        assert "önemli bir gelişme yok" in body
        assert "Bu Ay Yapılması Gerekenler" not in body
        assert "Sakin ay" in sent["Subject"]

    @patch("src.email_notifier.smtplib.SMTP")
    def test_quiet_force_full_sends_full(self, mock_smtp):
        server = self._captured_msg(mock_smtp)
        n = self._notifier()
        n.send_monthly_report(
            self._RESULT, significance={"level": "quiet", "score": 0, "reasons": []},
            force_full=True,
        )
        sent = server.send_message.call_args[0][0]
        body = sent.get_payload()[0].get_payload(decode=True).decode("utf-8")
        assert "Bu Ay Yapılması Gerekenler" in body

    @patch("src.email_notifier.smtplib.SMTP")
    def test_action_shows_reasons_and_prefix(self, mock_smtp):
        server = self._captured_msg(mock_smtp)
        n = self._notifier()
        n.send_monthly_report(
            self._RESULT,
            significance={"level": "action", "score": 80,
                          "reasons": ["Kriz rejimi tespit edildi"]},
        )
        sent = server.send_message.call_args[0][0]
        body = sent.get_payload()[0].get_payload(decode=True).decode("utf-8")
        assert "Bu ay önemli" in body
        assert "Kriz rejimi tespit edildi" in body
        assert sent["Subject"].startswith("⚠️")

    @patch("src.email_notifier.smtplib.SMTP")
    def test_no_significance_defaults_full(self, mock_smtp):
        server = self._captured_msg(mock_smtp)
        n = self._notifier()
        n.send_monthly_report(self._RESULT)   # significance None -> tam rapor
        sent = server.send_message.call_args[0][0]
        body = sent.get_payload()[0].get_payload(decode=True).decode("utf-8")
        assert "Bu Ay Yapılması Gerekenler" in body
