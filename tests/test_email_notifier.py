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
