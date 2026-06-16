import json

import pytest

from src import notification_prefs as np_


class TestIsValidEmail:
    @pytest.mark.parametrize("addr", ["a@b.com", "x.y@z.co.uk", " ok@d.com "])
    def test_valid(self, addr):
        assert np_.is_valid_email(addr) is True

    @pytest.mark.parametrize("addr", ["", "noat", "a@b", "a b@c.com", "@x.com", None])
    def test_invalid(self, addr):
        assert np_.is_valid_email(addr) is False


class TestLoadPrefs:
    def test_missing_returns_defaults(self, tmp_path):
        prefs = np_.load_notification_prefs(tmp_path / "none.json")
        assert prefs["email_enabled"] is False
        assert prefs["critical_signal"] is True

    def test_non_dict_returns_defaults(self, tmp_path):
        p = tmp_path / "p.json"
        p.write_text("[1,2,3]")
        assert np_.load_notification_prefs(p)["email_enabled"] is False

    def test_corrupt_returns_defaults(self, tmp_path):
        p = tmp_path / "p.json"
        p.write_text("{bad json")
        assert np_.load_notification_prefs(p)["email_address"] == ""

    def test_valid_merge(self, tmp_path):
        p = tmp_path / "p.json"
        p.write_text(json.dumps({"email_enabled": True, "email_address": "a@b.com"}))
        prefs = np_.load_notification_prefs(p)
        assert prefs["email_enabled"] is True
        assert prefs["email_address"] == "a@b.com"
        assert prefs["critical_signal"] is True  # default korunur

    def test_type_safe_merge(self, tmp_path):
        # email_enabled string "true" -> bool'a cevrilir, cokmemeli
        p = tmp_path / "p.json"
        p.write_text(json.dumps({"email_enabled": "yes"}))
        prefs = np_.load_notification_prefs(p)
        assert isinstance(prefs["email_enabled"], bool)


class TestSavePrefs:
    def test_save_and_reload(self, tmp_path):
        p = tmp_path / "p.json"
        prefs = {
            "email_enabled": True,
            "email_address": "x@y.com",
            "on_regime_change": False,
            "weekly_summary": True,
            "critical_signal": False,
        }
        assert np_.save_notification_prefs(prefs, p) is True
        reloaded = np_.load_notification_prefs(p)
        assert reloaded == prefs

    def test_save_returns_false_on_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            np_, "atomic_write_text",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )
        assert np_.save_notification_prefs({"x": 1}, tmp_path / "p.json") is False
