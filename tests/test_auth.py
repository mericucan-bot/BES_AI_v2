import json

import pytest

from src import auth


# --- get_app_password ---

class TestGetAppPassword:
    def test_from_secrets(self):
        assert auth.get_app_password({"APP_PASSWORD": "s3cret"}) == "s3cret"

    def test_secrets_missing_key_falls_to_env(self, monkeypatch):
        monkeypatch.setenv("APP_PASSWORD", "envpw")
        assert auth.get_app_password({"OTHER": "x"}) == "envpw"

    def test_from_env_when_no_secrets(self, monkeypatch):
        monkeypatch.setenv("APP_PASSWORD", "envpw")
        assert auth.get_app_password(None) == "envpw"

    def test_empty_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("APP_PASSWORD", raising=False)
        assert auth.get_app_password(None) == ""

    def test_bad_secrets_object_does_not_raise(self, monkeypatch):
        monkeypatch.delenv("APP_PASSWORD", raising=False)

        class Boom:
            def __contains__(self, k):
                raise RuntimeError("no secrets file")

        assert auth.get_app_password(Boom()) == ""


# --- throttle ---

class TestThrottle:
    def test_load_missing_returns_default(self, tmp_path):
        st = auth.load_auth_throttle(tmp_path / "none.json")
        assert st == {"failed_attempts": [], "lockout_until": 0.0}

    def test_load_corrupt_returns_default(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text("not json{{{")
        assert auth.load_auth_throttle(p)["lockout_until"] == 0.0

    def test_save_load_roundtrip(self, tmp_path):
        p = tmp_path / "t.json"
        auth.save_auth_throttle({"failed_attempts": [1.0, 2.0], "lockout_until": 5.0}, p)
        st = auth.load_auth_throttle(p)
        assert st["failed_attempts"] == [1.0, 2.0]
        assert st["lockout_until"] == 5.0

    def test_record_increments_and_returns_remaining(self, tmp_path, monkeypatch):
        monkeypatch.setattr(auth.time, "time", lambda: 1000.0)
        p = tmp_path / "t.json"
        remaining, lockout = auth.record_failed_attempt(p)
        assert remaining == auth.AUTH_MAX_FAILED - 1
        assert lockout == 0.0

    def test_lockout_after_max_failed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(auth.time, "time", lambda: 1000.0)
        p = tmp_path / "t.json"
        last = (0, 0.0)
        for _ in range(auth.AUTH_MAX_FAILED):
            last = auth.record_failed_attempt(p)
        remaining, lockout = last
        assert lockout == 1000.0 + auth.AUTH_LOCKOUT_SEC
        # Lockout sonrasi sayac sifirlanir
        assert auth.load_auth_throttle(p)["failed_attempts"] == []

    def test_old_attempts_pruned_outside_window(self, tmp_path, monkeypatch):
        p = tmp_path / "t.json"
        # Pencere disinda eski deneme
        auth.save_auth_throttle(
            {"failed_attempts": [100.0], "lockout_until": 0.0}, p
        )
        monkeypatch.setattr(
            auth.time, "time", lambda: 100.0 + auth.AUTH_WINDOW_SEC + 10
        )
        remaining, _ = auth.record_failed_attempt(p)
        # Eski deneme atildi, yalniz yeni 1 deneme sayilir
        assert remaining == auth.AUTH_MAX_FAILED - 1

    def test_reset_clears(self, tmp_path):
        p = tmp_path / "t.json"
        auth.save_auth_throttle({"failed_attempts": [1.0], "lockout_until": 9.0}, p)
        auth.reset_auth_throttle(p)
        st = auth.load_auth_throttle(p)
        assert st == {"failed_attempts": [], "lockout_until": 0.0}
