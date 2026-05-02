import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_collector import TEFASCollector, POPULAR_BES_FUNDS


class TestTEFASCollector:
    def test_init(self, tmp_path):
        with patch.object(TEFASCollector, "_init_session"):
            collector = TEFASCollector(cache_dir=str(tmp_path))
        assert collector.cache_dir.exists()

    def test_normalize_date_yyyymmdd(self):
        c = TEFASCollector.__new__(TEFASCollector)
        c.cache_dir = None
        assert c._normalize_date("20250115") == "20250115"

    def test_normalize_date_iso(self):
        c = TEFASCollector.__new__(TEFASCollector)
        assert c._normalize_date("2025-01-15") == "20250115"

    def test_normalize_date_dotted(self):
        c = TEFASCollector.__new__(TEFASCollector)
        assert c._normalize_date("15.01.2025") == "20250115"

    def test_normalize_date_iso_method(self):
        c = TEFASCollector.__new__(TEFASCollector)
        assert c._normalize_date_iso("20250115") == "2025-01-15"
        assert c._normalize_date_iso("2025-01-15") == "2025-01-15"
        assert c._normalize_date_iso("15.01.2025") == "2025-01-15"

    def test_popular_funds_not_empty(self):
        assert len(POPULAR_BES_FUNDS) >= 10

    def test_popular_funds_has_required_categories(self):
        names = list(POPULAR_BES_FUNDS.values())
        assert any("Altin" in n or "Altn" in n for n in names)
        assert any("Kamu" in n for n in names)


class TestParseSnapshot:
    def setup_method(self):
        with patch.object(TEFASCollector, "_init_session"):
            self.collector = TEFASCollector()

    def test_standard_response(self):
        items = [
            {"fonKodu": "AEA", "fonUnvan": "Test Fon A", "fonTurAciklama": "Altin",
             "riskDegeri": "3", "getiri1a": "2.5", "getiri1y": "45.0"},
            {"fonKodu": "IPB", "fonUnvan": "Test Fon B", "fonTurAciklama": "Hisse",
             "riskDegeri": "5", "getiri1a": "3.1", "getiri1y": "60.0"},
        ]
        df = self.collector._parse_snapshot(items, "20250115")
        assert df is not None
        assert len(df) == 2
        assert set(df.columns) >= {"date", "fund_code", "fund_name", "return_1m", "return_1y"}
        assert df.loc[df["fund_code"] == "AEA", "return_1m"].iloc[0] == pytest.approx(2.5)

    def test_empty_list(self):
        df = self.collector._parse_snapshot([], "20250115")
        assert df is None

    def test_missing_fund_code_skipped(self):
        items = [
            {"fonKodu": "", "fonUnvan": "No Code"},
            {"fonKodu": "AEA", "fonUnvan": "Valid", "getiri1a": "1.0"},
        ]
        df = self.collector._parse_snapshot(items, "20250115")
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["fund_code"] == "AEA"

    def test_null_return_handled(self):
        items = [{"fonKodu": "AEA", "fonUnvan": "Test", "getiri1a": None, "getiri1y": None}]
        df = self.collector._parse_snapshot(items, "20250115")
        assert df is not None
        assert pd.isna(df.iloc[0]["return_1m"])

    def test_comma_decimal_price(self):
        items = [{"fonKodu": "AEA", "fonUnvan": "Test", "getiri1a": "2,5"}]
        df = self.collector._parse_snapshot(items, "20250115")
        assert df is not None
        assert df.iloc[0]["return_1m"] == pytest.approx(2.5)

    def test_date_parsed_correctly(self):
        items = [{"fonKodu": "AEA", "fonUnvan": "Test", "getiri1a": "1.0"}]
        df = self.collector._parse_snapshot(items, "20250115")
        assert isinstance(df.iloc[0]["date"], pd.Timestamp)
        assert df.iloc[0]["date"].year == 2025
        assert df.iloc[0]["date"].month == 1
        assert df.iloc[0]["date"].day == 15

    def test_duplicate_fund_code_deduplicated(self):
        items = [
            {"fonKodu": "AEA", "fonUnvan": "A", "getiri1a": "1.0"},
            {"fonKodu": "AEA", "fonUnvan": "A-dup", "getiri1a": "2.0"},
        ]
        df = self.collector._parse_snapshot(items, "20250115")
        assert df is not None
        assert len(df) == 1


class TestFetchFundSnapshot:
    def setup_method(self):
        with patch.object(TEFASCollector, "_init_session"):
            self.collector = TEFASCollector(rate_limit_sec=0)

    def test_success(self, tmp_path):
        self.collector.cache_dir = tmp_path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"fonKodu": "AEA", "fonUnvan": "Test", "getiri1a": "2.5", "getiri1y": "45.0"}
        ]
        mock_resp.raise_for_status.return_value = None

        with patch.object(self.collector.session, "post", return_value=mock_resp):
            df = self.collector.fetch_fund_snapshot("20250115", use_cache=False)

        assert df is not None
        assert len(df) == 1

    def test_empty_response_returns_none(self, tmp_path):
        self.collector.cache_dir = tmp_path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None

        with patch.object(self.collector.session, "post", return_value=mock_resp):
            df = self.collector.fetch_fund_snapshot("20250115", use_cache=False)

        assert df is None

    def test_http_error_returns_none(self, tmp_path):
        self.collector.cache_dir = tmp_path
        with patch.object(self.collector.session, "post",
                          side_effect=requests_exception()):
            df = self.collector.fetch_fund_snapshot("20250115", use_cache=False)
        assert df is None

    def test_cache_used_on_second_call(self, tmp_path):
        self.collector.cache_dir = tmp_path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"fonKodu": "AEA", "fonUnvan": "Test", "getiri1a": "2.5"}
        ]
        mock_resp.raise_for_status.return_value = None

        with patch.object(self.collector.session, "post", return_value=mock_resp) as mock_post:
            self.collector.fetch_fund_snapshot("20250115", use_cache=True)
            self.collector.fetch_fund_snapshot("20250115", use_cache=True)
            assert mock_post.call_count == 1


class TestCacheOperations:
    def setup_method(self):
        with patch.object(TEFASCollector, "_init_session"):
            self.collector = TEFASCollector()

    def test_write_and_read(self, tmp_path):
        self.collector.cache_dir = tmp_path
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=3),
            "fund_code": "AEA",
            "return_1m": [1.0, 1.5, 2.0],
        })
        self.collector._write_cache("test_key", df)
        cached = self.collector._read_cache("test_key")
        assert cached is not None
        assert len(cached) == 3

    def test_cache_miss_when_not_exists(self, tmp_path):
        self.collector.cache_dir = tmp_path
        result = self.collector._read_cache("nonexistent_key")
        assert result is None


class TestFetchMultipleFunds:
    def setup_method(self):
        with patch.object(TEFASCollector, "_init_session"):
            self.collector = TEFASCollector()

    def test_filters_by_fund_code(self, tmp_path):
        self.collector.cache_dir = tmp_path
        combined = pd.DataFrame({
            "date": [pd.Timestamp("2025-01-31")] * 3,
            "fund_code": ["AEA", "IPB", "GAE"],
            "fund_name": ["A", "B", "C"],
            "return_1m": [1.0, 2.0, 3.0],
        })
        with patch.object(self.collector, "fetch_monthly_series", return_value=combined):
            result = self.collector.fetch_multiple_funds(
                ["AEA", "IPB"], "2025-01-01", "2025-01-31"
            )
        assert len(result) == 2
        assert set(result["fund_code"]) == {"AEA", "IPB"}

    def test_returns_empty_on_no_data(self, tmp_path):
        self.collector.cache_dir = tmp_path
        with patch.object(self.collector, "fetch_monthly_series", return_value=pd.DataFrame()):
            result = self.collector.fetch_multiple_funds(
                ["AEA"], "2025-01-01", "2025-01-31"
            )
        assert result.empty


def requests_exception():
    import requests
    return requests.exceptions.ConnectionError("test error")


import requests  # noqa: E402 — needed for requests_exception helper
