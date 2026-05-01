import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_collector import TEFASCollector, POPULAR_BES_FUNDS


class TestTEFASCollector:
    def test_init(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path))
        assert collector.cache_dir.exists()

    def test_parse_fund_data_standard_format(self):
        collector = TEFASCollector()
        items = [
            {"TARIH": "15.01.2025", "FIYAT": "1.234567"},
            {"TARIH": "16.01.2025", "FIYAT": "1.245678"},
            {"TARIH": "17.01.2025", "FIYAT": "1.256789"},
        ]
        df = collector._parse_fund_data(items, "TEST")
        assert df is not None
        assert len(df) == 3
        assert "date" in df.columns
        assert "nav" in df.columns
        assert df["fund_code"].iloc[0] == "TEST"

    def test_parse_fund_data_iso_format(self):
        collector = TEFASCollector()
        items = [
            {"TARIH": "2025-01-15T00:00:00", "FIYAT": 1.234567},
        ]
        df = collector._parse_fund_data(items, "TEST")
        assert df is not None
        assert len(df) == 1

    def test_parse_fund_data_alternative_keys(self):
        collector = TEFASCollector()
        items = [
            {"Tarih": "2025-01-15T00:00:00", "ToplamDeger": "1.234567"},
        ]
        df = collector._parse_fund_data(items, "TEST")
        assert df is not None
        assert len(df) == 1

    def test_parse_fund_data_empty(self):
        collector = TEFASCollector()
        df = collector._parse_fund_data([], "TEST")
        assert df is None

    def test_parse_fund_data_invalid_price(self):
        collector = TEFASCollector()
        items = [
            {"TARIH": "15.01.2025", "FIYAT": "0"},
            {"TARIH": "16.01.2025", "FIYAT": "-1"},
        ]
        df = collector._parse_fund_data(items, "TEST")
        assert df is None

    def test_parse_fund_data_sorted_by_date(self):
        collector = TEFASCollector()
        items = [
            {"TARIH": "17.01.2025", "FIYAT": "1.3"},
            {"TARIH": "15.01.2025", "FIYAT": "1.1"},
            {"TARIH": "16.01.2025", "FIYAT": "1.2"},
        ]
        df = collector._parse_fund_data(items, "TEST")
        assert df is not None
        dates = df["date"].tolist()
        assert dates == sorted(dates)

    def test_parse_fund_data_deduplicates(self):
        collector = TEFASCollector()
        items = [
            {"TARIH": "15.01.2025", "FIYAT": "1.1"},
            {"TARIH": "15.01.2025", "FIYAT": "1.2"},  # duplicate
        ]
        df = collector._parse_fund_data(items, "TEST")
        assert df is not None
        assert len(df) == 1

    def test_cache_write_and_read(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path))
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=5),
            "nav": [1.0, 1.1, 1.2, 1.3, 1.4],
            "fund_code": "TEST",
        })
        collector._write_cache("test_key", df)
        cached = collector._read_cache("test_key")
        assert cached is not None
        assert len(cached) == 5

    def test_cache_miss_when_not_exists(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path))
        result = collector._read_cache("nonexistent_key")
        assert result is None

    def test_popular_funds_not_empty(self):
        assert len(POPULAR_BES_FUNDS) >= 10

    def test_popular_funds_has_required_categories(self):
        codes = list(POPULAR_BES_FUNDS.keys())
        names = list(POPULAR_BES_FUNDS.values())
        assert any("Hisse" in n for n in names)
        assert any("Borç" in n or "Kamu" in n for n in names)
        assert any("Altın" in n for n in names)


class TestFetchFundHistory:
    def test_returns_none_on_http_error(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path), rate_limit_sec=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")

        with patch.object(collector.session, "post", return_value=mock_resp):
            result = collector.fetch_fund_history("AEA", "01.01.2025", "30.04.2025", use_cache=False)
        assert result is None

    def test_returns_none_on_empty_response(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path), rate_limit_sec=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status.return_value = None

        with patch.object(collector.session, "post", return_value=mock_resp):
            result = collector.fetch_fund_history("AEA", "01.01.2025", "30.04.2025", use_cache=False)
        assert result is None

    def test_returns_dataframe_on_success(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path), rate_limit_sec=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {
            "data": [
                {"TARIH": "15.01.2025", "FIYAT": "1.234"},
                {"TARIH": "16.01.2025", "FIYAT": "1.245"},
            ]
        }
        mock_resp.raise_for_status.return_value = None

        with patch.object(collector.session, "post", return_value=mock_resp):
            result = collector.fetch_fund_history("AEA", "01.01.2025", "30.04.2025", use_cache=False)

        assert result is not None
        assert len(result) == 2
        assert "nav" in result.columns

    def test_uses_cache_on_second_call(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path), rate_limit_sec=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = {
            "data": [{"TARIH": "15.01.2025", "FIYAT": "1.234"}]
        }
        mock_resp.raise_for_status.return_value = None

        with patch.object(collector.session, "post", return_value=mock_resp) as mock_post:
            collector.fetch_fund_history("AEA", "01.01.2025", "30.04.2025", use_cache=True)
            collector.fetch_fund_history("AEA", "01.01.2025", "30.04.2025", use_cache=True)
            assert mock_post.call_count == 1  # ikinci çağrı cache'ten geldi


class TestFetchMultipleFunds:
    def test_combines_results(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path))
        df1 = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=3),
            "nav": [1.0, 1.1, 1.2],
            "fund_code": "FUND1",
        })
        df2 = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=3),
            "nav": [2.0, 2.1, 2.2],
            "fund_code": "FUND2",
        })

        with patch.object(collector, "fetch_fund_history", side_effect=[df1, df2]):
            result = collector.fetch_multiple_funds(
                ["FUND1", "FUND2"], "01.01.2025", "31.03.2025"
            )

        assert len(result) == 6
        assert result["fund_code"].nunique() == 2

    def test_returns_empty_on_all_failures(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path))
        with patch.object(collector, "fetch_fund_history", return_value=None):
            result = collector.fetch_multiple_funds(
                ["FUND1", "FUND2"], "01.01.2025", "31.03.2025"
            )
        assert result.empty

    def test_partial_failure_still_returns_data(self, tmp_path):
        collector = TEFASCollector(cache_dir=str(tmp_path))
        df1 = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=3),
            "nav": [1.0, 1.1, 1.2],
            "fund_code": "FUND1",
        })
        with patch.object(collector, "fetch_fund_history", side_effect=[df1, None]):
            result = collector.fetch_multiple_funds(
                ["FUND1", "FUND2"], "01.01.2025", "31.03.2025"
            )
        assert len(result) == 3
        assert result["fund_code"].unique()[0] == "FUND1"
