import os
import time
from datetime import datetime

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
        # Guncel tarih: gecmis tarihler artik ag fetch'i yapmaz (PLAN-01)
        today = datetime.now().strftime("%Y%m%d")
        self.collector.cache_dir = tmp_path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"fonKodu": "AEA", "fonUnvan": "Test", "getiri1a": "2.5", "getiri1y": "45.0"}
        ]
        mock_resp.raise_for_status.return_value = None

        with patch.object(self.collector.session, "post", return_value=mock_resp):
            df = self.collector.fetch_fund_snapshot(today, use_cache=False)

        assert df is not None
        assert len(df) == 1

    def test_empty_response_returns_none(self, tmp_path):
        # Guncel tarih: gecmis tarihler artik ag fetch'i yapmaz (PLAN-01)
        today = datetime.now().strftime("%Y%m%d")
        self.collector.cache_dir = tmp_path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None

        with patch.object(self.collector.session, "post", return_value=mock_resp):
            df = self.collector.fetch_fund_snapshot(today, use_cache=False)

        assert df is None

    def test_http_error_returns_none(self, tmp_path):
        # Guncel tarih: gecmis tarihler artik ag fetch'i yapmaz (PLAN-01)
        today = datetime.now().strftime("%Y%m%d")
        self.collector.cache_dir = tmp_path
        with patch.object(self.collector.session, "post",
                          side_effect=requests_exception()):
            df = self.collector.fetch_fund_snapshot(today, use_cache=False)
        assert df is None

    def test_cache_used_on_second_call(self, tmp_path):
        # Guncel tarih: gecmis tarihler artik ag fetch'i yapmaz (PLAN-01)
        today = datetime.now().strftime("%Y%m%d")
        self.collector.cache_dir = tmp_path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"fonKodu": "AEA", "fonUnvan": "Test", "getiri1a": "2.5"}
        ]
        mock_resp.raise_for_status.return_value = None

        with patch.object(self.collector.session, "post", return_value=mock_resp) as mock_post:
            self.collector.fetch_fund_snapshot(today, use_cache=True)
            self.collector.fetch_fund_snapshot(today, use_cache=True)
            assert mock_post.call_count == 1


class TestHistoricalSnapshotProtection:
    """PLAN-01: gecmis tarihli snapshot'lar guncel veriyle ezilmemeli."""

    def setup_method(self):
        with patch.object(TEFASCollector, "_init_session"):
            self.collector = TEFASCollector(rate_limit_sec=0)

    def _write_snapshot(self, tmp_path, filename, date_iso):
        df = pd.DataFrame({
            "date": [pd.Timestamp(date_iso)],
            "fund_code": ["AEA"],
            "fund_name": ["Test Fon"],
            "category": ["Altin"],
            "return_1m": [2.5],
        })
        path = tmp_path / filename
        df.to_parquet(path, index=False)
        return path

    def test_historical_cache_never_expires(self, tmp_path):
        # 10 gunluk mtime normal TTL'yi (24h) coktan asmis olsa bile
        # gecmis tarihli cache okunmali ve aga cikilmamali.
        self.collector.cache_dir = tmp_path
        path = self._write_snapshot(tmp_path, "snapshot_EMK_20240115.parquet", "2024-01-15")
        old = time.time() - 10 * 24 * 3600
        os.utime(path, (old, old))

        with patch.object(self.collector.session, "post") as mock_post:
            df = self.collector.fetch_fund_snapshot("2024-01-15")

        assert df is not None
        assert not df.empty
        assert df.iloc[0]["fund_code"] == "AEA"
        mock_post.assert_not_called()

    def test_historical_miss_refuses_network(self, tmp_path):
        # Cache'te olmayan gecmis tarih: None donmeli, aga cikilmamali.
        self.collector.cache_dir = tmp_path
        with patch.object(self.collector.session, "post") as mock_post:
            df = self.collector.fetch_fund_snapshot("2024-03-29")
        assert df is None
        mock_post.assert_not_called()

    def test_historical_refuses_network_even_without_cache(self, tmp_path):
        # use_cache=False bile olsa gecmis tarih icin ag fetch'i yapilmamali
        # (aksi halde _write_cache eski dosyayi guncel veriyle ezerdi).
        self.collector.cache_dir = tmp_path
        self._write_snapshot(tmp_path, "snapshot_EMK_20240115.parquet", "2024-01-15")
        with patch.object(self.collector.session, "post") as mock_post:
            df = self.collector.fetch_fund_snapshot("2024-01-15", use_cache=False)
        assert df is None
        mock_post.assert_not_called()
        # Dosya icerigi degismemis olmali
        untouched = pd.read_parquet(tmp_path / "snapshot_EMK_20240115.parquet")
        assert untouched.iloc[0]["fund_code"] == "AEA"

    def test_recent_date_still_fetches(self, tmp_path):
        # Guncel tarih + cache yok: ag fetch'i calismali ve df donmeli.
        self.collector.cache_dir = tmp_path
        today = datetime.now().strftime("%Y-%m-%d")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"fonKodu": "AEA", "fonUnvan": "Test", "getiri1a": "2.5"}
        ]
        mock_resp.raise_for_status.return_value = None

        with patch.object(self.collector.session, "post", return_value=mock_resp) as mock_post:
            df = self.collector.fetch_fund_snapshot(today, use_cache=True)

        assert df is not None
        assert len(df) == 1
        mock_post.assert_called_once()
        # Guncel tarih icin cache yazimi da korunmus olmali
        cache_file = tmp_path / f"snapshot_EMK_{today.replace('-', '')}.parquet"
        assert cache_file.exists()


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


class TestDiscoverAllBesFunds:
    def setup_method(self):
        with patch.object(TEFASCollector, "_init_session"):
            self.collector = TEFASCollector(rate_limit_sec=0)

    def test_returns_snapshot_dataframe(self):
        mock_df = pd.DataFrame({
            "fund_code": ["AEA", "IPB", "GAE"],
            "fund_name": ["Fon A", "Fon B", "Fon C"],
            "return_1m": [2.5, 3.1, 1.8],
        })
        with patch.object(self.collector, "fetch_fund_snapshot", return_value=mock_df):
            result = self.collector.discover_all_bes_funds()
        assert not result.empty
        assert len(result) == 3
        assert "fund_code" in result.columns

    def test_falls_back_to_yesterday_on_empty(self):
        empty_df = pd.DataFrame()
        mock_df = pd.DataFrame({"fund_code": ["AEA"], "fund_name": ["Fon A"], "return_1m": [2.5]})
        call_count = [0]

        def side_effect(date, **kwargs):
            call_count[0] += 1
            return mock_df if call_count[0] > 1 else empty_df

        with patch.object(self.collector, "fetch_fund_snapshot", side_effect=side_effect):
            result = self.collector.discover_all_bes_funds()
        assert not result.empty
        assert call_count[0] == 2

    def test_returns_empty_when_all_fail(self):
        with patch.object(self.collector, "fetch_fund_snapshot", return_value=pd.DataFrame()):
            result = self.collector.discover_all_bes_funds()
        assert result.empty


def requests_exception():
    import requests
    return requests.exceptions.ConnectionError("test error")


import requests  # noqa: E402 — needed for requests_exception helper


class TestGetFundReturns:
    def _make_snapshot(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "date": ["2026-05-15", "2026-05-15"],
            "fund_code": ["AHS", "BGL"],
            "fund_name": ["A", "B"],
            "category": ["Stock Fund", "Gold Fund"],
            "return_1m": [3.5, -1.2],
            "return_3m": [10.0, 2.0],
        })
        df.to_parquet(tmp_path / "snapshot_EMK_20260515.parquet")

    def test_returns_as_fraction(self, tmp_path):
        with patch.object(TEFASCollector, "_init_session"):
            c = TEFASCollector(cache_dir=str(tmp_path))
        self._make_snapshot(tmp_path)
        out = c.get_fund_returns(period="return_1m")
        assert abs(out["AHS"] - 0.035) < 1e-9
        assert abs(out["BGL"] - (-0.012)) < 1e-9

    def test_filter_by_codes(self, tmp_path):
        with patch.object(TEFASCollector, "_init_session"):
            c = TEFASCollector(cache_dir=str(tmp_path))
        self._make_snapshot(tmp_path)
        out = c.get_fund_returns(codes=["ahs"], period="return_1m")
        assert set(out.keys()) == {"AHS"}

    def test_no_snapshot_returns_empty(self, tmp_path):
        with patch.object(TEFASCollector, "_init_session"):
            c = TEFASCollector(cache_dir=str(tmp_path))
        assert c.get_fund_returns() == {}


class TestFetchNavHistory:
    def test_parses_and_merges_windows(self, tmp_path):
        from unittest.mock import MagicMock, patch
        with patch.object(TEFASCollector, "_init_session"):
            c = TEFASCollector(cache_dir=str(tmp_path))

        def fake_post(url, json=None, timeout=None):
            bas = json["basTarih"]
            resp = MagicMock(); resp.status_code = 200; resp.raise_for_status = lambda: None
            # pencereye göre farklı gün döndür (örtüşme dedup edilmeli)
            d = "2025-01-02" if bas <= "20250102" else "2025-01-30"
            resp.json = lambda: {"resultList": [
                {"fonKodu":"AAA","fonUnvan":"A","tarih":d,"fiyat":1.5,
                 "tedPaySayisi":1,"kisiSayisi":2,"portfoyBuyukluk":3},
            ]}
            return resp
        sess = MagicMock(); sess.post.side_effect = fake_post; sess.get.return_value = MagicMock()
        with patch("src.data_collector.requests.Session", return_value=sess):
            df = c.fetch_nav_history("2025-01-01", "2025-02-15", sleep_sec=0)

        assert not df.empty
        assert set(df.columns) >= {"fund_code","fund_name","date","price"}
        assert df["price"].iloc[0] == 1.5
        # iki pencere -> iki farklı tarih, dedup sonrası 2 satır
        assert df["date"].nunique() == 2

    def test_empty_resultList_returns_empty_df(self, tmp_path):
        from unittest.mock import MagicMock, patch
        with patch.object(TEFASCollector, "_init_session"):
            c = TEFASCollector(cache_dir=str(tmp_path))
        resp = MagicMock(); resp.status_code = 200; resp.raise_for_status = lambda: None
        resp.json = lambda: {"resultList": []}
        sess = MagicMock(); sess.post.return_value = resp; sess.get.return_value = MagicMock()
        with patch("src.data_collector.requests.Session", return_value=sess):
            df = c.fetch_nav_history("2025-01-01", "2025-01-20", sleep_sec=0)
        assert df.empty


class TestUpdateNavHistory:
    def test_incremental_merge(self, tmp_path):
        from unittest.mock import patch
        import pandas as pd
        with patch.object(TEFASCollector, "_init_session"):
            c = TEFASCollector(cache_dir=str(tmp_path))
        navp = tmp_path / "nav_history.parquet"
        pd.DataFrame({"fund_code":["AAA","AAA"],"fund_name":["A","A"],
                      "date":pd.to_datetime(["2025-01-01","2025-01-02"]),
                      "price":[1.0,1.1]}).to_parquet(navp)
        # Yeni çekim: bir örtüşen + bir yeni gün
        new = pd.DataFrame({"fund_code":["AAA","AAA"],"fund_name":["A","A"],
                            "date":pd.to_datetime(["2025-01-02","2025-01-03"]),
                            "price":[1.1,1.2]})
        with patch.object(c, "fetch_nav_history", return_value=new):
            added = c.update_nav_history(path=str(navp))
        merged = pd.read_parquet(navp)
        assert added == 1                       # yalnız 2025-01-03 yeni
        assert len(merged) == 3
        assert merged["date"].nunique() == 3

    def test_no_new_data_returns_zero(self, tmp_path):
        from unittest.mock import patch
        import pandas as pd
        with patch.object(TEFASCollector, "_init_session"):
            c = TEFASCollector(cache_dir=str(tmp_path))
        navp = tmp_path / "nav_history.parquet"
        pd.DataFrame({"fund_code":["AAA"],"fund_name":["A"],
                      "date":pd.to_datetime(["2025-01-01"]),"price":[1.0]}).to_parquet(navp)
        with patch.object(c, "fetch_nav_history", return_value=pd.DataFrame()):
            added = c.update_nav_history(path=str(navp))
        assert added == 0
