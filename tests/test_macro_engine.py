import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.macro_engine import TCMBClient, MacroEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_policy_df():
    """evdspy'nin politika faizi icin dondurecegi DataFrame."""
    return pd.DataFrame({
        "Tarih":    ["01-03-2024", "08-03-2024", "15-03-2024", "22-03-2024", "29-03-2024"],
        "TP_AOFOD": ["45.0",       "45.0",       "50.0",       "50.0",       "50.0"],
    })


@pytest.fixture
def mock_usd_df():
    """evdspy'nin USD/TRY icin dondurecegi DataFrame."""
    return pd.DataFrame({
        "Tarih":           ["01-04-2026", "02-04-2026", "03-04-2026"],
        "TP_DK_USD_A_YTL": ["44.50",      "44.75",      "44.90"],
    })


@pytest.fixture
def cpi_13months():
    """13 aylik CPI indeks verisi — yoy hesaplamasini destekler."""
    # Ocak 2024'ten Ocak 2025'e: 1000 -> 1300 = %30 artis
    data = [{"date": f"2024-{i:02d}-01", "value": 1000.0 + i * 25} for i in range(1, 13)]
    data.append({"date": "2025-01-01", "value": 1300.0})
    return data  # data[-13]["value"]=1025, data[-1]["value"]=1300 → yoy≈26.8%


# ---------------------------------------------------------------------------
# TestTCMBClient
# ---------------------------------------------------------------------------

class TestTCMBClient:
    def test_init_without_key_warns(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TCMB_API_KEY", raising=False)
        with patch("src.macro_engine.load_dotenv"):
            client = TCMBClient(api_key=None, cache_dir=str(tmp_path))
        # evdspy kendi key yönetimini kullanıyor; api_key artık TCMBClient'ta saklanmıyor
        assert not hasattr(client, "api_key")

    def test_fetch_series_uses_cache(self, tmp_path):
        client = TCMBClient(api_key="dummy", cache_dir=str(tmp_path))
        client._write_cache("TP.AOFOD", [{"date": "2024-03-15", "value": 50.0}])

        with patch("evdspy.get_series") as mock_gs:
            data = client.fetch_series("TP.AOFOD")
            mock_gs.assert_not_called()

        assert len(data) == 1
        assert data[0]["value"] == 50.0

    def test_fetch_series_api_call(self, tmp_path, mock_policy_df):
        client = TCMBClient(api_key="testkey", cache_dir=str(tmp_path))

        with patch("evdspy.get_series", return_value=mock_policy_df):
            data = client.fetch_series("TP.AOFOD", use_cache=False)

        assert len(data) == 5
        assert data[0]["date"] == "2024-03-01"
        assert data[-1]["value"] == 50.0

    def test_api_failure_falls_back_to_stale_cache(self, tmp_path):
        client = TCMBClient(api_key="testkey", cache_dir=str(tmp_path))
        client._write_cache("TP.AOFOD", [{"date": "2024-01-01", "value": 40.0}])

        cache_path = client._cache_path("TP.AOFOD")
        old_data = json.loads(cache_path.read_text(encoding="utf-8"))
        old_data["fetched_at"] = "2020-01-01T00:00:00"
        cache_path.write_text(json.dumps(old_data), encoding="utf-8")

        with patch("evdspy.get_series", side_effect=Exception("Network down")):
            data = client.fetch_series("TP.AOFOD")

        assert len(data) == 1
        assert data[0]["value"] == 40.0

    def test_api_failure_no_cache_returns_empty(self, tmp_path):
        client = TCMBClient(api_key="testkey", cache_dir=str(tmp_path))

        with patch("evdspy.get_series", side_effect=Exception("Network down")):
            data = client.fetch_series("TP.AOFOD")

        assert data == []

    def test_evdspy_returns_bool_false_handled(self, tmp_path):
        """evdspy bazi serilerde bool False dondurur — bos liste olmali."""
        client = TCMBClient(api_key="testkey", cache_dir=str(tmp_path))

        with patch("evdspy.get_series", return_value=False):
            data = client.fetch_series("TP.AOFOD", use_cache=False)

        assert data == []


# ---------------------------------------------------------------------------
# TestDataframeToList
# ---------------------------------------------------------------------------

class TestDataframeToList:
    def test_dd_mm_yyyy_format(self, tmp_path):
        """Gunluk USD/TRY formati: DD-MM-YYYY."""
        client = TCMBClient(api_key="dummy", cache_dir=str(tmp_path))
        df = pd.DataFrame({
            "Tarih":           ["01-01-2026", "02-01-2026"],
            "TP_DK_USD_A_YTL": ["42.5",       "43.0"],
        })
        data = client._dataframe_to_list(df, "TP.DK.USD.A.YTL")
        assert len(data) == 2
        assert data[0]["date"] == "2026-01-01"
        assert data[0]["value"] == 42.5

    def test_yyyy_m_format(self, tmp_path):
        """Aylik CPI formati: YYYY-M."""
        client = TCMBClient(api_key="dummy", cache_dir=str(tmp_path))
        df = pd.DataFrame({
            "Tarih":   ["2025-1", "2025-10"],
            "TP_FG_J0": ["2819.65", "3453.09"],
        })
        data = client._dataframe_to_list(df, "TP.FG.J0")
        assert len(data) == 2
        assert data[0]["date"] == "2025-01-01"
        assert data[1]["date"] == "2025-10-01"

    def test_nan_values_filtered(self, tmp_path):
        """NaN degerler filtrelenmeli."""
        client = TCMBClient(api_key="dummy", cache_dir=str(tmp_path))
        df = pd.DataFrame({
            "Tarih":           ["01-04-2026", "02-04-2026"],
            "TP_DK_USD_A_YTL": ["44.90",      None],
        })
        data = client._dataframe_to_list(df, "TP.DK.USD.A.YTL")
        assert len(data) == 1
        assert data[0]["value"] == 44.90


# ---------------------------------------------------------------------------
# TestMacroEngine
# ---------------------------------------------------------------------------

class TestMacroEngine:
    def test_get_macro_snapshot_structure(self, tmp_path):
        """API down → tum alanlar var, rate_change=0."""
        client = TCMBClient(api_key="dummy", cache_dir=str(tmp_path))
        engine = MacroEngine(tcmb_client=client)

        with patch("evdspy.get_series", side_effect=Exception("Offline test")):
            snapshot = engine.get_macro_snapshot()

        assert "tcmb_rate_change" in snapshot
        assert "current_policy_rate" in snapshot
        assert "cpi_yoy" in snapshot
        assert "usdtry_official" in snapshot
        assert "bond_2y" in snapshot
        assert "data_quality" in snapshot
        assert snapshot["tcmb_rate_change"] == 0.0

    def test_rate_change_calculation(self):
        data = [
            {"date": "2024-01-01", "value": 45.0},
            {"date": "2024-02-01", "value": 45.0},
            {"date": "2024-03-01", "value": 50.0},
        ]
        change = MacroEngine._calculate_rate_change(data, days=30)
        assert change == pytest.approx(5.0)

    def test_rate_change_no_change(self):
        data = [
            {"date": "2024-01-01", "value": 50.0},
            {"date": "2024-02-01", "value": 50.0},
            {"date": "2024-03-01", "value": 50.0},
        ]
        change = MacroEngine._calculate_rate_change(data, days=30)
        assert change == pytest.approx(0.0)

    def test_cpi_yoy_calculation(self):
        """13 veri noktasiyla yoy dogru hesaplanmali."""
        # data[-13]=1000, data[-1]=1300 → yoy=0.30
        data = [{"date": f"2024-{i:02d}-01", "value": 1000.0 + i * 25} for i in range(1, 13)]
        data.append({"date": "2025-01-01", "value": 1300.0})
        yoy = MacroEngine._calculate_cpi_yoy(data)
        # data[-13] = 1000+1*25=1025, data[-1]=1300 → yoy=(1300/1025)-1≈0.268
        assert yoy == pytest.approx((1300 / 1025) - 1, rel=0.01)

    def test_cpi_yoy_insufficient_data(self):
        """12 veya daha az veri noktasinda None donmeli."""
        data = [{"date": f"2024-{i:02d}-01", "value": 100.0 * i} for i in range(1, 12)]
        assert MacroEngine._calculate_cpi_yoy(data) is None

    def test_full_snapshot_with_real_data(self, tmp_path, cpi_13months):
        client = TCMBClient(api_key="dummy", cache_dir=str(tmp_path))

        # Politika faizi: 5pp artis (45→50)
        client._write_cache(TCMBClient.SERIES["policy_rate"], [
            {"date": "2024-01-01", "value": 45.0},
            {"date": "2024-03-15", "value": 50.0},
        ])
        # CPI: 13 aylik indeks → yoy hesaplanacak
        client._write_cache(TCMBClient.SERIES["cpi_yoy"], cpi_13months)
        # USD/TRY
        client._write_cache(TCMBClient.SERIES["usdtry_official"], [
            {"date": "2024-03-15", "value": 32.5},
        ])
        # 2Y tahvil
        client._write_cache(TCMBClient.SERIES["bond_2y"], [
            {"date": "2024-03-15", "value": 47.0},
        ])

        engine = MacroEngine(tcmb_client=client)
        snapshot = engine.get_macro_snapshot()

        assert snapshot["current_policy_rate"] == 50.0
        assert snapshot["usdtry_official"] == 32.5
        assert snapshot["bond_2y"] == 47.0
        assert snapshot["tcmb_rate_change"] == pytest.approx(0.05)
        # CPI yoy: data[-13]=1025, data[-1]=1300
        expected_yoy = (1300 / 1025) - 1
        assert snapshot["cpi_yoy"] == pytest.approx(expected_yoy, rel=0.01)
