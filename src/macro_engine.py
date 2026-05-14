import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class TCMBClient:
    """
    TCMB EVDS API istemcisi — evdspy kutuphanesi uzerinden.

    Seri kodu kurali:
    - API isteginde NOKTA: TP.TRY.MT01
    - API response kolonu ALT CIZGI: TP_TRY_MT01
    - Cache dosya adinda ALT CIZGI: macro_TP_TRY_MT01.json

    Cekilen seriler:
    - TP.TRY.MT01:     Politika faizi (TCMB gecelik faiz, aylik) — aktif seri
    - TP.FG.J0:        TUFE indeksi (aylik, yoy hesaplanir)
    - TP.DK.USD.A.YTL: USD/TRY alis kuru (gunluk)
    - TP.AKM.B070:     2Y devlet tahvili faizi (gunluk) — erisilebilirse
    """

    SERIES = {
        "policy_rate":     "TP.TRY.MT01",
        "cpi_yoy":         "TP.FG.J0",
        "usdtry_official": "TP.DK.USD.A.YTL",
        "bond_2y":         "TP.AKM.B070",
    }

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/cache"):
        # evdspy kendi key yönetimini kullanıyor (APIKEY_FOLDER/api_key.txt)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, series_code: str) -> Path:
        safe_name = series_code.replace(".", "_")
        return self.cache_dir / f"macro_{safe_name}.json"

    def _read_cache(self, series_code: str, max_age_hours: int = 24) -> Optional[List[Dict]]:
        path = self._cache_path(series_code)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            age = datetime.now() - datetime.fromisoformat(cached["fetched_at"])
            if age > timedelta(hours=max_age_hours):
                logger.debug(f"Cache eski ({series_code}): {age}")
                return None
            logger.debug(f"Cache HIT: {series_code} (age={age})")
            return cached["data"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache okuma hatasi ({series_code}): {e}")
            return None

    def _write_cache(self, series_code: str, data: List[Dict]) -> None:
        path = self._cache_path(series_code)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "fetched_at": datetime.now().isoformat(),
                    "series_code": series_code,
                    "data": data,
                }, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Cache yazma hatasi ({series_code}): {e}")

    def _read_stale_cache(self, series_code: str) -> Optional[List[Dict]]:
        """API down oldugunda son care: yasli cache'i bile kullan."""
        path = self._cache_path(series_code)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            age = datetime.now() - datetime.fromisoformat(cached["fetched_at"])
            logger.warning(f"Stale cache kullaniliyor ({series_code}, age={age})")
            return cached["data"]
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def fetch_series(
        self,
        series_code: str,
        lookback_days: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[Dict]:
        """
        evdspy ile TCMB serisini cek.
        Donus formati: [{"date": "2024-01-15", "value": 50.0}, ...]
        """
        # Streamlit Cloud: env var'dan key yükle, dosya yoksa yaz
        # Lokalde dosya zaten varsa dokunma (evdspy kendi formatını kullanıyor)
        # evdspy base64-kodlu key bekliyor — raw key'i encode ederek yaz
        env_key = os.environ.get("TCMB_API_KEY") or os.environ.get("EVDS_API_KEY")
        if env_key:
            # evdspy EVDS_API_KEY env var'ını okur — ham key olarak set et
            os.environ["EVDS_API_KEY"] = env_key
            try:
                import base64 as _b64
                apikey_dir = Path("APIKEY_FOLDER")
                key_file = apikey_dir / "api_key.txt"
                # Her zaman yaz: evdspy WriteBytes gibi binary mod + base64-encoded bytes
                # (var olan bozuk dosyayı da düzeltir)
                apikey_dir.mkdir(exist_ok=True)
                key_file.write_bytes(_b64.b64encode(env_key.encode()))
                logger.debug("TCMB API key environment'tan yuklendi (binary format)")
            except Exception:
                pass

        if use_cache:
            cached = self._read_cache(series_code)
            if cached is not None:
                return cached

        lookback = lookback_days or 400  # aylik seri icin 13+ ay gerek
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=lookback)
        start_str  = start_date.strftime("%d-%m-%Y")
        end_str    = end_date.strftime("%d-%m-%Y")

        try:
            from evdspy import get_series
            from evdspy.EVDSlocal.initial_setup.api_key_save import check_api_key_on_load
            check_api_key_on_load()

            df = get_series(
                series_code,
                start_date=start_str,
                end_date=end_str,
                cache=False,
            )

            if df is None or isinstance(df, bool) or (isinstance(df, pd.DataFrame) and df.empty):
                logger.warning(f"evdspy bos/False dondu: {series_code}")
                stale = self._read_stale_cache(series_code)
                return stale if stale else []

            data = self._dataframe_to_list(df, series_code)

            if data:
                self._write_cache(series_code, data)
                logger.info(f"TCMB cekildi (evdspy): {series_code} ({len(data)} satir)")

            return data

        except ImportError:
            logger.error("evdspy yuklu degil: pip install evdspy")
        except Exception as e:
            logger.error(f"TCMB hatasi ({series_code}): {e}")

        stale = self._read_stale_cache(series_code)
        return stale if stale else []

    def _dataframe_to_list(self, df: pd.DataFrame, series_code: str) -> List[Dict]:
        """
        evdspy DataFrame ciktisini [{"date": ISO, "value": float}] listesine cevir.

        Desteklenen Tarih formatlari:
        - "DD-MM-YYYY" (gunluk seriler: USD/TRY)
        - "YYYY-M" / "YYYY-MM" (aylik seriler: CPI)
        """
        data = []
        try:
            if "Tarih" not in df.columns:
                logger.warning(f"'Tarih' kolonu yok ({series_code}): {df.columns.tolist()}")
                return []

            # Response kolonu: TP.AOFOD → TP_AOFOD
            response_key = series_code.replace(".", "_")
            val_col = response_key if response_key in df.columns else None
            if val_col is None:
                # Fallback: Tarih disindaki ilk kolonu al
                for col in df.columns:
                    if col != "Tarih":
                        val_col = col
                        break
            if val_col is None:
                logger.warning(f"Value kolonu bulunamadi ({series_code}): {df.columns.tolist()}")
                return []

            for _, row in df.iterrows():
                date_raw = str(row["Tarih"]).strip()
                value_raw = row[val_col]

                if pd.isna(value_raw):
                    continue

                try:
                    value = float(value_raw)
                except (ValueError, TypeError):
                    continue

                # Tarih parse: DD-MM-YYYY veya YYYY-M / YYYY-MM
                try:
                    parts = date_raw.split("-")
                    if len(parts) == 3:
                        if len(parts[0]) == 4:
                            # YYYY-MM-DD formati
                            iso_date = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                        else:
                            # DD-MM-YYYY formati
                            d, m, y = parts
                            iso_date = f"{y}-{int(m):02d}-{int(d):02d}"
                    elif len(parts) == 2:
                        # YYYY-M formati (aylik)
                        y, m = parts
                        iso_date = f"{y}-{int(m):02d}-01"
                    else:
                        continue
                except (ValueError, TypeError):
                    continue

                data.append({"date": iso_date, "value": value})

            data.sort(key=lambda x: x["date"])

        except Exception as e:
            logger.error(f"DataFrame donusturme hatasi ({series_code}): {e}")

        return data


class MacroEngine:
    """Ust seviye macro veri saglayici. RegimeEngine icin macro indikatörler uretir."""

    def __init__(self, tcmb_client: Optional[TCMBClient] = None):
        self.client = tcmb_client or TCMBClient()

    def get_macro_snapshot(self) -> Dict:
        """
        RegimeEngine'in compute_composite_score'a verebilecegi macro_data dict'i uret.

        Donus yapisi:
            tcmb_rate_change:    Son 30 gundeki politika faizi degisimi (oran, 0.05=+5pp)
            current_policy_rate: Su anki politika faizi (%)
            cpi_yoy:             12 aylik CPI degisimi (oran, 0.30=%30)
            usdtry_official:     TCMB resmi USD/TRY
            bond_2y:             2Y tahvil faizi (%)
            data_quality:        Meta bilgi
        """
        policy_data = self.client.fetch_series(TCMBClient.SERIES["policy_rate"])
        cpi_data    = self.client.fetch_series(TCMBClient.SERIES["cpi_yoy"], lookback_days=700)
        usd_data    = self.client.fetch_series(TCMBClient.SERIES["usdtry_official"])
        bond_data   = self.client.fetch_series(TCMBClient.SERIES["bond_2y"])

        rate_change   = self._calculate_rate_change(policy_data, days=30)
        current_rate  = self._get_latest_value(policy_data)
        usd_latest    = self._get_latest_value(usd_data)
        bond_latest   = self._get_latest_value(bond_data)

        # CPI: TP.FG.J0 indeks degerinden 12 aylik yoy hesapla
        cpi_normalized = self._calculate_cpi_yoy(cpi_data)

        rate_change_normalized = (rate_change / 100.0) if rate_change is not None else 0.0

        snapshot = {
            "tcmb_rate_change":    rate_change_normalized,
            "current_policy_rate": current_rate,
            "cpi_yoy":             cpi_normalized,
            "usdtry_official":     usd_latest,
            "bond_2y":             bond_latest,
            "data_quality": {
                "policy_rate_n": len(policy_data),
                "cpi_n":         len(cpi_data),
                "as_of": max(
                    [d["date"] for d in usd_data] +
                    [d["date"] for d in cpi_data] +
                    ["1900-01-01"]
                ),
            },
        }

        logger.info(
            f"Macro snapshot: faiz={current_rate}%, "
            f"30g degisim={rate_change_normalized * 100:+.2f}pp, "
            f"CPI={cpi_normalized * 100 if cpi_normalized else 'N/A'}%, "
            f"USD/TRY={usd_latest}"
        )
        return snapshot

    @staticmethod
    def _get_latest_value(data: List[Dict]) -> Optional[float]:
        if not data:
            return None
        return data[-1]["value"]

    @staticmethod
    def _calculate_rate_change(data: List[Dict], days: int = 30) -> Optional[float]:
        """Son 'days' gundeki faiz degisimini puan (pp) olarak hesapla."""
        if len(data) < 2:
            return None
        latest = data[-1]
        latest_date = datetime.fromisoformat(latest["date"])
        cutoff = latest_date - timedelta(days=days)

        prior = None
        for point in reversed(data[:-1]):
            point_date = datetime.fromisoformat(point["date"])
            if point_date <= cutoff:
                prior = point
                break

        if prior is None:
            prior = data[0]

        return latest["value"] - prior["value"]

    @staticmethod
    def _calculate_cpi_yoy(data: List[Dict]) -> Optional[float]:
        """
        CPI indeks serisinden 12 aylik yillik degisim hesapla.
        En az 13 aylik veri gerekir (yil once + 12 ay + son ay).
        Donus: oran (0.30 = %30)
        """
        if len(data) < 13:
            return None
        latest   = data[-1]["value"]
        year_ago = data[-13]["value"]
        if year_ago == 0:
            return None
        return (latest / year_ago) - 1.0
