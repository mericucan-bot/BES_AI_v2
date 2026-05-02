import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class TEFASCollector:
    """
    fundturkey.com.tr'den BES fon verilerini ceken collector.

    Calistigini dogruladigimiz endpoint:
      POST https://fundturkey.com.tr/api/fund-returns/export
      Content-Type: application/json
      Body: {format, listingType, fundType, locale, fonKodu, basTarih, bitTarih, calismaTipi}

    Bu endpoint EMK fon listesi + donem getirileri donduruyor.
    Eski BindHistoryInfo/BindHistoryAllocation endpointleri kalici olarak kapanmistir.
    """

    BASE_URL = "https://fundturkey.com.tr"
    EXPORT_ENDPOINT = "/api/fund-returns/export"

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Referer": "https://fundturkey.com.tr/en/historical-data",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    def __init__(self, cache_dir: str = "data/tefas_cache", rate_limit_sec: float = 1.5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_sec = rate_limit_sec
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self._init_session()

    def _init_session(self) -> None:
        """Ana sayfaya gidip cookie al."""
        try:
            self.session.get(f"{self.BASE_URL}/en/historical-data", timeout=15)
        except Exception as e:
            logger.warning(f"Session init basarisiz (devam ediyor): {e}")

    # ------------------------------------------------------------------
    # Ana metod: tek gün için tüm EMK fon listesi
    # ------------------------------------------------------------------

    def fetch_fund_snapshot(
        self,
        date: str,
        fund_type: str = "EMK",
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Belirli bir tarih icin tum EMK fon bilgilerini cek.

        date: "YYYYMMDD" veya "YYYY-MM-DD" formatinda
        Returns: DataFrame(date, fund_code, fund_name, category,
                           return_1m, return_3m, return_6m,
                           return_ytd, return_1y, return_3y, return_5y)
        """
        date_fmt = self._normalize_date(date)
        cache_key = f"snapshot_{fund_type}_{date_fmt}"

        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        payload = {
            "format": "json",
            "listingType": "return",
            "fundType": fund_type,
            "locale": "en",
            "filters": {
                "kurucuKodu": None,
                "fonTurKod": None,
                "fonGrubu": None,
                "fonTurAciklama": None,
                "sfonTurKod": None,
                "basTarih": date_fmt,
                "bitTarih": date_fmt,
                "calismaTipi": 2,
                "donemGetiri1a": "1",
                "donemGetiri3a": "1",
                "donemGetiri6a": "1",
                "donemGetiriyb": "1",
                "donemGetiri1y": "1",
                "donemGetiri3y": "1",
                "donemGetiri5y": "1",
                "getiriOrani": "1",
            },
        }

        try:
            resp = self.session.post(
                f"{self.BASE_URL}{self.EXPORT_ENDPOINT}",
                json=payload,
                timeout=20,
            )
            resp.raise_for_status()
            raw = resp.json()

            if not isinstance(raw, list) or not raw:
                logger.warning(f"TEFAS bos liste: {date_fmt}")
                return None

            df = self._parse_snapshot(raw, date_fmt)

            if df is not None and not df.empty:
                self._write_cache(cache_key, df)
                logger.info(f"TEFAS snapshot: {date_fmt} — {len(df)} fon")

            time.sleep(self.rate_limit_sec)
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"TEFAS baglanti hatasi ({date_fmt}): {e}")
            return None
        except Exception as e:
            logger.error(f"TEFAS beklenmedik hata ({date_fmt}): {e}")
            return None

    def _parse_snapshot(self, items: List[Dict], date_str: str) -> Optional[pd.DataFrame]:
        """API listesini standart DataFrame'e donustur."""
        rows = []
        for item in items:
            try:
                code = item.get("fonKodu") or item.get("fonKod") or ""
                if not code:
                    continue

                def to_float(v):
                    if v is None:
                        return None
                    if isinstance(v, str):
                        v = v.replace(",", ".")
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None

                rows.append({
                    "date": pd.Timestamp(
                        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    ),
                    "fund_code":  code.strip(),
                    "fund_name":  item.get("fonUnvan", ""),
                    "category":   item.get("fonTurAciklama", ""),
                    "risk":       to_float(item.get("riskDegeri")),
                    "return_1m":  to_float(item.get("getiri1a")),
                    "return_3m":  to_float(item.get("getiri3a")),
                    "return_6m":  to_float(item.get("getiri6a")),
                    "return_ytd": to_float(item.get("getiriyb")),
                    "return_1y":  to_float(item.get("getiri1y")),
                    "return_3y":  to_float(item.get("getiri3y")),
                    "return_5y":  to_float(item.get("getiri5y")),
                })
            except Exception as e:
                logger.debug(f"Satir parse edilemedi: {item} -> {e}")

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["fund_code"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Zaman serisi: aylik snapshot'lari birlestir
    # ------------------------------------------------------------------

    def fetch_monthly_series(
        self,
        start: str,
        end: str,
        fund_type: str = "EMK",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Verilen tarih araliginda aylik sonlarda snapshot cek ve birlestir.

        start/end: "YYYY-MM-DD" veya "YYYYMMDD"
        Returns: DataFrame(date, fund_code, fund_name, category, return_1m, ...)
                 uzun format — her satir bir fon x tarih kombinasyonu
        """
        dates = self._generate_month_end_dates(start, end)
        logger.info(f"Aylik seri: {len(dates)} snapshot x {fund_type}")

        all_dfs = []
        for i, date in enumerate(dates):
            logger.info(f"[{i+1}/{len(dates)}] {date}")
            df = self.fetch_fund_snapshot(date, fund_type, use_cache=use_cache)
            if df is not None and not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            logger.error("Hicbir snapshot alinamadi!")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Toplam: {len(combined)} satir, "
                    f"{combined['fund_code'].nunique()} fon, "
                    f"{combined['date'].nunique()} tarih")
        return combined

    def _generate_month_end_dates(self, start: str, end: str) -> List[str]:
        """Tarih araligindaki ay sonu is gunlerini uret."""
        start_dt = pd.Timestamp(self._normalize_date_iso(start))
        end_dt   = pd.Timestamp(self._normalize_date_iso(end))

        try:
            dates = pd.date_range(start_dt, end_dt, freq="BME")
        except ValueError:
            dates = pd.date_range(start_dt, end_dt, freq="BM")

        return [d.strftime("%Y%m%d") for d in dates]

    # ------------------------------------------------------------------
    # Eski arayüz uyumu (fetch_fund_history, fetch_multiple_funds)
    # ------------------------------------------------------------------

    def fetch_fund_history(
        self,
        fund_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Belirli bir fonun aylik return serisini cek.

        fund_code: TEFAS fon kodu (ornegin "AEA")
        start_date / end_date: "YYYY-MM-DD" veya "DD.MM.YYYY"
        Returns: DataFrame(date, return_1m, fund_code) — aylik frekans
        """
        start_iso = self._normalize_date_iso(start_date)
        end_iso   = self._normalize_date_iso(end_date)

        full = self.fetch_monthly_series(start_iso, end_iso, use_cache=use_cache)
        if full.empty:
            return None

        result = full[full["fund_code"] == fund_code.upper()].copy()

        if result.empty:
            logger.warning(f"Fon bulunamadi snapshot'larda: {fund_code}")
            return None

        return result.reset_index(drop=True)

    def fetch_multiple_funds(
        self,
        fund_codes: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Birden fazla fonun aylik return verisini cek.

        Returns: long format DataFrame(date, fund_code, return_1m, ...)
        """
        start_iso = self._normalize_date_iso(start_date)
        end_iso   = self._normalize_date_iso(end_date)

        full = self.fetch_monthly_series(start_iso, end_iso, use_cache=use_cache)
        if full.empty:
            return pd.DataFrame()

        codes_upper = [c.upper() for c in fund_codes]
        result = full[full["fund_code"].isin(codes_upper)].copy()

        missing = set(codes_upper) - set(result["fund_code"].unique())
        if missing:
            logger.warning(f"Bulunamayan fonlar: {missing}")

        return result.reset_index(drop=True)

    def get_return_pivot(
        self,
        fund_codes: List[str],
        start_date: str,
        end_date: str,
        return_col: str = "return_1m",
    ) -> pd.DataFrame:
        """
        Pivot format: index=date, columns=fund_code, values=return_1m

        Backtest ve feature engineering icin hazir format.
        """
        long = self.fetch_multiple_funds(fund_codes, start_date, end_date)
        if long.empty:
            return pd.DataFrame()

        if return_col not in long.columns:
            logger.error(f"Sutun bulunamadi: {return_col}")
            return pd.DataFrame()

        pivot = long.pivot_table(
            index="date",
            columns="fund_code",
            values=return_col,
            aggfunc="last",
        )
        return pivot

    # ------------------------------------------------------------------
    # Yardimci metodlar
    # ------------------------------------------------------------------

    def _normalize_date(self, date_str: str) -> str:
        """Her formati YYYYMMDD'ye cevir."""
        date_str = str(date_str).strip()
        if len(date_str) == 8 and date_str.isdigit():
            return date_str
        ts = pd.Timestamp(date_str.replace(".", "-") if "." in date_str else date_str)
        return ts.strftime("%Y%m%d")

    def _normalize_date_iso(self, date_str: str) -> str:
        """Her formati YYYY-MM-DD'ye cevir."""
        date_str = str(date_str).strip()
        if "." in date_str:
            parts = date_str.split(".")
            if len(parts) == 3:
                return f"{parts[2]}-{parts[1]}-{parts[0]}"
        return pd.Timestamp(date_str).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Cache yonetimi
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe}.parquet"

    def _read_cache(self, key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=max_age_hours):
                return None
            df = pd.read_parquet(path)
            logger.debug(f"Cache HIT: {key} ({len(df)} satir)")
            return df
        except Exception as e:
            logger.warning(f"Cache okuma hatasi ({key}): {e}")
            return None

    def _write_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_path(key), index=False)
        except Exception as e:
            logger.error(f"Cache yazma hatasi ({key}): {e}")


# --- HAZIR BES FON LISTESI ---
# Tum kodlar TEFAS EMK listesinde dogrulanmistir (21 aylik snapshot'ta mevcut).

POPULAR_BES_FUNDS = {
    # Anadolu Hayat - Altin
    "AEA": "Anadolu Hayat Altin Katilim",
    "BGL": "Anadolu Hayat Altin",
    # Anadolu Hayat - Hisse
    "AH5": "Anadolu Hayat Hisse Senedi",
    "AHB": "Anadolu Hayat Ikinci Hisse",
    # Anadolu Hayat - Degisken
    "AHL": "Anadolu Hayat Agresif Degisken",
    "AH6": "Anadolu Hayat Birinci Degisken",
    # Anadolu Hayat - Borclanma
    "HS1": "Anadolu Hayat Kamu Borc",
    "AH1": "Anadolu Hayat Ikinci Kamu Borc",
    # Anadolu Hayat - Para Piyasasi / Doviz
    "AH2": "Anadolu Hayat Para Piyasasi",
    "AH3": "Anadolu Hayat Kamu Dis Borc 1",
    "AH4": "Anadolu Hayat Kamu Dis Borc 2",
    # Anadolu Hayat - Standart / Katki / Katilim
    "ATK": "Anadolu Hayat Standart",
    "AET": "Anadolu Hayat Katki",
    "AGE": "Anadolu Hayat Katilim Standart",
    "AER": "Anadolu Hayat Katilim Katki",
    # Agesa (eski AXA/Aviva) - cesitli kategoriler
    "AEK": "Agesa Kamu Borc Grup",
    "AEI": "Agesa Katki",
    "AEH": "Agesa Hisse",
    "AE1": "Agesa Para Piyasasi",
    "AE2": "Agesa Kamu Borc",
    "AE3": "Agesa Dinamik Degisken",
    # Garanti Emeklilik ve Hayat
    "GEK": "Garanti Emeklilik Kamu Borc",
    "GEH": "Garanti Emeklilik Hisse",
    "GEL": "Garanti Emeklilik Para Piyasasi",
}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.logging_config import configure_logging

    configure_logging(level="INFO")

    collector = TEFASCollector()

    print("=== TEFAS Collector Gercek Veri Testi ===")

    # Test 1: Tek gun snapshot
    print("\n[1] Bugunun snapshot'i (EMK fonlari)...")
    today = datetime.now().strftime("%Y%m%d")
    df = collector.fetch_fund_snapshot(today, use_cache=False)
    if df is not None:
        print(f"  Fon sayisi: {len(df)}")
        print(f"  Kolonlar  : {df.columns.tolist()}")
        print(f"  Ilk 5 satir:")
        display = df[["fund_code", "return_1m", "return_1y"]].head()
        print(display.to_string())
    else:
        print("  BASARISIZ - bos snapshot")

    # Test 2: 3 aylik seri (AEA, IPB)
    print("\n[2] 3 aylik getiri serisi (AEA, IPB)...")
    series = collector.fetch_multiple_funds(
        ["AEA", "IPB"], "2026-01-01", "2026-04-30"
    )
    if not series.empty:
        print(f"  Toplam satir: {len(series)}")
        print(series[["date", "fund_code", "return_1m", "return_1y"]].to_string())
    else:
        print("  BASARISIZ - bos seri")

    print("\n=== Tamamlandi ===")
