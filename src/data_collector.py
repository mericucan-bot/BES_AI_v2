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
    TEFAS'tan BES fon verilerini çeken collector.

    TEFAS'ın arka plan API'si:
    - Fon geçmişi: https://www.tefas.gov.tr/api/DB/BindHistoryInfo
    - Fon dağılımı: https://www.tefas.gov.tr/api/DB/BindHistoryAllocation

    NOT: Bu resmi bir API değil, web sitesinin kullandığı endpoint.
    Rate limiting ve hata yönetimi kritik.
    """

    BASE_URL = "https://www.tefas.gov.tr/api/DB/BindHistoryInfo"
    ALLOCATION_URL = "https://www.tefas.gov.tr/api/DB/BindHistoryAllocation"

    BES_FUND_TYPES = {
        "EMK": "Emeklilik Yatırım Fonu",
    }

    FUND_CATEGORIES = {
        "HIS": "Hisse Senedi",
        "BYF": "Borçlanma Araçları",
        "ALT": "Altın / Kıymetli Maden",
        "KAR": "Karma / Değişken",
        "PPF": "Para Piyasası",
        "KAT": "Katılım",
        "STF": "Standart",
    }

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.tefas.gov.tr/TarihselVeriler.aspx",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    def __init__(self, cache_dir: str = "data/tefas_cache", rate_limit_sec: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_sec = rate_limit_sec
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def fetch_fund_history(
        self,
        fund_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Tek bir fonun günlük NAV geçmişini çek.

        fund_code: TEFAS fon kodu (örn: "AEA", "IPB")
        start_date: DD.MM.YYYY formatında
        end_date: DD.MM.YYYY formatında

        Returns: DataFrame(date, nav, fund_code) veya None
        """
        cache_key = f"{fund_code}_{start_date}_{end_date}".replace(".", "")

        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        try:
            payload = {
                "fontip": "EMK",
                "fonkod": fund_code,
                "baession_date": start_date,
                "bession_date": end_date,
                "fontupinitial": "EMK",
            }

            response = self.session.post(self.BASE_URL, data=payload, timeout=15)

            if response.status_code != 200:
                logger.warning(f"TEFAS HTTP {response.status_code}: {fund_code}")
                response = self.session.get(
                    f"{self.BASE_URL}?fonkod={fund_code}"
                    f"&baession_date={start_date}&bession_date={end_date}&fontip=EMK",
                    timeout=15,
                )

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type.lower() and "javascript" not in content_type.lower():
                logger.error(f"TEFAS JSON dönmedi ({fund_code}): {content_type}")
                return None

            data = response.json()

            if not data or (isinstance(data, dict) and not data.get("data")):
                logger.warning(f"TEFAS boş veri ({fund_code})")
                return None

            items = data if isinstance(data, list) else data.get("data", [])

            if not items:
                logger.warning(f"TEFAS boş items ({fund_code})")
                return None

            df = self._parse_fund_data(items, fund_code)

            if df is not None and not df.empty:
                self._write_cache(cache_key, df)
                logger.info(f"TEFAS çekildi: {fund_code} ({len(df)} gün)")

            time.sleep(self.rate_limit_sec)
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"TEFAS bağlantı hatası ({fund_code}): {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"TEFAS parse hatası ({fund_code}): {e}")
            return None
        except Exception as e:
            logger.error(f"TEFAS beklenmedik hata ({fund_code}): {e}")
            return None

    def _parse_fund_data(self, items: List[Dict], fund_code: str) -> Optional[pd.DataFrame]:
        """TEFAS response'unu DataFrame'e çevir."""
        rows = []
        for item in items:
            try:
                date_raw = (
                    item.get("TARIH")
                    or item.get("Tarih")
                    or item.get("tarih")
                    or item.get("date")
                )
                price_raw = (
                    item.get("FIYAT")
                    or item.get("Fiyat")
                    or item.get("fiyat")
                    or item.get("price")
                    or item.get("ToplamDeger")
                )

                if date_raw is None or price_raw is None:
                    continue

                if isinstance(date_raw, str):
                    if "T" in date_raw:
                        date = pd.Timestamp(date_raw.split("T")[0])
                    elif "." in date_raw:
                        parts = date_raw.split(".")
                        if len(parts) == 3:
                            date = pd.Timestamp(f"{parts[2]}-{parts[1]}-{parts[0]}")
                        else:
                            continue
                    else:
                        date = pd.Timestamp(date_raw)
                else:
                    date = pd.Timestamp(date_raw)

                if isinstance(price_raw, str):
                    price = float(price_raw.replace(",", "."))
                else:
                    price = float(price_raw)

                if price <= 0:
                    continue

                rows.append({"date": date, "nav": price, "fund_code": fund_code})

            except (ValueError, TypeError) as e:
                logger.debug(f"Satır parse edilemedi ({fund_code}): {item} → {e}")
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        return df

    def fetch_multiple_funds(
        self,
        fund_codes: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Birden fazla fonun verisini çek ve birleştir.

        Returns: DataFrame(date, fund_code, nav) — long format
        """
        all_dfs = []
        failed = []

        for i, code in enumerate(fund_codes):
            logger.info(f"[{i+1}/{len(fund_codes)}] Çekiliyor: {code}")
            df = self.fetch_fund_history(code, start_date, end_date, use_cache=use_cache)

            if df is not None and not df.empty:
                all_dfs.append(df)
            else:
                failed.append(code)

        if failed:
            logger.warning(f"Başarısız fonlar ({len(failed)}): {failed}")

        if not all_dfs:
            logger.error("Hiçbir fon verisi çekilemedi!")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Toplam: {len(combined)} satır, {combined['fund_code'].nunique()} fon")
        return combined

    def get_nav_pivot(
        self,
        fund_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        NAV verilerini pivot tablo formatında döndür.

        Returns: DataFrame(index=date, columns=fund_codes, values=nav)
        """
        long_df = self.fetch_multiple_funds(fund_codes, start_date, end_date)

        if long_df.empty:
            return pd.DataFrame()

        pivot = long_df.pivot_table(
            index="date",
            columns="fund_code",
            values="nav",
            aggfunc="last",
        )

        pivot = pivot.asfreq("B")
        pivot = pivot.ffill()
        return pivot

    # --- Cache yönetimi ---

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def _read_cache(self, key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=max_age_hours):
                return None
            df = pd.read_parquet(path)
            logger.debug(f"TEFAS cache HIT: {key} ({len(df)} satır)")
            return df
        except Exception as e:
            logger.warning(f"TEFAS cache okuma hatası ({key}): {e}")
            return None

    def _write_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_path(key), index=False)
        except Exception as e:
            logger.error(f"TEFAS cache yazma hatası ({key}): {e}")


# --- HAZIR BES FON LİSTESİ ---

POPULAR_BES_FUNDS = {
    "AEA": "Ak Emeklilik Hisse Senedi",
    "IPB": "İş Portföy BİST Emeklilik",
    "GAE": "Garanti Emeklilik Hisse",
    "AEK": "Ak Emeklilik Kamu Borç",
    "IPK": "İş Portföy Kamu Borç Emeklilik",
    "GEK": "Garanti Emeklilik Kamu Borç",
    "AES": "Ak Emeklilik Altın",
    "IPA": "İş Portföy Altın Emeklilik",
    "AED": "Ak Emeklilik Değişken",
    "IPD": "İş Portföy Değişken Emeklilik",
    "AET": "Ak Emeklilik Standart",
    "IPT": "İş Portföy Standart Emeklilik",
    "AEI": "Ak Emeklilik Katılım",
}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.logging_config import configure_logging

    configure_logging(level="DEBUG")

    collector = TEFASCollector()

    print("=== TEFAS Endpoint Kesfi ===")

    test_fund = "AEA"
    start = "01.01.2025"
    end = "30.04.2025"

    print(f"\nTest fonu: {test_fund} ({start} -> {end})")

    # Yontem 1: Standart POST
    print("\n--- Yontem 1: POST (standart payload) ---")
    try:
        payload = {
            "fontip": "EMK",
            "fonkod": test_fund,
            "baession_date": start,
            "bession_date": end,
        }
        resp = requests.post(
            "https://www.tefas.gov.tr/api/DB/BindHistoryInfo",
            data=payload,
            headers=TEFASCollector.DEFAULT_HEADERS,
            timeout=15,
        )
        print(f"  Status       : {resp.status_code}")
        print(f"  Content-Type : {resp.headers.get('Content-Type', '?')}")
        body = resp.text[:500].encode("ascii", errors="replace").decode("ascii")
        print(f"  Body (500c)  : {body}")

        if resp.status_code == 200 and resp.text.strip():
            try:
                data = resp.json()
                if isinstance(data, dict):
                    print(f"  JSON keys    : {list(data.keys())}")
                    inner = data.get("data") or data.get("Data") or []
                    if inner:
                        print(f"  Ilk kayit    : {inner[0]}")
                        print(f"  Kayit sayisi : {len(inner)}")
                elif isinstance(data, list) and data:
                    print(f"  Liste uzunlugu: {len(data)}")
                    print(f"  Ilk item keys : {list(data[0].keys())}")
                    print(f"  Ilk item      : {data[0]}")
            except Exception as e:
                print(f"  JSON parse hatasi: {e}")
    except Exception as e:
        print(f"  HATA: {e}")

    # Yontem 2: Alternatif key isimleri
    print("\n--- Yontem 2: POST (alternatif payload) ---")
    try:
        alt_payload = {
            "fontip": "EMK",
            "session_date": start,
            "fonkod": test_fund,
        }
        resp2 = requests.post(
            "https://www.tefas.gov.tr/api/DB/BindHistoryInfo",
            data=alt_payload,
            headers=TEFASCollector.DEFAULT_HEADERS,
            timeout=15,
        )
        print(f"  Status       : {resp2.status_code}")
        body2 = resp2.text[:500].encode("ascii", errors="replace").decode("ascii")
        print(f"  Body (500c)  : {body2}")
    except Exception as e:
        print(f"  HATA: {e}")

    # Yontem 3: Allocation endpoint
    print("\n--- Yontem 3: BindHistoryAllocation ---")
    try:
        alloc_payload = {
            "fontip": "EMK",
            "fonkod": test_fund,
            "baession_date": start,
            "bession_date": end,
        }
        resp3 = requests.post(
            "https://www.tefas.gov.tr/api/DB/BindHistoryAllocation",
            data=alloc_payload,
            headers=TEFASCollector.DEFAULT_HEADERS,
            timeout=15,
        )
        print(f"  Status       : {resp3.status_code}")
        body3 = resp3.text[:500].encode("ascii", errors="replace").decode("ascii")
        print(f"  Body (500c)  : {body3}")
    except Exception as e:
        print(f"  HATA: {e}")

    print("\n=== Kesif tamamlandi ===")
    print("Yukaridaki ciktiyi paylas -- response formatina gore parser'i guncelleyecegiz.")
