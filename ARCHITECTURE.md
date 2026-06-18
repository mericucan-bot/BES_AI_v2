# BES AI — Mimari

Yapay zeka destekli Bireysel Emeklilik Sistemi (BES) fon danışmanı. Türkiye piyasa
verisinden (BIST, USD/TRY, altın, TCMB makro) **piyasa rejimi** tespit eder, rejime
göre **portföy hedef ağırlıkları** üretir, AL/SAT/TUT önerir ve **XGBoost ile fon
getirisi tahmini** yapar. Çıktı: Streamlit dashboard, CLI, PDF rapor, e-posta.

> Bu belge kod tabanının güncel halini yansıtır. Yatırım tavsiyesi değildir.

---

## 1. Giriş noktaları

| Dosya | Rol |
|-------|-----|
| [`app.py`](app.py) | Streamlit web dashboard (kullanıcıya bakan yüz, render katmanı) |
| [`main.py`](main.py) | CLI / aylık otomasyon (cron motoru): pipeline, ML, backtest, rapor, e-posta |
| [`scripts/monthly_run.sh`](scripts/monthly_run.sh) | macOS launchd ile aylık otomatik çalışma |

Tüm iş mantığı `src/`'tedir; `app.py` ve `main.py` yalnızca orkestrasyon + sunum.

---

## 2. Katmanlı mimari

```
┌──────────────────────────────────────────────────────────────┐
│ SUNUM            app.py (Streamlit, 5 sekme)   main.py (CLI)   │
│                  report_generator   email_notifier             │
├──────────────────────────────────────────────────────────────┤
│ ORKESTRASYON     pipeline.MonthlyPipeline   ml_pipeline.MLPipeline
│                  backtest_engine.BacktestEngine                │
├──────────────────────────────────────────────────────────────┤
│ İŞ MANTIĞI       regime_engine   learning_engine   cost_model  │
│                  performance_tracker   feature_engineer        │
│                  ml_model   macro_engine   portfolio_manager   │
├──────────────────────────────────────────────────────────────┤
│ SAF YARDIMCILAR  auth   notification_prefs   ui_format         │
│ (UI'dan ayrık)   io_utils   logging_config   cache_manager     │
├──────────────────────────────────────────────────────────────┤
│ VERİ ERİŞİMİ     data_collector (TEFAS)   regime fetch (yfinance)
│                  macro_engine.TCMBClient (EVDS)                │
├──────────────────────────────────────────────────────────────┤
│ KAYNAKLAR        Yahoo Finance · tefas.gov.tr · TCMB EVDS      │
│ DEPOLAMA         data/  (parquet + JSON snapshot)              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. İki ana akış

### A) Aylık portföy pipeline'ı — `MonthlyPipeline.run()` ([pipeline.py](src/pipeline.py))

`python main.py` / cron ile tetiklenir. Sıralama **kritiktir**:

```
1. Portföy yükle              data/my_portfolio.json → {holdings_tl: {FONKODU: TL}}
2. Değerleme                  PerformanceTracker (toplam + ağırlıklar)
3. ÖNCEKİ AYI DEĞERLENDİR     ← snapshot YAZILMADAN önce
   ├─ PİYASA getirisi: önceki fon bakiyelerini gerçekleşen getiriyle revalüe et
   │     (katkı/çıkışlardan arındırılmış — alfa için doğru sinyal)
   │     öncelik: gerçek NAV tam-dönem → ~aylık return_1m → nominal
   ├─ benchmark: gerçek-NAV karma sepet → ~aylık sepet → periyot-doğru BIST 100
   └─ net_alpha = getiri − benchmark − önceki rebalance maliyeti → LearningEngine'e gözlem
4. Rejim tespiti              RegimeEngineV2.compute_composite_score()
5. Hedef ağırlıklar           LearningEngine.get_optimized_weights(regime) (shrinkage'lı)
6. Öneriler                   BUY/SELL/HOLD + aylık switch limiti + maliyet
7. Reel getiri                Fisher denklemi (TÜFE düzeltmeli)
8. Snapshot kaydet            data/history/YYYY_MM_snapshot.json
```

Bu, **kendi kendini denetleyen kapalı döngü**: her ay önceki tahminin tutup tutmadığını
ölçüp `learning_history.json`'a yazar; sonraki ayki ağırlıklar bu birikimden beslenir.

### B) ML tahmin pipeline'ı — `MLPipeline.run_full_pipeline()` ([ml_pipeline.py](src/ml_pipeline.py))

`python main.py --ml-train` ile, portföy pipeline'ından bağımsız:

```
1. Fon NAV verisi topla   →  2. Piyasa verisi  →  3. Feature matrisi (FeatureEngineer)
4. Model eğit (BESPredictor: XGBoost / RandomForest / Ridge, tarih-bazlı walk-forward)
5. En iyi modeli IC ile seç  →  6. Tahminler + model (.joblib) + özet kaydet
```

---

## 4. Modül modül roller

| Modül | Rol |
|-------|-----|
| **regime_engine** | 4 rejim: `CRISIS / RISK_ON / RATE_HIKE / STABLE`. BIST drawdown+vol, USD/altın momentum, TCMB faiz değişiminden kompozit skor → olasılık. Altın momentumu CRISIS sinyaline dahil. |
| **learning_engine** | Rejime göre hedef ağırlık. Yeterli gözlemde **alpha-ağırlıklı öğrenilmiş** ağırlık, ama küçük örneklemde **prior'a shrinkage** (λ=n/(n+12)); yoksa `STATIC_PRIORS`. Risk profili çarpanı. |
| **pipeline** | A akışı orkestratörü; piyasa-getirisi revalüasyon + karma benchmark + periyot gate. |
| **performance_tracker** | Değerleme, **reel getiri** (Fisher), `revalue_holdings` (fon getirileriyle yeniden değerleme). |
| **cost_model** | İşlem maliyeti (slippage), aylık switch limiti, **`holding_cost_pct`** (yönetim ücreti — yalnız proxy modunda; gerçek-NAV net). |
| **macro_engine** | TCMB EVDS: politika faizi, TÜFE, USD/TRY, 2Y tahvil. Disk cache + stale fallback. Explicit api_key öncelikli. |
| **data_collector** | `TEFASCollector` — güncel getiriler (fundturkey), **gerçek tarihsel günlük NAV** (`fetch_nav_history`, tefas.gov.tr), artımlı güncelleme (`update_nav_history`). |
| **feature_engineer** | NAV + makro → ML feature'ları + forward-return hedefleri (anti-leakage). |
| **ml_model** | `BESPredictor` — **tarih-bazlı + embargo'lu** walk-forward CV, **kesitsel IC**, XGBoost/RF/Ridge. |
| **backtest_engine** | Walk-forward backtest; **gerçek-NAV tam-dönem getirisi** (`RealNavReturnProvider`), tarihsel makro (look-ahead'siz). |
| **portfolio_manager** | Çoklu portföy CRUD, atomik JSON, slug. |
| **report_generator** / **email_notifier** | PDF rapor (reportlab) / SMTP aylık rapor. |
| **auth** | Uygulama şifresi + brute-force throttle (sabit-zamanlı karşılaştırma). |
| **notification_prefs** / **ui_format** / **io_utils** | Bildirim tercihleri / sunum formatlama / atomik dosya yazımı (app.py'den ayrıştırıldı, test edilebilir). |
| **cache_manager** / **logging_config** | Disk cache + akıllı TTL / renkli loglama. |

---

## 5. Veri kaynakları

| Kaynak | Ne için | Not |
|--------|---------|-----|
| **Yahoo Finance** (`yfinance`) | BIST `XU100.IS`, USD/TRY, altın `GC=F` | Rejim + ML piyasa feature'ları; 3 deneme + backoff |
| **tefas.gov.tr** `/api/funds/fonGnlBlgSiraliGetir` | **Gerçek tarihsel günlük NAV** | Tarihe saygılı, **~1 ay sert pencere** → ay-ay döngü; 429 backoff |
| **fundturkey.com.tr** `/api/fund-returns/export` | Güncel dönem getirileri + fon listesi | ⚠️ İstenen geçmiş tarihi **yok sayar** — tarihsel için kullanılmaz |
| **TCMB EVDS** | Makro seriler | API key opsiyonel (`.env` → `TCMB_API_KEY`) |

> Eski `tefas.gov.tr/api/DB/BindHistoryInfo` kapatılmıştır (404). Gerçek geçmiş
> yalnızca `fonGnlBlgSiraliGetir` üzerinden alınır.

---

## 6. Veri depolama (`data/`)

| Yol | İçerik |
|-----|--------|
| `tefas_cache/nav_history.parquet` | **Gerçek günlük fon NAV geçmişi** (artımlı büyür) — backtest + piyasa getirisi |
| `tefas_cache/snapshot_EMK_*.parquet` | Güncel fon snapshot'ı (kategori haritası + return_1m) |
| `history/YYYY_MM_snapshot.json` | Aylık portföy durumu (öğrenme döngüsünün hafızası) |
| `learning_history.json` | Rejim × ağırlık × alpha gözlemleri |
| `portfolios/*.json` | Çoklu kullanıcı portföyleri |
| `cache/macro_*.json` | TCMB makro cache |
| `ml/` | Eğitilmiş modeller (`.joblib`), tahminler (`.csv`), dataset (`.parquet`) |

`nav_history.parquet` ve `cache/`, `history/` vb. **gitignore'da** (büyük/kişisel veri);
`fetch_nav_history` / aylık cron ile yeniden üretilir.

---

## 7. Doğruluk güvenceleri (denetim sonrası)

- **ML anti-leakage:** walk-forward CV **takvim tarihine** göre bölünür + forward-return
  ufku kadar **embargo/purge**; IC **kesitsel** (her tarih için) hesaplanıp ortalanır.
- **Backtest gerçek veriyle:** proxy yerine TEFAS gerçek günlük NAV'dan **tam-dönem**
  getiri; altın TL-bazlı; TCMB faiz değişimi her karar tarihine göre tarihsel
  (look-ahead yok). Veri kopya/eksikse yüksek-sesle uyarı.
- **Pipeline alfası:** katkı/çıkışlardan arındırılmış **piyasa getirisi** üzerinden;
  benchmark çok-varlıklı karma BES sepeti (BIST 100 değil); `return_1m` yalnız ~1 aylık
  periyotta, NAV tam-dönem her periyotta geçerli.
- **Öğrenme:** küçük örneklem overfit'ine karşı prior'a shrinkage.

---

## 8. Dashboard ([app.py](app.py)) — 5 sekme

`Piyasa Şu An Nasıl?` · `Ne Yapmalıyım?` · `Geçmiş Performans` · `AI Fon Tahminleri` (+ ayarlar).
Şifre koruması + sunucu-tarafı brute-force throttle; `DEV_BYPASS_AUTH=true` ile yerel
preview'da auth atlanır. `@st.cache_data/resource` ile pahalı çağrılar önbelleklenir.

---

## 9. Test & otomasyon

- **`tests/`** — ~300 test (pytest); her motor modülünün karşılığı + saf yardımcılar
  (auth, notification_prefs, ui_format, io_utils) test kapsamında.
- **Aylık cron** (`scripts/monthly_run.sh` → launchd → `main.py`): pipeline + PDF + e-posta
  çalıştırır ve `update_nav_history` ile gerçek NAV geçmişini taze tutar.
- **Streamlit Cloud** uyumlu (`runtime.txt`, `packages.txt`, `.streamlit/secrets.toml`).

---

## 10. Çalıştırma

```bash
streamlit run app.py                          # Dashboard
python main.py                                # Aylık pipeline
python main.py --ml-train                     # ML model eğitimi
python main.py --backtest --bt-start 2024-06-01 --bt-end 2026-06-01
python main.py --report                       # PDF rapor
pytest tests/ -q                              # Testler

# Gerçek tarihsel NAV çek (Python):
python -c "from src.data_collector import TEFASCollector; \
TEFASCollector().update_nav_history()"
```
