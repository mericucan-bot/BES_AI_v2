# BES Akıllı Fon Danışmanı

Yapay zeka destekli Bireysel Emeklilik Sistemi (BES) portföy yönetim ve analiz platformu.

## Özellikler

- **Piyasa Analizi:** BIST 100, USD/TRY, Altın, TCMB makro verileri ile piyasa rejimi tespiti
- **AI Fon Tahminleri:** XGBoost ile ~400 BES fonunun getiri tahmini — tarih-bazlı + embargo'lu (purged) walk-forward CV, kesitsel IC ile değerlendirilir (look-ahead'siz)
- **Portföy Önerisi:** Kişisel portföye göre AL/SAT/TUT önerileri
- **Reel Getiri:** Enflasyon düzeltmeli performans analizi (Fisher denklemi)
- **Backtest:** Gerçek günlük NAV (TEFAS tarihsel verisi) ile walk-forward geriye dönük test; karma BES benchmark'ı, look-ahead'siz makro
- **PDF Rapor:** Otomatik aylık profesyonel rapor üretimi
- **~300 Test:** Kapsamlı test suite ile doğrulanmış

## Dashboard

| Tab | İçerik |
|-----|--------|
| Piyasa Şu An Nasıl? | Rejim tespiti, makro göstergeler |
| Ne Yapmalıyım? | AL/SAT önerileri, maliyet analizi |
| Geçmiş Performans | Backtest, reel getiri, öğrenme durumu |
| AI Fon Tahminleri | ML model sonuçları, top fonlar |

## Kurulum

```bash
git clone https://github.com/mericucan-bot/BES_AI_v2.git
cd BES_AI_v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Konfigürasyon

```bash
# TCMB API key (opsiyonel, makro veriler için)
echo "TCMB_API_KEY=your_key" > .env
# veya evdspy native key yönetimi:
python3 -c "from evdspy import save_apikey; save_apikey('your_key')"
```

Streamlit Cloud için `.streamlit/secrets.toml.example` dosyasına bakın.

## Çalıştırma

```bash
# Dashboard
streamlit run app.py

# Aylık pipeline
python main.py

# ML model eğitimi (~400 BES fonu)
python main.py --ml-train

# 12 aylık model de eğit
python main.py --ml-train --ml-12m

# Backtest (gerçek NAV ile)
python main.py --backtest --bt-start 2024-06-01 --bt-end 2026-06-01

# PDF rapor
python main.py --report

# Gerçek tarihsel günlük NAV'ı çek/güncelle (backtest verisi)
python -c "from src.data_collector import TEFASCollector; TEFASCollector().update_nav_history()"
```

## Mimari

Detaylı mimari için [ARCHITECTURE.md](ARCHITECTURE.md).

## Testler

```bash
pytest tests/ -q  # ~300 test
```

## Yasal Uyarı

Bu sistem yatırım tavsiyesi niteliği taşımaz. Geçmiş performans gelecek sonuçları garanti etmez.
