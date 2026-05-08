# BES Akıllı Fon Danışmanı

Yapay zeka destekli Bireysel Emeklilik Sistemi (BES) portföy yönetim ve analiz platformu.

## Özellikler

- **Piyasa Analizi:** BIST 100, USD/TRY, Altın, TCMB makro verileri ile piyasa rejimi tespiti
- **AI Fon Tahminleri:** XGBoost ile 390 BES fonunun 3 aylık getiri tahmini (IC=0.80)
- **Portföy Önerisi:** Kişisel portföye göre AL/SAT/TUT önerileri
- **Reel Getiri:** Enflasyon düzeltmeli performans analizi (Fisher denklemi)
- **Backtest:** 21 aylık walk-forward geriye dönük test
- **PDF Rapor:** Otomatik aylık profesyonel rapor üretimi
- **200 Test:** Kapsamlı test suite ile doğrulanmış

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

# ML model eğitimi (390 BES fonu)
python main.py --ml-train

# 12 aylık model de eğit
python main.py --ml-train --ml-12m

# Backtest
python main.py --backtest --bt-start 2024-06-01 --bt-end 2026-04-01

# PDF rapor
python main.py --report
```

## Testler

```bash
pytest tests/ -q  # 200 test
```

## Yasal Uyarı

Bu sistem yatırım tavsiyesi niteliği taşımaz. Geçmiş performans gelecek sonuçları garanti etmez.
