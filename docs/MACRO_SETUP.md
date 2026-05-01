# TCMB Makro Veri Kurulumu

## 1. EVDS Hesabı Oluştur

1. https://evds2.tcmb.gov.tr/ adresine git
2. Sağ üstten "Üye Ol" ile ücretsiz hesap oluştur
3. Giriş yaptıktan sonra sağ üstten **Profil** → **API Anahtarı** sekmesine git
4. "Anahtarı Göster" butonuna tıkla (8 karakter, örn. `aBcD3fGh`)

## 2. .env Dosyası Oluştur

Proje kökünde `.env` dosyası oluştur (`.env.example`'ı kopyala):

```
TCMB_API_KEY=aBcD3fGh
```

> `.env` dosyası `.gitignore`'da tanımlı — commit edilmez. `.env.example` commit edilebilir.

## 3. Doğrula

```bash
python -c "from src.macro_engine import MacroEngine; import pprint; pprint.pprint(MacroEngine().get_macro_snapshot())"
```

Beklenen çıktı (API key geçerliyse):
```
{'bond_2y': 47.8,
 'cpi_yoy': 0.38,
 'current_policy_rate': 42.5,
 'data_quality': {'as_of': '2026-04-29', 'cpi_n': 12, 'policy_rate_n': 90},
 'tcmb_rate_change': -0.025,
 'usdtry_official': 38.5}
```

## Çekilen Seriler

| Seri Kodu       | İçerik                    | Frekans |
|-----------------|---------------------------|---------|
| TP_AOFOD        | Politika faizi (repo)     | Günlük  |
| TP_FG_J0        | TÜFE yıllık değişim       | Aylık   |
| TP_DK_USD_A_YTL | USD/TRY satış kuru (TCMB) | Günlük  |
| TP_AKM_B070     | 2Y devlet tahvili faizi   | Günlük  |

## Cache Davranışı

- İlk çekimden sonra `data/cache/macro_*.json` dosyalarına kaydedilir
- 24 saat cache geçerli (tekrar API çağrısı yapılmaz)
- API down ise eski cache kullanılır (stale fallback)
- API down + cache yok ise nötr değerler (`tcmb_rate_change=0`)

## API Key Yoksa Ne Olur?

Sistem yine de çalışır. `TCMB_API_KEY` bulunamazsa:
- `tcmb_rate_change = 0` (faiz hareketi sinyali yok)
- RATE_HIKE rejimi yalnızca USD momentum üzerinden tetiklenebilir
- Diğer rejimler (CRISIS, STABLE, RISK_ON) etkilenmez
