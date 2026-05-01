import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.regime_engine import RegimeEngineV2
from src.learning_engine import LearningEngineV2
from src.cache_manager import get_smart_ttl, is_market_hours
from src.logging_config import configure_logging

# --- SAYFA KONFİGÜRASYONU ---
st.set_page_config(page_title="BES Akıllı Fon Danışmanı", page_icon="🛡️", layout="wide")

if "logging_configured" not in st.session_state:
    configure_logging(log_file="streamlit.log", level="INFO")
    st.session_state.logging_configured = True

st.markdown("""<style>.stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}</style>""", unsafe_allow_html=True)


# --- YARDIMCI FONKSİYONLAR ---

def explain_regime(regime: str) -> dict:
    explanations = {
        "STABLE": {
            "emoji": "😌",
            "label": "Sakin Piyasa",
            "color": "blue",
            "border": "#3b82f6",
            "summary": "Piyasalarda belirgin bir yön yok. Dengeli bir dağılım mantıklı.",
            "action": "Portföyünü dengeli tut, panik yapma.",
            "detail": "Volatilite düşük, BIST belirgin bir trend göstermiyor, döviz sakin. Bu ortamda aşırı agresif veya defansif olmaya gerek yok.",
        },
        "CRISIS": {
            "emoji": "🚨",
            "label": "Kriz Modu",
            "color": "red",
            "border": "#ef4444",
            "summary": "Piyasalarda ciddi düşüş veya belirsizlik var. Korunma öncelikli.",
            "action": "Altın ve sabit getirili fonlara ağırlık ver, hisse oranını azalt.",
            "detail": "BIST'te sert düşüş, dövizde hızlı yükseliş veya yüksek volatilite tespit edildi. Bu dönemlerde sermayeyi korumak önceliklidir.",
        },
        "RISK_ON": {
            "emoji": "🚀",
            "label": "Yükseliş Trendi",
            "color": "green",
            "border": "#22c55e",
            "summary": "Piyasalar yukarı yönlü. Hisse ağırlığını artırma fırsatı.",
            "action": "Hisse ve karma fonlara ağırlık ver.",
            "detail": "BIST güçlü momentum gösteriyor, volatilite makul seviyelerde. Tarihsel olarak bu dönemlerde hisse fonları iyi performans gösterir.",
        },
        "RATE_HIKE": {
            "emoji": "🏦",
            "label": "Faiz Artışı Dönemi",
            "color": "orange",
            "border": "#f59e0b",
            "summary": "Merkez bankası faiz artırıyor. Sabit getirili fonlar öne çıkıyor.",
            "action": "Kamu borçlanma (KTS) fonlarına ağırlık ver.",
            "detail": "TCMB politika faizini artırma eğiliminde. Yüksek faiz ortamında tahvil/bono fonları cazip getiri sunuyor.",
        },
    }
    return explanations.get(regime, explanations["STABLE"])


def confidence_to_text(confidence: float) -> str:
    if confidence >= 0.80:
        return "Yüksek güven — sinyal güçlü"
    elif confidence >= 0.60:
        return "Orta güven — makul sinyal"
    elif confidence >= 0.40:
        return "Düşük güven — belirsiz ortam"
    else:
        return "Çok düşük güven — dikkatli ol"


def format_tl(value: float) -> str:
    return f"{value:,.0f} TL".replace(",", ".")


def action_text(action: str) -> str:
    return {"BUY": "EKLE", "SELL": "AZALT", "HOLD": "DEĞİŞTİRME"}.get(action, action)


# --- VERİ YÜKLEME ---

@st.cache_data(ttl=get_smart_ttl())
def get_market_analysis():
    engine = RegimeEngineV2()
    return engine.compute_composite_score()


def load_my_portfolio():
    try:
        with open("data/my_portfolio.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


result = get_market_analysis()
regime = result["detected"]
metrics = result["metrics"]
my_data = load_my_portfolio()

# --- SIDEBAR ---
with st.sidebar:
    st.write("### 🛡️ BES Akıllı Fon Danışmanı")
    st.caption("v2.0 • 119 test ile doğrulanmış")

    st.divider()

    st.write("**Nasıl Çalışır?**")
    st.markdown("""
    1. 📊 Piyasa verilerini analiz eder
    2. 🎯 Piyasa ortamını sınıflandırır
    3. ⚖️ Optimal fon dağılımı önerir
    4. 📈 Performansını ölçer ve öğrenir
    """)

    st.divider()

    if st.button("🔄 Veriyi Yenile"):
        st.cache_data.clear()
        st.success("Veriler yenilendi!")
        st.rerun()

    market_status = "🟢 Açık" if is_market_hours() else "🔴 Kapalı"
    st.caption(f"BIST: {market_status}")
    st.caption(f"Sonraki güncelleme: {get_smart_ttl() // 60} dk")

    st.divider()
    st.caption("⚠️ Bu sistem yatırım tavsiyesi vermez. Kararlarınızdan siz sorumlusunuz.")

# --- BAŞLIK ---
st.title("🛡️ BES Akıllı Fon Danışmanı")
st.caption("Yapay zeka destekli BES portföy yönetim sistemi • Yatırım tavsiyesi değildir")

# --- SEKMELER ---
tab1, tab2, tab3 = st.tabs(["📊 Piyasa Şu An Nasıl?", "💼 Ne Yapmalıyım?", "📈 Geçmiş Performans"])


# ══════════════════════════════════════════════════════
# TAB 1 — Piyasa Şu An Nasıl?
# ══════════════════════════════════════════════════════
with tab1:
    regime_info = explain_regime(regime)
    macro = result.get("macro", {})

    # === ANA MESAJ ===
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 30px; border-radius: 16px; margin-bottom: 20px;
                border-left: 6px solid {regime_info['border']};">
        <h1 style="margin:0; font-size: 2.5em;">{regime_info['emoji']} {regime_info['label']}</h1>
        <p style="font-size: 1.3em; margin: 10px 0 5px 0; color: #374151;">{regime_info['summary']}</p>
        <p style="font-size: 1.1em; color: #6b7280;">💡 <strong>Öneri:</strong> {regime_info['action']}</p>
    </div>
    """, unsafe_allow_html=True)

    conf = result.get("confidence", 0)
    st.progress(conf, text=f"Sinyal Güveni: %{conf*100:.0f} — {confidence_to_text(conf)}")

    # === PİYASA ÖZETİ ===
    st.write("### 📊 Piyasa Özeti")
    m1, m2, m3, m4 = st.columns(4)

    dd_val  = metrics["dd"] * 100
    vol_val = metrics["vol"] * 100
    usd_val = metrics["usd_mom"] * 100
    cpi     = macro.get("cpi_yoy")

    m1.metric(
        "BIST 100 Durumu",
        "Düşüşte" if dd_val < -10 else ("Hafif Düşüş" if dd_val < -5 else "Normal"),
        f"%{dd_val:.1f} zirveden",
        delta_color="inverse",
    )
    m2.metric(
        "Piyasa Hareketliliği",
        "Yüksek" if vol_val > 30 else ("Normal" if vol_val < 20 else "Orta"),
        f"%{vol_val:.1f} yıllık",
    )
    m3.metric(
        "Dolar/TL Trendi",
        "Yükseliyor" if usd_val > 3 else ("Düşüyor" if usd_val < -3 else "Sabit"),
        f"%{usd_val:.1f} aylık",
    )
    m4.metric(
        "Enflasyon (TÜFE)",
        f"%{cpi*100:.1f}" if cpi else "Veri yok",
        "Yıllık" if cpi else None,
    )

    # === EKONOMİK GÖSTERGELER ===
    if macro and macro.get("usdtry_official"):
        st.write("### 🏛️ Ekonomik Göstergeler")
        mc1, mc2, mc3 = st.columns(3)

        rate        = macro.get("current_policy_rate")
        usd_off     = macro.get("usdtry_official")
        bond        = macro.get("bond_2y")
        rate_change = macro.get("tcmb_rate_change", 0) * 100

        mc1.metric(
            "TCMB Politika Faizi",
            f"%{rate:.1f}" if rate else "Veri yok",
            delta=f"{rate_change:+.2f} pp (30g)" if rate_change else None,
        )
        mc2.metric("Dolar/TL (Resmi)", f"₺{usd_off:.2f}" if usd_off else "Veri yok")
        mc3.metric("2 Yıllık Tahvil Faizi", f"%{bond:.1f}" if bond else "Veri yok")

        as_of = macro.get("data_quality", {}).get("as_of", "?")
        st.caption(f"📅 TCMB verisi: {as_of}")
    else:
        st.info("⚠️ TCMB verisi yok — .env dosyasında TCMB_API_KEY tanımlı mı?")

    # === TEKNİK DETAYLAR (gizli) ===
    with st.expander("🔧 Teknik Detaylar (ileri düzey)"):
        st.write("**Rejim Skorları** — Her piyasa durumunun olasılık puanı (0–1 arası):")
        st.bar_chart(pd.DataFrame.from_dict(result["scores"], orient="index", columns=["Skor"]))

        if result.get("probabilities"):
            st.write("**Olasılık Dağılımı** — Softmax ile normalize edilmiş rejim olasılıkları:")
            prob_df = pd.DataFrame.from_dict(result["probabilities"], orient="index", columns=["Olasılık"])
            st.dataframe(prob_df.style.format({"Olasılık": "{:.1%}"}))

        st.write("**Ham Metrikler:**")
        st.json({
            "Drawdown": f"%{metrics['dd']*100:.2f}",
            "Volatilite (yıllık)": f"%{metrics['vol']*100:.2f}",
            "USD/TRY Momentum (20g)": f"%{metrics['usd_mom']*100:.2f}",
            "Veri Kalitesi": result.get("data_quality", {}),
        })


# ══════════════════════════════════════════════════════
# TAB 2 — Ne Yapmalıyım?
# ══════════════════════════════════════════════════════
with tab2:
    if not my_data:
        st.warning("⚠️ Portföy dosyası bulunamadı. 'data/my_portfolio.json' dosyasını kontrol et.")
    else:
        holdings     = my_data["holdings_tl"]
        total_value  = sum(holdings.values())
        current_weights = {k: v / total_value for k, v in holdings.items()}

        learning       = LearningEngineV2()
        target_weights = learning.get_optimized_weights(regime)
        regime_info    = explain_regime(regime)

        asset_names = {
            "VEF":  "Hisse Senedi Fonu",
            "ALT":  "Altın Fonu",
            "KTS":  "Kamu Borç. Fonu",
            "KCH":  "Karma/Değişken Fon",
            "CASH": "Para Piyasası",
        }

        # === ANA MESAJ ===
        st.markdown(f"""
        <div style="background: #f0f9ff; padding: 20px; border-radius: 12px;
                    border-left: 5px solid #3b82f6; margin-bottom: 20px;">
            <h2 style="margin:0;">💼 Portföyün: {format_tl(total_value)}</h2>
            <p style="font-size: 1.1em; margin: 8px 0 0 0; color: #374151;">
                {regime_info['emoji']} Piyasa <strong>{regime_info['label']}</strong> modunda →
                {regime_info['action']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # === AKSİYONLAR + MALİYET HESABI ===
        from src.cost_model import TransactionCostModel
        cost_model = TransactionCostModel()

        analysis_data   = []
        recommendations = []
        all_assets = sorted(set(list(target_weights.keys()) + list(holdings.keys())))

        for asset in all_assets:
            curr_w  = current_weights.get(asset, 0)
            target_w = target_weights.get(asset, 0)
            diff_tl  = (target_w - curr_w) * total_value

            if abs(diff_tl) < 100:
                action = "HOLD"
            elif diff_tl > 0:
                action = "BUY"
            else:
                action = "SELL"

            analysis_data.append({
                "asset":      asset,
                "name":       asset_names.get(asset, asset),
                "current_tl": holdings.get(asset, 0),
                "current_w":  curr_w,
                "target_w":   target_w,
                "diff_tl":    diff_tl,
                "action":     action,
            })
            recommendations.append({"asset": asset, "action": action, "diff_tl": diff_tl})

        # === YAPMAN GEREKENLER ===
        st.write("### 📋 Bu Ay Yapman Gerekenler")

        has_action = False
        for item in sorted(analysis_data, key=lambda x: -abs(x["diff_tl"])):
            if item["action"] == "HOLD":
                continue
            has_action = True
            if item["action"] == "BUY":
                st.success(
                    f"🟢 **{item['name']}** ({item['asset']}) fonuna "
                    f"**{format_tl(abs(item['diff_tl']))}** ekle\n\n"
                    f"Şu an: %{item['current_w']*100:.0f} → Hedef: %{item['target_w']*100:.0f}"
                )
            else:
                st.error(
                    f"🔴 **{item['name']}** ({item['asset']}) fonundan "
                    f"**{format_tl(abs(item['diff_tl']))}** azalt\n\n"
                    f"Şu an: %{item['current_w']*100:.0f} → Hedef: %{item['target_w']*100:.0f}"
                )

        if not has_action:
            st.success("✅ Portföyün şu an dengeli görünüyor, değişiklik gerekmiyor.")

        # === MALİYET BİLGİSİ ===
        cost = cost_model.calculate_rebalance_cost(recommendations, total_value)

        st.divider()
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Tahmini Maliyet", format_tl(cost["total_cost_tl"]))
        cc2.metric("İşlem Sayısı", f"{cost['switch_count']} / 6 (aylık limit)")
        cc3.metric("Portföy Değişimi", f"%{cost['turnover_pct']*100:.0f}")

        if cost["exceeds_monthly_limit"]:
            st.error("⚠️ Aylık fon değişikliği limiti (6) aşılıyor! En önemli değişiklikler önceliklendirildi.")

        st.caption("💡 BES'te fon geçişi ücretsizdir. Maliyet sadece tahmini slippage'dir (%0.2).")

        # === DETAY TABLO (gizli) ===
        with st.expander("📊 Detaylı Portföy Tablosu"):
            df_detail = pd.DataFrame([{
                "Fon":           f"{item['name']} ({item['asset']})",
                "Mevcut":        format_tl(item["current_tl"]),
                "Mevcut %":      f"%{item['current_w']*100:.1f}",
                "Hedef %":       f"%{item['target_w']*100:.1f}",
                "Değişiklik":    format_tl(item["diff_tl"]),
                "İşlem":         action_text(item["action"]),
            } for item in analysis_data])
            st.dataframe(df_detail, hide_index=True, use_container_width=True)

        # === EĞİTİCİ EXPANDER ===
        with st.expander("❓ Bu öneriler ne anlama geliyor?"):
            st.markdown(f"""
**Sistem nasıl çalışıyor?**

1. Piyasa verilerine bakarak ortamı sınıflandırıyoruz:
   şu an **{regime_info['label']}** ({regime_info['emoji']})

2. Her piyasa ortamı için tarihsel olarak en iyi çalışan
   fon dağılımını hesaplıyoruz

3. Senin mevcut dağılımınla hedef dağılımı karşılaştırıp
   "ne kadar al, ne kadar sat" önerisi üretiyoruz

**Fon tipleri nedir?**
- 🏢 **VEF (Hisse Fonu):** Borsa İstanbul hisselerine yatırım. Yüksek risk, yüksek getiri potansiyeli.
- 🥇 **ALT (Altın Fonu):** Altın fiyatına endeksli. Kriz dönemlerinde koruma sağlar.
- 🏛️ **KTS (Kamu Borç.):** Devlet tahvili ve bonosu. Düşük risk, sabit getiri.
- 🔄 **KCH (Karma):** Hisse + tahvil + altın karışımı. Orta risk.
- 💵 **CASH (Para Piy.):** En düşük riskli, mevduat benzeri getiri.

**Önemli:** Bu öneriler kesin yatırım tavsiyesi değildir.
Kendi durumunuza göre değerlendirin.
            """)


# ══════════════════════════════════════════════════════
# TAB 3 — Geçmiş Performans
# ══════════════════════════════════════════════════════
with tab3:
    # === SAYFA AÇIKLAMASI ===
    st.markdown("""
    <div style="background: #fefce8; padding: 15px; border-radius: 10px;
                border-left: 5px solid #eab308; margin-bottom: 20px;">
        <p style="margin: 0; color: #854d0e;">
            📚 <strong>Bu sayfa ne gösteriyor?</strong> Sistemimiz geçmişte nasıl çalışırdı?
            Gerçek piyasa verisiyle geriye dönük test yaparak, önerilerimizin ne kadar
            isabetli olduğunu ölçüyoruz.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # === ÖĞRENME DURUMU ===
    learning = LearningEngineV2()
    stats = learning.get_regime_stats()
    total_obs = sum(s["n"] for s in stats.values())

    if total_obs < 6:
        st.info(
            f"🧠 **Sistem öğrenme aşamasında.** Henüz {total_obs} aylık gözlem var. "
            f"6 aydan sonra sistem kendi geçmiş performansından öğrenmeye başlayacak. "
            f"Şu an sabit (uzman görüşü bazlı) ağırlıklar kullanılıyor."
        )
    else:
        st.success(f"🧠 **Sistem öğreniyor!** {total_obs} aylık gözlem mevcut.")

    with st.expander("📋 Rejim Bazlı Öğrenme İstatistikleri", expanded=False):
        stats_df = pd.DataFrame.from_dict(stats, orient="index")
        stats_df.index.name = "Rejim"
        st.dataframe(stats_df.style.format({
            "win_rate":  "{:.1%}",
            "avg_alpha": "{:.2%}",
            "confidence": "{:.1%}",
        }, na_rep="—"))

    # === ENFLASYONs ETKİSİ ===
    st.divider()
    st.write("### 💰 Enflasyon Etkisi")
    st.markdown("""
Türkiye'de yüksek enflasyon nedeniyle **nominal (görünen) getiri yanıltıcı** olabilir.
Örneğin portföyün %20 kazanmış görünse bile, enflasyon %30 ise **gerçekte %10 kaybetmişsindir**.
Aşağıdaki tablo farklı getiri senaryolarında gerçek (reel) kazancını gösteriyor.
    """)

    macro = result.get("macro", {})
    cpi = macro.get("cpi_yoy")

    if cpi is not None:
        from src.performance_tracker import PerformanceTracker
        tracker = PerformanceTracker()

        scenarios = [0.01, 0.02, 0.03, 0.05, 0.08]
        scenario_data = []
        for nom in scenarios:
            calc = tracker.calculate_real_return(nom, cpi, period_months=1)
            scenario_data.append({
                "Nominal Getiri":  f"%{nom*100:.1f}",
                "Aylık Enflasyon": f"%{calc['inflation_period']*100:.2f}" if calc["inflation_period"] else "N/A",
                "Reel Getiri":     f"%{calc['real_return']*100:.2f}" if calc["real_return"] else "N/A",
                "Enflasyon Etkisi": f"%{calc['inflation_drag']*100:.2f}" if calc["inflation_drag"] else "N/A",
            })

        st.dataframe(pd.DataFrame(scenario_data), hide_index=True)
        monthly_inf = ((1 + cpi) ** (1 / 12) - 1) * 100
        st.caption(
            f"💡 Yıllık TÜFE: %{cpi*100:.1f} | "
            f"Aylık enflasyon etkisi: ~%{monthly_inf:.2f} | "
            f"Enflasyonun üstünde getiri için aylık minimum ~%{monthly_inf:.2f} getiri gerekli."
        )
    else:
        st.info("⚠️ CPI verisi yok — reel getiri hesaplanamıyor. TCMB_API_KEY tanımlı mı?")

    # === BACKTEST BÖLÜMÜ ===
    st.divider()
    st.write("### 🔬 Geriye Dönük Test (Backtest)")
    st.markdown("""
Aşağıdaki test, sistemin **geçmişteki gerçek piyasa verisiyle** ne yapacağını simüle eder.

**Nasıl çalışıyor?**
- Her ayın sonunda piyasayı analiz eder (sadece o güne kadar olan veriyle — hile yok!)
- Rejim belirler ve fon dağılımı önerir
- Bir sonraki ayın gerçek getirisini hesaplar
- Sonunda "AI portföy mü yoksa eşit dağılım mı daha iyi?" sorusunu cevaplar

**Önemli terimler:**
- **Benchmark:** Tüm fonlara eşit dağılım — "hiçbir şey yapmasan ne olurdu?"
- **Alpha (Piyasaya göre fark):** AI portföyünün benchmark'tan farkı (+ iyi, − kötü)
- **Drawdown (Düşüş):** Zirvedeki değerinden ne kadar düştü
- **Sharpe (Risk-getiri dengesi):** Ne kadar risk alarak ne kadar kazanıldı; yüksek = iyi
- **CAGR (Yıllık büyüme):** Yılda ortalama kaç % kazanıldı
    """)

    with st.expander("⚙️ Backtest Ayarları", expanded=False):
        bc1, bc2 = st.columns(2)
        bt_start = bc1.date_input("Başlangıç", value=pd.Timestamp("2024-06-01"))
        bt_end   = bc2.date_input("Bitiş",     value=pd.Timestamp("2026-04-01"))
        run_backtest = st.button("🚀 Backtest Çalıştır", type="primary")

    if run_backtest:
        with st.spinner("Backtest çalışıyor... (yfinance'tan veri çekiliyor, 1-2 dk sürebilir)"):
            from src.backtest_engine import BacktestEngine, BacktestConfig
            bt_config = BacktestConfig(start_date=str(bt_start), end_date=str(bt_end))
            bt_engine = BacktestEngine(bt_config)
            bt_result = bt_engine.run()
            st.session_state.bt_result = bt_result
            st.session_state.bt_engine = bt_engine

    if "bt_result" in st.session_state and st.session_state.bt_result.steps:
        bt_result = st.session_state.bt_result
        bt_engine = st.session_state.bt_engine
        df_bt     = bt_engine.to_dataframe(bt_result)

        # === BASİT YORUM ===
        total_ret = bt_result.total_return
        bench_ret = bt_result.benchmark_total_return
        diff      = total_ret - bench_ret

        if diff > 0.02:
            st.success(
                f"✅ **AI portföy benchmark'ı geçti!** "
                f"AI: %{total_ret*100:.1f} vs Eşit Dağılım: %{bench_ret*100:.1f} "
                f"→ %{diff*100:.1f} daha iyi performans."
            )
        elif diff > -0.02:
            st.info(
                f"🟡 **AI ve benchmark yakın performans gösterdi.** "
                f"AI: %{total_ret*100:.1f} vs Eşit Dağılım: %{bench_ret*100:.1f} "
                f"→ Fark: %{diff*100:.1f}"
            )
        else:
            st.warning(
                f"⚠️ **Bu dönemde benchmark daha iyi performans gösterdi.** "
                f"AI: %{total_ret*100:.1f} vs Eşit Dağılım: %{bench_ret*100:.1f} "
                f"→ %{abs(diff)*100:.1f} geride. Bu normal olabilir — proxy model "
                f"kullanılıyor ve sistem henüz öğrenme aşamasında."
            )

        # === ÖZET METRİKLER ===
        st.write("### 📊 Performans Özeti")
        km1, km2, km3, km4 = st.columns(4)
        km1.metric(
            "AI Toplam Getiri",
            f"%{total_ret*100:.1f}",
            delta=f"%{diff*100:.1f} vs benchmark",
        )
        km2.metric("Yıllık Büyüme (CAGR)", f"%{bt_result.cagr*100:.1f}")
        km3.metric("Risk-Getiri Dengesi (Sharpe)", f"{bt_result.sharpe_ratio:.2f}")
        km4.metric("En Derin Düşüş (Drawdown)", f"%{bt_result.max_drawdown*100:.1f}")
        km5, km6, km7, km8 = st.columns(4)
        km5.metric("Kazanan Ay Oranı", f"%{bt_result.win_rate*100:.0f}")
        km6.metric("Piyasaya Göre Fark (Net)", f"%{bt_result.avg_net_alpha*100:.2f}")
        km7.metric("Toplam Maliyet", f"%{bt_result.total_cost_pct*100:.2f}")
        km8.metric("Test Süresi", f"{bt_result.months_count} ay")

        # --- 1. Equity Curve ---
        st.write("### 📈 Portföy Değeri: AI vs Benchmark")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df_bt.index, y=df_bt["portfolio_value"], name="AI Portföy",
            line=dict(color="#2563eb", width=2.5),
            hovertemplate="Tarih: %{x}<br>Değer: %{y:,.0f} TL<extra></extra>",
        ))
        fig_equity.add_trace(go.Scatter(
            x=df_bt.index, y=df_bt["benchmark_value"], name="Benchmark (Eşit Ağırlık)",
            line=dict(color="#9ca3af", width=2, dash="dash"),
            hovertemplate="Tarih: %{x}<br>Değer: %{y:,.0f} TL<extra></extra>",
        ))
        fig_equity.add_hline(
            y=bt_engine.config.initial_capital, line_dash="dot", line_color="#d1d5db",
            annotation_text=f"Başlangıç: {bt_engine.config.initial_capital:,.0f} TL",
        )
        fig_equity.update_layout(
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Portföy Değeri (TL)", yaxis_tickformat=",",
            hovermode="x unified",
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # --- 2. Rejim Zaman Çizelgesi ---
        st.write("### 🎯 Piyasa Ortamı — Zaman Çizelgesi")
        st.markdown("""
**Rejim ne demek?** Sistem piyasayı 4 kategoride sınıflandırıyor:
- 🔴 **CRISIS:** Sert düşüş dönemi — altın ve nakit ağırlıklı
- 🟢 **RISK_ON:** Yükseliş trendi — hisse ağırlıklı
- 🔵 **STABLE:** Sakin dönem — dengeli dağılım
- 🟠 **RATE_HIKE:** Faiz artışı — tahvil ağırlıklı
        """)
        regime_colors = {
            "CRISIS":    "#ef4444",
            "RATE_HIKE": "#f59e0b",
            "RISK_ON":   "#22c55e",
            "STABLE":    "#3b82f6",
        }
        fig_regime = go.Figure()
        seen = set()
        for i, row in df_bt.iterrows():
            reg   = row["regime"]
            color = regime_colors.get(reg, "#6b7280")
            fig_regime.add_trace(go.Bar(
                x=[i], y=[1], marker_color=color, name=reg,
                showlegend=(reg not in seen),
                hovertemplate=f"Tarih: {i.strftime('%Y-%m')}<br>Rejim: {reg}<br>Güven: {row['confidence']:.0%}<extra></extra>",
            ))
            seen.add(reg)
        fig_regime.update_layout(
            height=120, margin=dict(l=20, r=20, t=10, b=20),
            barmode="stack", bargap=0.05,
            yaxis=dict(visible=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_regime, use_container_width=True)

        # --- 3. Aylık Alpha ---
        st.write("### 📊 Piyasaya Göre Aylık Fark (Alpha)")
        colors_alpha = ["#22c55e" if a > 0 else "#ef4444" for a in df_bt["alpha"]]
        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Bar(
            x=df_bt.index, y=df_bt["alpha"] * 100, marker_color=colors_alpha, name="Alpha",
            hovertemplate="Tarih: %{x}<br>Alpha: %{y:.2f}%<extra></extra>",
        ))
        fig_alpha.add_hline(y=0, line_color="#6b7280", line_width=1)
        fig_alpha.add_hline(
            y=bt_result.avg_alpha * 100, line_dash="dash", line_color="#2563eb",
            annotation_text=f"Ort: {bt_result.avg_alpha*100:.2f}%",
            annotation_position="bottom right",
        )
        fig_alpha.update_layout(
            height=300, margin=dict(l=20, r=20, t=10, b=20),
            yaxis_title="Fark (%)", hovermode="x unified",
        )
        st.plotly_chart(fig_alpha, use_container_width=True)

        # --- 4. Drawdown ---
        st.write("### 📉 Zirveden Düşüş (Drawdown)")
        pf_returns = [s.portfolio_return - s.rebalance_cost_pct for s in bt_result.steps]
        eq = [bt_engine.config.initial_capital]
        for r in pf_returns:
            eq.append(eq[-1] * (1 + r))
        eq_s = pd.Series(eq[1:], index=df_bt.index)
        dd   = (eq_s - eq_s.expanding().max()) / eq_s.expanding().max() * 100

        bq = [bt_engine.config.initial_capital]
        for s in bt_result.steps:
            bq.append(bq[-1] * (1 + s.benchmark_return))
        bq_s = pd.Series(bq[1:], index=df_bt.index)
        bdd  = (bq_s - bq_s.expanding().max()) / bq_s.expanding().max() * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)", line=dict(color="#ef4444", width=2),
            name="AI Düşüşü",
            hovertemplate="Tarih: %{x}<br>Düşüş: %{y:.2f}%<extra></extra>",
        ))
        fig_dd.add_trace(go.Scatter(
            x=bdd.index, y=bdd.values,
            line=dict(color="#9ca3af", width=1.5, dash="dash"),
            name="Benchmark Düşüşü",
            hovertemplate="Tarih: %{x}<br>Düşüş: %{y:.2f}%<extra></extra>",
        ))
        fig_dd.update_layout(
            height=250, margin=dict(l=20, r=20, t=10, b=20),
            yaxis_title="Düşüş (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # --- 5. Rejim Bazlı Performans ---
        st.write("### 🏷️ Her Piyasa Ortamında Performans")
        regime_stats = {}
        for reg in regime_colors:
            rs = [s for s in bt_result.steps if s.regime == reg]
            if not rs:
                continue
            regime_stats[reg] = {
                "Ay Sayısı":  len(rs),
                "Ort. Getiri": f"%{sum(s.portfolio_return for s in rs)/len(rs)*100:.2f}",
                "Ort. Fark":  f"%{sum(s.alpha for s in rs)/len(rs)*100:.2f}",
                "Kazanma Oranı": f"%{sum(1 for s in rs if s.alpha > 0)/len(rs)*100:.0f}",
                "Ort. Güven": f"%{sum(s.confidence for s in rs)/len(rs)*100:.0f}",
            }
        if regime_stats:
            st.dataframe(
                pd.DataFrame.from_dict(regime_stats, orient="index"),
                use_container_width=True,
            )

        # --- 6. Aylık Detay ---
        with st.expander("📋 Aylık Detay Tablosu", expanded=False):
            detail_df = df_bt[["regime", "confidence", "portfolio_return", "benchmark_return",
                               "alpha", "net_alpha", "cost_pct", "portfolio_value"]].copy()
            detail_df.columns = ["Rejim", "Güven", "Getiri", "Benchmark", "Fark (Alpha)",
                                 "Net Fark", "Maliyet", "Portföy Değeri"]
            st.dataframe(detail_df.style.format({
                "Güven":         "{:.0%}",
                "Getiri":        "{:+.2%}",
                "Benchmark":     "{:+.2%}",
                "Fark (Alpha)":  "{:+.2%}",
                "Net Fark":      "{:+.2%}",
                "Maliyet":       "{:.3%}",
                "Portföy Değeri": "{:,.0f} TL",
            }), use_container_width=True)

    elif "bt_result" not in st.session_state:
        st.info("⬆️ Yukarıdaki 'Backtest Çalıştır' butonuna tıkla.")
