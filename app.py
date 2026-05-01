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

# Sayfa Konfigürasyonu
st.set_page_config(page_title="BES Regime Intelligence v2.0", page_icon="🛡️", layout="wide")

# Streamlit her sayfa yenilemesinde import'ları yeniden çalıştırır
# session_state ile idempotent yapıyoruz
if "logging_configured" not in st.session_state:
    configure_logging(log_file="streamlit.log", level="INFO")
    st.session_state.logging_configured = True

# Stil Ayarları
st.markdown("""<style>.stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}</style>""", unsafe_allow_html=True)

# Başlık
st.title("🛡️ BES Regime Intelligence v2.0")

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

# Analizi ve Portföyü Al
result = get_market_analysis()
regime = result['detected']
metrics = result['metrics']
my_data = load_my_portfolio()

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("⚙️ Sistem")
    if st.button("🔄 Cache Temizle ve Yenile"):
        st.cache_data.clear()
        st.success("Cache temizlendi, sayfa yenileniyor...")
        st.rerun()
    st.caption(f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}")

# --- SEKME YAPISI ---
tab1, tab2, tab3 = st.tabs(["🌍 Mevcut Piyasa İklimi", "🔮 Kişisel Rebalance Önerisi", "📊 Performans"])

with tab1:
    market_status = "🟢 Açık" if is_market_hours() else "🔴 Kapalı"
    st.caption(f"BIST Durumu: {market_status} • Veri yenileme: {get_smart_ttl() // 60} dk")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mevcut Rejim", regime, delta=f"%{result['confidence']*100:.0f} güven")
    m2.metric("BIST Drawdown", f"%{metrics['dd']*100:.2f}")
    m3.metric("BIST Volatilite", f"%{metrics['vol']*100:.2f}")
    m4.metric("USD/TRY Momentum", f"%{metrics['usd_mom']*100:.2f}")
    st.divider()
    st.write("### AI Rejim Karar Matrisi")
    st.bar_chart(pd.DataFrame.from_dict(result['scores'], orient='index', columns=['Skor']))

    st.divider()
    st.write("### 🏛️ TCMB Makro Göstergeler")

    macro = result.get("macro")
    if macro:
        mc1, mc2, mc3, mc4 = st.columns(4)
        rate        = macro.get("current_policy_rate")
        rate_change = macro.get("tcmb_rate_change", 0) * 100
        cpi         = macro.get("cpi_yoy")
        usd         = macro.get("usdtry_official")

        mc1.metric(
            "Politika Faizi",
            f"%{rate:.2f}" if rate else "N/A",
            delta=f"{rate_change:+.2f} pp (30g)" if rate_change else None,
        )
        mc2.metric("TÜFE (Yıllık)", f"%{cpi*100:.1f}" if cpi else "N/A")
        mc3.metric("USD/TRY (TCMB)", f"{usd:.2f}" if usd else "N/A")
        mc4.metric(
            "2Y Tahvil",
            f"%{macro.get('bond_2y'):.2f}" if macro.get("bond_2y") else "N/A",
        )
        as_of = macro.get("data_quality", {}).get("as_of", "?")
        st.caption(f"📅 TCMB verisi: {as_of}")
    else:
        st.info("⚠️ TCMB verisi yok. .env dosyasında TCMB_API_KEY tanımlı mı?")

with tab2:
    if not my_data:
        st.warning("⚠️ 'data/my_portfolio.json' dosyası okunamadı. Lütfen dosyanın varlığını ve formatını kontrol et.")
    else:
        st.subheader("🛠️ Sana Özel Portföy Optimizasyonu")
        
        learning = LearningEngineV2()
        target_weights = learning.get_optimized_weights(regime)
        
        # Hesaplamalar
        holdings = my_data['holdings_tl']
        total_value = sum(holdings.values())
        current_weights = {k: v / total_value for k, v in holdings.items()}
        
        # Karşılaştırma Tablosu Oluşturma
        analysis_data = []
        all_assets = set(list(target_weights.keys()) + list(holdings.keys()))
        
        for asset in all_assets:
            curr_w = current_weights.get(asset, 0)
            target_w = target_weights.get(asset, 0)
            diff_w = target_w - curr_w
            diff_tl = diff_w * total_value
            
            analysis_data.append({
                "Varlık Sınıfı": asset,
                "Mevcut (TL)": holdings.get(asset, 0),
                "Mevcut (%)": f"%{curr_w*100:.1f}",
                "Hedef (%)": f"%{target_w*100:.1f}",
                "Fark (TL)": diff_tl
            })
            
        df_analysis = pd.DataFrame(analysis_data)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"**Toplam Portföy Değeri:** {total_value:,.2f} TL")
            # Fark sütununu renklendirme
            def color_diff(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'
            
            st.dataframe(df_analysis.style.map(color_diff, subset=['Fark (TL)']).format({"Fark (TL)": "{:,.2f} TL", "Mevcut (TL)": "{:,.2f} TL"}))
            
        with c2:
            st.write("### 📝 Ne Yapmalısın?")
            for index, row in df_analysis.iterrows():
                if row['Fark (TL)'] > 100:
                    st.success(f"**AL:** {row['Varlık Sınıfı']} fonuna {row['Fark (TL)']:,.0f} TL ekle.")
                elif row['Fark (TL)'] < -100:
                    st.error(f"**SAT:** {row['Varlık Sınıfı']} fonundan {abs(row['Fark (TL)']):,.0f} TL azalt.")
            
            st.info(f"💡 Portföyün şu an `{regime}` rejimine göre optimize ediliyor.")

        st.divider()
        st.write("### 💰 İşlem Maliyet Tahmini")

        from src.cost_model import TransactionCostModel
        cost_model = TransactionCostModel()

        recs_for_cost = [
            {
                "asset": row["Varlık Sınıfı"],
                "action": "BUY" if row["Fark (TL)"] > 100 else ("SELL" if row["Fark (TL)"] < -100 else "HOLD"),
                "diff_tl": row["Fark (TL)"],
            }
            for _, row in df_analysis.iterrows()
        ]
        cost = cost_model.calculate_rebalance_cost(recs_for_cost, total_value)

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Toplam Maliyet", f"{cost['total_cost_tl']:,.2f} TL")
        cc2.metric("Turnover", f"%{cost['turnover_pct']*100:.1f}")
        cc3.metric("İşlem Sayısı", f"{cost['switch_count']} / {cost_model.config.max_monthly_switches}")

        if cost["exceeds_monthly_limit"]:
            st.error(
                f"⚠️ Aylık fon değişikliği limiti ({cost_model.config.max_monthly_switches}) aşılıyor! "
                f"En büyük {cost_model.config.max_monthly_switches} işlem önceliklendirildi."
            )

        st.caption(
            f"💡 Tahmini slippage: %{cost_model.config.slippage_pct*100:.1f} | "
            f"BES fon geçişi: {'ücretsiz' if cost_model.config.switch_fee_pct == 0 else f'%{cost_model.config.switch_fee_pct*100:.2f}'}"
        )

with tab3:
    learning = LearningEngineV2()
    stats = learning.get_regime_stats()

    st.subheader("📊 Rejim Bazlı Performans Geçmişi")

    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.index.name = "Rejim"
    st.dataframe(stats_df.style.format({
        "win_rate": "{:.1%}",
        "avg_alpha": "{:.2%}",
        "confidence": "{:.1%}",
    }, na_rep="—"))

    total_obs = sum(s["n"] for s in stats.values())
    if total_obs < 6:
        st.warning(
            f"⚠️ Henüz {total_obs} gözlem var. Sistem statik prior'ları kullanıyor. "
            f"Öğrenilmiş ağırlıklar için minimum 6 gözlem/rejim gerekli."
        )
    else:
        st.success(f"✅ Toplam {total_obs} gözlem mevcut. Sistem öğrenmeye başladı.")

    st.info("Performans takibi için her ay sonu portföy değerini kaydetmeyi unutma.")

    st.divider()
    st.subheader("📈 Reel vs Nominal Getiri")

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
                "Nominal Getiri": f"%{nom*100:.1f}",
                "Aylık Enflasyon": f"%{calc['inflation_period']*100:.2f}" if calc["inflation_period"] else "N/A",
                "Reel Getiri": f"%{calc['real_return']*100:.2f}" if calc["real_return"] else "N/A",
                "Enflasyon Etkisi": f"%{calc['inflation_drag']*100:.2f}" if calc["inflation_drag"] else "N/A",
            })

        st.dataframe(pd.DataFrame(scenario_data), hide_index=True)
        monthly_inf = ((1 + cpi) ** (1 / 12) - 1) * 100
        st.caption(
            f"💡 Yıllık TÜFE: %{cpi*100:.1f} | "
            f"Aylık enflasyon etkisi: ~%{monthly_inf:.2f} | "
            f"Enflasyonun üstünde getiri sağlamak için aylık minimum ~%{monthly_inf:.2f} getiri gerekli."
        )
    else:
        st.info("⚠️ CPI verisi yok, reel getiri hesaplanamıyor. .env dosyasında TCMB_API_KEY tanımlı mı?")

    st.divider()
    st.subheader("📈 Backtest Sonuçları")

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
        df_bt = bt_engine.to_dataframe(bt_result)

        # --- Özet metrikler ---
        st.write("### 📊 Performans Özeti")
        km1, km2, km3, km4 = st.columns(4)
        km1.metric("AI Toplam Getiri", f"%{bt_result.total_return*100:.1f}",
                   delta=f"%{(bt_result.total_return - bt_result.benchmark_total_return)*100:.1f} vs benchmark")
        km2.metric("CAGR", f"%{bt_result.cagr*100:.1f}")
        km3.metric("Sharpe", f"{bt_result.sharpe_ratio:.2f}")
        km4.metric("Max Drawdown", f"%{bt_result.max_drawdown*100:.1f}")
        km5, km6, km7, km8 = st.columns(4)
        km5.metric("Win Rate", f"%{bt_result.win_rate*100:.0f}")
        km6.metric("Ort. Alpha (net)", f"%{bt_result.avg_net_alpha*100:.2f}")
        km7.metric("Toplam Maliyet", f"%{bt_result.total_cost_pct*100:.2f}")
        km8.metric("Süre", f"{bt_result.months_count} ay")

        # --- 1. Equity Curve ---
        st.write("### 📈 Equity Curve: AI vs Benchmark")
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
        fig_equity.add_hline(y=bt_engine.config.initial_capital, line_dash="dot",
                             line_color="#d1d5db",
                             annotation_text=f"Başlangıç: {bt_engine.config.initial_capital:,.0f} TL")
        fig_equity.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20),
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                 yaxis_title="Portföy Değeri (TL)", yaxis_tickformat=",",
                                 hovermode="x unified")
        st.plotly_chart(fig_equity, use_container_width=True)

        # --- 2. Rejim Zaman Çizelgesi ---
        st.write("### 🎯 Rejim Zaman Çizelgesi")
        regime_colors = {"CRISIS": "#ef4444", "RATE_HIKE": "#f59e0b",
                         "RISK_ON": "#22c55e", "STABLE": "#3b82f6"}
        fig_regime = go.Figure()
        seen = set()
        for i, row in df_bt.iterrows():
            reg = row["regime"]
            color = regime_colors.get(reg, "#6b7280")
            fig_regime.add_trace(go.Bar(
                x=[i], y=[1], marker_color=color, name=reg,
                showlegend=(reg not in seen),
                hovertemplate=f"Tarih: {i.strftime('%Y-%m')}<br>Rejim: {reg}<br>Güven: {row['confidence']:.0%}<extra></extra>",
            ))
            seen.add(reg)
        fig_regime.update_layout(height=120, margin=dict(l=20, r=20, t=10, b=20),
                                 barmode="stack", bargap=0.05,
                                 yaxis=dict(visible=False),
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_regime, use_container_width=True)

        # --- 3. Aylık Alpha ---
        st.write("### 📊 Aylık Alpha (AI - Benchmark)")
        colors_alpha = ["#22c55e" if a > 0 else "#ef4444" for a in df_bt["alpha"]]
        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Bar(
            x=df_bt.index, y=df_bt["alpha"] * 100, marker_color=colors_alpha, name="Alpha",
            hovertemplate="Tarih: %{x}<br>Alpha: %{y:.2f}%<extra></extra>",
        ))
        fig_alpha.add_hline(y=0, line_color="#6b7280", line_width=1)
        fig_alpha.add_hline(y=bt_result.avg_alpha * 100, line_dash="dash", line_color="#2563eb",
                            annotation_text=f"Ort: {bt_result.avg_alpha*100:.2f}%",
                            annotation_position="bottom right")
        fig_alpha.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20),
                                yaxis_title="Alpha (%)", hovermode="x unified")
        st.plotly_chart(fig_alpha, use_container_width=True)

        # --- 4. Drawdown ---
        st.write("### 📉 Drawdown")
        pf_returns = [s.portfolio_return - s.rebalance_cost_pct for s in bt_result.steps]
        eq = [bt_engine.config.initial_capital]
        for r in pf_returns:
            eq.append(eq[-1] * (1 + r))
        eq_s = pd.Series(eq[1:], index=df_bt.index)
        dd = (eq_s - eq_s.expanding().max()) / eq_s.expanding().max() * 100

        bq = [bt_engine.config.initial_capital]
        for s in bt_result.steps:
            bq.append(bq[-1] * (1 + s.benchmark_return))
        bq_s = pd.Series(bq[1:], index=df_bt.index)
        bdd = (bq_s - bq_s.expanding().max()) / bq_s.expanding().max() * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)", line=dict(color="#ef4444", width=2),
            name="AI Drawdown",
            hovertemplate="Tarih: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))
        fig_dd.add_trace(go.Scatter(
            x=bdd.index, y=bdd.values,
            line=dict(color="#9ca3af", width=1.5, dash="dash"),
            name="Benchmark DD",
            hovertemplate="Tarih: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))
        fig_dd.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=20),
                             yaxis_title="Drawdown (%)",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                             hovermode="x unified")
        st.plotly_chart(fig_dd, use_container_width=True)

        # --- 5. Rejim Bazlı Performans Tablosu ---
        st.write("### 🏷️ Rejim Bazlı Performans")
        regime_stats = {}
        for reg in regime_colors:
            rs = [s for s in bt_result.steps if s.regime == reg]
            if not rs:
                continue
            regime_stats[reg] = {
                "Ay Sayısı": len(rs),
                "Ort. Getiri": f"%{sum(s.portfolio_return for s in rs)/len(rs)*100:.2f}",
                "Ort. Alpha": f"%{sum(s.alpha for s in rs)/len(rs)*100:.2f}",
                "Win Rate": f"%{sum(1 for s in rs if s.alpha > 0)/len(rs)*100:.0f}",
                "Ort. Güven": f"%{sum(s.confidence for s in rs)/len(rs)*100:.0f}",
            }
        if regime_stats:
            st.dataframe(pd.DataFrame.from_dict(regime_stats, orient="index"),
                         use_container_width=True)

        # --- 6. Aylık Detay Tablosu ---
        with st.expander("📋 Aylık Detay Tablosu", expanded=False):
            detail_df = df_bt[["regime", "confidence", "portfolio_return", "benchmark_return",
                               "alpha", "net_alpha", "cost_pct", "portfolio_value"]].copy()
            detail_df.columns = ["Rejim", "Güven", "Getiri", "Benchmark", "Alpha",
                                 "Net Alpha", "Maliyet", "Portföy Değeri"]
            st.dataframe(detail_df.style.format({
                "Güven": "{:.0%}", "Getiri": "{:+.2%}", "Benchmark": "{:+.2%}",
                "Alpha": "{:+.2%}", "Net Alpha": "{:+.2%}", "Maliyet": "{:.3%}",
                "Portföy Değeri": "{:,.0f} TL",
            }), use_container_width=True)

    elif "bt_result" not in st.session_state:
        st.info("⬆️ Yukarıdaki 'Backtest Çalıştır' butonuna tıkla.")