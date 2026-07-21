import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# EVDS_API_KEY → TCMB_API_KEY'den fallback (evdspy için)
if not os.environ.get("EVDS_API_KEY") and os.environ.get("TCMB_API_KEY"):
    os.environ["EVDS_API_KEY"] = os.environ["TCMB_API_KEY"]

from src.logging_config import configure_logging, get_logger
from src.pipeline import MonthlyPipeline
from src.performance_tracker import PerformanceTracker


def _format_candidate_funds(candidates: list) -> str:
    """BUY onerisi icin aday fonlari tek satirda formatla (en fazla 3, '·' ile).
    Ornek: "KTB (AI 3A:+%12.3 | 1Y:%48.2) · AEK (1Y:%44.0)". predicted_3m None
    ise yalniz 1Y; held=True ise kodun yanina '(elinde)' eklenir."""
    parts = []
    for c in (candidates or [])[:3]:
        code = str(c.get("fund_code", "?"))
        if c.get("held"):
            code += " (elinde)"
        segs = []
        p3 = c.get("predicted_3m")
        if p3 is not None:
            p = p3 * 100
            segs.append(f"AI 3A:{'+' if p >= 0 else '-'}%{abs(p):.1f}")
        r1y = c.get("return_1y")
        if r1y is not None:
            segs.append(f"1Y:{'-' if r1y < 0 else ''}%{abs(r1y):.1f}")
        parts.append(f"{code} ({' | '.join(segs)})" if segs else code)
    return " · ".join(parts)


def print_summary(result: dict, narrative: Optional[str] = None) -> None:
    """Insan-okunabilir ozet (stdout'a)."""
    if result.get("status") != "SUCCESS":
        print(f"\nHATA: {result.get('message', 'Bilinmeyen hata')}\n")
        return

    pv  = result["portfolio_value"]
    rg  = result["regime"]
    rec = result["recommendation"]
    dq  = rg.get("data_quality", {})

    print()
    print("=" * 64)
    print("BES AI - Aylik Analiz Raporu")
    print("=" * 64)
    if narrative:
        print("Özet         :")
        for _line in narrative.splitlines():
            print(f"   {_line}")
        print("-" * 64)
    print(f"Tarih        : {result['run_date'][:10]}")
    print(f"Toplam Deger : {pv['total_value']:>14,.2f} TL")
    print(f"Rejim        : {rg['detected']} (guven: {rg['confidence']:.1%})")
    print(f"Veri Kalite  : {dq.get('rows_count', '?')} gun, "
          f"%{dq.get('missing_pct', 0)*100:.1f} eksik, "
          f"as_of: {dq.get('as_of', '?')}")
    macro = rg.get("macro", {})
    if macro and macro.get("current_policy_rate"):
        rate_change_pp = macro.get("tcmb_rate_change", 0) * 100
        print(f"TCMB         : faiz=%{macro['current_policy_rate']:.2f} "
              f"({rate_change_pp:+.2f}pp/30g), "
              f"TÜFE=%{(macro.get('cpi_yoy', 0) or 0)*100:.1f}")

    sig = result.get("significance")
    if sig:
        _reasons = "; ".join(sig.get("reasons", [])) or "sakin ay"
        print(f"Önemlilik    : {sig['score']}/100 ({sig['level']}) — {_reasons}")

    sc = result.get("state_contribution")
    if sc:
        if sc.get("at_cap"):
            print(
                f"Devlet Katkısı: yılda {sc['annual_match']:,.0f} TL "
                f"(tavan {sc['max_annual_match']:,.0f}) — tavandasın"
            )
        else:
            print(
                f"Devlet Katkısı: yılda {sc['annual_match']:,.0f} TL "
                f"(tavan {sc['max_annual_match']:,.0f}) — "
                f"ayda {sc['suggested_extra_monthly']:,.0f} TL daha koy, "
                f"{sc['match_gap']:,.0f} TL kaçırıyorsun"
            )

    if result.get("previous_evaluation"):
        ev = result["previous_evaluation"]
        status_map = {"WIN": "[WIN]", "LOSS": "[LOSS]", "NEUTRAL": "[NEUTRAL]"}
        tag = status_map.get(ev["status"], "[?]")
        print()
        print(f"{tag} Gecen Ay    : {ev['status']}")
        print(f"   Getiri     : {ev['monthly_return']:+.2%}")
        print(f"   BIST Bench : {ev['benchmark_return']:+.2%}")
        if ev.get("net_alpha") is not None and ev.get("rebalance_cost_pct", 0) > 0:
            print(f"   Brut Alpha : {ev.get('gross_alpha', 0):+.2%}")
            print(f"   Maliyet    : -{ev['rebalance_cost_pct']:.4%}")
            print(f"   Net Alpha  : {ev['net_alpha']:+.2%}")
        else:
            print(f"   Alpha      : {ev['alpha_vs_benchmark']:+.2%}")
        real = ev.get("real_metrics")
        if real and real.get("real_return") is not None:
            print(f"   Reel Getiri: {real['real_return']:+.2%} "
                  f"(enflasyon: {real['inflation_period']:+.2%}/ay)")

    real_pf = result.get("real_portfolio")
    if real_pf and real_pf.get("real_total_return") is not None:
        print()
        print(f"Reel Durum   : {real_pf['months_elapsed']} ayda "
              f"nominal {real_pf['nominal_total_return']:+.2%}, "
              f"reel {real_pf['real_total_return']:+.2%}")
        if real_pf.get("real_value"):
            print(f"   Reel Deger : {real_pf['real_value']:>14,.2f} TL "
                  f"(satin alma gucu bazinda)")

    print()
    print("Aksiyonlar:")
    has_action = False
    for action in rec["actions"]:
        if action["action"] == "HOLD":
            continue
        has_action = True
        sign = "+" if action["diff_tl"] > 0 else ""
        tag  = "[AL ]" if action["action"] == "BUY" else "[SAT]"
        print(f"   {tag} {action['asset']:8s}: "
              f"{sign}{action['diff_tl']:>12,.0f} TL "
              f"(su an: %{action['current_weight']*100:.1f} -> "
              f"hedef: %{action['target_weight']*100:.1f})")
        if action["action"] == "BUY":
            cand_line = _format_candidate_funds(action.get("candidate_funds"))
            if cand_line:
                print(f"          → aday: {cand_line}")

    if not has_action:
        print("   Portfoy hedefe yakin, aksiyon gerekmiyor")

    cost = rec.get("cost_analysis")
    if cost:
        print(f"\nMaliyet      : {cost['total_cost_tl']:,.2f} TL "
              f"(%{cost['total_cost_pct']*100:.3f}), "
              f"turnover %{cost['turnover_pct']*100:.1f}, "
              f"{cost['switch_count']} islem")
        if cost["exceeds_monthly_limit"]:
            print("   UYARI: Aylik limit asimi! Bazi islemler ertelendi.")

    print()
    print(f"Snapshot     : {result.get('snapshot_path', '?')}")

    tracker = PerformanceTracker()
    history = tracker.get_portfolio_history()
    if len(history) >= 2:
        first = history.iloc[0]
        last  = history.iloc[-1]
        ret   = (last["total_value"] / first["total_value"]) - 1
        print(f"📈 Geçmiş     : {len(history)} ay takip, "
              f"{first['total_value']:,.0f} → {last['total_value']:,.0f} TL "
              f"({ret:+.1%})")

    from src.data_health import check_data_health
    _health = check_data_health()
    if not _health["ok"]:
        print("\n⚠️ Veri Sağlık Uyarıları:")
        for w in _health["warnings"]:
            print(f"   - {w}")

    print("=" * 64)
    print()


def run_all_portfolios(args, logger):
    """Tum portfoyleri sirayla kosar. Returns: [{slug, name, result}, ...]."""
    from src.portfolio_manager import PortfolioManager
    from src.pipeline import MonthlyPipeline
    pm = PortfolioManager()
    portfolios = pm.list_portfolios()
    if not portfolios:
        logger.error("Hicbir portfoy bulunamadi (data/portfolios/)")
        return []
    results = []
    for p in portfolios:
        slug = p["slug"]
        pf_path = f"data/portfolios/{slug}.json"
        logger.info(f"=== Portfoy: {p['name']} ({slug}) ===")
        try:
            pipeline = MonthlyPipeline(
                portfolio_path=pf_path,
                history_dir=f"data/history/{slug}",
                learning_path=f"data/history/{slug}/learning_history.json",
            )
            result = pipeline.run()
        except Exception as e:
            logger.exception(f"Portfoy hatasi ({slug}): {e}")
            result = {"status": "ERROR", "message": str(e)}
        results.append({"slug": slug, "name": p["name"], "result": result})
    return results


def _send_combined_email(all_results, args, logger):
    try:
        from src.email_notifier import EmailNotifier
        notifier = EmailNotifier()
        ok = notifier.send_multi_portfolio_report(all_results)
        if ok:
            print(f"\n📧 Birlesik e-posta gonderildi: {', '.join(notifier.recipients)}")
    except Exception as e:
        logger.error(f"Birlesik e-posta hatasi: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="BES AI Aylik Pipeline - Otomatik portfoy analizi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python main.py                     # Normal calistirma
  python main.py --verbose           # Detayli log
  python main.py --quiet             # Sadece hata goster
  python main.py --json > rapor.json # JSON ciktisi dosyaya
  python main.py --portfolio data/test.json
  python main.py --all-portfolios    # Tum portfoyleri kos
        """,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG seviyesi log")
    parser.add_argument("--quiet",   "-q", action="store_true", help="Sadece WARNING+ log")
    parser.add_argument("--json",          action="store_true", help="Sadece JSON ciktisi")
    parser.add_argument("--portfolio", default="data/my_portfolio.json", help="Portfoy JSON yolu")
    parser.add_argument("--all-portfolios", action="store_true",
                        help="data/portfolios/ altindaki TUM portfoyleri sirayla kosar")
    parser.add_argument("--backtest",      action="store_true", help="Walk-forward backtest calistir")
    parser.add_argument("--bt-start", default="2024-01-01", help="Backtest baslangic (YYYY-MM-DD)")
    parser.add_argument("--bt-end",   default=None, help="Backtest bitis (YYYY-MM-DD, varsayilan: icinde bulunulan ayin 1'i)")
    parser.add_argument("--learn-from-backtest", action="store_true",
                        help="Backtest sonuclarini learning history'ye yaz ve ogrenilmis agirliklarla tekrar backtest calistir")
    parser.add_argument("--ml-train",      action="store_true", help="ML model egitimi calistir (TEFAS verisiyle)")
    parser.add_argument("--ml-funds", type=int, default=None,
                        help="Kac fon ile egitim (None=POPULAR_BES_FUNDS, -1=TEFAS'taki tum EMK, sayi=ilk N)")
    parser.add_argument("--ml-12m", action="store_true",
                        help="12 aylik uzun vadeli model de egit (3M + 12M)")
    parser.add_argument("--email", action="store_true",
                        help="Sonuçları e-posta ile gönder")
    parser.add_argument("--email-full", action="store_true",
                        help="Sessiz ayda bile tam rapor gönder (önemlilik eşiğini atla)")
    parser.add_argument("--test-email", action="store_true",
                        help="Test e-postası gönder (yapılandırma kontrolü)")
    parser.add_argument("--report", action="store_true",
                        help="Aylik PDF rapor uret")
    parser.add_argument("--narrative", action="store_true",
                        help="Rapora LLM anlati ozeti ekle (ANTHROPIC_API_KEY gerekir; yoksa sablon)")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Telegram uyarisi gonderme (varsayilan: yapilandirilmissa gonder)")
    parser.add_argument("--backup", action="store_true",
                        help="Kisisel veriyi zip yedekle ve cik (data/backups veya BES_BACKUP_DIR)")
    args = parser.parse_args()

    if args.quiet:
        log_level = "WARNING"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    # JSON modunda console'u sustur (stdout temiz JSON icin)
    configure_logging(
        log_file="pipeline.log",
        level=log_level,
        quiet_console=args.json,
    )

    logger = get_logger(__name__)
    logger.info(f"main.py basladi (verbose={args.verbose}, json={args.json})")

    # PLAN-24: yalniz yedek alip cik (ml_train/backtest'ten once)
    if args.backup:
        from src.backup import backup_personal_data
        p = backup_personal_data()
        print(f"Yedek: {p}" if p else "Yedek alınamadı")
        sys.exit(0 if p else 1)

    if args.ml_train:
        from src.ml_pipeline import MLPipeline
        from src.data_collector import POPULAR_BES_FUNDS

        logger.info("ML egitim modu baslatiliyor...")
        ml = MLPipeline()

        fund_codes = None
        if args.ml_funds == -1:
            fund_codes = None  # nav_history'deki tum fonlar
            logger.info("Tum fonlar: nav_history.parquet icindeki tum EMK fonlari kullanilacak")
        elif args.ml_funds:
            fund_codes = list(POPULAR_BES_FUNDS.keys())[: args.ml_funds]
            logger.info(f"Test modu: sadece {len(fund_codes)} fon ({fund_codes})")

        targets = ["fwd_return_3m"]
        if args.ml_12m:
            targets.append("fwd_return_12m")

        result = ml.run_full_pipeline(fund_codes=fund_codes, targets=targets)

        if args.json:
            output = {k: v for k, v in result.items() if k != "predictions"}
            if "predictions" in result and not result["predictions"].empty:
                output["predictions"] = result["predictions"].to_dict(orient="records")
            print(json.dumps(output, indent=2, default=str))
        else:
            if result.get("status") == "SUCCESS":
                print()
                print("=" * 60)
                print("ML EGITIM SONUCLARI")
                print("=" * 60)
                print(f"Fon sayisi      : {result['fund_count']}")
                print(f"Dataset         : {result['dataset_shape']}")
                print(f"En iyi model    : {result['best_model']}")
                print(f"IC              : {result['best_ic']:.3f}")
                print(f"MAE             : {result['best_mae']:.4f}")
                print(f"DirAcc          : {result['best_dir_acc']:.1%}")
                print(f"Sure            : {result['run_time_sec']}s")

                preds = result.get("predictions")
                if preds is not None and not preds.empty:
                    pred_col = "predicted_fwd_return_3m"
                    print(f"\nTahmin sayisi   : {result['predictions_count']}")
                    print("\nEn iyi 5 fon (3M tahmini):")
                    for _, row in preds.head(5).iterrows():
                        val = row.get(pred_col, 0) * 100
                        print(f"  {row['fund_code']:8s}: %{val:+.1f}")

                if result.get("top_features"):
                    print("\nTop 5 Feature:")
                    for i, (feat, imp) in enumerate(list(result["top_features"].items())[:5]):
                        print(f"  {i+1}. {feat}: {imp:.4f}")

                print("=" * 60)
            else:
                print(f"\nML Pipeline hatasi: {result.get('message', '?')}")

        sys.exit(0 if result.get("status") == "SUCCESS" else 1)

    if args.backtest:
        from src.backtest_engine import BacktestEngine, BacktestConfig
        bt_config = BacktestConfig(start_date=args.bt_start, **({"end_date": args.bt_end} if args.bt_end else {}))
        bt_engine = BacktestEngine(bt_config)
        logger.info(f"Backtest modu: {args.bt_start} -> {bt_config.end_date}")
        bt_result = bt_engine.run()
        if args.json:
            output = {
                "type": "backtest",
                "months": bt_result.months_count,
                "total_return": bt_result.total_return,
                "benchmark_total_return": bt_result.benchmark_total_return,
                "cagr": bt_result.cagr,
                "sharpe": bt_result.sharpe_ratio,
                "max_drawdown": bt_result.max_drawdown,
                "win_rate": bt_result.win_rate,
                "avg_alpha": bt_result.avg_alpha,
                "avg_net_alpha": bt_result.avg_net_alpha,
                "steps": [
                    {
                        "date": s.date,
                        "regime": s.regime,
                        "portfolio_return": s.portfolio_return,
                        "benchmark_return": s.benchmark_return,
                        "alpha": s.alpha,
                        "portfolio_value": s.portfolio_value,
                    }
                    for s in bt_result.steps
                ],
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            print(bt_engine.print_summary(bt_result))

        if args.learn_from_backtest:
            # Adim 1: Backtest sonuclarini learning history'ye yaz
            n_new = bt_engine.export_to_learning_history(bt_result)
            print(f"\n📝 {n_new} gozlem learning_history.json'a yazildi")

            # Adim 2: Ogrenilmis agirliklarla yeniden backtest
            print("\n🔄 Ogrenilmis agirliklarla yeniden backtest calistiriliyor...\n")
            bt_config_v2 = BacktestConfig(
                start_date=args.bt_start,
                use_learning=True,
                **({"end_date": args.bt_end} if args.bt_end else {}),
            )
            bt_engine_v2 = BacktestEngine(bt_config_v2)
            bt_result_v2 = bt_engine_v2.run()

            if not args.json:
                # Adim 3: Karsilastirma tablosu
                print("\n" + "=" * 64)
                print("OGRENME ONCESI vs SONRASI KARSILASTIRMA")
                print("=" * 64)
                print(f"{'':25s} {'Statik':>12s} {'Ogrenilmis':>12s} {'Fark':>10s}")
                print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")

                pct_metrics = [
                    ("Toplam Getiri", bt_result.total_return, bt_result_v2.total_return),
                    ("CAGR", bt_result.cagr, bt_result_v2.cagr),
                    ("Max Drawdown", bt_result.max_drawdown, bt_result_v2.max_drawdown),
                    ("Win Rate", bt_result.win_rate, bt_result_v2.win_rate),
                    ("Ort. Alpha", bt_result.avg_alpha, bt_result_v2.avg_alpha),
                ]
                for name, before, after in pct_metrics:
                    diff = after - before
                    sign = "+" if diff > 0 else ""
                    print(f"{name:25s} {before:>12.2%} {after:>12.2%} {sign}{diff:>9.2%}")

                s1, s2 = bt_result.sharpe_ratio, bt_result_v2.sharpe_ratio
                sd = s2 - s1
                sign = "+" if sd > 0 else ""
                print(f"{'Sharpe':25s} {s1:>12.2f} {s2:>12.2f} {sign}{sd:>9.2f}")

                print("=" * 64)

        sys.exit(0)

    if args.test_email:
        from src.email_notifier import EmailNotifier
        notifier = EmailNotifier()
        if notifier.send_test_email():
            print("✅ Test e-postası gönderildi!")
        else:
            print("❌ Test e-postası gönderilemedi. .env ayarlarını kontrol et.")
        sys.exit(0)

    # TEFAS cache kontrolü — pipeline öncesi
    try:
        from src.data_collector import TEFASCollector as _TC_MAIN
        _tc_main = _TC_MAIN()
        if _tc_main.is_cache_stale(max_age_days=7):
            logger.info("TEFAS cache eski — otomatik güncelleniyor...")
            _tc_main.auto_refresh_cache(max_age_days=7)
        # Gercek gunluk NAV gecmisini artimli guncelle (backtest + piyasa getirisi
        # icin). Genelde 1-2 pencere; her ay calistikca gercek gecmis birikir.
        try:
            _added = _tc_main.update_nav_history()
            if _added:
                logger.info(f"nav_history: +{_added} satir eklendi")
        except Exception as _e2:
            logger.warning(f"nav_history güncelleme hatası: {_e2}")
    except Exception as _e:
        logger.warning(f"TEFAS cache kontrol hatası: {_e}")

    # PLAN-18: tum portfoyleri sirayla kos (tek-portfoy yolundan once)
    if args.all_portfolios:
        all_results = run_all_portfolios(args, logger)
        ok = sum(1 for r in all_results if r["result"].get("status") == "SUCCESS")
        if args.json:
            print(json.dumps(
                [{"slug": r["slug"], "name": r["name"], "result": r["result"]}
                 for r in all_results], ensure_ascii=False, indent=2, default=str))
        elif not args.quiet:
            for r in all_results:
                print(f"\n{'#'*64}\n# {r['name']} ({r['slug']})\n{'#'*64}")
                print_summary(r["result"])
        # E-posta: tum portfoyleri tek mailde
        if args.email:
            _send_combined_email(all_results, args, logger)
        # PLAN-23: Telegram birlesik uyari (e-postadan bagimsiz)
        if not args.no_telegram:
            try:
                from src.telegram_notifier import TelegramNotifier, build_multi_message
                _tg_msg = build_multi_message(all_results)
                if _tg_msg and TelegramNotifier().send(_tg_msg):
                    if not args.json:
                        print("\n📱 Telegram uyarısı gönderildi")
            except Exception as e:
                logger.warning(f"Telegram hatasi: {e}")
        sys.exit(0 if ok == len(all_results) and all_results else 1)

    try:
        pipeline = MonthlyPipeline(portfolio_path=args.portfolio)
        result = pipeline.run()
    except Exception as e:
        logger.exception(f"Pipeline beklenmedik hata: {e}")
        result = {"status": "ERROR", "message": f"Beklenmedik hata: {e}"}

    # LLM/sablon anlati ozeti (yalniz --narrative veya --email istenirse uret)
    _narrative = None
    if (args.narrative or args.email) and result.get("status") == "SUCCESS":
        try:
            from src.narrative import generate_narrative
            _ml_for_narr = None
            _ml_p = Path("data/ml/latest_run_summary.json")
            if _ml_p.exists():
                with open(_ml_p) as _f:
                    _ml_for_narr = json.load(_f)
            _narrative = generate_narrative(result, _ml_for_narr)
        except Exception as e:
            logger.warning(f"Anlati uretilemedi: {e}")

    if args.json:
        # stdout'a sadece JSON
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif not args.quiet:
        # Normal mod: insan-okunabilir ozet
        print_summary(result, narrative=_narrative)
    # quiet modda: hicbir sey basma, sadece log dosyasina yazildi

    # PDF rapor
    if args.report and result.get("status") == "SUCCESS":
        try:
            from src.report_generator import ReportGenerator
            import pandas as pd

            gen = ReportGenerator()

            ml_summary_path = Path("data/ml/latest_run_summary.json")
            ml_summary = None
            if ml_summary_path.exists():
                with open(ml_summary_path) as f:
                    ml_summary = json.load(f)

            pred_files = sorted(Path("data/ml").glob("predictions_fwd_return_3m_*.csv"))
            predictions_df = None
            if pred_files:
                predictions_df = pd.read_csv(pred_files[-1])

            pdf_path = gen.generate(
                pipeline_result=result,
                ml_summary=ml_summary,
                predictions_df=predictions_df,
            )

            if pdf_path:
                print(f"\nPDF Rapor: {pdf_path}")
        except ImportError:
            logger.warning("reportlab yuklu degil, PDF uretilemiyor: pip install reportlab")
        except Exception as e:
            logger.error(f"PDF uretim hatasi: {e}")

    if args.email and result.get("status") == "SUCCESS":
        try:
            from src.email_notifier import EmailNotifier

            notifier   = EmailNotifier()
            ml_summary = None
            ml_path    = Path("data/ml/latest_run_summary.json")
            if ml_path.exists():
                with open(ml_path) as f:
                    ml_summary = json.load(f)

            pdf_files = sorted(Path("data/reports").glob("BES_AI_Rapor_*.pdf"))
            pdf_path  = str(pdf_files[-1]) if pdf_files else None

            if notifier.send_monthly_report(
                result, pdf_path, ml_summary,
                significance=result.get("significance"),
                force_full=args.email_full,
                narrative=_narrative,
            ):
                print(f"\n📧 E-posta gönderildi: {', '.join(notifier.recipients)}")
            else:
                print("\n⚠️ E-posta gönderilemedi. Logs'a bak.")
        except Exception as e:
            logger.error(f"E-posta hatası: {e}")

    # PLAN-23: Telegram uyari (e-postadan bagimsiz; yalniz SUCCESS)
    if result.get("status") == "SUCCESS" and not args.no_telegram:
        try:
            from src.telegram_notifier import TelegramNotifier, build_alert_message
            _tg_msg = build_alert_message(result)
            if _tg_msg and TelegramNotifier().send(_tg_msg):
                if not args.json:
                    print("\n📱 Telegram uyarısı gönderildi")
        except Exception as e:
            logger.warning(f"Telegram hatasi: {e}")

    sys.exit(0 if result.get("status") == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
