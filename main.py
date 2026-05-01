import argparse
import json
import sys

from src.logging_config import configure_logging, get_logger
from src.pipeline import MonthlyPipeline


def print_summary(result: dict) -> None:
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
    print("=" * 64)
    print()


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
        """,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG seviyesi log")
    parser.add_argument("--quiet",   "-q", action="store_true", help="Sadece WARNING+ log")
    parser.add_argument("--json",          action="store_true", help="Sadece JSON ciktisi")
    parser.add_argument("--portfolio", default="data/my_portfolio.json", help="Portfoy JSON yolu")
    parser.add_argument("--backtest",      action="store_true", help="Walk-forward backtest calistir")
    parser.add_argument("--bt-start", default="2024-01-01", help="Backtest baslangic (YYYY-MM-DD)")
    parser.add_argument("--bt-end",   default="2026-04-01", help="Backtest bitis (YYYY-MM-DD)")
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

    if args.backtest:
        from src.backtest_engine import BacktestEngine, BacktestConfig
        bt_config = BacktestConfig(start_date=args.bt_start, end_date=args.bt_end)
        bt_engine = BacktestEngine(bt_config)
        logger.info(f"Backtest modu: {args.bt_start} -> {args.bt_end}")
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
        sys.exit(0)

    try:
        pipeline = MonthlyPipeline(portfolio_path=args.portfolio)
        result = pipeline.run()
    except Exception as e:
        logger.exception(f"Pipeline beklenmedik hata: {e}")
        result = {"status": "ERROR", "message": f"Beklenmedik hata: {e}"}

    if args.json:
        # stdout'a sadece JSON
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif not args.quiet:
        # Normal mod: insan-okunabilir ozet
        print_summary(result)
    # quiet modda: hicbir sey basma, sadece log dosyasina yazildi

    sys.exit(0 if result.get("status") == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
