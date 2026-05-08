import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("reportlab yüklü değil: pip install reportlab")


class ReportGenerator:
    """
    Aylık BES portföy analiz raporu üretici.

    Kullanım:
        gen = ReportGenerator()
        gen.generate(pipeline_result, ml_summary, output_path)
    """

    PRIMARY   = HexColor("#1e40af") if HAS_REPORTLAB else None
    SECONDARY = HexColor("#6b7280") if HAS_REPORTLAB else None
    SUCCESS   = HexColor("#16a34a") if HAS_REPORTLAB else None
    DANGER    = HexColor("#dc2626") if HAS_REPORTLAB else None
    WARNING   = HexColor("#d97706") if HAS_REPORTLAB else None
    LIGHT_BG  = HexColor("#f8fafc") if HAS_REPORTLAB else None
    WHITE     = HexColor("#ffffff") if HAS_REPORTLAB else None

    def __init__(self):
        if not HAS_REPORTLAB:
            raise ImportError("reportlab gerekli: pip install reportlab")

        self._register_turkish_font()

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _register_turkish_font(self):
        """Türkçe karakterleri destekleyen font kaydet."""
        import os

        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ]

        project_font      = os.path.join(os.path.dirname(__file__), "..", "data", "fonts", "DejaVuSans.ttf")
        project_font_bold = os.path.join(os.path.dirname(__file__), "..", "data", "fonts", "DejaVuSans-Bold.ttf")

        self.font_name      = "Helvetica"
        self.font_name_bold = "Helvetica-Bold"

        try:
            if os.path.exists(project_font):
                pdfmetrics.registerFont(TTFont("DejaVuSans", project_font))
                if os.path.exists(project_font_bold):
                    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", project_font_bold))
                self.font_name      = "DejaVuSans"
                self.font_name_bold = "DejaVuSans-Bold" if os.path.exists(project_font_bold) else "DejaVuSans"
                logger.info("Türkçe font yüklendi: DejaVuSans (proje)")
                return

            for path in font_paths:
                if os.path.exists(path):
                    pdfmetrics.registerFont(TTFont("TurkishFont", path))
                    self.font_name      = "TurkishFont"
                    self.font_name_bold = "TurkishFont"
                    logger.info(f"Türkçe font yüklendi: {path}")
                    return

            logger.warning("Türkçe font bulunamadı, Helvetica kullanılacak (bazı karakterler bozuk görünebilir)")

        except Exception as e:
            logger.warning(f"Font kayıt hatası: {e}")

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name="ReportTitle",
            fontName=self.font_name_bold,
            fontSize=22,
            textColor=self.PRIMARY,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeader",
            fontName=self.font_name_bold,
            fontSize=14,
            textColor=self.PRIMARY,
            spaceBefore=16,
            spaceAfter=8,
        ))
        self.styles.add(ParagraphStyle(
            name="SubHeader",
            fontName=self.font_name_bold,
            fontSize=11,
            textColor=self.SECONDARY,
            spaceBefore=8,
            spaceAfter=4,
        ))
        self.styles.add(ParagraphStyle(
            name="BodyText2",
            fontName=self.font_name,
            fontSize=10,
            leading=14,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name="SmallText",
            fontName=self.font_name,
            fontSize=8,
            textColor=self.SECONDARY,
        ))

    def generate(
        self,
        pipeline_result: Optional[Dict] = None,
        ml_summary: Optional[Dict] = None,
        predictions_df=None,
        output_path: str = "data/reports/",
    ) -> Optional[str]:
        """
        PDF rapor üret.

        pipeline_result : MonthlyPipeline.run() çıktısı
        ml_summary      : data/ml/latest_run_summary.json içeriği
        predictions_df  : ML tahminleri DataFrame

        Returns: oluşturulan PDF dosya yolu
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y_%m_%d")
        filename = output_dir / f"BES_AI_Rapor_{today}.pdf"

        doc = SimpleDocTemplate(
            str(filename),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        story = []

        # === KAPAK ===
        story.append(Spacer(1, 2 * cm))
        story.append(Paragraph("BES Akilli Fon Danismani", self.styles["ReportTitle"]))
        story.append(Paragraph(
            f"Aylik Portfoy Analiz Raporu - {datetime.now().strftime('%B %Y')}",
            self.styles["SubHeader"],
        ))
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=2, color=self.PRIMARY))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            f"Rapor tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            self.styles["SmallText"],
        ))
        story.append(Spacer(1, 1 * cm))

        # === 1. PİYASA ÖZETİ ===
        if pipeline_result and pipeline_result.get("status") == "SUCCESS":
            story.append(Paragraph("1. Piyasa Ozeti", self.styles["SectionHeader"]))

            regime = pipeline_result.get("regime", {})
            detected  = regime.get("detected", "?")
            confidence = regime.get("confidence", 0)

            regime_names = {
                "STABLE":    "Sakin Piyasa",
                "CRISIS":    "Kriz Modu",
                "RISK_ON":   "Yukselis Trendi",
                "RATE_HIKE": "Faiz Artisi Donemi",
            }
            regime_descriptions = {
                "STABLE":    "Piyasalarda belirgin bir yon yok. Dengeli dagilim mantikli.",
                "CRISIS":    "Ciddi dusus veya belirsizlik var. Korunma oncelikli.",
                "RISK_ON":   "Piyasalar yukari yonlu. Hisse agirligini artirma firsati.",
                "RATE_HIKE": "Merkez bankasi faiz artiriyor. Sabit getirili fonlar one cikiyor.",
            }

            story.append(Paragraph(
                f"<b>Piyasa Durumu:</b> {regime_names.get(detected, detected)} "
                f"(Guven: %{confidence * 100:.0f})",
                self.styles["BodyText2"],
            ))
            story.append(Paragraph(
                regime_descriptions.get(detected, ""),
                self.styles["BodyText2"],
            ))

            metrics = regime.get("metrics", {})
            macro   = regime.get("macro", {})

            metrics_data = [
                ["Gosterge", "Deger"],
                ["BIST Drawdown",       f"%{metrics.get('dd', 0) * 100:.2f}"],
                ["Volatilite (Yillik)", f"%{metrics.get('vol', 0) * 100:.2f}"],
                ["USD/TRY Momentum",    f"%{metrics.get('usd_mom', 0) * 100:.2f}"],
            ]
            if macro.get("cpi_yoy"):
                metrics_data.append(["Enflasyon (TUFE Yillik)", f"%{macro['cpi_yoy'] * 100:.1f}"])
            if macro.get("usdtry_official"):
                metrics_data.append(["USD/TRY (TCMB)", f"{macro['usdtry_official']:.2f} TL"])

            metrics_table = Table(metrics_data, colWidths=[8 * cm, 6 * cm])
            metrics_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), self.PRIMARY),
                ("TEXTCOLOR",    (0, 0), (-1, 0), self.WHITE),
                ("FONTNAME",     (0, 0), (-1, -1), self.font_name),
                ("FONTSIZE",     (0, 0), (-1, -1), 10),
                ("ALIGN",        (1, 0), (1, -1), "RIGHT"),
                ("GRID",         (0, 0), (-1, -1), 0.5, self.SECONDARY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [self.WHITE, self.LIGHT_BG]),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 0.5 * cm))

            # === 2. PORTFÖY DURUMU ===
            story.append(Paragraph("2. Portfoy Durumu", self.styles["SectionHeader"]))

            pv    = pipeline_result.get("portfolio_value", {})
            total = pv.get("total_value", 0)
            story.append(Paragraph(
                f"<b>Toplam Portfoy Degeri:</b> {total:,.0f} TL",
                self.styles["BodyText2"],
            ))

            real_pf = pipeline_result.get("real_portfolio")
            if real_pf and real_pf.get("real_total_return") is not None:
                story.append(Paragraph(
                    f"<b>Reel Durum:</b> {real_pf['months_elapsed']} ayda "
                    f"nominal {real_pf['nominal_total_return']:+.2%}, "
                    f"reel {real_pf['real_total_return']:+.2%}",
                    self.styles["BodyText2"],
                ))
                if real_pf.get("real_value"):
                    story.append(Paragraph(
                        f"<b>Reel Deger (satin alma gucu):</b> {real_pf['real_value']:,.0f} TL",
                        self.styles["BodyText2"],
                    ))

            # === 3. REBALANCE ÖNERİLERİ ===
            story.append(Paragraph("3. Bu Ay Yapilmasi Gerekenler", self.styles["SectionHeader"]))

            rec     = pipeline_result.get("recommendation", {})
            actions = rec.get("actions", [])

            asset_names = {
                "VEF":  "Hisse Senedi Fonu",
                "ALT":  "Altin Fonu",
                "KTS":  "Kamu Borc. Fonu",
                "KCH":  "Karma/Degisken Fon",
                "CASH": "Para Piyasasi",
            }

            action_data = [["Fon", "Islem", "Tutar", "Mevcut %", "Hedef %"]]
            has_action = False

            for a in sorted(actions, key=lambda x: -abs(x.get("diff_tl", 0))):
                if a.get("action") == "HOLD":
                    continue
                has_action = True
                action_text = "EKLE" if a["action"] == "BUY" else "AZALT"
                name = asset_names.get(a["asset"], a["asset"])
                action_data.append([
                    f"{name} ({a['asset']})",
                    action_text,
                    f"{abs(a['diff_tl']):,.0f} TL",
                    f"%{a['current_weight'] * 100:.0f}",
                    f"%{a['target_weight'] * 100:.0f}",
                ])

            if not has_action:
                story.append(Paragraph(
                    "Portfoy dengeli, bu ay degisiklik gerekmiyor.",
                    self.styles["BodyText2"],
                ))
            else:
                action_table = Table(action_data, colWidths=[5 * cm, 2.5 * cm, 3 * cm, 2 * cm, 2 * cm])
                action_table.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0), self.PRIMARY),
                    ("TEXTCOLOR",     (0, 0), (-1, 0), self.WHITE),
                    ("FONTNAME",      (0, 0), (-1, -1), self.font_name),
                    ("FONTSIZE",      (0, 0), (-1, -1), 9),
                    ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
                    ("GRID",          (0, 0), (-1, -1), 0.5, self.SECONDARY),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [self.WHITE, self.LIGHT_BG]),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(action_table)

            cost = rec.get("cost_analysis", {})
            if cost:
                story.append(Spacer(1, 0.3 * cm))
                story.append(Paragraph(
                    f"<b>Tahmini Islem Maliyeti:</b> {cost.get('total_cost_tl', 0):,.0f} TL "
                    f"(%{cost.get('total_cost_pct', 0) * 100:.3f}) | "
                    f"Islem Sayisi: {cost.get('switch_count', 0)} / 6",
                    self.styles["SmallText"],
                ))

        # === 4. AI FON TAHMİNLERİ ===
        if ml_summary and ml_summary.get("status") == "SUCCESS":
            story.append(Paragraph("4. AI Fon Tahminleri (3 Aylik)", self.styles["SectionHeader"]))

            story.append(Paragraph(
                f"<b>Model:</b> {ml_summary.get('best_model', '?').upper()} | "
                f"<b>Sinyal Gucu (IC):</b> {ml_summary.get('best_ic', 0):.2f} | "
                f"<b>Yon Dogrulugu:</b> %{ml_summary.get('best_dir_acc', 0) * 100:.0f} | "
                f"<b>Analiz Edilen Fon:</b> {ml_summary.get('fund_count', 0)}",
                self.styles["BodyText2"],
            ))

            if predictions_df is not None and not predictions_df.empty:
                from src.data_collector import POPULAR_BES_FUNDS

                pred_col = "predicted_fwd_return_3m"
                if pred_col in predictions_df.columns:
                    story.append(Paragraph("En Yuksek Tahmini Getiri (Top 10)", self.styles["SubHeader"]))

                    top10    = predictions_df.nlargest(10, pred_col)
                    top_data = [["#", "Fon Kodu", "Fon Adi", "3M Tahmin"]]
                    for i, (_, row) in enumerate(top10.iterrows()):
                        code = row["fund_code"]
                        name = POPULAR_BES_FUNDS.get(code, code)
                        ret  = row[pred_col]
                        top_data.append([str(i + 1), code, name[:30], f"%{ret * 100:+.1f}"])

                    top_table = Table(top_data, colWidths=[1 * cm, 2.5 * cm, 7 * cm, 3 * cm])
                    top_table.setStyle(TableStyle([
                        ("BACKGROUND",    (0, 0), (-1, 0), self.SUCCESS),
                        ("TEXTCOLOR",     (0, 0), (-1, 0), self.WHITE),
                        ("FONTNAME",      (0, 0), (-1, -1), self.font_name),
                        ("FONTSIZE",      (0, 0), (-1, -1), 9),
                        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
                        ("ALIGN",         (3, 0), (3, -1), "RIGHT"),
                        ("GRID",          (0, 0), (-1, -1), 0.5, self.SECONDARY),
                        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [self.WHITE, self.LIGHT_BG]),
                        ("TOPPADDING",    (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]))
                    story.append(top_table)

                    story.append(Spacer(1, 0.3 * cm))
                    story.append(Paragraph("En Dusuk Tahmini Getiri (Alt 5)", self.styles["SubHeader"]))

                    bottom5  = predictions_df.nsmallest(5, pred_col)
                    bot_data = [["Fon Kodu", "Fon Adi", "3M Tahmin"]]
                    for _, row in bottom5.iterrows():
                        code = row["fund_code"]
                        name = POPULAR_BES_FUNDS.get(code, code)
                        ret  = row[pred_col]
                        bot_data.append([code, name[:30], f"%{ret * 100:+.1f}"])

                    bot_table = Table(bot_data, colWidths=[2.5 * cm, 7 * cm, 3 * cm])
                    bot_table.setStyle(TableStyle([
                        ("BACKGROUND",    (0, 0), (-1, 0), self.DANGER),
                        ("TEXTCOLOR",     (0, 0), (-1, 0), self.WHITE),
                        ("FONTNAME",      (0, 0), (-1, -1), self.font_name),
                        ("FONTSIZE",      (0, 0), (-1, -1), 9),
                        ("ALIGN",         (2, 0), (2, -1), "RIGHT"),
                        ("GRID",          (0, 0), (-1, -1), 0.5, self.SECONDARY),
                        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [self.WHITE, self.LIGHT_BG]),
                        ("TOPPADDING",    (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]))
                    story.append(bot_table)

            top_features = ml_summary.get("top_features", {})
            if top_features:
                story.append(Spacer(1, 0.3 * cm))
                feature_names = {
                    "return_1m":  "Son 1 Ay Getirisi",
                    "return_3m":  "Son 3 Ay Getirisi",
                    "return_6m":  "Son 6 Ay Getirisi",
                    "return_12m": "Son 1 Yil Getirisi",
                    "sharpe_6m":  "6 Aylik Risk-Getiri",
                    "vol_6m":     "6 Aylik Oynaklik",
                    "vol_3m":     "3 Aylik Oynaklik",
                }
                active = {k: float(v) for k, v in top_features.items() if float(v) > 0.001}
                if active:
                    story.append(Paragraph("Model Neye Bakiyor?", self.styles["SubHeader"]))
                    for feat, imp in list(active.items())[:5]:
                        name = feature_names.get(feat, feat)
                        bar  = "#" * int(imp * 50)
                        story.append(Paragraph(
                            f"{name}: {bar} ({imp:.1%})",
                            self.styles["SmallText"],
                        ))

        # === 12 AYLIK TAHMİNLER (varsa) ===
        pred_files_12m = sorted(Path("data/ml").glob("predictions_fwd_return_12m_*.csv"))
        if pred_files_12m:
            import pandas as _pd
            pred_12m = _pd.read_csv(pred_files_12m[-1])
            if not pred_12m.empty and "predicted_fwd_return_12m" in pred_12m.columns:
                from src.data_collector import POPULAR_BES_FUNDS as _BES_FUNDS
                story.append(Spacer(1, 0.5 * cm))
                story.append(Paragraph("12 Aylik Uzun Vadeli Tahminler (Top 5)", self.styles["SubHeader"]))

                top5_12m = pred_12m.nlargest(5, "predicted_fwd_return_12m")
                data_12m = [["Fon Kodu", "Fon Adi", "12M Tahmin"]]
                for _, row in top5_12m.iterrows():
                    code = row["fund_code"]
                    name = _BES_FUNDS.get(code, code)
                    ret  = row["predicted_fwd_return_12m"]
                    data_12m.append([code, name[:30], f"%{ret * 100:+.1f}"])

                table_12m = Table(data_12m, colWidths=[2.5 * cm, 7 * cm, 3 * cm])
                table_12m.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0), self.PRIMARY),
                    ("TEXTCOLOR",     (0, 0), (-1, 0), self.WHITE),
                    ("FONTNAME",      (0, 0), (-1, -1), self.font_name),
                    ("FONTSIZE",      (0, 0), (-1, -1), 9),
                    ("ALIGN",         (2, 0), (2, -1), "RIGHT"),
                    ("GRID",          (0, 0), (-1, -1), 0.5, self.SECONDARY),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [self.WHITE, self.LIGHT_BG]),
                    ("TOPPADDING",    (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.append(table_12m)

        # === YASAL UYARI ===
        story.append(Spacer(1, 1 * cm))
        story.append(HRFlowable(width="100%", thickness=1, color=self.SECONDARY))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "<b>Yasal Uyari:</b> Bu rapor yapay zeka destekli otomatik analiz sistemi tarafından "
            "üretilmistir. Yatirim tavsiyesi niteligli taşımaz. Gecmis performans gelecek sonuclari "
            "garanti etmez. Yatirim kararlarinizdan yalnizca siz sorumlusunuz. "
            "Detayli bilgi icin bir yatirim daniSmanina basvurunuz.",
            self.styles["SmallText"],
        ))
        story.append(Paragraph(
            f"BES Akilli Fon Danismani v2.0 - {datetime.now().strftime('%d.%m.%Y')} - 194 test ile dogrulanmis",
            self.styles["SmallText"],
        ))

        try:
            doc.build(story)
            logger.info(f"PDF rapor üretildi: {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"PDF üretim hatasi: {e}")
            return None
