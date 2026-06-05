import logging
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class EmailNotifier:
    """
    Aylık BES AI rapor e-postası gönderici.

    Gmail SMTP kullanır. Ayarlar .env'den okunur:
    - EMAIL_SENDER: gönderen e-posta (Gmail)
    - EMAIL_PASSWORD: Gmail App Password (normal şifre DEĞİL)
    - EMAIL_RECIPIENTS: alıcı e-postalar (virgülle ayrılmış)

    Gmail App Password almak için:
    1. Google Account → Security → 2-Step Verification (aktif olmalı)
    2. App passwords → "Mail" seç → 16 karakterli şifre oluştur
    """

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    _TR_MONTHS = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
        5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
        9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
    }

    @classmethod
    def _tr_month_year(cls, dt: datetime) -> str:
        return f"{cls._TR_MONTHS[dt.month]} {dt.year}"

    def __init__(
        self,
        sender: Optional[str] = None,
        password: Optional[str] = None,
        recipients: Optional[List[str]] = None,
    ):
        self.sender = sender or os.getenv("EMAIL_SENDER")
        self.password = password or os.getenv("EMAIL_PASSWORD")

        recipients_env = os.getenv("EMAIL_RECIPIENTS", "")
        self.recipients = recipients or [r.strip() for r in recipients_env.split(",") if r.strip()]

        self.is_configured = bool(self.sender and self.password and self.recipients)

        if not self.is_configured:
            logger.info(
                "E-posta bildirimi yapılandırılmamış. .env dosyasına ekle:\n"
                "  EMAIL_SENDER=senin@gmail.com\n"
                "  EMAIL_PASSWORD=xxxx xxxx xxxx xxxx  (Gmail App Password)\n"
                "  EMAIL_RECIPIENTS=alici1@email.com,alici2@email.com"
            )

    def send_monthly_report(
        self,
        pipeline_result: Optional[Dict] = None,
        pdf_path: Optional[str] = None,
        ml_summary: Optional[Dict] = None,
    ) -> bool:
        """
        Aylık rapor e-postası gönder.

        Returns: True başarılı, False başarısız
        """
        if not self.is_configured:
            logger.warning("E-posta yapılandırılmamış, gönderim atlandı")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = self._build_subject(pipeline_result)

            html_body = self._build_html_body(pipeline_result, ml_summary)
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as f:
                    pdf_attachment = MIMEApplication(f.read(), _subtype="pdf")
                    pdf_filename = Path(pdf_path).name
                    pdf_attachment.add_header(
                        "Content-Disposition", "attachment", filename=pdf_filename
                    )
                    msg.attach(pdf_attachment)
                logger.info(f"PDF eki eklendi: {pdf_filename}")

            with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.send_message(msg)

            logger.info(f"E-posta gönderildi: {', '.join(self.recipients)}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "Gmail kimlik doğrulama hatası. Kontrol et:\n"
                "1. EMAIL_SENDER doğru mu?\n"
                "2. EMAIL_PASSWORD App Password mi (normal şifre değil)?\n"
                "3. Google 2FA aktif mi?"
            )
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"E-posta gönderim hatası: {e}")
            return False

    def _build_subject(self, pipeline_result: Optional[Dict]) -> str:
        month = self._tr_month_year(datetime.now())

        if pipeline_result and pipeline_result.get("status") == "SUCCESS":
            regime = pipeline_result.get("regime", {}).get("detected", "?")
            regime_labels = {
                "STABLE":    "Sakin Piyasa",
                "CRISIS":    "Kriz Modu",
                "RISK_ON":   "Yükseliş",
                "RATE_HIKE": "Faiz Artışı",
            }
            regime_text = regime_labels.get(regime, regime)
            total = pipeline_result.get("portfolio_value", {}).get("total_value", 0)
            return f"🛡️ BES AI Rapor — {month} | {regime_text} | {total:,.0f} TL"

        return f"🛡️ BES AI Aylık Rapor — {month}"

    def _build_html_body(
        self,
        pipeline_result: Optional[Dict],
        ml_summary: Optional[Dict],
    ) -> str:
        html = """
        <html>
        <head>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a2e; line-height: 1.6; }
                .header { background: linear-gradient(135deg, #1e40af, #3b82f6); color: white;
                          padding: 24px; border-radius: 12px; text-align: center; }
                .header h1 { margin: 0; font-size: 24px; }
                .header p  { margin: 8px 0 0; opacity: 0.9; }
                .section { background: #f8fafc; border-radius: 10px; padding: 20px; margin: 16px 0;
                           border-left: 4px solid #3b82f6; }
                .section h2 { color: #1e40af; font-size: 18px; margin: 0 0 12px; }
                .metric { display: inline-block; text-align: center; padding: 12px 20px;
                          background: white; border-radius: 8px; margin: 4px; min-width: 120px;
                          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                .metric .value { font-size: 22px; font-weight: 700; color: #1e40af; }
                .metric .label { font-size: 12px; color: #6b7280; text-transform: uppercase; }
                .action-buy  { color: #16a34a; font-weight: 600; }
                .action-sell { color: #dc2626; font-weight: 600; }
                .footer { text-align: center; color: #9ca3af; font-size: 12px; margin-top: 24px; padding: 16px; }
                table { border-collapse: collapse; width: 100%; }
                th { background: #1e40af; color: white; padding: 10px; text-align: left; }
                td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }
                tr:nth-child(even) { background: #f8fafc; }
            </style>
        </head>
        <body>
        <div class="header">
            <h1>🛡️ BES Akıllı Fon Danışmanı</h1>
            <p>Aylık Portföy Analiz Raporu</p>
        </div>
        """

        if pipeline_result and pipeline_result.get("status") == "SUCCESS":
            regime_obj = pipeline_result.get("regime", {})
            detected   = regime_obj.get("detected", "?")
            confidence = regime_obj.get("confidence", 0)
            total      = pipeline_result.get("portfolio_value", {}).get("total_value", 0)

            regime_meta = {
                "STABLE":    ("😌 Sakin Piyasa", "#3b82f6"),
                "CRISIS":    ("🚨 Kriz Modu",    "#ef4444"),
                "RISK_ON":   ("🚀 Yükseliş",     "#22c55e"),
                "RATE_HIKE": ("🏦 Faiz Artışı",  "#f59e0b"),
            }
            regime_label, regime_color = regime_meta.get(detected, ("?", "#6b7280"))

            html += f"""
            <div class="section" style="border-left-color: {regime_color};">
                <h2>📊 Piyasa Durumu</h2>
                <div class="metric">
                    <div class="value" style="color: {regime_color};">{regime_label}</div>
                    <div class="label">Güven: %{confidence*100:.0f}</div>
                </div>
                <div class="metric">
                    <div class="value">{total:,.0f} TL</div>
                    <div class="label">Portföy Değeri</div>
                </div>
            </div>
            """

            real_pf = pipeline_result.get("real_portfolio")
            if real_pf and real_pf.get("real_total_return") is not None:
                rr_color = "#16a34a" if real_pf["real_total_return"] > 0 else "#dc2626"
                html += f"""
                <div class="section" style="border-left-color: #f59e0b;">
                    <h2>💰 Reel Durum</h2>
                    <div class="metric">
                        <div class="value">{real_pf['nominal_total_return']:+.1%}</div>
                        <div class="label">Nominal Getiri</div>
                    </div>
                    <div class="metric">
                        <div class="value" style="color: {rr_color};">{real_pf['real_total_return']:+.1%}</div>
                        <div class="label">Reel Getiri</div>
                    </div>
                    <div class="metric">
                        <div class="value">{real_pf.get('real_value', 0):,.0f} TL</div>
                        <div class="label">Reel Değer</div>
                    </div>
                </div>
                """

            actions    = pipeline_result.get("recommendation", {}).get("actions", [])
            actionable = [a for a in actions if a.get("action") != "HOLD"]

            if actionable:
                asset_names = {
                    "VEF": "Hisse Fonu", "ALT": "Altın Fonu", "KTS": "Kamu Borç.",
                    "KCH": "Karma Fon",  "CASH": "Para Piy.",
                }
                html += """
                <div class="section" style="border-left-color: #22c55e;">
                    <h2>📋 Bu Ay Yapılması Gerekenler</h2>
                    <table><tr><th>Fon</th><th>İşlem</th><th>Tutar</th></tr>
                """
                for a in sorted(actionable, key=lambda x: -abs(x.get("diff_tl", 0))):
                    css   = "action-buy" if a["action"] == "BUY" else "action-sell"
                    label = "EKLE" if a["action"] == "BUY" else "AZALT"
                    name  = asset_names.get(a["asset"], a["asset"])
                    html += f"""
                        <tr>
                            <td>{name} ({a['asset']})</td>
                            <td class="{css}">{label}</td>
                            <td>{abs(a['diff_tl']):,.0f} TL</td>
                        </tr>
                    """
                html += "</table></div>"

        if ml_summary and ml_summary.get("status") == "SUCCESS":
            html += f"""
            <div class="section" style="border-left-color: #7c3aed;">
                <h2>🤖 AI Model</h2>
                <div class="metric">
                    <div class="value">{ml_summary.get('best_model', '?').upper()}</div>
                    <div class="label">Model</div>
                </div>
                <div class="metric">
                    <div class="value">{ml_summary.get('best_ic', 0):.2f}</div>
                    <div class="label">Sinyal Gücü (IC)</div>
                </div>
                <div class="metric">
                    <div class="value">{ml_summary.get('fund_count', 0)}</div>
                    <div class="label">Analiz Edilen Fon</div>
                </div>
            </div>
            """

        html += f"""
        <div class="footer">
            <p>BES Akıllı Fon Danışmanı v2.0 — {datetime.now().strftime('%d.%m.%Y')}</p>
            <p>⚠️ Bu e-posta yatırım tavsiyesi niteliği taşımaz.</p>
        </div>
        </body></html>
        """
        return html

    def send_test_email(self) -> bool:
        """Yapılandırma testi için basit e-posta gönder."""
        if not self.is_configured:
            logger.error("E-posta yapılandırılmamış")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"]    = self.sender
            msg["To"]      = ", ".join(self.recipients)
            msg["Subject"] = "🛡️ BES AI — Test E-postası"
            msg.attach(MIMEText(
                "<h2>✅ E-posta bildirimi çalışıyor!</h2>"
                "<p>BES Akıllı Fon Danışmanı aylık raporları bu adrese gönderilecek.</p>",
                "html", "utf-8",
            ))

            with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.send_message(msg)

            logger.info(f"Test e-postası gönderildi: {', '.join(self.recipients)}")
            return True
        except Exception as e:
            logger.error(f"Test e-postası başarısız: {e}")
            return False
