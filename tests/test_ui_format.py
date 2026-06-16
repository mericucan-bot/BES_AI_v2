import pytest

from src.ui_format import explain_regime, confidence_to_text, format_tl, action_text


class TestExplainRegime:
    @pytest.mark.parametrize("regime", ["STABLE", "CRISIS", "RISK_ON", "RATE_HIKE"])
    def test_known_regimes_have_fields(self, regime):
        d = explain_regime(regime)
        for key in ("symbol", "label", "color", "border", "summary", "action", "detail"):
            assert key in d and d[key]

    def test_unknown_falls_back_to_stable(self):
        assert explain_regime("WAT") == explain_regime("STABLE")


class TestConfidenceToText:
    @pytest.mark.parametrize("c,frag", [
        (0.95, "Yüksek"), (0.80, "Yüksek"),
        (0.70, "Orta"), (0.60, "Orta"),
        (0.50, "Düşük"), (0.40, "Düşük"),
        (0.10, "Çok düşük"), (0.0, "Çok düşük"),
    ])
    def test_thresholds(self, c, frag):
        assert frag in confidence_to_text(c)


class TestFormatTL:
    def test_thousands_use_dots(self):
        assert format_tl(1234567) == "1.234.567 TL"

    def test_zero(self):
        assert format_tl(0) == "0 TL"

    def test_rounds(self):
        assert format_tl(1234.7) == "1.235 TL"


class TestActionText:
    @pytest.mark.parametrize("action,expected", [
        ("BUY", "EKLE"), ("SELL", "AZALT"), ("HOLD", "DEĞİŞTİRME"),
    ])
    def test_known(self, action, expected):
        assert action_text(action) == expected

    def test_unknown_passthrough(self):
        assert action_text("XYZ") == "XYZ"
