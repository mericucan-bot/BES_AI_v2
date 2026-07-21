import json
import os
import pytest
from pathlib import Path
from src.portfolio_manager import PortfolioManager


class TestPortfolioManager:
    def test_init_creates_dir(self, tmp_path):
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        assert (tmp_path / "portfolios").exists()

    def test_save_and_get(self, tmp_path):
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        pm.save_portfolio("test", "Test Portföy", {"AHB": 50000, "BGL": 30000})

        pf = pm.get_portfolio("test")
        assert pf is not None
        assert pf["name"] == "Test Portföy"
        assert pf["holdings_tl"]["AHB"] == 50000

    def test_list_portfolios(self, tmp_path):
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        pm.save_portfolio("a", "Portföy A", {"AHB": 100000})
        pm.save_portfolio("b", "Portföy B", {"BGL": 50000})

        portfolios = pm.list_portfolios()
        assert len(portfolios) >= 2
        names = {p["name"] for p in portfolios}
        assert "Portföy A" in names
        assert "Portföy B" in names

    def test_total_tl_calculated(self, tmp_path):
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        pm.save_portfolio("test", "Test", {"A": 30000, "B": 20000, "C": 50000})

        portfolios = pm.list_portfolios()
        assert portfolios[0]["total_tl"] == 100000

    def test_delete_portfolio(self, tmp_path):
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        pm.save_portfolio("deleteme", "Silinecek", {"A": 1000})
        assert pm.delete_portfolio("deleteme") is True
        assert pm.get_portfolio("deleteme") is None

    def test_delete_nonexistent(self, tmp_path):
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        assert pm.delete_portfolio("yok") is False

    def test_create_slug(self, tmp_path):
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        assert pm.create_slug("Eşimin BES'i")    == "esimin_bes_i"
        assert pm.create_slug("Benim Portföyüm") == "benim_portfoyum"
        assert pm.create_slug("  Test 123  ")    == "test_123"

    def test_migrate_legacy(self, tmp_path):
        legacy_dir = tmp_path / "data"
        legacy_dir.mkdir()
        (legacy_dir / "my_portfolio.json").write_text(json.dumps({
            "holdings_tl": {"VEF": 30000, "ALT": 20000}
        }))

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            pm = PortfolioManager(str(tmp_path / "portfolios"))
        finally:
            os.chdir(old_cwd)

        # Temel CRUD çalışıyor
        pm.save_portfolio("slug_test", "Slug Test", {"X": 1000})
        assert pm.get_portfolio("slug_test") is not None

    def test_updated_at_preserved(self, tmp_path):
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        pm.save_portfolio("test", "İlk", {"A": 1000})

        pf1     = pm.get_portfolio("test")
        created = pf1["created_at"]

        pm.save_portfolio("test", "Güncellendi", {"A": 2000})
        pf2 = pm.get_portfolio("test")

        assert pf2["created_at"] == created
        assert pf2["name"] == "Güncellendi"
        assert pf2["holdings_tl"]["A"] == 2000

    def test_monthly_contribution_tl_saved(self, tmp_path):
        """PLAN-19: monthly_contribution_tl kaydedilir; parametresiz cagri 0 yazar."""
        pm = PortfolioManager(str(tmp_path / "portfolios"))
        pm.save_portfolio("test", "Test", {"A": 1000}, monthly_contribution_tl=5000)
        pf = pm.get_portfolio("test")
        assert pf is not None
        assert pf["monthly_contribution_tl"] == 5000

        # Parametresiz cagri geriye uyumlu: 0 yazar
        pm.save_portfolio("test", "Test", {"A": 1000})
        pf2 = pm.get_portfolio("test")
        assert pf2["monthly_contribution_tl"] == 0
