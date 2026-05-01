import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def synthetic_market_data():
    """Test için deterministik market data üret."""
    dates = pd.date_range(end="2024-12-31", periods=300, freq="B")
    np.random.seed(42)

    bist   = 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 300))
    usdtry =    30 * np.cumprod(1 + np.random.normal(0.001,  0.008, 300))
    gold   =  2000 * np.cumprod(1 + np.random.normal(0.0003, 0.010, 300))

    return pd.DataFrame({"BIST": bist, "USDTRY": usdtry, "GOLD": gold}, index=dates)


@pytest.fixture
def crisis_market_data():
    """Kriz senaryosu: BIST son 100 günde sert düşüş, vol yüksek."""
    dates = pd.date_range(end="2024-12-31", periods=300, freq="B")
    np.random.seed(42)

    normal = 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 200))
    crisis = normal[-1] * np.cumprod(1 + np.random.normal(-0.005, 0.035, 100))
    bist   = np.concatenate([normal, crisis])

    usdtry = 30 * np.cumprod(1 + np.abs(np.random.normal(0.003, 0.012, 300)))
    gold   = 2000 * np.cumprod(1 + np.random.normal(0.001, 0.012, 300))

    return pd.DataFrame({"BIST": bist, "USDTRY": usdtry, "GOLD": gold}, index=dates)


@pytest.fixture
def temp_history_path(tmp_path):
    """Her test için izole bir geçmiş dosyası."""
    return tmp_path / "test_learning_history.json"


@pytest.fixture
def sample_observations():
    """Learning engine testleri için 8 pozitif-alpha gözlemi."""
    return [
        {
            "date": f"2024-{i + 1:02d}-01",
            "regime": "CRISIS",
            "weights_used": {"ALT": 0.55 + i * 0.01, "KTS": 0.35 - i * 0.01, "CASH": 0.10},
            "monthly_return": 0.018 + i * 0.002,
            "alpha_vs_benchmark": 0.008 + i * 0.001,
        }
        for i in range(8)
    ]
