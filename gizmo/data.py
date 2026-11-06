from dataclasses import dataclass
import numpy as np
from typing import List, Optional

@dataclass
class PortfolioData:
    tickers: List[str]
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    description: str

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

def generate_realistic_market_data(
    n_assets: int = 8,
    seed: int = 42,
    base_return: float = 0.12,
    base_volatility: float = 0.20
) -> PortfolioData:
    rng = np.random.default_rng(seed)
    expected_returns = rng.uniform(0.06, 0.22, size=n_assets)
    n_factors = max(2, n_assets // 3)
    factor_loadings = rng.normal(loc=0.5, scale=0.25, size=(n_assets, n_factors))
    factor_cov = np.eye(n_factors) * (base_volatility ** 2)
    specific_var = rng.uniform(0.04, 0.10, size=n_assets) ** 2
    cov_matrix = (factor_loadings @ factor_cov @ factor_loadings.T) + np.diag(specific_var)
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
    if min_eig < 1e-6:
        cov_matrix += (1e-6 - min_eig + 1e-4) * np.eye(n_assets)
    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]
    return PortfolioData(
        tickers=tickers,
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        description=f"Synthetic Factor Model ({n_assets} assets)"
    )

def load_portfolio_data(
    tickers: Optional[List[str]] = None,
    n_assets: int = 8,
    period: str = "1y",
    fallback: bool = True
) -> PortfolioData:
    default_tickers = [
        "AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "PG", "AMZN",
        "GOOGL", "BRK-B", "LLY", "TSLA", "V", "UNH", "HD", "CVX"
    ]
    if tickers is None:
        target_tickers = default_tickers[:n_assets]
    else:
        target_tickers = tickers
        n_assets = len(target_tickers)

    try:
        import yfinance as yf
        df = yf.download(target_tickers, period=period, progress=False)
        if "Adj Close" in df:
            prices = df["Adj Close"]
        elif "Close" in df:
            prices = df["Close"]
        else:
            raise ValueError()
        prices = prices.dropna(axis=1, how="all")
        if prices.shape[1] < 2 or prices.shape[0] < 30:
            raise ValueError()
        avail_tickers = [t for t in target_tickers if t in prices.columns]
        prices = prices[avail_tickers].dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        mu = returns.mean().values * 252.0
        sigma = returns.cov().values * 252.0
        return PortfolioData(
            tickers=avail_tickers,
            expected_returns=mu,
            covariance_matrix=sigma,
            description=f"Live Historical Data ({period})"
        )
    except Exception:
        if not fallback:
            raise
        return generate_realistic_market_data(n_assets=n_assets)
