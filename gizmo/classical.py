from dataclasses import dataclass
import itertools
import numpy as np
from typing import List, Dict, Optional

@dataclass
class MarkowitzSolution:
    best_bitstring: str
    best_indices: List[int]
    best_tickers: List[str]
    best_cost: float
    worst_cost: float
    expected_return: float
    portfolio_variance: float
    portfolio_volatility: float
    sharpe_ratio: float
    all_subspace_evaluations: Dict[str, float]

def evaluate_portfolio_cost(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float
) -> float:
    return 0.5 * q * float(x.T @ sigma @ x) - float(mu.T @ x)

def solve_markowitz_brute_force(
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float = 1.0,
    K: int = 2,
    tickers: Optional[List[str]] = None,
    risk_free_rate: float = 0.03
) -> MarkowitzSolution:
    N = len(mu)
    if tickers is None:
        tickers = [f"ASSET_{i}" for i in range(N)]
    
    all_evals = {}
    best_cost = float("inf")
    worst_cost = float("-inf")
    best_x = None
    best_combo = []
    best_bitstring = ""

    for combo in itertools.combinations(range(N), K):
        x = np.zeros(N, dtype=float)
        x[list(combo)] = 1.0
        bitstring = "".join(str(int(x[N - 1 - i])) for i in range(N))
        cost = evaluate_portfolio_cost(x, mu, sigma, q)
        all_evals[bitstring] = cost
        
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()
            best_combo = list(combo)
            best_bitstring = bitstring
        if cost > worst_cost:
            worst_cost = cost

    w = best_x / K
    port_return = float(w.T @ mu)
    port_var = float(w.T @ sigma @ w)
    port_vol = np.sqrt(max(1e-9, port_var))
    sharpe = (port_return - risk_free_rate) / port_vol

    return MarkowitzSolution(
        best_bitstring=best_bitstring,
        best_indices=best_combo,
        best_tickers=[tickers[i] for i in best_combo],
        best_cost=best_cost,
        worst_cost=worst_cost,
        expected_return=port_return,
        portfolio_variance=port_var,
        portfolio_volatility=port_vol,
        sharpe_ratio=sharpe,
        all_subspace_evaluations=all_evals
    )
