#!/usr/bin/env python3
# ==============================================================================
#   ██████╗ ███████╗██████╗  █████╗ ██╗██╗
#  ██╔════╝ ██╔════╝██╔══██╗██╔══██╗██║██║
#  ██║  ███╗███████╗██████╔╝███████║██║██║
#  ██║   ██║╚════██║██╔══██╗██╔══██║██║██║
#  ╚██████╔╝███████║██████╔╝██║  ██║██║███████╗
#   ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝
#
#  PROJECT     : AI-Assisted Portfolio Construction & Robo-Advisory Engine
#  MODULE      : portfolio_optimizer.py — Mean-Variance + Black-Litterman +
#                Risk-Profiling Robo-Advisor
#  DESCRIPTION : Given a customer's risk profile (via questionnaire) and a
#                universe of investable assets (GSB mutual funds, bonds,
#                deposits, ETFs), constructs a personalized, constrained,
#                efficient portfolio — with an optional Black-Litterman
#                layer to blend market equilibrium with house-view tilts
#                from the investment research desk.
#
#  AUTHOR      : Teerapong Panboonyuen (Kao)
#  AFFILIATION : GSBAIL — Government Savings Bank AI Lab
#                (Government Savings Bank, Thailand)
#  ROLE        : Deputy Director of AI
#  STATUS      : Mockup / Reference Architecture — synthetic market data
#                only. Not investment advice; illustrative engineering
#                reference for a future GSB robo-advisory product.
# ==============================================================================

"""
AI-Assisted Portfolio Construction & Robo-Advisory Engine
=============================================================

Pipeline overview
------------------
    1. RISK PROFILING
       A short questionnaire (age, horizon, income stability, loss
       tolerance, investment experience) is scored into a 1-5 risk
       tier, which maps to a target volatility / target return band.

    2. ASSET UNIVERSE & EXPECTED RETURNS
       Synthetic multi-asset universe (money market, government bonds,
       corporate bonds, Thai equity fund, global equity fund, gold,
       REITs) with a covariance matrix estimated from (synthetic)
       historical returns.

    3. BLACK-LITTERMAN VIEW BLENDING (optional)
       Instead of relying purely on noisy historical means (classic
       mean-variance's Achilles heel), we start from market-implied
       equilibrium returns (reverse-optimization from market-cap
       weights) and blend in explicit "house views" from the research
       desk (e.g. "Thai equities will outperform government bonds by
       2% annually, with 60% confidence").

    4. CONSTRAINED MEAN-VARIANCE OPTIMIZATION
       Solves for the minimum-variance portfolio achieving the target
       return implied by the customer's risk tier, subject to
       real-world constraints: no shorting, per-asset-class caps,
       minimum deposit/money-market floor (capital preservation
       requirement typical of a retail savings-bank client base).

    5. REBALANCING SIGNAL
       Periodic drift-based rebalancing check: if any asset weight
       drifts beyond a tolerance band from target, flag for rebalance.

This is a reference architecture — for production, expected returns
and covariances would come from the investment research desk's models
and a licensed market-data feed, and the optimizer would be wrapped in
full suitability / KYC compliance workflows (Thai SEC robo-advisory
regulations).
"""

from __future__ import annotations

import dataclasses
import logging
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    raise SystemExit("scipy is required. Install with: pip install scipy numpy pandas")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [GSBAIL::PortfolioAI] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("portfolio_optimization")


# ------------------------------------------------------------------------------
# Step 1 — Risk profiling
# ------------------------------------------------------------------------------
class RiskTier(IntEnum):
    CONSERVATIVE = 1
    MODERATE_CONSERVATIVE = 2
    MODERATE = 3
    MODERATE_AGGRESSIVE = 4
    AGGRESSIVE = 5


@dataclasses.dataclass
class RiskQuestionnaire:
    age: int
    investment_horizon_years: int
    income_stability_score: int   # 1 (unstable) - 5 (very stable)
    loss_tolerance_score: int     # 1 (cannot tolerate any loss) - 5 (comfortable with large swings)
    investment_experience_score: int  # 1 (none) - 5 (expert)


# Target annualized volatility & return bands per risk tier — illustrative.
RISK_TIER_TARGETS = {
    RiskTier.CONSERVATIVE: {"target_vol": 0.03, "target_return": 0.030},
    RiskTier.MODERATE_CONSERVATIVE: {"target_vol": 0.06, "target_return": 0.045},
    RiskTier.MODERATE: {"target_vol": 0.10, "target_return": 0.060},
    RiskTier.MODERATE_AGGRESSIVE: {"target_vol": 0.15, "target_return": 0.080},
    RiskTier.AGGRESSIVE: {"target_vol": 0.22, "target_return": 0.100},
}


def score_risk_profile(q: RiskQuestionnaire) -> RiskTier:
    """Simple weighted-sum scoring -> 5-tier bucket. A production system
    would validate this against a psychometrically-tested questionnaire
    and periodic re-assessment (SEC robo-advisor requirement).
    """
    horizon_score = min(5, max(1, round(q.investment_horizon_years / 4)))
    age_penalty = 1 if q.age >= 60 else (0 if q.age >= 45 else -1)  # older -> more conservative

    raw = (
        0.25 * horizon_score
        + 0.20 * q.income_stability_score
        + 0.30 * q.loss_tolerance_score
        + 0.25 * q.investment_experience_score
        - 0.3 * age_penalty
    )
    tier_value = int(np.clip(round(raw / 5 * 5), 1, 5))  # normalize into 1-5
    tier = RiskTier(tier_value)
    logger.info(f"Risk questionnaire scored -> {tier.name} (raw={raw:.2f})")
    return tier


# ------------------------------------------------------------------------------
# Step 2 — Asset universe & synthetic covariance estimation
# ------------------------------------------------------------------------------
ASSET_UNIVERSE = [
    "money_market_fund",     # capital-preservation anchor
    "government_bond_fund",
    "corporate_bond_fund",
    "thai_equity_fund",
    "global_equity_fund",
    "gold_etf",
    "reit_fund",
]

# Illustrative long-run assumptions (annualized). Production values would
# come from the research desk's capital market assumptions.
EXPECTED_RETURNS = pd.Series(
    {
        "money_market_fund": 0.018,
        "government_bond_fund": 0.028,
        "corporate_bond_fund": 0.038,
        "thai_equity_fund": 0.075,
        "global_equity_fund": 0.085,
        "gold_etf": 0.045,
        "reit_fund": 0.060,
    }
)

ANNUAL_VOLATILITY = pd.Series(
    {
        "money_market_fund": 0.005,
        "government_bond_fund": 0.035,
        "corporate_bond_fund": 0.055,
        "thai_equity_fund": 0.180,
        "global_equity_fund": 0.160,
        "gold_etf": 0.150,
        "reit_fund": 0.140,
    }
)

# Illustrative correlation matrix (symmetric, positive semi-definite by construction).
CORRELATION = pd.DataFrame(
    [
        [1.00, 0.30, 0.20, 0.05, 0.05, 0.00, 0.10],
        [0.30, 1.00, 0.70, 0.10, 0.05, 0.10, 0.20],
        [0.20, 0.70, 1.00, 0.20, 0.15, 0.10, 0.30],
        [0.05, 0.10, 0.20, 1.00, 0.65, 0.10, 0.40],
        [0.05, 0.05, 0.15, 0.65, 1.00, 0.15, 0.35],
        [0.00, 0.10, 0.10, 0.10, 0.15, 1.00, 0.20],
        [0.10, 0.20, 0.30, 0.40, 0.35, 0.20, 1.00],
    ],
    index=ASSET_UNIVERSE,
    columns=ASSET_UNIVERSE,
)


def build_covariance_matrix() -> pd.DataFrame:
    vol = ANNUAL_VOLATILITY[ASSET_UNIVERSE]
    cov = CORRELATION.loc[ASSET_UNIVERSE, ASSET_UNIVERSE].values * np.outer(vol, vol)
    return pd.DataFrame(cov, index=ASSET_UNIVERSE, columns=ASSET_UNIVERSE)


# ------------------------------------------------------------------------------
# Step 3 — Black-Litterman view blending (optional layer)
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class HouseView:
    """A single research-desk 'view', e.g.:
    'thai_equity_fund will outperform government_bond_fund by 2%/yr,
    confidence=0.6'
    """
    asset_long: str
    asset_short: Optional[str]  # None => absolute view on asset_long
    expected_outperformance: float
    confidence: float  # 0..1


def black_litterman_blend(
    market_cap_weights: pd.Series,
    cov: pd.DataFrame,
    views: list[HouseView],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> pd.Series:
    """Blends market-implied equilibrium returns with explicit house views.
    Simplified single-factor-per-view implementation of Black & Litterman
    (1992) — production would use the full matrix formulation with a
    proper view-uncertainty (Omega) matrix.
    """
    logger.info(f"Blending {len(views)} house view(s) via Black-Litterman ...")
    pi = risk_aversion * cov.values @ market_cap_weights.values  # equilibrium returns

    n = len(ASSET_UNIVERSE)
    P = np.zeros((len(views), n))
    Q = np.zeros(len(views))
    omega_diag = np.zeros(len(views))

    idx = {a: i for i, a in enumerate(ASSET_UNIVERSE)}
    for i, v in enumerate(views):
        P[i, idx[v.asset_long]] = 1
        if v.asset_short:
            P[i, idx[v.asset_short]] = -1
        Q[i] = v.expected_outperformance
        # Lower confidence -> higher view uncertainty (variance).
        omega_diag[i] = (1 - v.confidence) * (P[i] @ (tau * cov.values) @ P[i].T) + 1e-6

    omega = np.diag(omega_diag)
    tau_cov = tau * cov.values

    middle = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P)
    blended = middle @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(omega) @ Q)

    return pd.Series(blended, index=ASSET_UNIVERSE)


# ------------------------------------------------------------------------------
# Step 4 — Constrained mean-variance optimization
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class OptimizationConstraints:
    min_money_market_weight: float = 0.05   # capital-preservation floor
    max_single_asset_weight: float = 0.40
    max_equity_weight: float = 0.70         # thai_equity + global_equity combined


def optimize_portfolio(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    target_return: float,
    constraints: OptimizationConstraints,
) -> pd.Series:
    """Minimizes portfolio variance subject to:
        - weights sum to 1, all weights >= 0 (no shorting)
        - achieves at least `target_return`
        - per-constraint caps/floors defined in `constraints`
    """
    n = len(ASSET_UNIVERSE)
    mu = expected_returns[ASSET_UNIVERSE].values
    Sigma = cov.loc[ASSET_UNIVERSE, ASSET_UNIVERSE].values

    def portfolio_variance(w):
        return w @ Sigma @ w

    equity_idx = [ASSET_UNIVERSE.index("thai_equity_fund"), ASSET_UNIVERSE.index("global_equity_fund")]
    mm_idx = ASSET_UNIVERSE.index("money_market_fund")

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w @ mu - target_return},
        {"type": "ineq", "fun": lambda w: w[mm_idx] - constraints.min_money_market_weight},
        {"type": "ineq", "fun": lambda w: constraints.max_equity_weight - sum(w[i] for i in equity_idx)},
    ]
    bounds = [(0.0, constraints.max_single_asset_weight) for _ in range(n)]

    w0 = np.repeat(1 / n, n)
    result = minimize(
        portfolio_variance, w0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if not result.success:
        logger.warning(f"Optimizer did not fully converge: {result.message}. "
                        "Falling back to closest feasible solution found.")

    weights = pd.Series(np.clip(result.x, 0, None), index=ASSET_UNIVERSE)
    weights = weights / weights.sum()  # renormalize for numerical safety
    return weights


# ------------------------------------------------------------------------------
# Step 5 — Rebalancing signal
# ------------------------------------------------------------------------------
def check_rebalance_needed(
    current_weights: pd.Series, target_weights: pd.Series, tolerance: float = 0.05
) -> pd.DataFrame:
    drift = (current_weights - target_weights).abs()
    report = pd.DataFrame(
        {
            "current_weight": current_weights,
            "target_weight": target_weights,
            "drift": drift,
            "needs_rebalance": drift > tolerance,
        }
    )
    return report


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
def build_portfolio_for_customer(
    questionnaire: RiskQuestionnaire,
    house_views: Optional[list[HouseView]] = None,
) -> pd.Series:
    logger.info("=" * 78)
    logger.info(" GSBAIL Robo-Advisory Portfolio Construction — START")
    logger.info("=" * 78)

    tier = score_risk_profile(questionnaire)
    target_return = RISK_TIER_TARGETS[tier]["target_return"]
    logger.info(f"Target annual return for {tier.name}: {target_return:.1%}")

    cov = build_covariance_matrix()

    if house_views:
        market_cap_weights = pd.Series(1 / len(ASSET_UNIVERSE), index=ASSET_UNIVERSE)
        expected_returns = black_litterman_blend(market_cap_weights, cov, house_views)
        logger.info("Using Black-Litterman blended returns (house views applied).")
    else:
        expected_returns = EXPECTED_RETURNS
        logger.info("Using baseline capital-market-assumption expected returns.")

    weights = optimize_portfolio(
        expected_returns, cov, target_return, OptimizationConstraints()
    )

    port_return = weights @ expected_returns[ASSET_UNIVERSE]
    port_vol = np.sqrt(weights.values @ cov.values @ weights.values)
    logger.info(f"Constructed portfolio: expected return={port_return:.2%}, "
                f"expected volatility={port_vol:.2%}")
    logger.info("Recommended allocation:\n" + weights.round(4).to_string())

    logger.info("=" * 78)
    logger.info(" GSBAIL Robo-Advisory Portfolio Construction — COMPLETE")
    logger.info("=" * 78)
    return weights


if __name__ == "__main__":
    customer = RiskQuestionnaire(
        age=35,
        investment_horizon_years=15,
        income_stability_score=4,
        loss_tolerance_score=3,
        investment_experience_score=2,
    )

    house_views = [
        HouseView(
            asset_long="thai_equity_fund",
            asset_short="government_bond_fund",
            expected_outperformance=0.02,
            confidence=0.6,
        ),
        HouseView(
            asset_long="gold_etf",
            asset_short=None,
            expected_outperformance=0.05,
            confidence=0.4,
        ),
    ]

    recommended_weights = build_portfolio_for_customer(customer, house_views=house_views)
    print(recommended_weights.round(4))
