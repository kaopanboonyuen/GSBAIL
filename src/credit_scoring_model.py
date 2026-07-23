#!/usr/bin/env python3
# ==============================================================================
#   ██████╗ ███████╗██████╗  █████╗ ██╗██╗
#  ██╔════╝ ██╔════╝██╔══██╗██╔══██╗██║██║
#  ██║  ███╗███████╗██████╔╝███████║██║██║
#  ██║   ██║╚════██║██╔══██╗██╔══██║██║██║
#  ╚██████╔╝███████║██████╔╝██║  ██║██║███████╗
#   ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝
#
#  PROJECT     : Explainable Alternative-Data Credit Scoring
#  MODULE      : credit_scoring_model.py — Scorecard + Gradient Boosting +
#                SHAP Explainability
#  DESCRIPTION : Builds an interpretable credit score (300–850 scale, like
#                a FICO-style score) that blends traditional bureau-style
#                features with alternative data (e-wallet cashflow, utility
#                bill payment history, savings behavior) — aimed at fairly
#                scoring GSB's underserved / informal-income customer base
#                (a core part of GSB's social-mission mandate).
#
#  AUTHOR      : Teerapong Panboonyuen (Kao)
#  AFFILIATION : GSBAIL — Government Savings Bank AI Lab
#                (Government Savings Bank, Thailand)
#  ROLE        : Deputy Director of AI
#  STATUS      : Mockup / Reference Architecture — synthetic data only.
# ==============================================================================

"""
Explainable Alternative-Data Credit Scoring
==============================================

Design goals
------------
1. FAIR & INCLUSIVE
   GSB's mission skews toward savings, grassroots, and micro/SME lending —
   many applicants are thin-file or no-file (no formal credit bureau
   history). We therefore engineer features from ALTERNATIVE DATA:
       - Savings account behavior (balance stability, deposit regularity)
       - Utility-bill / mobile-top-up payment punctuality
       - Cashflow proxies from linked e-wallet / PromptPay transaction history
       - Government welfare-card / social program enrollment signals
         (used only as *positive* stability indicators, never punitive)

2. INTERPRETABLE BY DESIGN
   A single opaque "black box score" is not acceptable for a regulated
   lending decision. We combine:
       - A traditional points-based SCORECARD (WoE/IV binning) — the
         auditable, regulator-friendly baseline.
       - A Gradient Boosting model for uplift, with SHAP values attached
         to every prediction so any loan officer / rejected applicant can
         see WHY a score came out the way it did (aligned with Thailand's
         PDPA and fair-lending expectations).

3. CALIBRATED SCALE
   Final output is calibrated onto a familiar 300–850 scale so business
   teams and partners can reason about it the same way they would a
   bureau score, with configurable cut-off bands (e.g. Reject / Manual
   Review / Auto-Approve / Auto-Approve-Premium-Rate).

Pipeline
--------
    generate_synthetic_applicants()
        -> engineer_features()
        -> compute_woe_iv_scorecard()   # interpretable baseline
        -> train_gbm_uplift_model()     # ML refinement
        -> explain_with_shap()          # per-applicant reason codes
        -> calibrate_to_score_scale()
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [GSBAIL::CreditScore] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("credit_scoring")


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class ScoringConfig:
    n_applicants: int = 8_000
    random_seed: int = 7
    test_size: float = 0.2
    score_min: int = 300
    score_max: int = 850
    # Business-defined risk bands on the final calibrated score.
    bands: tuple = (
        ("Reject", 300, 579),
        ("Manual Review", 580, 649),
        ("Auto-Approve (Standard Rate)", 650, 749),
        ("Auto-Approve (Preferred Rate)", 750, 850),
    )


# ------------------------------------------------------------------------------
# Step 1 — Synthetic applicant data (bureau + alternative data)
# ------------------------------------------------------------------------------
def generate_synthetic_applicants(cfg: ScoringConfig) -> pd.DataFrame:
    logger.info(f"Generating {cfg.n_applicants} synthetic applicant profiles ...")
    rng = np.random.default_rng(cfg.random_seed)
    n = cfg.n_applicants

    df = pd.DataFrame(
        {
            "applicant_id": [f"APP{100000+i}" for i in range(n)],
            "age": rng.integers(20, 65, size=n),
            "monthly_income_thb": np.round(rng.lognormal(mean=9.8, sigma=0.5, size=n), -2),
            "employment_years": rng.exponential(scale=4, size=n).clip(0, 40),
            "has_formal_bureau_history": rng.choice([0, 1], size=n, p=[0.35, 0.65]),
            # --- Alternative data (key for thin-file / informal-income segment) ---
            "avg_savings_balance_6m": np.round(rng.lognormal(mean=8.5, sigma=1.0, size=n), -2),
            "savings_deposit_regularity_score": rng.uniform(0, 1, size=n),  # 0=erratic, 1=very regular
            "utility_bill_ontime_ratio": rng.beta(a=6, b=1.5, size=n),
            "ewallet_avg_monthly_inflow": np.round(rng.lognormal(mean=8.0, sigma=0.8, size=n), -2),
            "mobile_topup_frequency_per_month": rng.poisson(lam=4, size=n),
            "existing_loan_count": rng.poisson(lam=0.8, size=n),
            "debt_service_ratio": rng.beta(a=2, b=5, size=n),  # existing debt / income
        }
    )

    # Latent "true creditworthiness" that drives synthetic default outcomes.
    latent = (
        0.6 * (df.savings_deposit_regularity_score)
        + 0.5 * (df.utility_bill_ontime_ratio)
        + 0.3 * np.log1p(df.monthly_income_thb) / 12
        - 1.2 * df.debt_service_ratio
        - 0.15 * df.existing_loan_count
        + rng.normal(0, 0.3, size=n)
    )
    default_prob = 1 / (1 + np.exp(3 * (latent - latent.mean())))
    df["defaulted_within_12m"] = (rng.random(n) < default_prob).astype(int)

    logger.info(
        f"Synthetic default rate: {df['defaulted_within_12m'].mean():.2%} "
        "(target book quality for calibration sanity-check)"
    )
    return df


# ------------------------------------------------------------------------------
# Step 2 — Feature engineering
# ------------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["income_to_debt_ratio"] = df["monthly_income_thb"] / (
        1 + df["debt_service_ratio"] * df["monthly_income_thb"]
    )
    df["savings_stability_index"] = (
        0.5 * df["savings_deposit_regularity_score"]
        + 0.5 * (df["avg_savings_balance_6m"] / df["avg_savings_balance_6m"].median()).clip(0, 3) / 3
    )
    df["alt_data_composite_score"] = (
        0.4 * df["utility_bill_ontime_ratio"]
        + 0.3 * df["savings_deposit_regularity_score"]
        + 0.3 * (df["mobile_topup_frequency_per_month"] / df["mobile_topup_frequency_per_month"].max())
    )
    return df


FEATURE_COLUMNS = [
    "age",
    "monthly_income_thb",
    "employment_years",
    "has_formal_bureau_history",
    "avg_savings_balance_6m",
    "savings_deposit_regularity_score",
    "utility_bill_ontime_ratio",
    "ewallet_avg_monthly_inflow",
    "mobile_topup_frequency_per_month",
    "existing_loan_count",
    "debt_service_ratio",
    "income_to_debt_ratio",
    "savings_stability_index",
    "alt_data_composite_score",
]


# ------------------------------------------------------------------------------
# Step 3 — Interpretable scorecard baseline (WoE-style, simplified)
# ------------------------------------------------------------------------------
def compute_woe_iv_scorecard(
    df: pd.DataFrame, target_col: str, n_bins: int = 5
) -> tuple[pd.DataFrame, LogisticRegression]:
    """Builds a simplified Weight-of-Evidence (WoE) binned scorecard —
    the classic, highly-auditable approach regulators and credit-risk
    committees are most comfortable with. Each feature is bucketed into
    quantile bins, and WoE = ln(%good / %bad) per bin becomes the model
    input, so the final logistic regression coefficients map directly to
    interpretable "points" per bin (standard FICO-style scorecard math).
    """
    logger.info("Computing WoE / IV scorecard baseline ...")
    woe_df = pd.DataFrame(index=df.index)
    iv_report = {}

    good = (df[target_col] == 0).sum()
    bad = (df[target_col] == 1).sum()

    for col in FEATURE_COLUMNS:
        try:
            bins = pd.qcut(df[col], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        grp = df.groupby(bins, observed=True)[target_col].agg(["count", "sum"])
        grp["good"] = grp["count"] - grp["sum"]
        grp["bad"] = grp["sum"]
        grp["pct_good"] = (grp["good"] / max(good, 1)).clip(lower=1e-4)
        grp["pct_bad"] = (grp["bad"] / max(bad, 1)).clip(lower=1e-4)
        grp["woe"] = np.log(grp["pct_good"] / grp["pct_bad"])
        grp["iv_component"] = (grp["pct_good"] - grp["pct_bad"]) * grp["woe"]
        iv_report[col] = grp["iv_component"].sum()

        woe_map = grp["woe"].to_dict()
        woe_df[f"woe_{col}"] = bins.map(woe_map).astype(float)

    iv_series = pd.Series(iv_report).sort_values(ascending=False)
    logger.info("Information Value (IV) by feature (predictive strength ranking):\n"
                + iv_series.to_string())

    woe_df[target_col] = df[target_col].values
    X = woe_df.drop(columns=[target_col]).fillna(0.0)
    y = woe_df[target_col]

    scorecard_model = LogisticRegression(max_iter=1000)
    scorecard_model.fit(X, y)

    return woe_df, scorecard_model


# ------------------------------------------------------------------------------
# Step 4 — ML refinement model (Gradient Boosting)
# ------------------------------------------------------------------------------
def train_gbm_uplift_model(df: pd.DataFrame, cfg: ScoringConfig):
    logger.info("Training Gradient Boosting uplift model ...")
    X = df[FEATURE_COLUMNS]
    y = df["defaulted_within_12m"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y
    )

    model = GradientBoostingClassifier(random_state=cfg.random_seed, n_estimators=200)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logger.info(f"GBM refinement model AUC-ROC = {auc:.4f}")
    return model, X_test, y_test


# ------------------------------------------------------------------------------
# Step 5 — Explainability (SHAP-style reason codes)
# ------------------------------------------------------------------------------
def explain_with_shap(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Attempts real SHAP explanations; falls back to a permutation-style
    approximate contribution if the `shap` package isn't installed, so the
    pipeline still produces adverse-action reason codes either way.
    """
    try:
        import shap  # type: ignore

        logger.info("Computing SHAP values for per-applicant explainability ...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return pd.DataFrame(shap_values, columns=X_sample.columns, index=X_sample.index)
    except ImportError:
        logger.warning("`shap` not installed — falling back to approximate "
                        "feature-importance-weighted contribution estimate.")
        importances = pd.Series(model.feature_importances_, index=X_sample.columns)
        centered = X_sample - X_sample.mean()
        approx = centered * importances
        return approx


def generate_reason_codes(contrib_row: pd.Series, top_k: int = 3) -> list[str]:
    """Turns the top negative-contribution features into human-readable
    'adverse action' reason codes — required for regulatory transparency
    when an applicant is declined or scored lower than expected.
    """
    worst = contrib_row.sort_values().head(top_k)
    readable = {
        "debt_service_ratio": "High existing debt relative to income",
        "existing_loan_count": "Multiple existing loan obligations",
        "utility_bill_ontime_ratio": "Inconsistent utility bill payment history",
        "savings_deposit_regularity_score": "Irregular savings deposit pattern",
        "monthly_income_thb": "Income level relative to requested facility",
    }
    return [readable.get(f.replace("woe_", ""), f) for f in worst.index]


# ------------------------------------------------------------------------------
# Step 6 — Calibrate probability -> familiar 300-850 score scale
# ------------------------------------------------------------------------------
def calibrate_to_score_scale(default_proba: np.ndarray, cfg: ScoringConfig) -> np.ndarray:
    """Simple log-odds linear calibration onto a bureau-style score range.
    Lower default probability -> higher score.
    """
    eps = 1e-6
    proba = np.clip(default_proba, eps, 1 - eps)
    log_odds = np.log((1 - proba) / proba)  # higher = safer
    lo, hi = log_odds.min(), log_odds.max()
    scaled = (log_odds - lo) / (hi - lo + eps)
    scores = cfg.score_min + scaled * (cfg.score_max - cfg.score_min)
    return np.round(scores).astype(int)


def assign_band(score: int, cfg: ScoringConfig) -> str:
    for label, lo, hi in cfg.bands:
        if lo <= score <= hi:
            return label
    return "Unclassified"


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
def run_pipeline(cfg: Optional[ScoringConfig] = None) -> pd.DataFrame:
    cfg = cfg or ScoringConfig()
    logger.info("=" * 78)
    logger.info(" GSBAIL Explainable Credit Scoring — Pipeline START")
    logger.info("=" * 78)

    df = generate_synthetic_applicants(cfg)
    df = engineer_features(df)

    compute_woe_iv_scorecard(df, target_col="defaulted_within_12m")
    model, X_test, y_test = train_gbm_uplift_model(df, cfg)

    contrib = explain_with_shap(model, X_test.head(200))
    sample_reason_codes = generate_reason_codes(contrib.iloc[0])
    logger.info(f"Example adverse-action reason codes for applicant #0: {sample_reason_codes}")

    full_proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df["credit_score"] = calibrate_to_score_scale(full_proba, cfg)
    df["risk_band"] = df["credit_score"].apply(lambda s: assign_band(s, cfg))

    logger.info("Score distribution by risk band:\n"
                + df["risk_band"].value_counts().to_string())

    logger.info("=" * 78)
    logger.info(" GSBAIL Explainable Credit Scoring — Pipeline COMPLETE")
    logger.info("=" * 78)
    return df[["applicant_id", "credit_score", "risk_band", "defaulted_within_12m"]]


if __name__ == "__main__":
    result_df = run_pipeline()
    print(result_df.head(10).to_string(index=False))
