#!/usr/bin/env python3
# ==============================================================================
#   ██████╗ ███████╗██████╗  █████╗ ██╗██╗
#  ██╔════╝ ██╔════╝██╔══██╗██╔══██╗██║██║
#  ██║  ███╗███████╗██████╔╝███████║██║██║
#  ██║   ██║╚════██║██╔══██╗██╔══██║██║██║
#  ╚██████╔╝███████║██████╔╝██║  ██║██║███████╗
#   ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝
#
#  PROJECT     : Mule Account ("บัญชีม้า") Detection Engine
#  MODULE      : train.py — Graph + Behavioral Anomaly Training Pipeline
#  DESCRIPTION : Trains a hybrid (graph-topology + transactional-behavior)
#                model to flag bank accounts that are likely being used as
#                "mule accounts" (บัญชีม้า) — i.e. accounts rented, bought,
#                or coerced from real owners and used as pass-through
#                conduits for laundering scam / fraud proceeds.
#
#  AUTHOR      : Teerapong Panboonyuen (Kao)
#  AFFILIATION : GSBAIL — Government Savings Bank AI Lab
#                (Government Savings Bank, Thailand)
#  ROLE        : Deputy Director of AI
#  STATUS      : Mockup / Reference Architecture — synthetic data only.
#                No production PII, no real transaction data is used or
#                required to run this script.
#
#  NOTE TO REVIEWERS
#  ------------------
#  This file is intentionally self-contained and runnable end-to-end on
#  synthetic data so it can serve as a design reference / conversation
#  starter for the real GSBAIL pipeline (which would plug into core
#  banking, AMLO watchlists, and the National ID / mobile-number graph).
# ==============================================================================

"""
Mule Account Detection — Training Pipeline
===========================================

High-level idea
----------------
Mule accounts rarely look suspicious in isolation — a single transaction
can look completely normal. What gives them away is *pattern*:

    1. GRAPH STRUCTURE
       - Fan-in / fan-out bursts: money arrives from many unrelated senders
         and is quickly forwarded onward ("smurfing" / layering).
       - Short holding time: funds don't rest in the account (low dwell time
         between credit and debit).
       - Circular / relay chains: A -> B -> C -> ... -> A style loops that
         are common in layering networks.
       - Degree anomalies: an otherwise dormant account suddenly becomes a
         high-degree hub in the transaction graph.

    2. BEHAVIORAL / TEMPORAL FEATURES
       - Sudden activity spike after long dormancy.
       - Transactions clustered at odd hours (e.g. 02:00–04:00).
       - Round-number transfers, structuring just below reporting thresholds.
       - New device / new location just before the spike (device-graph tie-in).

    3. HYBRID MODEL
       - We combine graph-derived features (via NetworkX) with tabular
         behavioral features and feed them into an ensemble anomaly /
         classification model (Isolation Forest for unsupervised triage +
         Gradient Boosting for supervised refinement once labels/AMLO
         confirmations are available).

This script demonstrates the full pipeline on SYNTHETIC data:
    generate_synthetic_transaction_graph()
        -> build_graph_features()
        -> build_behavioral_features()
        -> train_unsupervised_triage_model()
        -> train_supervised_refinement_model()
        -> evaluate_and_report()

Real deployment would replace `generate_synthetic_transaction_graph()` with
a connector to the core banking data warehouse / AMLO STR feed.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

try:
    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required. Install with: "
        "pip install scikit-learn networkx pandas numpy"
    )


# ------------------------------------------------------------------------------
# Logging setup — GSBAIL house style: structured, timestamped, single-line.
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [GSBAIL::MuleDetect] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mule_account_detection")


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class TrainConfig:
    """Central config object so the whole pipeline is reproducible & auditable
    (important for a bank — model risk management wants every run traceable).
    """

    n_accounts: int = 5_000
    n_transactions: int = 40_000
    mule_ratio: float = 0.03          # ~3% of accounts are synthetic mules
    random_seed: int = 42
    contamination: float = 0.03       # expected anomaly rate for IsolationForest
    test_size: float = 0.25
    output_dir: Path = Path("./artifacts")


# ------------------------------------------------------------------------------
# Step 1 — Synthetic data generation
# ------------------------------------------------------------------------------
def generate_synthetic_transaction_graph(cfg: TrainConfig) -> pd.DataFrame:
    """Generate a synthetic transaction log with an injected mule-ring pattern.

    In production this is replaced by a query against the core banking
    transaction warehouse (e.g. Oracle FLEXCUBE / Temenos extract) joined
    with KYC and device-fingerprint tables.

    Returns
    -------
    pd.DataFrame with columns:
        [tx_id, src_account, dst_account, amount, timestamp, channel]
    """
    rng = np.random.default_rng(cfg.random_seed)
    logger.info("Generating synthetic transaction graph: "
                f"{cfg.n_accounts} accounts / {cfg.n_transactions} transactions")

    accounts = np.array([f"ACC{100000 + i}" for i in range(cfg.n_accounts)])
    n_mules = max(1, int(cfg.n_accounts * cfg.mule_ratio))
    mule_accounts = set(rng.choice(accounts, size=n_mules, replace=False))

    rows = []
    start_time = datetime(2026, 1, 1)

    for tx_id in range(cfg.n_transactions):
        if rng.random() < 0.35 and mule_accounts:
            # Inject "fan-in -> quick fan-out" mule ring pattern.
            dst = rng.choice(list(mule_accounts))
            src = rng.choice(accounts)
            amount = rng.choice([4_900, 9_800, 14_700, 19_500])  # structuring-ish
            ts = start_time + timedelta(
                minutes=int(rng.integers(0, 60 * 24 * 90)),
                hours=int(rng.integers(0, 3)),  # bias toward odd hours below
            )
        else:
            # Normal, benign transaction.
            src, dst = rng.choice(accounts, size=2, replace=False)
            amount = float(np.round(rng.lognormal(mean=6.5, sigma=1.0), 2))
            ts = start_time + timedelta(minutes=int(rng.integers(0, 60 * 24 * 90)))

        rows.append(
            {
                "tx_id": f"TX{tx_id:08d}",
                "src_account": src,
                "dst_account": dst,
                "amount": amount,
                "timestamp": ts,
                "channel": rng.choice(
                    ["mobile_banking", "internet_banking", "atm", "branch", "api"]
                ),
            }
        )

    df = pd.DataFrame(rows)
    df["is_synthetic_mule_label"] = df["dst_account"].isin(mule_accounts).astype(int)
    logger.info(f"Synthetic dataset ready. Injected {n_mules} candidate mule accounts.")
    return df


# ------------------------------------------------------------------------------
# Step 2 — Graph feature engineering
# ------------------------------------------------------------------------------
def build_graph_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Derive account-level graph features from the transaction network.

    Features
    --------
    in_degree / out_degree   : fan-in / fan-out counts
    net_flow                 : total credit - total debit
    pass_through_ratio       : how much of what comes in goes back out
                                (classic mule signature: ratio near 1.0)
    avg_dwell_time_hours     : average time between a credit and the next
                                debit — mules tend to move money FAST
    unique_counterparties    : distinct accounts transacted with
    """
    if nx is None:
        raise ImportError("networkx is required for graph feature extraction.")

    logger.info("Building transaction graph with NetworkX ...")
    G = nx.DiGraph()
    for _, row in tx_df.iterrows():
        G.add_edge(row.src_account, row.dst_account, amount=row.amount, ts=row.timestamp)

    accounts = list(G.nodes())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    inflow = tx_df.groupby("dst_account")["amount"].sum()
    outflow = tx_df.groupby("src_account")["amount"].sum()

    dwell_times = _compute_dwell_times(tx_df)
    unique_cp = tx_df.groupby("dst_account")["src_account"].nunique()

    feats = pd.DataFrame({"account": accounts})
    feats["in_degree"] = feats["account"].map(in_deg).fillna(0)
    feats["out_degree"] = feats["account"].map(out_deg).fillna(0)
    feats["inflow"] = feats["account"].map(inflow).fillna(0.0)
    feats["outflow"] = feats["account"].map(outflow).fillna(0.0)
    feats["net_flow"] = feats["inflow"] - feats["outflow"]
    feats["pass_through_ratio"] = (
        feats[["inflow", "outflow"]].min(axis=1)
        / feats["inflow"].replace(0, np.nan)
    ).fillna(0.0)
    feats["avg_dwell_time_hours"] = feats["account"].map(dwell_times).fillna(999.0)
    feats["unique_counterparties"] = feats["account"].map(unique_cp).fillna(0)

    logger.info(f"Graph features built for {len(feats)} accounts.")
    return feats


def _compute_dwell_times(tx_df: pd.DataFrame) -> pd.Series:
    """Approximate average hours between money arriving at an account and
    the next time that account sends money out. Short dwell time is one of
    the strongest mule signals (money doesn't 'rest').
    """
    dwell = {}
    for acct, grp in tx_df.groupby("dst_account"):
        credits = grp.sort_values("timestamp")["timestamp"].tolist()
        debits = tx_df[tx_df.src_account == acct].sort_values("timestamp")["timestamp"].tolist()
        if not credits or not debits:
            continue
        gaps = []
        di = 0
        for c in credits:
            while di < len(debits) and debits[di] < c:
                di += 1
            if di < len(debits):
                gaps.append((debits[di] - c).total_seconds() / 3600.0)
        if gaps:
            dwell[acct] = float(np.mean(gaps))
    return pd.Series(dwell)


# ------------------------------------------------------------------------------
# Step 3 — Behavioral / temporal feature engineering
# ------------------------------------------------------------------------------
def build_behavioral_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Derive account-level temporal / behavioral red flags."""
    logger.info("Building behavioral / temporal features ...")
    tx_df = tx_df.copy()
    tx_df["hour"] = tx_df["timestamp"].dt.hour
    tx_df["is_odd_hour"] = tx_df["hour"].between(1, 4).astype(int)
    tx_df["is_round_number"] = (tx_df["amount"] % 100 == 0).astype(int)
    tx_df["near_threshold"] = (tx_df["amount"].between(19_000, 19_999)).astype(int)

    beh = (
        tx_df.groupby("dst_account")
        .agg(
            n_credits=("tx_id", "count"),
            odd_hour_ratio=("is_odd_hour", "mean"),
            round_number_ratio=("is_round_number", "mean"),
            near_threshold_ratio=("near_threshold", "mean"),
            avg_amount=("amount", "mean"),
            std_amount=("amount", "std"),
        )
        .reset_index()
        .rename(columns={"dst_account": "account"})
        .fillna(0.0)
    )
    return beh


# ------------------------------------------------------------------------------
# Step 4 — Unsupervised triage (works even with zero confirmed labels)
# ------------------------------------------------------------------------------
def train_unsupervised_triage_model(features: pd.DataFrame, cfg: TrainConfig):
    """Isolation Forest for day-1 deployment — flags outlier accounts for
    human / AMLO analyst review *before* any confirmed labels exist.
    """
    logger.info("Training unsupervised triage model (IsolationForest) ...")
    feature_cols = [c for c in features.columns if c != "account"]
    X = features[feature_cols].fillna(0.0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=cfg.contamination,
        random_state=cfg.random_seed,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    scores = -model.score_samples(X_scaled)  # higher = more anomalous
    features = features.copy()
    features["anomaly_score"] = scores
    features["flagged_for_review"] = model.predict(X_scaled) == -1

    n_flagged = int(features["flagged_for_review"].sum())
    logger.info(f"Triage complete: {n_flagged} accounts flagged for analyst review.")
    return model, scaler, features


# ------------------------------------------------------------------------------
# Step 5 — Supervised refinement (once AMLO / analyst-confirmed labels exist)
# ------------------------------------------------------------------------------
def train_supervised_refinement_model(
    features: pd.DataFrame, labels: pd.Series, cfg: TrainConfig
):
    """Gradient Boosting classifier trained on analyst-confirmed mule labels.
    Meant to run periodically (e.g. weekly) as confirmed cases accumulate,
    progressively improving precision and reducing false positives sent to
    the investigations team.
    """
    logger.info("Training supervised refinement model (GradientBoosting) ...")
    feature_cols = [c for c in features.columns if c not in ("account", "flagged_for_review")]
    X = features[feature_cols].fillna(0.0)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y
    )

    clf = GradientBoostingClassifier(random_state=cfg.random_seed)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    logger.info(f"Supervised model AUC-ROC={auc:.4f} | AUC-PR={ap:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, digits=3))

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    logger.info("Top feature importances:\n" + importances.head(10).to_string())

    return clf, {"auc_roc": auc, "auc_pr": ap}


# ------------------------------------------------------------------------------
# Step 6 — Reporting
# ------------------------------------------------------------------------------
def evaluate_and_report(features: pd.DataFrame, cfg: TrainConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "mule_account_risk_scores.csv"
    features.sort_values("anomaly_score", ascending=False).to_csv(out_path, index=False)
    logger.info(f"Risk-scored account list written to: {out_path}")

    top10 = features.sort_values("anomaly_score", ascending=False).head(10)
    logger.info(
        "Top-10 highest-risk accounts for analyst / AMLO review:\n"
        + top10[["account", "anomaly_score", "pass_through_ratio",
                 "avg_dwell_time_hours", "in_degree", "out_degree"]].to_string(index=False)
    )


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
def run_pipeline(cfg: TrainConfig, supervised: bool = True) -> None:
    logger.info("=" * 78)
    logger.info(" GSBAIL Mule Account Detection — Training Pipeline START")
    logger.info("=" * 78)

    tx_df = generate_synthetic_transaction_graph(cfg)

    graph_feats = build_graph_features(tx_df)
    beh_feats = build_behavioral_features(tx_df)
    features = graph_feats.merge(beh_feats, on="account", how="left").fillna(0.0)

    _, _, scored = train_unsupervised_triage_model(features, cfg)

    if supervised:
        # In production these labels come from AMLO STR confirmations /
        # fraud-ops case outcomes. Here we approximate with the synthetic
        # injected ground truth for demonstration purposes only.
        label_map = tx_df.groupby("dst_account")["is_synthetic_mule_label"].max()
        y = scored["account"].map(label_map).fillna(0).astype(int)
        if y.sum() >= 2:  # need at least a couple positive examples
            train_supervised_refinement_model(scored, y, cfg)
        else:
            logger.warning("Not enough positive labels to train supervised model this run.")

    evaluate_and_report(scored, cfg)

    logger.info("=" * 78)
    logger.info(" GSBAIL Mule Account Detection — Training Pipeline COMPLETE")
    logger.info("=" * 78)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="GSBAIL Mule Account (บัญชีม้า) Detection — Training Pipeline"
    )
    parser.add_argument("--n-accounts", type=int, default=5_000)
    parser.add_argument("--n-transactions", type=int, default=40_000)
    parser.add_argument("--mule-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument("--no-supervised", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    args = parser.parse_args(argv)

    return TrainConfig(
        n_accounts=args.n_accounts,
        n_transactions=args.n_transactions,
        mule_ratio=args.mule_ratio,
        random_seed=args.seed,
        contamination=args.contamination,
        output_dir=Path(args.output_dir),
    ), args.no_supervised


if __name__ == "__main__":
    cfg, no_supervised = parse_args(sys.argv[1:])
    run_pipeline(cfg, supervised=not no_supervised)
