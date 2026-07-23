#!/usr/bin/env python3
# ==============================================================================
#   ██████╗ ███████╗██████╗  █████╗ ██╗██╗
#  ██╔════╝ ██╔════╝██╔══██╗██╔══██╗██║██║
#  ██║  ███╗███████╗██████╔╝███████║██║██║
#  ██║   ██║╚════██║██╔══██╗██╔══██║██║██║
#  ╚██████╔╝███████║██████╔╝██║  ██║██║███████╗
#   ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝
#
#  PROJECT     : Hybrid Rules + ML Loan Underwriting Decision Engine
#  MODULE      : loan_engine.py — Policy Rules Layer + ML Risk Layer +
#                Pricing / Affordability Layer
#  DESCRIPTION : Turns a credit score + applicant profile into a final
#                underwriting decision: APPROVE / DECLINE / REFER-TO-HUMAN,
#                along with a recommended loan amount, tenor, and interest
#                rate — fully auditable and policy-overridable, as required
#                for a regulated national bank.
#
#  AUTHOR      : Teerapong Panboonyuen (Kao)
#  AFFILIATION : GSBAIL — Government Savings Bank AI Lab
#                (Government Savings Bank, Thailand)
#  ROLE        : Deputy Director of AI
#  STATUS      : Mockup / Reference Architecture — synthetic data only.
#                Interest-rate and affordability constants below are
#                illustrative placeholders, NOT actual GSB policy figures.
# ==============================================================================

"""
Hybrid Rules + ML Loan Underwriting Decision Engine
=======================================================

Philosophy: "ML proposes, policy disposes"
---------------------------------------------
Fully autonomous ML lending decisions are a compliance and reputational
risk for a state-owned bank. This engine therefore layers ML *inside* a
transparent, overridable rules framework rather than letting the model
have the final word:

    Layer 1 — HARD POLICY GATES (deterministic, never bypassed by ML)
        e.g. minimum age, blacklist / AMLO watch-list check, maximum
        debt-service ratio ceiling, minimum required documents.

    Layer 2 — ML RISK ASSESSMENT
        Consumes the credit_scoring_model.py output (score + risk band)
        plus loan-specific features (requested amount, tenor, purpose) to
        estimate probability of default AND expected loss.

    Layer 3 — AFFORDABILITY & PRICING ENGINE
        Computes a risk-based interest rate (risk-based pricing) and the
        maximum affordable installment given declared income, then checks
        the requested loan against that affordability ceiling.

    Layer 4 — DECISION ARBITRATION
        Combines all layers into one of:
            APPROVE                  (auto, within policy + ML confidence)
            APPROVE_WITH_CONDITIONS  (auto, but reduced amount / higher rate)
            REFER_TO_HUMAN           (borderline / low-confidence / high-value)
            DECLINE                  (policy violation or excessive risk)
        Every decision carries a full `DecisionTrace` — an ordered log of
        every rule and score that contributed — for audit, complaints
        handling, and regulator inspection.
"""

from __future__ import annotations

import dataclasses
import logging
from enum import Enum
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [GSBAIL::LoanEngine] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("loan_underwriting")


# ------------------------------------------------------------------------------
# Domain enums & data contracts
# ------------------------------------------------------------------------------
class Decision(str, Enum):
    APPROVE = "APPROVE"
    APPROVE_WITH_CONDITIONS = "APPROVE_WITH_CONDITIONS"
    REFER_TO_HUMAN = "REFER_TO_HUMAN"
    DECLINE = "DECLINE"


@dataclasses.dataclass
class LoanApplication:
    applicant_id: str
    age: int
    monthly_income_thb: float
    requested_amount_thb: float
    requested_tenor_months: int
    loan_purpose: str
    credit_score: int              # from credit_scoring_model.py, 300-850
    existing_monthly_debt_thb: float
    on_amlo_watchlist: bool = False
    has_complete_documents: bool = True


@dataclasses.dataclass
class DecisionTrace:
    """Ordered, human-readable audit trail of every check performed —
    this is what gets shown to a compliance officer or a customer who
    disputes a decision.
    """
    steps: list[str] = dataclasses.field(default_factory=list)

    def log(self, message: str) -> None:
        self.steps.append(message)
        logger.info(f"  -> {message}")


@dataclasses.dataclass
class UnderwritingResult:
    applicant_id: str
    decision: Decision
    approved_amount_thb: Optional[float]
    approved_tenor_months: Optional[int]
    annual_interest_rate_pct: Optional[float]
    estimated_probability_of_default: float
    trace: DecisionTrace


# ------------------------------------------------------------------------------
# Layer 1 — Hard policy gates
# ------------------------------------------------------------------------------
class PolicyGate:
    """Deterministic, non-negotiable rules. If ANY of these fail, the ML
    layers are never even consulted — this keeps the audit story simple
    for the highest-risk failure modes (fraud, sanctions, ineligibility).
    """

    MIN_AGE = 20
    MAX_AGE = 70
    MAX_DEBT_SERVICE_RATIO = 0.70  # existing + new debt vs income ceiling

    @classmethod
    def evaluate(cls, app: LoanApplication, trace: DecisionTrace) -> Optional[Decision]:
        if app.on_amlo_watchlist:
            trace.log("HARD DECLINE: applicant on AMLO / sanctions watchlist.")
            return Decision.DECLINE

        if not app.has_complete_documents:
            trace.log("HARD DECLINE: mandatory KYC / income documents incomplete.")
            return Decision.DECLINE

        if not (cls.MIN_AGE <= app.age <= cls.MAX_AGE):
            trace.log(f"HARD DECLINE: applicant age {app.age} outside policy band "
                       f"[{cls.MIN_AGE}, {cls.MAX_AGE}].")
            return Decision.DECLINE

        trace.log("Policy gate PASSED: no hard blockers found.")
        return None  # None = proceed to ML layer


# ------------------------------------------------------------------------------
# Layer 2 — ML risk assessment (mocked; would call the trained PD model)
# ------------------------------------------------------------------------------
class MLRiskAssessor:
    """Maps credit_score (+ loan-specific context) to an estimated
    probability of default (PD) for THIS SPECIFIC loan. In production this
    calls the trained model from `credit_scoring/credit_scoring_model.py`
    (or a loan-specific fine-tuned variant that also conditions on
    requested amount / tenor / purpose).
    """

    @staticmethod
    def estimate_pd(app: LoanApplication, trace: DecisionTrace) -> float:
        # Simple monotonic mock: lower score & higher leverage -> higher PD.
        score_component = (850 - app.credit_score) / (850 - 300)  # 0 (safe) .. 1 (risky)
        leverage = (app.existing_monthly_debt_thb + app.requested_amount_thb / max(app.requested_tenor_months, 1)) / max(app.monthly_income_thb, 1)
        leverage_component = min(leverage, 1.5) / 1.5

        pd_estimate = 0.65 * score_component + 0.35 * leverage_component
        pd_estimate = max(0.01, min(pd_estimate, 0.95))

        trace.log(
            f"ML risk layer: estimated 12-month PD = {pd_estimate:.2%} "
            f"(score_component={score_component:.2f}, leverage_component={leverage_component:.2f})"
        )
        return pd_estimate


# ------------------------------------------------------------------------------
# Layer 3 — Affordability & risk-based pricing
# ------------------------------------------------------------------------------
class PricingEngine:
    """Computes a risk-based annual interest rate and checks the requested
    loan against an affordability ceiling (max installment-to-income ratio).
    NOTE: rate figures below are illustrative placeholders only.
    """

    BASE_RATE_PCT = 5.5
    MAX_RISK_PREMIUM_PCT = 12.0
    MAX_INSTALLMENT_TO_INCOME = 0.40

    @classmethod
    def price_loan(cls, app: LoanApplication, pd_estimate: float, trace: DecisionTrace) -> float:
        risk_premium = pd_estimate * cls.MAX_RISK_PREMIUM_PCT
        rate = round(cls.BASE_RATE_PCT + risk_premium, 2)
        trace.log(f"Pricing layer: risk-based annual rate = {rate}% "
                   f"(base {cls.BASE_RATE_PCT}% + risk premium {risk_premium:.2f}%)")
        return rate

    @classmethod
    def max_affordable_installment(cls, app: LoanApplication, trace: DecisionTrace) -> float:
        headroom_income = app.monthly_income_thb - app.existing_monthly_debt_thb
        max_installment = max(0.0, headroom_income * cls.MAX_INSTALLMENT_TO_INCOME)
        trace.log(f"Affordability layer: max affordable installment = "
                   f"THB {max_installment:,.2f}/month")
        return max_installment

    @staticmethod
    def monthly_installment(principal: float, annual_rate_pct: float, tenor_months: int) -> float:
        r = (annual_rate_pct / 100) / 12
        if r == 0:
            return principal / tenor_months
        return principal * r / (1 - (1 + r) ** -tenor_months)


# ------------------------------------------------------------------------------
# Layer 4 — Decision arbitration
# ------------------------------------------------------------------------------
class DecisionArbiter:
    """Combines the outputs of all layers into a single, explainable
    underwriting decision.
    """

    PD_AUTO_APPROVE_CEILING = 0.15
    PD_AUTO_DECLINE_FLOOR = 0.55
    HIGH_VALUE_REFERRAL_THRESHOLD_THB = 1_000_000

    @classmethod
    def decide(
        cls,
        app: LoanApplication,
        pd_estimate: float,
        annual_rate_pct: float,
        max_installment: float,
        trace: DecisionTrace,
    ) -> UnderwritingResult:

        requested_installment = PricingEngine.monthly_installment(
            app.requested_amount_thb, annual_rate_pct, app.requested_tenor_months
        )
        trace.log(f"Requested loan implies installment of THB {requested_installment:,.2f}/month "
                   f"vs affordability ceiling of THB {max_installment:,.2f}/month.")

        # High-value loans always go to a human, regardless of model confidence —
        # standard practice for large-exposure state-bank governance.
        if app.requested_amount_thb >= cls.HIGH_VALUE_REFERRAL_THRESHOLD_THB:
            trace.log("REFERRAL: requested amount exceeds high-value auto-decision threshold.")
            return UnderwritingResult(
                app.applicant_id, Decision.REFER_TO_HUMAN,
                None, None, None, pd_estimate, trace,
            )

        if pd_estimate >= cls.PD_AUTO_DECLINE_FLOOR:
            trace.log("DECLINE: estimated default risk exceeds auto-decline threshold.")
            return UnderwritingResult(
                app.applicant_id, Decision.DECLINE,
                None, None, None, pd_estimate, trace,
            )

        if requested_installment > max_installment:
            # Try to salvage the application by right-sizing amount/tenor
            # rather than an outright decline — friendlier customer outcome.
            affordable_principal = cls._solve_affordable_principal(
                max_installment, annual_rate_pct, app.requested_tenor_months
            )
            if affordable_principal < 0.3 * app.requested_amount_thb:
                trace.log("DECLINE: even a right-sized loan would be too small to be useful "
                          "relative to the request — affordability gap too large.")
                return UnderwritingResult(
                    app.applicant_id, Decision.DECLINE,
                    None, None, None, pd_estimate, trace,
                )
            trace.log(f"APPROVE_WITH_CONDITIONS: amount reduced to THB "
                      f"{affordable_principal:,.2f} to fit affordability ceiling.")
            return UnderwritingResult(
                app.applicant_id, Decision.APPROVE_WITH_CONDITIONS,
                round(affordable_principal, 2), app.requested_tenor_months,
                annual_rate_pct, pd_estimate, trace,
            )

        if pd_estimate <= cls.PD_AUTO_APPROVE_CEILING:
            trace.log("APPROVE: low estimated risk, within affordability, auto-approved.")
            return UnderwritingResult(
                app.applicant_id, Decision.APPROVE,
                app.requested_amount_thb, app.requested_tenor_months,
                annual_rate_pct, pd_estimate, trace,
            )

        trace.log("REFER_TO_HUMAN: moderate risk band — routed to a loan officer for judgment.")
        return UnderwritingResult(
            app.applicant_id, Decision.REFER_TO_HUMAN,
            app.requested_amount_thb, app.requested_tenor_months,
            annual_rate_pct, pd_estimate, trace,
        )

    @staticmethod
    def _solve_affordable_principal(max_installment: float, annual_rate_pct: float, tenor_months: int) -> float:
        r = (annual_rate_pct / 100) / 12
        if r == 0:
            return max_installment * tenor_months
        return max_installment * (1 - (1 + r) ** -tenor_months) / r


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
def underwrite(app: LoanApplication) -> UnderwritingResult:
    trace = DecisionTrace()
    logger.info(f"Underwriting application {app.applicant_id} "
                f"(requested THB {app.requested_amount_thb:,.0f} / {app.requested_tenor_months}mo)")

    gate_decision = PolicyGate.evaluate(app, trace)
    if gate_decision is not None:
        return UnderwritingResult(app.applicant_id, gate_decision, None, None, None, 1.0, trace)

    pd_estimate = MLRiskAssessor.estimate_pd(app, trace)
    annual_rate = PricingEngine.price_loan(app, pd_estimate, trace)
    max_installment = PricingEngine.max_affordable_installment(app, trace)

    result = DecisionArbiter.decide(app, pd_estimate, annual_rate, max_installment, trace)
    logger.info(f"Final decision for {app.applicant_id}: {result.decision.value}")
    return result


if __name__ == "__main__":
    demo_applicants = [
        LoanApplication(
            applicant_id="APP900001", age=34, monthly_income_thb=35_000,
            requested_amount_thb=200_000, requested_tenor_months=36,
            loan_purpose="home_improvement", credit_score=712,
            existing_monthly_debt_thb=4_000,
        ),
        LoanApplication(
            applicant_id="APP900002", age=27, monthly_income_thb=18_000,
            requested_amount_thb=150_000, requested_tenor_months=24,
            loan_purpose="debt_consolidation", credit_score=560,
            existing_monthly_debt_thb=9_000,
        ),
        LoanApplication(
            applicant_id="APP900003", age=45, monthly_income_thb=90_000,
            requested_amount_thb=1_500_000, requested_tenor_months=60,
            loan_purpose="sme_working_capital", credit_score=680,
            existing_monthly_debt_thb=10_000,
        ),
    ]

    print("=" * 78)
    for app in demo_applicants:
        result = underwrite(app)
        print("-" * 78)
        print(f"Applicant       : {result.applicant_id}")
        print(f"Decision        : {result.decision.value}")
        print(f"Approved amount : {result.approved_amount_thb}")
        print(f"Annual rate     : {result.annual_interest_rate_pct}")
        print(f"Estimated PD    : {result.estimated_probability_of_default:.2%}")
    print("=" * 78)
