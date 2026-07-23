# GSBAIL — AI Project Portfolio

**Author:** Teerapong Panboonyuen (Kao)
**Affiliation:** GSBAIL — Government Savings Bank AI Lab (Government Savings Bank, Thailand)
**Role:** Deputy Director of AI

> A reference portfolio of AI system designs for GSB's AI Lab, spanning
> fraud/AML detection, intelligent document processing, credit risk, loan
> underwriting, and wealth advisory. Every module is a runnable, fully
> commented **mockup on synthetic data** — intended as an architecture
> discussion-starter, not a production system.

---

## 📂 Projects

| # | Project | Path | Core idea |
|---|---------|------|-----------|
| 1 | **Mule Account Detection** (บัญชีม้า) | [`mule_account_detection/train.py`](./mule_account_detection/train.py) | Graph + behavioral anomaly detection to flag accounts used as laundering pass-throughs (fan-in/fan-out bursts, short dwell time, structuring patterns), combining unsupervised triage (Isolation Forest) with supervised refinement (Gradient Boosting) as AMLO-confirmed labels accumulate. |
| 2 | **Intelligent Document Processing for Bank Documents** | [`ocr_bank_documents/ocr_pipeline.py`](./ocr_bank_documents/ocr_pipeline.py) | Hybrid OCR: classical OCR for printed text + modern multimodal LLM vision for handwritten Thai/English bank forms, deposit slips, cheques, and KYC documents — structured JSON extraction with National ID checksum & amount-consistency validation. |
| 3 | **Explainable Alternative-Data Credit Scoring** | [`credit_scoring/credit_scoring_model.py`](./credit_scoring/credit_scoring_model.py) | Bureau + alternative-data (savings behavior, utility bill history, e-wallet cashflow) credit scorecard, blending an auditable WoE/IV scorecard with a Gradient Boosting uplift model and SHAP-based adverse-action reason codes, calibrated to a 300–850 score scale. |
| 4 | **Hybrid Rules + ML Loan Underwriting Engine** | [`loan_underwriting/loan_engine.py`](./loan_underwriting/loan_engine.py) | "ML proposes, policy disposes" underwriting: hard policy gates → ML default-risk estimate → risk-based pricing & affordability check → arbitrated decision (Approve / Approve-with-Conditions / Refer / Decline), with a full audit trace per application. |
| 5 | **AI-Assisted Portfolio Construction (Robo-Advisory)** | [`portfolio_optimization/portfolio_optimizer.py`](./portfolio_optimization/portfolio_optimizer.py) | Risk-profiling questionnaire → Black-Litterman view blending → constrained mean-variance optimization (capital-preservation floor, per-asset caps) → drift-based rebalancing signals, for a GSB fund/deposit/ETF product shelf. |

---

## 🧭 Design principles applied across every project

- **Auditability first** — every decision-making module (loan engine, scoring)
  produces a human-readable trace / reason codes, not just a number.
- **Human-in-the-loop by design** — ML never has unilateral authority over a
  regulated decision; policy gates and referral thresholds are explicit and
  overridable.
- **Built for GSB's mission** — alternative-data credit scoring and
  affordability-first underwriting specifically target the thin-file,
  grassroots, and SME customer base central to GSB's social mandate.
- **Synthetic data only** — every script is self-contained and runnable with
  `python3 <script>.py`; no real customer or transaction data is used,
  read, or required.

## ⚙️ Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run any project directly, e.g.:

```bash
python3 mule_account_detection/train.py --n-accounts 5000 --n-transactions 40000
python3 ocr_bank_documents/ocr_pipeline.py
python3 credit_scoring/credit_scoring_model.py
python3 loan_underwriting/loan_engine.py
python3 portfolio_optimization/portfolio_optimizer.py
```

## 🗺️ Roadmap ideas for GSBAIL

- Replace synthetic generators with governed data connectors (core banking
  warehouse, AMLO STR feed, bureau API, market-data feed) behind the same
  function signatures.
- Add a shared `gsbail_common/` package (logging, config, model registry,
  drift monitoring) once a second project is promoted past prototype stage.
- Wrap the loan/credit pipelines in a model-risk-management (MRM) validation
  harness ahead of any production pilot.

---

*This repository is a personal/portfolio reference project and does not
represent GSB's official systems, data, or policies.*
