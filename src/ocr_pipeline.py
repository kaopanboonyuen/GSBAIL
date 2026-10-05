#!/usr/bin/env python3
# ==============================================================================
#   ██████╗ ███████╗██████╗  █████╗ ██╗██╗
#  ██╔════╝ ██╔════╝██╔══██╗██╔══██╗██║██║
#  ██║  ███╗███████╗██████╔╝███████║██║██║
#  ██║   ██║╚════██║██╔══██╗██╔══██║██║██║
#  ╚██████╔╝███████║██████╔╝██║  ██║██║███████╗
#   ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝
#
#  PROJECT     : Intelligent Document Processing (IDP) for Banking Documents
#  MODULE      : ocr_pipeline.py — Hybrid OCR + LLM Vision Extraction Engine
#  DESCRIPTION : End-to-end pipeline for reading BOTH printed-font and
#                handwritten Thai/English bank documents (deposit slips,
#                loan applications, KYC forms, cheques, passbooks) and
#                converting them into structured, validated JSON records.
#
#  AUTHOR      : Teerapong Panboonyuen (Kao)
#  AFFILIATION : GSBAIL — Government Savings Bank AI Lab
#                (Government Savings Bank, Thailand)
#  ROLE        : Vice President of AI
#  STATUS      : Mockup / Reference Architecture — no real customer
#                documents are read or stored by this script.
# ==============================================================================

"""
Intelligent Document Processing for Bank Documents
=====================================================

Why "classic OCR" alone isn't enough
--------------------------------------
Traditional OCR engines (Tesseract, PaddleOCR, EasyOCR) are excellent at
*printed* text but degrade sharply on:
    - Thai cursive / fast handwriting on deposit slips and loan forms
    - Mixed Thai-English forms with stamps, signatures, and noisy scans
    - Structured layouts where FIELD MEANING matters more than raw text
      (e.g. distinguishing "ชื่อผู้ฝาก" from "จำนวนเงิน" from a signature box)

Our approach: a THREE-STAGE HYBRID pipeline
----------------------------------------------
    Stage 1 — Layout & Field Localization
        A layout-detection model (e.g. LayoutLMv3 / Donut-style) segments
        the document into semantic regions: header, field labels, field
        values, signature box, stamp, table rows.

    Stage 2 — Dual-Path Text Recognition
        (a) Printed text  -> fast classical OCR (PaddleOCR/Tesseract) for
            speed and cost efficiency on machine-printed regions.
        (b) Handwritten / low-confidence text -> routed to a modern
            multimodal LLM (vision-language model) which is dramatically
            more robust to handwriting, cross-outs, and stylistic variance
            than classical OCR engines.

    Stage 3 — Structured Extraction + Validation
        The LLM is prompted with a strict JSON schema per document type
        (deposit slip / loan application / KYC form / cheque) and asked to
        both TRANSCRIBE and NORMALIZE (e.g. Thai date -> ISO date, Thai
        digits -> Arabic digits, running total 'ตัวอักษร' -> numeric).
        A rule-based validator then cross-checks extracted fields
        (checksum on account numbers, amount-in-words vs amount-in-digits
        consistency, National ID checksum, etc.) before the record is
        accepted into the downstream core-banking workflow.

This script is a MOCKUP: the actual LLM vision call is abstracted behind
`LLMVisionClient`, which you would point at your provider of choice
(e.g. an Anthropic-compatible /v1/messages endpoint with image input).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [GSBAIL::IDP] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ocr_bank_documents")


# ------------------------------------------------------------------------------
# Document taxonomy
# ------------------------------------------------------------------------------
class DocumentType(str, Enum):
    DEPOSIT_SLIP = "deposit_slip"          # ใบนำฝาก
    WITHDRAWAL_SLIP = "withdrawal_slip"    # ใบถอนเงิน
    LOAN_APPLICATION = "loan_application"  # ใบสมัครสินเชื่อ
    KYC_FORM = "kyc_form"                  # แบบฟอร์ม KYC
    CHEQUE = "cheque"                      # เช็ค
    PASSBOOK_PAGE = "passbook_page"        # สมุดบัญชี


# Per-document-type JSON extraction schema. This is what we send to the LLM
# as the "contract" it must fill in — keeps outputs structured & auditable.
DOCUMENT_SCHEMAS: dict[DocumentType, dict[str, str]] = {
    DocumentType.DEPOSIT_SLIP: {
        "depositor_name": "string",
        "account_number": "string (10 digits, may contain hyphens in source)",
        "branch_code": "string",
        "amount_digits": "number",
        "amount_in_words_thai": "string",
        "date_iso": "string (YYYY-MM-DD, convert from Thai Buddhist Era)",
        "deposit_channel": "string (cash | cheque | transfer)",
    },
    DocumentType.LOAN_APPLICATION: {
        "applicant_full_name": "string",
        "national_id": "string (13 digits)",
        "requested_amount": "number",
        "loan_purpose": "string",
        "monthly_income_declared": "number",
        "employer_name": "string",
        "signature_present": "boolean",
    },
    DocumentType.KYC_FORM: {
        "full_name": "string",
        "national_id": "string (13 digits)",
        "date_of_birth_iso": "string (YYYY-MM-DD)",
        "address": "string",
        "occupation": "string",
        "estimated_monthly_income": "number",
        "pep_declared": "boolean",  # politically exposed person self-declaration
    },
    DocumentType.CHEQUE: {
        "payee_name": "string",
        "amount_digits": "number",
        "amount_in_words_thai": "string",
        "date_iso": "string (YYYY-MM-DD)",
        "cheque_number": "string",
        "drawer_signature_present": "boolean",
    },
    DocumentType.PASSBOOK_PAGE: {
        "account_number": "string",
        "account_holder_name": "string",
        "transactions": "array of {date_iso, description, amount, balance}",
    },
    DocumentType.WITHDRAWAL_SLIP: {
        "account_holder_name": "string",
        "account_number": "string",
        "amount_digits": "number",
        "amount_in_words_thai": "string",
        "date_iso": "string (YYYY-MM-DD)",
    },
}


# ------------------------------------------------------------------------------
# Data contracts
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class OCRRegion:
    """A single localized region on the document image."""
    region_id: str
    label: str                 # e.g. "field_label", "field_value", "signature", "stamp"
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1
    is_handwritten: bool
    classic_ocr_text: Optional[str] = None
    classic_ocr_confidence: Optional[float] = None


@dataclasses.dataclass
class ExtractionResult:
    document_type: DocumentType
    fields: dict[str, Any]
    field_confidence: dict[str, float]
    validation_errors: list[str]
    raw_regions: list[OCRRegion]

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0


# ------------------------------------------------------------------------------
# Stage 1 — Layout detection (mocked)
# ------------------------------------------------------------------------------
class LayoutDetector:
    """Wraps a layout-detection model (e.g. LayoutLMv3, Donut, or a
    fine-tuned YOLO for form-field boxes). Mocked here with deterministic
    dummy regions so the pipeline is runnable without model weights.
    """

    def detect(self, image_path: Path) -> list[OCRRegion]:
        logger.info(f"[Stage 1] Detecting layout regions for {image_path.name}")
        # In production: run the layout model and return real bounding boxes.
        return [
            OCRRegion("r1", "field_value", (100, 50, 400, 90), is_handwritten=False),
            OCRRegion("r2", "field_value", (100, 100, 400, 140), is_handwritten=True),
            OCRRegion("r3", "field_value", (100, 150, 400, 190), is_handwritten=True),
            OCRRegion("r4", "signature", (100, 400, 400, 460), is_handwritten=True),
            OCRRegion("r5", "stamp", (420, 400, 520, 460), is_handwritten=False),
        ]


# ------------------------------------------------------------------------------
# Stage 2a — Classical OCR for printed text
# ------------------------------------------------------------------------------
class ClassicOCREngine:
    """Wraps a fast classical OCR engine (PaddleOCR / Tesseract) used for
    machine-printed regions where speed & cost matter more than robustness
    to messy handwriting.
    """

    def recognize(self, image_path: Path, region: OCRRegion) -> tuple[str, float]:
        # In production: crop region.bbox from image_path and run the OCR engine.
        logger.debug(f"[Stage 2a] Classic OCR on region {region.region_id}")
        return "ธนาคารออมสิน สาขาสำนักงานใหญ่", 0.94  # mocked printed-text result


# ------------------------------------------------------------------------------
# Stage 2b — Modern LLM vision client for handwriting & structured extraction
# ------------------------------------------------------------------------------
class LLMVisionClient(ABC):
    """Abstract interface for a modern multimodal LLM used to (a) transcribe
    handwritten regions and (b) perform full structured extraction with a
    strict JSON schema. Implement `call()` against your actual provider.
    """

    @abstractmethod
    def call(self, image_path: Path, prompt: str) -> str:
        """Returns raw text/JSON string response from the vision-language model."""
        raise NotImplementedError


class MockLLMVisionClient(LLMVisionClient):
    """Deterministic mock so this script runs end-to-end without API keys.
    Swap this out for a real client, e.g.:

        class AnthropicVisionClient(LLMVisionClient):
            def call(self, image_path, prompt):
                # base64-encode image_path, call /v1/messages with an
                # image content block + prompt, return response text.
                ...
    """

    def call(self, image_path: Path, prompt: str) -> str:
        logger.debug("[Stage 2b] (mock) LLM vision call for structured extraction")
        mock_response = {
            "depositor_name": "สมชาย ใจดี",
            "account_number": "0201234567",
            "branch_code": "0201",
            "amount_digits": 15000.00,
            "amount_in_words_thai": "หนึ่งหมื่นห้าพันบาทถ้วน",
            "date_iso": "2026-07-20",
            "deposit_channel": "cash",
        }
        return json.dumps(mock_response, ensure_ascii=False)


def build_extraction_prompt(document_type: DocumentType) -> str:
    """Builds a strict, schema-constrained prompt for the vision LLM. This is
    the actual prompt-engineering artifact you'd iterate on in production.
    """
    schema = DOCUMENT_SCHEMAS[document_type]
    schema_desc = "\n".join(f"  - {k}: {v}" for k, v in schema.items())
    return f"""
You are a meticulous bank-document transcriber. You will be shown an image
of a Thai bank document of type "{document_type.value}", which may contain
BOTH machine-printed text and handwritten Thai/English text.

Transcribe the document and return ONLY a single valid JSON object with
exactly these fields (no extra commentary, no markdown fences):

{schema_desc}

Rules:
  - Convert Thai Buddhist-Era years to Gregorian (subtract 543).
  - Convert Thai numerals to Arabic numerals.
  - If a field is illegible or absent, set its value to null.
  - Preserve Thai text in Thai script; do not transliterate names.
""".strip()


# ------------------------------------------------------------------------------
# Stage 3 — Field-level validation rules
# ------------------------------------------------------------------------------
class DocumentValidator:
    """Rule-based cross-checks applied after LLM extraction, before the
    record is allowed to flow into core-banking / loan-origination systems.
    """

    @staticmethod
    def validate_national_id(national_id: Optional[str]) -> Optional[str]:
        if national_id is None:
            return None
        digits = re.sub(r"\D", "", national_id)
        if len(digits) != 13:
            return "National ID must contain 13 digits."
        weights = list(range(13, 1, -1))
        checksum = sum(int(d) * w for d, w in zip(digits[:12], weights)) % 11
        expected_check_digit = (11 - checksum) % 10
        if expected_check_digit != int(digits[12]):
            return "National ID checksum failed — likely a transcription error."
        return None

    @staticmethod
    def validate_amount_consistency(
        amount_digits: Optional[float], amount_words: Optional[str]
    ) -> Optional[str]:
        if amount_digits is None or not amount_words:
            return None
        # Placeholder: a full implementation would parse Thai number-words
        # (บาทถ้วน / สตางค์) back into a numeric value and compare.
        if amount_digits <= 0:
            return "Amount in digits must be positive."
        return None

    @classmethod
    def validate(cls, document_type: DocumentType, fields: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if "national_id" in fields:
            err = cls.validate_national_id(fields.get("national_id"))
            if err:
                errors.append(err)
        if "amount_digits" in fields and "amount_in_words_thai" in fields:
            err = cls.validate_amount_consistency(
                fields.get("amount_digits"), fields.get("amount_in_words_thai")
            )
            if err:
                errors.append(err)
        return errors


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
class BankDocumentIDPPipeline:
    """Top-level pipeline tying together layout detection, dual-path OCR,
    LLM-based structured extraction, and validation.
    """

    def __init__(self, llm_client: Optional[LLMVisionClient] = None):
        self.layout_detector = LayoutDetector()
        self.classic_ocr = ClassicOCREngine()
        self.llm_client = llm_client or MockLLMVisionClient()

    def process(self, image_path: Path, document_type: DocumentType) -> ExtractionResult:
        logger.info(
            f"Processing '{image_path.name}' as document_type={document_type.value}"
        )
        regions = self.layout_detector.detect(image_path)

        for region in regions:
            if not region.is_handwritten and region.label == "field_value":
                text, conf = self.classic_ocr.recognize(image_path, region)
                region.classic_ocr_text = text
                region.classic_ocr_confidence = conf

        prompt = build_extraction_prompt(document_type)
        llm_raw_response = self.llm_client.call(image_path, prompt)

        try:
            fields = json.loads(llm_raw_response)
        except json.JSONDecodeError:
            logger.error("LLM response was not valid JSON — falling back to empty record.")
            fields = {}

        field_confidence = {k: 0.9 for k in fields}  # mocked; real system would
        # derive per-field confidence from LLM logprobs / self-consistency voting.

        validation_errors = DocumentValidator.validate(document_type, fields)

        result = ExtractionResult(
            document_type=document_type,
            fields=fields,
            field_confidence=field_confidence,
            validation_errors=validation_errors,
            raw_regions=regions,
        )

        status = "VALID" if result.is_valid else "NEEDS_REVIEW"
        logger.info(f"Extraction complete. Status={status}. Fields={list(fields.keys())}")
        return result


# ------------------------------------------------------------------------------
# Demo entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = BankDocumentIDPPipeline()
    demo_image = Path("./sample_documents/deposit_slip_001.jpg")  # not read by mock
    result = pipeline.process(demo_image, DocumentType.DEPOSIT_SLIP)

    print(json.dumps(dataclasses.asdict(result), indent=2, ensure_ascii=False, default=str))
