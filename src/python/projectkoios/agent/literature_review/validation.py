from __future__ import annotations

import json
import re
from typing import Any

from projectkoios.agent.literature_review.models import (
    AssessmentStatus,
    ClaimAssessmentProposal,
    ClaimEvidenceBundle,
)

_MAX_RESPONSE_BYTES = 64_000
_MAX_LIST_ITEMS = 8
_EVIDENCE_LABEL = re.compile(r"E[1-9][0-9]*")
_KEYS = {
    "assumptions",
    "citations",
    "corrected_claim",
    "source_requests",
    "status",
    "summary",
}


class AssessmentContractError(ValueError):
    pass


def parse_assessment_proposal(
    bundle: ClaimEvidenceBundle,
    response: bytes,
) -> ClaimAssessmentProposal:
    if len(response) > _MAX_RESPONSE_BYTES:
        raise AssessmentContractError("assessment exceeds its byte limit")
    raw = _json_object(response)
    if set(raw) != _KEYS:
        raise AssessmentContractError("assessment has unexpected fields")
    try:
        status = AssessmentStatus(_string(raw, "status", 32))
    except ValueError as error:
        raise AssessmentContractError("assessment status is invalid") from error
    summary = _string(raw, "summary", 2_000)
    corrected_raw = raw["corrected_claim"]
    corrected_claim = (
        None
        if corrected_raw is None
        else _bounded_string(corrected_raw, "corrected_claim", 4_096)
    )
    assumptions = _string_tuple(raw, "assumptions", 1_000)
    citations = _string_tuple(raw, "citations", 16, max_items=12)
    source_requests = _string_tuple(raw, "source_requests", 1_000)
    labels = {item.label for item in bundle.evidence}
    if any(
        _EVIDENCE_LABEL.fullmatch(label) is None or label not in labels
        for label in citations
    ):
        raise AssessmentContractError("assessment cites unavailable evidence")
    if len(citations) != len(set(citations)):
        raise AssessmentContractError("assessment citations must be unique")
    if status is not AssessmentStatus.UNRESOLVED and not citations:
        raise AssessmentContractError(
            "resolved assessment requires cited evidence"
        )
    if (
        status in {AssessmentStatus.QUALIFIED, AssessmentStatus.CONTRADICTED}
        and corrected_claim is None
    ):
        raise AssessmentContractError(
            "qualified or contradicted assessment requires correction"
        )
    return ClaimAssessmentProposal(
        claim_id=bundle.claim.claim_id,
        status=status,
        summary=summary,
        corrected_claim=corrected_claim,
        assumptions=assumptions,
        citations=citations,
        source_requests=source_requests,
    )


def assessment_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": sorted(status.value for status in AssessmentStatus),
            },
            "summary": {"type": "string"},
            "corrected_claim": {"type": ["string", "null"]},
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": _MAX_LIST_ITEMS,
            },
            "citations": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^E[1-9][0-9]*$"},
                "maxItems": 12,
            },
            "source_requests": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": _MAX_LIST_ITEMS,
            },
        },
        "required": sorted(_KEYS),
        "additionalProperties": False,
    }


def _json_object(payload: bytes) -> dict[str, Any]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise AssessmentContractError("assessment must not contain a BOM")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssessmentContractError(
                    f"assessment contains duplicate member {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssessmentContractError("assessment is not valid JSON") from error
    if not isinstance(value, dict):
        raise AssessmentContractError("assessment must be a JSON object")
    return value


def _string(value: dict[str, Any], key: str, limit: int) -> str:
    return _bounded_string(value[key], key, limit)


def _bounded_string(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise AssessmentContractError(f"{label} is invalid")
    return value


def _string_tuple(
    value: dict[str, Any],
    key: str,
    character_limit: int,
    *,
    max_items: int = _MAX_LIST_ITEMS,
) -> tuple[str, ...]:
    raw = value[key]
    if not isinstance(raw, list) or len(raw) > max_items:
        raise AssessmentContractError(f"{key} is invalid")
    result = tuple(
        _bounded_string(item, f"{key} item", character_limit) for item in raw
    )
    return result
