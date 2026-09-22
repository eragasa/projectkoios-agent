from __future__ import annotations

import json
from typing import Any

from projectkoios.agent.note_review.models import (
    NoteReviewProposal,
    NoteReviewRequest,
)

_MAX_RESPONSE_BYTES = 64_000
_MAX_LIST_ITEMS = 8
_MAX_ITEM_CHARACTERS = 1_000
_MAX_SUMMARY_CHARACTERS = 2_000
_KEYS = {
    "concerns",
    "missing_definitions",
    "questions",
    "suggested_revisions",
    "strengths",
    "summary",
}


class NoteReviewContractError(ValueError):
    pass


def parse_note_review_proposal(
    request: NoteReviewRequest,
    response: bytes,
) -> NoteReviewProposal:
    if len(response) > _MAX_RESPONSE_BYTES:
        raise NoteReviewContractError("note review exceeds its byte limit")
    raw = _json_object(response)
    if set(raw) != _KEYS:
        raise NoteReviewContractError("note review has unexpected fields")
    return NoteReviewProposal(
        request_id=request.request_id,
        summary=_bounded_line(
            raw["summary"], "summary", _MAX_SUMMARY_CHARACTERS
        ),
        strengths=_line_tuple(raw, "strengths"),
        concerns=_line_tuple(raw, "concerns"),
        missing_definitions=_line_tuple(raw, "missing_definitions"),
        suggested_revisions=_line_tuple(raw, "suggested_revisions"),
        questions=_line_tuple(raw, "questions"),
    )


def note_review_json_schema() -> dict[str, object]:
    properties: dict[str, object] = {
        "summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_SUMMARY_CHARACTERS,
        }
    }
    for key in sorted(_KEYS - {"summary"}):
        properties[key] = {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": _MAX_ITEM_CHARACTERS,
            },
            "maxItems": _MAX_LIST_ITEMS,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(_KEYS),
        "additionalProperties": False,
    }


def _json_object(payload: bytes) -> dict[str, Any]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise NoteReviewContractError("note review must not contain a BOM")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NoteReviewContractError(
                    f"note review contains duplicate member {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NoteReviewContractError(
            "note review is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise NoteReviewContractError("note review must be a JSON object")
    return value


def _line_tuple(value: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = value[key]
    if not isinstance(raw, list) or len(raw) > _MAX_LIST_ITEMS:
        raise NoteReviewContractError(f"{key} is invalid")
    result = tuple(
        _bounded_line(item, f"{key} item", _MAX_ITEM_CHARACTERS)
        for item in raw
    )
    if len(result) != len(set(result)):
        raise NoteReviewContractError(f"{key} items must be unique")
    return result


def _bounded_line(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise NoteReviewContractError(f"{label} is invalid")
    if value != value.strip():
        raise NoteReviewContractError(f"{label} has surrounding whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise NoteReviewContractError(f"{label} must be one printable line")
    return value
