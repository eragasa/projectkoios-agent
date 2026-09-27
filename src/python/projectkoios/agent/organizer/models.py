from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_RELATIVE_PATH_CHARACTERS = 1_024
_MAX_RELATIVE_PATH_BYTES = 4_096
_MAX_EXTENSION_CHARACTERS = 32
_MAX_GROUP_CHARACTERS = 120
_MAX_RATIONALE_CHARACTERS = 500
_MAX_MODEL_CHARACTERS = 500
_MAX_BYTE_SIZE = 2**63 - 1


class ParaCategory(StrEnum):
    PROJECT = "project"
    AREA = "area"
    RESOURCE = "resource"
    ARCHIVE = "archive"
    INBOX = "inbox"


class LifeDomain(StrEnum):
    RESEARCH = "research"
    TEACHING = "teaching"
    SOFTWARE = "software"
    BUSINESS = "business"
    PERSONAL = "personal"
    ADMINISTRATION = "administration"
    FINANCE = "finance"
    HEALTH = "health"
    MEDIA = "media"
    OTHER = "other"


@dataclass(frozen=True)
class FileObservation:
    """Bounded metadata supplied by a caller; no file access is implied."""

    file_id: str
    relative_path: str
    extension: str
    byte_size: int

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.file_id) is None:
            raise ValueError("file ID must be 64 lowercase hex")
        _validate_relative_path(self.relative_path)
        if len(self.extension) > _MAX_EXTENSION_CHARACTERS:
            raise ValueError("file extension exceeds its character limit")
        if self.extension and not self.extension.startswith("."):
            raise ValueError("file extension must be empty or start with a dot")
        _validate_line(self.extension, "file extension", allow_empty=True)
        if isinstance(self.byte_size, bool) or not isinstance(
            self.byte_size, int
        ):
            raise ValueError("file byte size must be an integer")
        if not 0 <= self.byte_size <= _MAX_BYTE_SIZE:
            raise ValueError("file byte size is outside its bound")


@dataclass(frozen=True)
class CategorizationProposal:
    file_id: str
    para_category: ParaCategory
    life_domain: LifeDomain
    confidence: float
    suggested_group: str
    rationale: str
    model: str
    model_digest: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.file_id) is None:
            raise ValueError("file ID must be 64 lowercase hex")
        if not isinstance(self.para_category, ParaCategory):
            raise ValueError("PARA category is invalid")
        if not isinstance(self.life_domain, LifeDomain):
            raise ValueError("life domain is invalid")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, int | float
        ):
            raise ValueError("confidence must be numeric")
        if (
            not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be between zero and one")
        _validate_bounded_line(
            self.suggested_group,
            "suggested group",
            _MAX_GROUP_CHARACTERS,
        )
        _validate_bounded_line(
            self.rationale,
            "rationale",
            _MAX_RATIONALE_CHARACTERS,
        )
        _validate_bounded_line(self.model, "model", _MAX_MODEL_CHARACTERS)
        if _SHA256.fullmatch(self.model_digest) is None:
            raise ValueError("model digest must be 64 lowercase hex")


def _validate_relative_path(value: str) -> None:
    if not value or len(value) > _MAX_RELATIVE_PATH_CHARACTERS:
        raise ValueError("relative path is invalid")
    _validate_line(value, "relative path")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("relative path is not valid UTF-8") from error
    if len(encoded) > _MAX_RELATIVE_PATH_BYTES:
        raise ValueError("relative path exceeds its byte limit")
    if value.startswith("/"):
        raise ValueError("relative path must not be absolute")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("relative path contains an invalid segment")


def _validate_bounded_line(value: str, label: str, limit: int) -> None:
    if not value or len(value) > limit:
        raise ValueError(f"{label} is invalid")
    if value != value.strip():
        raise ValueError(f"{label} has surrounding whitespace")
    _validate_line(value, label)


def _validate_line(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} is invalid")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{label} must be one printable line")
