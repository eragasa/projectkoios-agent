from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_RELATIVE_PATH_CHARACTERS = 1_024
_MAX_RELATIVE_PATH_BYTES = 4_096
_MAX_EXTENSION_CHARACTERS = 32
_MAX_GROUP_CHARACTERS = 120
_MAX_RATIONALE_CHARACTERS = 500
_MAX_MODEL_CHARACTERS = 500
_MAX_BYTE_SIZE = 2**63 - 1
_MIN_TIMESTAMP_NS = -(2**63)
_MAX_TIMESTAMP_NS = 2**63 - 1
_MAX_NAME_BYTES = 255


class OrganizerControlMode(StrEnum):
    ON = "on"
    PAUSE = "pause"
    OFF = "off"


class OrganizerActivity(StrEnum):
    OFF = "off"
    PAUSED = "paused"
    IDLE = "idle"
    DISCOVERING = "discovering"
    SCANNING = "scanning"
    CLASSIFYING = "classifying"
    FAILED = "failed"


class FileAvailability(StrEnum):
    LOCAL = "local"
    CLOUD_PLACEHOLDER = "cloud_placeholder"
    INACCESSIBLE = "inaccessible"


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
class CloudRoot:
    root_id: str
    label: str
    path: Path
    provider: str

    def __post_init__(self) -> None:
        _validate_sha256(self.root_id, "cloud root ID")
        _validate_bounded_line(self.label, "cloud root label", 255)
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("cloud root path must be absolute")
        _validate_bounded_line(self.provider, "cloud provider", 120)


@dataclass(frozen=True)
class FileObservation:
    """Bounded metadata supplied by a caller; no file access is implied."""

    file_id: str
    relative_path: str
    extension: str
    byte_size: int

    def __post_init__(self) -> None:
        _validate_sha256(self.file_id, "file ID")
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
class CatalogFileObservation(FileObservation):
    """A file observation enriched with local catalog metadata."""

    root_id: str
    name: str
    modified_ns: int
    availability: FileAvailability

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_sha256(self.root_id, "cloud root ID")
        _validate_line(self.name, "file name")
        try:
            encoded_name = self.name.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError("file name is not valid UTF-8") from error
        if (
            len(encoded_name) > _MAX_NAME_BYTES
            or self.name in {".", ".."}
            or "/" in self.name
            or self.relative_path.rsplit("/", 1)[-1] != self.name
        ):
            raise ValueError("file name is invalid")
        if isinstance(self.modified_ns, bool) or not isinstance(
            self.modified_ns, int
        ):
            raise ValueError("file modification time must be an integer")
        if not _MIN_TIMESTAMP_NS <= self.modified_ns <= _MAX_TIMESTAMP_NS:
            raise ValueError("file modification time is outside its bound")
        if not isinstance(self.availability, FileAvailability):
            raise ValueError("file availability is invalid")


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
        _validate_sha256(self.file_id, "file ID")
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
        _validate_sha256(self.model_digest, "model digest")


@dataclass(frozen=True)
class OrganizerEvent:
    sequence: int
    occurred_at: str
    kind: str
    message: str
    root_id: str | None = None
    file_id: str | None = None


@dataclass(frozen=True)
class OrganizerStatus:
    desired_mode: OrganizerControlMode
    activity: OrganizerActivity
    discovered_roots: int
    observed_files: int
    local_files: int
    placeholder_files: int
    proposed_files: int
    last_event_sequence: int
    current_root_id: str | None
    current_relative_path: str | None
    last_error: str | None


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hex")


def _validate_relative_path(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("relative path is invalid")
    if len(value) > _MAX_RELATIVE_PATH_CHARACTERS:
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
    if not isinstance(value, str) or not value or len(value) > limit:
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
