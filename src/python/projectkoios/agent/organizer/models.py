from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


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


@dataclass(frozen=True)
class FileObservation:
    file_id: str
    root_id: str
    relative_path: str
    name: str
    extension: str
    byte_size: int
    modified_ns: int
    availability: FileAvailability


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
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if not self.suggested_group.strip():
            raise ValueError("suggested_group must be nonempty")
        if not self.rationale.strip():
            raise ValueError("rationale must be nonempty")


@dataclass(frozen=True)
class OrganizerProposalView:
    file_id: str
    root_id: str
    relative_path: str
    name: str
    extension: str
    byte_size: int
    availability: FileAvailability
    para_category: ParaCategory
    life_domain: LifeDomain
    confidence: float
    suggested_group: str
    rationale: str
    model: str
    model_digest: str
    proposed_at: str


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
