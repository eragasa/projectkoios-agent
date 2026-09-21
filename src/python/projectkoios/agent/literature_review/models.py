from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AssessmentStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    QUALIFIED = "QUALIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"


class AssessmentBoundary(StrEnum):
    AUTOMATED_UNREVIEWED = "AUTOMATED_UNREVIEWED"


@dataclass(frozen=True)
class LiteratureClaim:
    claim_id: str
    section: str
    text: str


@dataclass(frozen=True)
class EvidenceExcerpt:
    label: str
    source_id: str
    passage_id: str
    citation_key: str | None
    physical_page: int
    text: str


@dataclass(frozen=True)
class ClaimEvidenceBundle:
    claim: LiteratureClaim
    evidence: tuple[EvidenceExcerpt, ...]


@dataclass(frozen=True)
class ClaimAssessmentProposal:
    claim_id: str
    status: AssessmentStatus
    summary: str
    corrected_claim: str | None
    assumptions: tuple[str, ...]
    citations: tuple[str, ...]
    source_requests: tuple[str, ...]
    boundary: AssessmentBoundary = AssessmentBoundary.AUTOMATED_UNREVIEWED
