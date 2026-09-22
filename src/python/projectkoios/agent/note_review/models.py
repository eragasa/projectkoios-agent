from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_SOURCE_CHARACTERS = 96_000
_MAX_SOURCE_BYTES = 128_000
_MAX_TITLE_CHARACTERS = 200


class ReviewProfile(StrEnum):
    SCIENTIFIC_NOTE = "SCIENTIFIC_NOTE"


class ReviewBoundary(StrEnum):
    AUTOMATED_UNREVIEWED = "AUTOMATED_UNREVIEWED"


@dataclass(frozen=True)
class NoteReviewRequest:
    request_id: str
    title: str
    source_sha256: str
    source_markdown: str
    profile: ReviewProfile = ReviewProfile.SCIENTIFIC_NOTE

    def __post_init__(self) -> None:
        if _REQUEST_ID.fullmatch(self.request_id) is None:
            raise ValueError("request ID is invalid")
        if _SHA256.fullmatch(self.source_sha256) is None:
            raise ValueError("source SHA-256 is invalid")
        if not self.title or len(self.title) > _MAX_TITLE_CHARACTERS:
            raise ValueError("title is invalid")
        if self.title != self.title.strip():
            raise ValueError("title has surrounding whitespace")
        if any(
            ord(character) < 32 or ord(character) == 127
            for character in self.title
        ):
            raise ValueError("title must be one printable line")
        if not isinstance(self.profile, ReviewProfile):
            raise ValueError("review profile is invalid")
        if not self.source_markdown:
            raise ValueError("source Markdown must be nonempty")
        if len(self.source_markdown) > _MAX_SOURCE_CHARACTERS:
            raise ValueError("source Markdown exceeds its character limit")
        try:
            source_bytes = self.source_markdown.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError("source Markdown is not valid UTF-8") from error
        if len(source_bytes) > _MAX_SOURCE_BYTES:
            raise ValueError("source Markdown exceeds its byte limit")
        if hashlib.sha256(source_bytes).hexdigest() != self.source_sha256:
            raise ValueError("source SHA-256 does not match source Markdown")


@dataclass(frozen=True)
class NoteReviewProposal:
    request_id: str
    summary: str
    strengths: tuple[str, ...]
    concerns: tuple[str, ...]
    missing_definitions: tuple[str, ...]
    suggested_revisions: tuple[str, ...]
    questions: tuple[str, ...]
    boundary: ReviewBoundary = ReviewBoundary.AUTOMATED_UNREVIEWED
