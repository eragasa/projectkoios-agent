from __future__ import annotations

from projectkoios.agent.note_review.models import (
    NoteReviewProposal,
    NoteReviewRequest,
)


def render_note_review(
    request: NoteReviewRequest,
    proposal: NoteReviewProposal,
) -> str:
    if proposal.request_id != request.request_id:
        raise ValueError("proposal request ID does not match request")

    lines = [
        "# Automated note review",
        "",
        f"- Source title: {_escape_inline(request.title)}",
        f"- Request: `{request.request_id}`",
        f"- Source SHA-256: `{request.source_sha256}`",
        f"- Profile: `{request.profile.value}`",
        f"- Boundary: `{proposal.boundary.value}`",
        "",
        "## Summary",
        "",
        _escape_inline(proposal.summary),
    ]
    sections = (
        ("Strengths", proposal.strengths),
        ("Concerns", proposal.concerns),
        ("Missing definitions", proposal.missing_definitions),
        ("Suggested revisions", proposal.suggested_revisions),
        ("Questions", proposal.questions),
    )
    for heading, items in sections:
        lines.extend(["", f"## {heading}", ""])
        if items:
            lines.extend(f"- {_escape_inline(item)}" for item in items)
        else:
            lines.append("- None proposed.")
    return "\n".join(lines) + "\n"


def _escape_inline(value: str) -> str:
    return value.replace("\\", "\\\\").replace("<", "&lt;").replace(">", "&gt;")
