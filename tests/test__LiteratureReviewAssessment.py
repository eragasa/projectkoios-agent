from __future__ import annotations

import json
from pathlib import Path

import pytest
from projectkoios.agent.literature_review import (
    ArchivedOllamaLiteratureAssessmentEngine,
    AssessmentBoundary,
    AssessmentContractError,
    AssessmentStatus,
    ClaimEvidenceBundle,
    EvidenceExcerpt,
    LiteratureClaim,
    OllamaAssessmentConfiguration,
    assessment_system_prompt,
    parse_assessment_proposal,
)


def _bundle() -> ClaimEvidenceBundle:
    return ClaimEvidenceBundle(
        claim=LiteratureClaim(
            claim_id="C-001",
            section="Thermodynamics",
            text=(
                "The proposed method applies universally and proves outcome X."
            ),
        ),
        evidence=(
            EvidenceExcerpt(
                label="E1",
                source_id="source:sha256:" + "b" * 64,
                passage_id="passage:sha256:" + "c" * 64,
                citation_key="Example1965",
                physical_page=2,
                text="The theorem applies to equilibrium ensembles.",
            ),
        ),
    )


def _content(**changes: object) -> bytes:
    value: dict[str, object] = {
        "status": "QUALIFIED",
        "summary": "The evidence supports only a narrower statement.",
        "corrected_claim": "The theorem applies to equilibrium ensembles.",
        "assumptions": ["Equilibrium"],
        "citations": ["E1"],
        "source_requests": [],
    }
    value.update(changes)
    return json.dumps(value).encode()


def test__parse_assessment_proposal__accepts_closed_bounded_result() -> None:
    result = parse_assessment_proposal(_bundle(), _content())

    assert result.status is AssessmentStatus.QUALIFIED
    assert result.citations == ("E1",)
    assert result.boundary is AssessmentBoundary.AUTOMATED_UNREVIEWED


@pytest.mark.parametrize(
    "payload",
    [
        b'{"status":"SUPPORTED","status":"QUALIFIED",'
        b'"summary":"x","corrected_claim":null,"assumptions":[],'
        b'"citations":["E1"],"source_requests":[]}',
        _content(extra="not allowed"),
        _content(citations=["E2"]),
        _content(status="SUPPORTED", citations=[]),
        _content(status="CONTRADICTED", corrected_claim=None),
    ],
)
def test__parse_assessment_proposal__rejects_contract_violations(
    payload: bytes,
) -> None:
    with pytest.raises(AssessmentContractError):
        parse_assessment_proposal(_bundle(), payload)


def test__assessment_system_prompt__requires_all_conjuncts() -> None:
    prompt = assessment_system_prompt()

    assert "every material clause" in prompt
    assert "Silence about a material clause is not support" in prompt
    assert "AUTOMATED_UNREVIEWED" in prompt


class _FakeTransport:
    def __init__(self) -> None:
        self.chat_calls = 0

    def request(
        self, route: str, payload: bytes | None, *, timeout_seconds: int
    ) -> bytes:
        assert timeout_seconds == 30
        if route == "/api/tags":
            return json.dumps(
                {"models": [{"name": "example:1b", "digest": "a" * 64}]}
            ).encode()
        if route == "/api/version":
            return b'{"version":"1.2.3"}'
        if route == "/api/chat":
            self.chat_calls += 1
            assert payload is not None
            request = json.loads(payload)
            assert request["stream"] is False
            assert request["think"] is False
            return json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": _content().decode(),
                    }
                }
            ).encode()
        raise AssertionError(route)


def test__archived_engine__pins_runtime_and_replays_exact_exchange(
    tmp_path: Path,
) -> None:
    transport = _FakeTransport()
    configuration = OllamaAssessmentConfiguration(
        model="example:1b",
        model_digest="a" * 64,
        minimum_runtime_version="1.2.0",
        timeout_seconds=30,
    )
    engine = ArchivedOllamaLiteratureAssessmentEngine(
        configuration,
        tmp_path / "archive",
        transport,
    )

    first = engine.propose(_bundle())
    second = engine.propose(_bundle())

    assert first == second
    assert transport.chat_calls == 1
    files = tuple((tmp_path / "archive").iterdir())
    assert len(files) == 3
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in files)
