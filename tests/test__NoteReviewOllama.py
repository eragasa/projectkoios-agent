from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from projectkoios.agent.note_review import (
    LocalNoteReviewService,
    NoteReviewApplicationError,
    OllamaNoteReviewBackend,
    OllamaNoteReviewConfiguration,
    OllamaNoteReviewError,
    note_review_json_schema,
)


def _content() -> str:
    return json.dumps(
        {
            "summary": "The note introduces entropy as a state function.",
            "strengths": ["The central statement is concise."],
            "concerns": ["The thermodynamic scope is not stated."],
            "missing_definitions": ["Define state function."],
            "suggested_revisions": [
                "State the intended thermodynamic scope."
            ],
            "questions": ["Which class of systems is intended?"],
        }
    )


class _FakeTransport:
    def __init__(
        self,
        *,
        digest: str = "a" * 64,
        version: str = "1.2.3",
    ) -> None:
        self.digest = digest
        self.version = version
        self.calls: list[tuple[str, bytes | None, int]] = []

    def request(
        self, route: str, payload: bytes | None, *, timeout_seconds: int
    ) -> bytes:
        self.calls.append((route, payload, timeout_seconds))
        if route == "/api/tags":
            return json.dumps(
                {
                    "models": [
                        {"name": "example:1b", "digest": self.digest}
                    ]
                }
            ).encode()
        if route == "/api/version":
            return json.dumps({"version": self.version}).encode()
        if route == "/api/chat":
            return json.dumps(
                {"message": {"role": "assistant", "content": _content()}}
            ).encode()
        raise AssertionError(route)


def _configuration() -> OllamaNoteReviewConfiguration:
    return OllamaNoteReviewConfiguration(
        model="example:1b",
        model_digest="a" * 64,
        minimum_runtime_version="1.2.0",
        seed=17,
        num_ctx=8_192,
        num_predict=1_024,
        timeout_seconds=30,
    )


def test__ollama_note_review_backend__pins_runtime_and_request() -> None:
    transport = _FakeTransport()
    backend = OllamaNoteReviewBackend(_configuration(), transport)
    schema = note_review_json_schema()

    completion = backend.complete(
        system_prompt="system",
        user_prompt="user",
        response_schema=schema,
    )

    assert completion.response == _content().encode()
    assert completion.evidence.backend == "ollama-loopback"
    assert completion.evidence.model == "example:1b"
    assert completion.evidence.model_digest == "a" * 64
    assert completion.evidence.runtime_version == "1.2.3"
    assert completion.exchange_request is not None
    request = json.loads(completion.exchange_request)
    assert request == {
        "format": schema,
        "messages": [
            {"content": "system", "role": "system"},
            {"content": "user", "role": "user"},
        ],
        "model": "example:1b",
        "options": {
            "num_ctx": 8_192,
            "num_predict": 1_024,
            "seed": 17,
            "temperature": 0,
        },
        "stream": False,
        "think": False,
    }
    assert [call[0] for call in transport.calls] == [
        "/api/tags",
        "/api/version",
        "/api/chat",
    ]
    assert all(call[2] == 30 for call in transport.calls)


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (_FakeTransport(digest="b" * 64), "digest does not match"),
        (_FakeTransport(version="1.1.9"), "below minimum"),
    ],
)
def test__ollama_note_review_backend__rejects_unpinned_runtime(
    transport: _FakeTransport,
    message: str,
) -> None:
    backend = OllamaNoteReviewBackend(_configuration(), transport)

    with pytest.raises(OllamaNoteReviewError, match=message):
        backend.complete(
            system_prompt="system",
            user_prompt="user",
            response_schema=note_review_json_schema(),
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434",
        "http://127.0.0.1:8080",
        "http://ollama.internal:11434",
    ],
)
def test__ollama_note_review_configuration__rejects_nonallowlisted_endpoint(
    endpoint: str,
) -> None:
    with pytest.raises(ValueError, match="loopback"):
        OllamaNoteReviewConfiguration(
            model="example:1b",
            model_digest="a" * 64,
            minimum_runtime_version="1.2.0",
            endpoint=endpoint,
        )


def test__local_note_review_service__archives_exact_ollama_exchange(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    source = tmp_path / "note.md"
    source.write_text(
        "# Entropy\n\nEntropy is a state function.", encoding="utf-8"
    )
    archive = root / "model-exchanges"
    output = root / "review.md"
    transport = _FakeTransport()
    service = LocalNoteReviewService(
        OllamaNoteReviewBackend(_configuration(), transport)
    )

    first = service.review(
        request_id="review:entropy-ollama-001",
        title="Entropy note",
        source_path=source,
        artifact_root=root,
        output_path=output,
        archive_directory=archive,
    )
    second = service.review(
        request_id="review:entropy-ollama-001",
        title="Entropy note",
        source_path=source,
        artifact_root=root,
        output_path=root / "replay.md",
        archive_directory=archive,
    )

    assert len(transport.calls) == 3
    assert len(tuple(archive.iterdir())) == 5
    exchange_request = next(archive.glob("*.exchange.request.json"))
    exchange_response = next(archive.glob("*.exchange.response.json"))
    receipt = json.loads(first.receipt_path.read_bytes())
    assert receipt["schema"] == "koios.note-review-receipt.v2"
    assert receipt["backend"]["model"] == "example:1b"
    assert receipt["backend"]["runtime_version"] == "1.2.3"
    assert receipt["backend"]["exchange_request_sha256"] == (
        hashlib.sha256(exchange_request.read_bytes()).hexdigest()
    )
    assert receipt["backend"]["exchange_response_sha256"] == (
        hashlib.sha256(exchange_response.read_bytes()).hexdigest()
    )
    assert first.review_sha256 == second.review_sha256
    assert first.backend_evidence == second.backend_evidence


def test__local_note_review_service__separates_backend_configurations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    source = tmp_path / "note.md"
    source.write_text("# Entropy", encoding="utf-8")
    archive = root / "model-exchanges"
    first_transport = _FakeTransport()
    second_transport = _FakeTransport()
    first = LocalNoteReviewService(
        OllamaNoteReviewBackend(_configuration(), first_transport)
    )
    changed_configuration = OllamaNoteReviewConfiguration(
        model="example:1b",
        model_digest="a" * 64,
        minimum_runtime_version="1.2.0",
        seed=18,
        num_ctx=8_192,
        num_predict=1_024,
        timeout_seconds=30,
    )
    second = LocalNoteReviewService(
        OllamaNoteReviewBackend(changed_configuration, second_transport)
    )

    first.review(
        request_id="review:configuration-001",
        title="Entropy note",
        source_path=source,
        artifact_root=root,
        output_path=root / "first.md",
        archive_directory=archive,
    )
    second.review(
        request_id="review:configuration-001",
        title="Entropy note",
        source_path=source,
        artifact_root=root,
        output_path=root / "second.md",
        archive_directory=archive,
    )

    assert len(first_transport.calls) == 3
    assert len(second_transport.calls) == 3
    assert len(tuple(archive.iterdir())) == 10


def test__local_note_review_service__rejects_modified_exchange_archive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    source = tmp_path / "note.md"
    source.write_text("# Entropy", encoding="utf-8")
    archive = root / "model-exchanges"
    output = root / "review.md"
    transport = _FakeTransport()
    service = LocalNoteReviewService(
        OllamaNoteReviewBackend(_configuration(), transport)
    )
    publication = service.review(
        request_id="review:archive-integrity-001",
        title="Entropy note",
        source_path=source,
        artifact_root=root,
        output_path=output,
        archive_directory=archive,
    )
    output.unlink()
    publication.receipt_path.unlink()
    exchange_response = next(archive.glob("*.exchange.response.json"))
    exchange_response.write_bytes(exchange_response.read_bytes() + b" ")

    with pytest.raises(
        NoteReviewApplicationError, match="identity does not match"
    ):
        service.review(
            request_id="review:archive-integrity-001",
            title="Entropy note",
            source_path=source,
            artifact_root=root,
            output_path=output,
            archive_directory=archive,
        )

    assert len(transport.calls) == 3


@pytest.mark.integration
def test__ollama_note_review_backend__configured_loopback_integration(
    tmp_path: Path,
) -> None:
    if os.environ.get("KOIOS_RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("set KOIOS_RUN_OLLAMA_INTEGRATION=1 to run")
    model = os.environ["KOIOS_OLLAMA_MODEL"]
    model_digest = os.environ["KOIOS_OLLAMA_MODEL_DIGEST"]
    minimum_version = os.environ.get("KOIOS_OLLAMA_MINIMUM_VERSION", "0.0.0")
    root = tmp_path / "artifacts"
    root.mkdir()
    source = tmp_path / "note.md"
    source.write_text(
        "# Entropy\n\nEntropy is a state function.", encoding="utf-8"
    )
    service = LocalNoteReviewService(
        OllamaNoteReviewBackend(
            OllamaNoteReviewConfiguration(
                model=model,
                model_digest=model_digest,
                minimum_runtime_version=minimum_version,
            )
        )
    )

    publication = service.review(
        request_id="review:ollama-integration-001",
        title="Entropy note",
        source_path=source,
        artifact_root=root,
        output_path=root / "review.md",
        archive_directory=root / "model-exchanges",
    )

    assert publication.output_path.is_file()
    assert publication.receipt_path.is_file()
    assert publication.backend_evidence.model_digest == model_digest
