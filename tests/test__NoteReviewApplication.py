from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from projectkoios.agent.note_review import (
    LocalNoteReviewService,
    NoteReviewApplicationError,
    NoteReviewBackendCompletion,
    NoteReviewBackendEvidence,
    application,
)


def _response() -> bytes:
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
    ).encode()


class _FakeBackend:
    def __init__(self, response: bytes | None = None) -> None:
        self.response = response or _response()
        self.calls = 0
        self.system_prompt = ""
        self.user_prompt = ""
        self.response_schema: dict[str, object] = {}

    def configuration(self) -> dict[str, object]:
        return {"backend": "test-fake", "fixture": "note-review-v1"}

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> NoteReviewBackendCompletion:
        self.calls += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.response_schema = response_schema
        return NoteReviewBackendCompletion(
            response=self.response,
            evidence=NoteReviewBackendEvidence(
                backend="test-fake",
                response_sha256=hashlib.sha256(self.response).hexdigest(),
            ),
        )


def _paths(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "artifacts"
    root.mkdir()
    source = tmp_path / "note.md"
    source.write_text(
        "# Entropy\n\nEntropy is a state function.", encoding="utf-8"
    )
    return {
        "root": root,
        "source": source,
        "output": root / "entropy-review.md",
        "archive": root / "model-exchanges",
    }


def _review(
    service: LocalNoteReviewService,
    paths: dict[str, Path],
    *,
    output: Path | None = None,
):
    return service.review(
        request_id="review:entropy-001",
        title="Entropy note",
        source_path=paths["source"],
        artifact_root=paths["root"],
        output_path=output or paths["output"],
        archive_directory=paths["archive"],
    )


def test__local_note_review_service__publishes_review_receipt_and_archive(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    backend = _FakeBackend()
    service = LocalNoteReviewService(backend)

    publication = _review(service, paths)

    assert backend.calls == 1
    assert "untrusted data" in backend.system_prompt
    assert backend.response_schema["additionalProperties"] is False
    assert publication.output_path == paths["output"]
    rendered = paths["output"].read_bytes()
    assert b"`AUTOMATED_UNREVIEWED`" in rendered
    receipt = json.loads(publication.receipt_path.read_bytes())
    source = paths["source"].read_bytes()
    request_archive = next(paths["archive"].glob("*.request.json"))
    assert receipt == {
        "backend": {
            "backend": "test-fake",
            "exchange_request_sha256": None,
            "exchange_response_sha256": None,
            "model": None,
            "model_digest": None,
            "response_sha256": hashlib.sha256(_response()).hexdigest(),
            "runtime_version": None,
        },
        "boundary": "AUTOMATED_UNREVIEWED",
        "model_request_sha256": request_archive.name[:64],
        "profile": "SCIENTIFIC_NOTE",
        "request_id": "review:entropy-001",
        "response_sha256": hashlib.sha256(_response()).hexdigest(),
        "review_sha256": hashlib.sha256(rendered).hexdigest(),
        "schema": "koios.note-review-receipt.v2",
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }
    archived = tuple(paths["archive"].iterdir())
    assert len(archived) == 3
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in archived)
    assert (paths["output"].stat().st_mode & 0o777) == 0o600
    assert (publication.receipt_path.stat().st_mode & 0o777) == 0o600


def test__local_note_review_service__replays_archived_exchange(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    backend = _FakeBackend()
    service = LocalNoteReviewService(backend)

    first = _review(service, paths)
    second = _review(service, paths, output=paths["root"] / "replayed.md")

    assert backend.calls == 1
    assert first.review_sha256 == second.review_sha256
    assert second.output_path.read_bytes() == first.output_path.read_bytes()
    assert second.receipt_path.read_bytes() == first.receipt_path.read_bytes()


def test__local_note_review_service__rejects_existing_output_before_backend(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["output"].write_text("existing", encoding="utf-8")
    backend = _FakeBackend()

    with pytest.raises(NoteReviewApplicationError, match="already exists"):
        _review(LocalNoteReviewService(backend), paths)

    assert backend.calls == 0
    assert not paths["archive"].exists()


def test__local_note_review_service__rejects_output_symlink(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    existing = paths["root"] / "existing.md"
    existing.write_text("existing", encoding="utf-8")
    paths["output"].symlink_to(existing)
    backend = _FakeBackend()

    with pytest.raises(NoteReviewApplicationError, match="already exists"):
        _review(LocalNoteReviewService(backend), paths)

    assert backend.calls == 0
    assert not paths["archive"].exists()


def test__local_note_review_service__rejects_source_symlink(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    link = tmp_path / "linked-note.md"
    link.symlink_to(paths["source"])
    paths["source"] = link
    backend = _FakeBackend()

    with pytest.raises(NoteReviewApplicationError, match="non-symlink"):
        _review(LocalNoteReviewService(backend), paths)

    assert backend.calls == 0
    assert not paths["archive"].exists()


def test__local_note_review_service__rejects_oversized_source(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["source"].write_bytes(b"x" * 128_001)
    backend = _FakeBackend()

    with pytest.raises(NoteReviewApplicationError, match="byte limit"):
        _review(LocalNoteReviewService(backend), paths)

    assert backend.calls == 0
    assert not paths["archive"].exists()


def test__local_note_review_service__rejects_invalid_utf8(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["source"].write_bytes(b"\xff")
    backend = _FakeBackend()

    with pytest.raises(NoteReviewApplicationError, match="UTF-8"):
        _review(LocalNoteReviewService(backend), paths)

    assert backend.calls == 0
    assert not paths["archive"].exists()


def test__local_note_review_service__rejects_output_outside_root(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    backend = _FakeBackend()

    with pytest.raises(NoteReviewApplicationError, match="below artifact root"):
        _review(
            LocalNoteReviewService(backend),
            paths,
            output=tmp_path / "outside.md",
        )

    assert backend.calls == 0


def test__local_note_review_service__rolls_back_partial_publication_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    backend = _FakeBackend()
    service = LocalNoteReviewService(backend)
    real_link = os.link
    link_calls = 0

    def fail_receipt_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 5:
            raise OSError("simulated receipt publication failure")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(application.os, "link", fail_receipt_link)

    with pytest.raises(NoteReviewApplicationError, match="publication failed"):
        _review(service, paths)

    assert backend.calls == 1
    assert not paths["output"].exists()
    receipt = paths["output"].with_name(
        f"{paths['output'].name}.receipt.json"
    )
    assert not receipt.exists()
    assert not tuple(paths["root"].glob(".*.tmp"))

    monkeypatch.setattr(application.os, "link", real_link)
    publication = _review(service, paths)

    assert backend.calls == 1
    assert publication.output_path.is_file()
    assert publication.receipt_path.is_file()


def test__local_note_review_service__rejects_incomplete_archive(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    backend = _FakeBackend()
    service = LocalNoteReviewService(backend)
    _review(service, paths)
    paths["output"].unlink()
    paths["output"].with_name(
        f"{paths['output'].name}.receipt.json"
    ).unlink()
    next(paths["archive"].glob("*.response.json")).unlink()

    with pytest.raises(
        NoteReviewApplicationError, match="archive is incomplete"
    ):
        _review(service, paths)

    assert backend.calls == 1
