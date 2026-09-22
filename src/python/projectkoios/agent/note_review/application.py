from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from projectkoios.agent.note_review.models import (
    NoteReviewRequest,
    ReviewProfile,
)
from projectkoios.agent.note_review.prompting import (
    note_review_system_prompt,
    note_review_user_prompt,
)
from projectkoios.agent.note_review.rendering import render_note_review
from projectkoios.agent.note_review.validation import (
    note_review_json_schema,
    parse_note_review_proposal,
)

_MAX_SOURCE_BYTES = 128_000
_MAX_RESPONSE_BYTES = 64_000


class NoteReviewApplicationError(RuntimeError):
    pass


class NoteReviewBackend(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> bytes: ...


@dataclass(frozen=True)
class NoteReviewPublication:
    request_id: str
    output_path: Path
    receipt_path: Path
    source_sha256: str
    response_sha256: str
    review_sha256: str


class LocalNoteReviewService:
    def __init__(self, backend: NoteReviewBackend) -> None:
        self.backend = backend

    def review(
        self,
        *,
        request_id: str,
        title: str,
        source_path: Path,
        artifact_root: Path,
        output_path: Path,
        archive_directory: Path,
        profile: ReviewProfile = ReviewProfile.SCIENTIFIC_NOTE,
    ) -> NoteReviewPublication:
        root = _artifact_root(artifact_root)
        output = _output_path(root, output_path, "review output")
        receipt = _output_path(
            root,
            output.with_name(f"{output.name}.receipt.json"),
            "review receipt",
        )
        _require_absent(output, "review output")
        _require_absent(receipt, "review receipt")

        source = _read_source(source_path)
        source_sha256 = hashlib.sha256(source).hexdigest()
        try:
            source_markdown = source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise NoteReviewApplicationError(
                "source note is not valid UTF-8"
            ) from error
        try:
            request = NoteReviewRequest(
                request_id=request_id,
                title=title,
                source_sha256=source_sha256,
                source_markdown=source_markdown,
                profile=profile,
            )
        except ValueError as error:
            raise NoteReviewApplicationError(
                "note review request is invalid"
            ) from error
        archive = _archive_directory(root, archive_directory)
        system_prompt = note_review_system_prompt()
        user_prompt = note_review_user_prompt(request)
        response_schema = note_review_json_schema()
        model_request = _canonical_bytes(
            {
                "schema": "koios.note-review-model-request.v1",
                "response_schema": response_schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        model_request_sha256 = hashlib.sha256(model_request).hexdigest()
        archive_paths = _archive_paths(archive, model_request_sha256)
        response = _archived_response(archive_paths, model_request)
        if response is None:
            response = self.backend.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
            )
            if not isinstance(response, bytes):
                raise NoteReviewApplicationError(
                    "note review backend must return bytes"
                )
            if len(response) > _MAX_RESPONSE_BYTES:
                raise NoteReviewApplicationError(
                    "note review response exceeds its byte limit"
                )
            proposal = parse_note_review_proposal(request, response)
            _publish_pair(
                (
                    (archive_paths["request"], model_request),
                    (archive_paths["response"], response),
                )
            )
        else:
            proposal = parse_note_review_proposal(request, response)

        rendered = render_note_review(request, proposal).encode("utf-8")
        response_sha256 = hashlib.sha256(response).hexdigest()
        review_sha256 = hashlib.sha256(rendered).hexdigest()
        receipt_payload = _canonical_bytes(
            {
                "boundary": proposal.boundary.value,
                "model_request_sha256": model_request_sha256,
                "profile": request.profile.value,
                "request_id": request.request_id,
                "response_sha256": response_sha256,
                "review_sha256": review_sha256,
                "schema": "koios.note-review-receipt.v1",
                "source_sha256": source_sha256,
            }
        )
        _publish_pair(((output, rendered), (receipt, receipt_payload)))
        return NoteReviewPublication(
            request_id=request.request_id,
            output_path=output,
            receipt_path=receipt,
            source_sha256=source_sha256,
            response_sha256=response_sha256,
            review_sha256=review_sha256,
        )


def _artifact_root(path: Path) -> Path:
    if not path.is_absolute():
        raise NoteReviewApplicationError("artifact root must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise NoteReviewApplicationError(
            "artifact root must be an existing non-symlink directory"
        )
    return path.resolve(strict=True)


def _output_path(root: Path, path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise NoteReviewApplicationError(f"{label} must be absolute")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NoteReviewApplicationError(
            f"{label} parent must be an existing non-symlink directory"
        )
    resolved_parent = path.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(root):
        raise NoteReviewApplicationError(f"{label} must be below artifact root")
    return resolved_parent / path.name


def _archive_directory(root: Path, path: Path) -> Path:
    if not path.is_absolute():
        raise NoteReviewApplicationError("archive directory must be absolute")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise NoteReviewApplicationError(
                "archive path must be a non-symlink directory"
            )
    else:
        parent = path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise NoteReviewApplicationError(
                "archive parent must be an existing non-symlink directory"
            )
        if not parent.resolve(strict=True).is_relative_to(root):
            raise NoteReviewApplicationError(
                "archive directory must be below artifact root"
            )
        path.mkdir(mode=0o700)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise NoteReviewApplicationError(
            "archive directory must be below artifact root"
        )
    return resolved


def _read_source(path: Path) -> bytes:
    if not path.is_absolute():
        raise NoteReviewApplicationError("source path must be absolute")
    if path.is_symlink():
        raise NoteReviewApplicationError(
            "source must be a readable non-symlink file"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise NoteReviewApplicationError(
            "source must be a readable non-symlink file"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise NoteReviewApplicationError("source must be a regular file")
        if metadata.st_size > _MAX_SOURCE_BYTES:
            raise NoteReviewApplicationError(
                "source note exceeds its byte limit"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(_MAX_SOURCE_BYTES + 1)
        if len(payload) > _MAX_SOURCE_BYTES:
            raise NoteReviewApplicationError(
                "source note exceeds its byte limit"
            )
        return payload
    finally:
        os.close(descriptor)


def _archive_paths(directory: Path, request_sha256: str) -> dict[str, Path]:
    return {
        "request": directory / f"{request_sha256}.request.json",
        "response": directory / f"{request_sha256}.response.json",
    }


def _archived_response(
    paths: dict[str, Path],
    expected_request: bytes,
) -> bytes | None:
    request_exists = paths["request"].exists() or paths["request"].is_symlink()
    response_exists = (
        paths["response"].exists() or paths["response"].is_symlink()
    )
    if not request_exists and not response_exists:
        return None
    if not request_exists or not response_exists:
        raise NoteReviewApplicationError("model exchange archive is incomplete")
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise NoteReviewApplicationError("model exchange archive is invalid")
    request = paths["request"].read_bytes()
    if request != expected_request:
        raise NoteReviewApplicationError(
            "archived model request does not match"
        )
    response = paths["response"].read_bytes()
    if len(response) > _MAX_RESPONSE_BYTES:
        raise NoteReviewApplicationError(
            "archived model response exceeds its byte limit"
        )
    return response


def _require_absent(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise NoteReviewApplicationError(f"{label} already exists")


def _publish_pair(entries: tuple[tuple[Path, bytes], ...]) -> None:
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        for path, _ in entries:
            _require_absent(path, path.name)
        for path, payload in entries:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
        for (path, _), temporary in zip(entries, temporary_paths, strict=True):
            os.link(temporary, path, follow_symlinks=False)
            published_paths.append(path)
        for temporary in temporary_paths:
            temporary.unlink()
    except OSError as error:
        for path in reversed(published_paths):
            path.unlink(missing_ok=True)
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise NoteReviewApplicationError(
            "artifact publication failed"
        ) from error


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
