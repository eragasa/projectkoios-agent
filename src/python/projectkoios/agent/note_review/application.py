from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
_MAX_MODEL_REQUEST_BYTES = 512_000
_MAX_BACKEND_CONFIGURATION_BYTES = 16_000
_MAX_BACKEND_EVIDENCE_BYTES = 16_000
_MAX_EXCHANGE_BYTES = 2_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")


class NoteReviewApplicationError(RuntimeError):
    pass


@dataclass(frozen=True)
class NoteReviewBackendEvidence:
    backend: str
    response_sha256: str
    model: str | None = None
    model_digest: str | None = None
    runtime_version: str | None = None
    exchange_request_sha256: str | None = None
    exchange_response_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.backend or len(self.backend) > 100:
            raise ValueError("backend evidence name is invalid")
        if _SHA256.fullmatch(self.response_sha256) is None:
            raise ValueError("backend response SHA-256 is invalid")
        for label, value in (
            ("model", self.model),
            ("model digest", self.model_digest),
            ("runtime version", self.runtime_version),
        ):
            if value is not None and (not value or len(value) > 500):
                raise ValueError(f"backend {label} is invalid")
        if self.model_digest is not None:
            if _SHA256.fullmatch(self.model_digest) is None:
                raise ValueError("backend model digest is invalid")
        exchange_hashes = (
            self.exchange_request_sha256,
            self.exchange_response_sha256,
        )
        if any(value is None for value in exchange_hashes) and any(
            value is not None for value in exchange_hashes
        ):
            raise ValueError("backend exchange evidence is incomplete")
        if any(
            value is not None and _SHA256.fullmatch(value) is None
            for value in exchange_hashes
        ):
            raise ValueError("backend exchange SHA-256 is invalid")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "backend": self.backend,
            "exchange_request_sha256": self.exchange_request_sha256,
            "exchange_response_sha256": self.exchange_response_sha256,
            "model": self.model,
            "model_digest": self.model_digest,
            "response_sha256": self.response_sha256,
            "runtime_version": self.runtime_version,
        }


@dataclass(frozen=True)
class NoteReviewBackendCompletion:
    response: bytes
    evidence: NoteReviewBackendEvidence
    exchange_request: bytes | None = None
    exchange_response: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response, bytes):
            raise ValueError("backend response must be bytes")
        if hashlib.sha256(self.response).hexdigest() != (
            self.evidence.response_sha256
        ):
            raise ValueError("backend response identity does not match")
        exchanges = (self.exchange_request, self.exchange_response)
        if any(value is None for value in exchanges) and any(
            value is not None for value in exchanges
        ):
            raise ValueError("backend exchange archive is incomplete")
        if self.exchange_request is None or self.exchange_response is None:
            if self.evidence.exchange_request_sha256 is not None:
                raise ValueError("backend exchange evidence has no archive")
            return
        if self.evidence.exchange_request_sha256 is None:
            raise ValueError("backend exchange archive has no evidence")
        if hashlib.sha256(self.exchange_request).hexdigest() != (
            self.evidence.exchange_request_sha256
        ):
            raise ValueError("backend exchange request identity does not match")
        if hashlib.sha256(self.exchange_response).hexdigest() != (
            self.evidence.exchange_response_sha256
        ):
            raise ValueError(
                "backend exchange response identity does not match"
            )


class NoteReviewBackend(Protocol):
    def configuration(self) -> dict[str, object]: ...

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> NoteReviewBackendCompletion: ...


@dataclass(frozen=True)
class NoteReviewPublication:
    request_id: str
    output_path: Path
    receipt_path: Path
    source_sha256: str
    response_sha256: str
    review_sha256: str
    backend_evidence: NoteReviewBackendEvidence


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
        backend_configuration = _backend_configuration(
            self.backend.configuration()
        )
        model_request = _canonical_bytes(
            {
                "backend": backend_configuration,
                "response_schema": response_schema,
                "schema": "koios.note-review-model-request.v2",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if len(model_request) > _MAX_MODEL_REQUEST_BYTES:
            raise NoteReviewApplicationError(
                "model request exceeds its byte limit"
            )
        model_request_sha256 = hashlib.sha256(model_request).hexdigest()
        archive_paths = _archive_paths(archive, model_request_sha256)
        completion = _archived_completion(archive_paths, model_request)
        if completion is None:
            completion = self.backend.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
            )
            _validate_completion(completion, backend_configuration)
            proposal = parse_note_review_proposal(request, completion.response)
            _publish_entries(
                _completion_archive_entries(
                    archive_paths, model_request, completion
                )
            )
        else:
            _validate_completion(completion, backend_configuration)
            proposal = parse_note_review_proposal(request, completion.response)

        rendered = render_note_review(request, proposal).encode("utf-8")
        response_sha256 = hashlib.sha256(completion.response).hexdigest()
        review_sha256 = hashlib.sha256(rendered).hexdigest()
        receipt_payload = _canonical_bytes(
            {
                "backend": completion.evidence.as_dict(),
                "boundary": proposal.boundary.value,
                "model_request_sha256": model_request_sha256,
                "profile": request.profile.value,
                "request_id": request.request_id,
                "response_sha256": response_sha256,
                "review_sha256": review_sha256,
                "schema": "koios.note-review-receipt.v2",
                "source_sha256": source_sha256,
            }
        )
        _publish_entries(((output, rendered), (receipt, receipt_payload)))
        return NoteReviewPublication(
            request_id=request.request_id,
            output_path=output,
            receipt_path=receipt,
            source_sha256=source_sha256,
            response_sha256=response_sha256,
            review_sha256=review_sha256,
            backend_evidence=completion.evidence,
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


def _backend_configuration(value: object) -> dict[str, Any]:
    payload = _canonical_bytes(value)
    if len(payload) > _MAX_BACKEND_CONFIGURATION_BYTES:
        raise NoteReviewApplicationError(
            "backend configuration exceeds its byte limit"
        )
    normalized = json.loads(payload)
    if not isinstance(normalized, dict):
        raise NoteReviewApplicationError(
            "backend configuration must be an object"
        )
    backend = normalized.get("backend")
    if not isinstance(backend, str) or not backend or len(backend) > 100:
        raise NoteReviewApplicationError(
            "backend configuration name is invalid"
        )
    return normalized


def _archive_paths(directory: Path, request_sha256: str) -> dict[str, Path]:
    return {
        "request": directory / f"{request_sha256}.request.json",
        "response": directory / f"{request_sha256}.response.json",
        "evidence": directory / f"{request_sha256}.evidence.json",
        "exchange_request": directory
        / f"{request_sha256}.exchange.request.json",
        "exchange_response": directory
        / f"{request_sha256}.exchange.response.json",
    }


def _archived_completion(
    paths: dict[str, Path],
    expected_request: bytes,
) -> NoteReviewBackendCompletion | None:
    present = {
        key: path.exists() or path.is_symlink() for key, path in paths.items()
    }
    if not any(present.values()):
        return None
    if not all(present[key] for key in ("request", "response", "evidence")):
        raise NoteReviewApplicationError("model exchange archive is incomplete")
    required_paths = (paths["request"], paths["response"], paths["evidence"])
    if any(path.is_symlink() or not path.is_file() for path in required_paths):
        raise NoteReviewApplicationError("model exchange archive is invalid")
    request = _read_archive(
        paths["request"], _MAX_MODEL_REQUEST_BYTES, "model request"
    )
    if request != expected_request:
        raise NoteReviewApplicationError(
            "archived model request does not match"
        )
    response = _read_archive(
        paths["response"], _MAX_RESPONSE_BYTES, "model response"
    )
    evidence = _backend_evidence(
        _read_archive(
            paths["evidence"],
            _MAX_BACKEND_EVIDENCE_BYTES,
            "backend evidence",
        )
    )
    has_exchange = evidence.exchange_request_sha256 is not None
    exchange_present = (
        present["exchange_request"],
        present["exchange_response"],
    )
    if has_exchange != all(exchange_present) or (
        not has_exchange and any(exchange_present)
    ):
        raise NoteReviewApplicationError(
            "backend exchange archive is incomplete"
        )
    exchange_request = None
    exchange_response = None
    if has_exchange:
        exchange_paths = (
            paths["exchange_request"],
            paths["exchange_response"],
        )
        if any(
            path.is_symlink() or not path.is_file() for path in exchange_paths
        ):
            raise NoteReviewApplicationError(
                "backend exchange archive is invalid"
            )
        exchange_request = _read_archive(
            paths["exchange_request"],
            _MAX_EXCHANGE_BYTES,
            "backend exchange request",
        )
        exchange_response = _read_archive(
            paths["exchange_response"],
            _MAX_EXCHANGE_BYTES,
            "backend exchange response",
        )
    try:
        return NoteReviewBackendCompletion(
            response=response,
            evidence=evidence,
            exchange_request=exchange_request,
            exchange_response=exchange_response,
        )
    except ValueError as error:
        raise NoteReviewApplicationError(
            "backend exchange archive identity does not match"
        ) from error


def _read_archive(path: Path, limit: int, label: str) -> bytes:
    if path.stat().st_size > limit:
        raise NoteReviewApplicationError(
            f"archived {label} exceeds its byte limit"
        )
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise NoteReviewApplicationError(
            f"archived {label} exceeds its byte limit"
        )
    return payload


def _backend_evidence(payload: bytes) -> NoteReviewBackendEvidence:
    try:
        raw = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NoteReviewApplicationError(
            "backend evidence is invalid JSON"
        ) from error
    keys = {
        "backend",
        "exchange_request_sha256",
        "exchange_response_sha256",
        "model",
        "model_digest",
        "response_sha256",
        "runtime_version",
    }
    if not isinstance(raw, dict) or set(raw) != keys:
        raise NoteReviewApplicationError("backend evidence is invalid")
    try:
        evidence = NoteReviewBackendEvidence(**raw)
    except (TypeError, ValueError) as error:
        raise NoteReviewApplicationError(
            "backend evidence is invalid"
        ) from error
    if payload != _canonical_bytes(evidence.as_dict()):
        raise NoteReviewApplicationError("backend evidence is not canonical")
    return evidence


def _validate_completion(
    completion: object,
    configuration: dict[str, Any],
) -> None:
    if not isinstance(completion, NoteReviewBackendCompletion):
        raise NoteReviewApplicationError(
            "note review backend returned an invalid completion"
        )
    if len(completion.response) > _MAX_RESPONSE_BYTES:
        raise NoteReviewApplicationError(
            "note review response exceeds its byte limit"
        )
    if completion.evidence.backend != configuration["backend"]:
        raise NoteReviewApplicationError(
            "backend evidence does not match configuration"
        )
    for key in ("model", "model_digest"):
        configured = configuration.get(key)
        observed = getattr(completion.evidence, key)
        if configured is not None and observed != configured:
            raise NoteReviewApplicationError(
                "backend evidence does not match configuration"
            )
    for payload in (
        completion.exchange_request,
        completion.exchange_response,
    ):
        if payload is not None and len(payload) > _MAX_EXCHANGE_BYTES:
            raise NoteReviewApplicationError(
                "backend exchange exceeds its byte limit"
            )


def _completion_archive_entries(
    paths: dict[str, Path],
    model_request: bytes,
    completion: NoteReviewBackendCompletion,
) -> tuple[tuple[Path, bytes], ...]:
    entries = [
        (paths["request"], model_request),
        (paths["response"], completion.response),
        (paths["evidence"], _canonical_bytes(completion.evidence.as_dict())),
    ]
    if completion.exchange_request is not None:
        assert completion.exchange_response is not None
        entries.extend(
            (
                (paths["exchange_request"], completion.exchange_request),
                (paths["exchange_response"], completion.exchange_response),
            )
        )
    return tuple(entries)


def _require_absent(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise NoteReviewApplicationError(f"{label} already exists")


def _publish_entries(entries: tuple[tuple[Path, bytes], ...]) -> None:
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
