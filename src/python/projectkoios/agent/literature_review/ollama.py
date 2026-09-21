from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from projectkoios.agent.literature_review.models import (
    ClaimAssessmentProposal,
    ClaimEvidenceBundle,
)
from projectkoios.agent.literature_review.prompting import (
    assessment_system_prompt,
    assessment_user_prompt,
)
from projectkoios.agent.literature_review.validation import (
    assessment_json_schema,
    parse_assessment_proposal,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_TRANSPORT_BYTES = 2_000_000


class OllamaRuntimeError(RuntimeError):
    pass


class OllamaTransport(Protocol):
    def request(
        self, route: str, payload: bytes | None, *, timeout_seconds: int
    ) -> bytes: ...


@dataclass(frozen=True)
class OllamaAssessmentConfiguration:
    model: str
    model_digest: str
    minimum_runtime_version: str
    seed: int = 20260921
    num_ctx: int = 32_768
    num_predict: int = 2_048
    timeout_seconds: int = 600

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be nonempty")
        if _SHA256.fullmatch(self.model_digest) is None:
            raise ValueError("model digest must be 64 lowercase hex")
        if not self.minimum_runtime_version.strip():
            raise ValueError("minimum runtime version must be nonempty")
        if not 4_096 <= self.num_ctx <= 131_072:
            raise ValueError("context bound is invalid")
        if not 256 <= self.num_predict <= 8_192:
            raise ValueError("prediction bound is invalid")
        if not 1 <= self.timeout_seconds <= 3_600:
            raise ValueError("timeout is invalid")


class LoopbackOllamaTransport:
    def __init__(self, endpoint: str = "http://127.0.0.1:11434") -> None:
        normalized = endpoint.rstrip("/")
        if normalized not in {
            "http://127.0.0.1:11434",
            "http://localhost:11434",
        }:
            raise ValueError("Ollama endpoint must be loopback")
        self.endpoint = normalized

    def request(
        self, route: str, payload: bytes | None, *, timeout_seconds: int
    ) -> bytes:
        headers = {"Accept": "application/json"}
        method = "GET"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = urllib.request.Request(
            f"{self.endpoint}{route}",
            data=payload,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_seconds
            ) as response:
                body = cast(bytes, response.read(_MAX_TRANSPORT_BYTES + 1))
        except urllib.error.HTTPError as error:
            body = error.read(_MAX_TRANSPORT_BYTES + 1)
            digest = hashlib.sha256(body).hexdigest()
            raise OllamaRuntimeError(
                f"Ollama returned HTTP {error.code}; body sha256={digest}"
            ) from error
        except (OSError, urllib.error.URLError) as error:
            raise OllamaRuntimeError(
                f"loopback Ollama request failed: {error}"
            ) from error
        if len(body) > _MAX_TRANSPORT_BYTES:
            raise OllamaRuntimeError("Ollama response exceeds its byte limit")
        return body


class ArchivedOllamaLiteratureAssessmentEngine:
    def __init__(
        self,
        configuration: OllamaAssessmentConfiguration,
        archive_directory: Path,
        transport: OllamaTransport | None = None,
    ) -> None:
        self.configuration = configuration
        self.archive_directory = archive_directory
        self.transport = transport or LoopbackOllamaTransport()

    def validate_runtime(self) -> str:
        tags = _json_object(
            self.transport.request(
                "/api/tags",
                None,
                timeout_seconds=self.configuration.timeout_seconds,
            ),
            "model inventory",
        )
        models = tags.get("models")
        if not isinstance(models, list):
            raise OllamaRuntimeError("Ollama model inventory is invalid")
        matches = [
            item
            for item in models
            if isinstance(item, dict)
            and item.get("name") == self.configuration.model
        ]
        if len(matches) != 1:
            raise OllamaRuntimeError("configured model is unavailable")
        if matches[0].get("digest") != self.configuration.model_digest:
            raise OllamaRuntimeError("configured model digest does not match")
        version_payload = _json_object(
            self.transport.request(
                "/api/version",
                None,
                timeout_seconds=self.configuration.timeout_seconds,
            ),
            "runtime version",
        )
        version = version_payload.get("version")
        if not isinstance(version, str):
            raise OllamaRuntimeError("runtime version is unavailable")
        if _version_tuple(version) < _version_tuple(
            self.configuration.minimum_runtime_version
        ):
            raise OllamaRuntimeError("runtime version is below minimum")
        return version

    def propose(self, bundle: ClaimEvidenceBundle) -> ClaimAssessmentProposal:
        runtime_version = self.validate_runtime()
        request_payload = {
            "model": self.configuration.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": assessment_system_prompt()},
                {"role": "user", "content": assessment_user_prompt(bundle)},
            ],
            "format": assessment_json_schema(),
            "options": {
                "temperature": 0,
                "seed": self.configuration.seed,
                "num_ctx": self.configuration.num_ctx,
                "num_predict": self.configuration.num_predict,
            },
        }
        request_bytes = _canonical_bytes(request_payload)
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        paths = self._archive_paths(request_sha256)
        response_bytes = self._response(request_bytes, paths)
        response_sha256 = hashlib.sha256(response_bytes).hexdigest()
        receipt = {
            "schema": "koios.ollama-literature-assessment-receipt.v1",
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "model": self.configuration.model,
            "model_digest": self.configuration.model_digest,
            "runtime_version": runtime_version,
            "claim_id": bundle.claim.claim_id,
        }
        if not paths["receipt"].exists():
            _atomic_write(paths["receipt"], _canonical_bytes(receipt))
        else:
            archived = _json_object(
                paths["receipt"].read_bytes(), "archived receipt"
            )
            if archived != receipt:
                raise OllamaRuntimeError("archived receipt does not match")
        envelope = _json_object(response_bytes, "Ollama response")
        message = envelope.get("message")
        if not isinstance(message, dict):
            raise OllamaRuntimeError("Ollama response message is invalid")
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaRuntimeError("Ollama response content is invalid")
        return parse_assessment_proposal(bundle, content.encode("utf-8"))

    def _response(self, request_bytes: bytes, paths: dict[str, Path]) -> bytes:
        self._prepare_archive()
        if paths["response"].exists():
            if not paths["request"].is_file():
                raise OllamaRuntimeError("archived request is incomplete")
            if paths["request"].read_bytes() != request_bytes:
                raise OllamaRuntimeError("archived request does not match")
            return paths["response"].read_bytes()
        if paths["request"].exists():
            raise OllamaRuntimeError("partial archived request exists")
        _atomic_write(paths["request"], request_bytes)
        response = self.transport.request(
            "/api/chat",
            request_bytes,
            timeout_seconds=self.configuration.timeout_seconds,
        )
        _atomic_write(paths["response"], response)
        return response

    def _archive_paths(self, request_sha256: str) -> dict[str, Path]:
        return {
            "request": self.archive_directory
            / f"{request_sha256}.request.json",
            "response": self.archive_directory
            / f"{request_sha256}.response.json",
            "receipt": self.archive_directory
            / f"{request_sha256}.receipt.json",
        }

    def _prepare_archive(self) -> None:
        if self.archive_directory.is_symlink():
            raise OllamaRuntimeError("archive directory must not be a symlink")
        self.archive_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.archive_directory, 0o700)


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise OllamaRuntimeError(f"archive output already exists: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise OllamaRuntimeError(
            f"archive temporary output already exists: {temporary.name}"
        )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise OllamaRuntimeError("archive publication failed") from error


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


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OllamaRuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise OllamaRuntimeError(f"{label} must be an object")
    return value


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part) for part in version.split("."))
    except ValueError as error:
        raise OllamaRuntimeError("runtime version is invalid") from error
    if not parts or any(part < 0 for part in parts):
        raise OllamaRuntimeError("runtime version is invalid")
    return parts + (0,) * (4 - len(parts))
