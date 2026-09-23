from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, cast

from projectkoios.agent.note_review.application import (
    NoteReviewBackendCompletion,
    NoteReviewBackendEvidence,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_TRANSPORT_BYTES = 2_000_000
_LOOPBACK_ENDPOINTS = {
    "http://127.0.0.1:11434",
    "http://localhost:11434",
}


class OllamaNoteReviewError(RuntimeError):
    pass


class OllamaNoteReviewTransport(Protocol):
    def request(
        self, route: str, payload: bytes | None, *, timeout_seconds: int
    ) -> bytes: ...


@dataclass(frozen=True)
class OllamaNoteReviewConfiguration:
    model: str
    model_digest: str
    minimum_runtime_version: str
    endpoint: str = "http://127.0.0.1:11434"
    seed: int = 20260922
    num_ctx: int = 32_768
    num_predict: int = 2_048
    timeout_seconds: int = 600

    def __post_init__(self) -> None:
        if not self.model.strip() or len(self.model) > 500:
            raise ValueError("model must be nonempty")
        if _SHA256.fullmatch(self.model_digest) is None:
            raise ValueError("model digest must be 64 lowercase hex")
        if not self.minimum_runtime_version.strip():
            raise ValueError("minimum runtime version must be nonempty")
        if self.endpoint.rstrip("/") not in _LOOPBACK_ENDPOINTS:
            raise ValueError("Ollama endpoint must be loopback")
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("seed is invalid")
        if not 4_096 <= self.num_ctx <= 131_072:
            raise ValueError("context bound is invalid")
        if not 256 <= self.num_predict <= 8_192:
            raise ValueError("prediction bound is invalid")
        if not 1 <= self.timeout_seconds <= 3_600:
            raise ValueError("timeout is invalid")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class LoopbackNoteReviewOllamaTransport:
    def __init__(self, endpoint: str = "http://127.0.0.1:11434") -> None:
        normalized = endpoint.rstrip("/")
        if normalized not in _LOOPBACK_ENDPOINTS:
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
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                body = cast(bytes, response.read(_MAX_TRANSPORT_BYTES + 1))
        except urllib.error.HTTPError as error:
            body = error.read(_MAX_TRANSPORT_BYTES + 1)
            digest = hashlib.sha256(body).hexdigest()
            raise OllamaNoteReviewError(
                f"Ollama returned HTTP {error.code}; body sha256={digest}"
            ) from error
        except (OSError, urllib.error.URLError) as error:
            raise OllamaNoteReviewError(
                f"loopback Ollama request failed: {error}"
            ) from error
        if len(body) > _MAX_TRANSPORT_BYTES:
            raise OllamaNoteReviewError(
                "Ollama response exceeds its byte limit"
            )
        return body


class OllamaNoteReviewBackend:
    def __init__(
        self,
        configuration: OllamaNoteReviewConfiguration,
        transport: OllamaNoteReviewTransport | None = None,
    ) -> None:
        self.configuration_value = configuration
        self.transport = transport or LoopbackNoteReviewOllamaTransport(
            configuration.endpoint
        )

    def configuration(self) -> dict[str, object]:
        configuration = self.configuration_value
        return {
            "backend": "ollama-loopback",
            "endpoint": configuration.endpoint.rstrip("/"),
            "minimum_runtime_version": configuration.minimum_runtime_version,
            "model": configuration.model,
            "model_digest": configuration.model_digest,
            "num_ctx": configuration.num_ctx,
            "num_predict": configuration.num_predict,
            "seed": configuration.seed,
            "timeout_seconds": configuration.timeout_seconds,
        }

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> NoteReviewBackendCompletion:
        runtime_version = self._validate_runtime()
        configuration = self.configuration_value
        request_payload = {
            "model": configuration.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": response_schema,
            "options": {
                "temperature": 0,
                "seed": configuration.seed,
                "num_ctx": configuration.num_ctx,
                "num_predict": configuration.num_predict,
            },
        }
        request_bytes = _canonical_bytes(request_payload)
        response_bytes = self.transport.request(
            "/api/chat",
            request_bytes,
            timeout_seconds=configuration.timeout_seconds,
        )
        envelope = _json_object(response_bytes, "Ollama response")
        message = envelope.get("message")
        if not isinstance(message, dict):
            raise OllamaNoteReviewError(
                "Ollama response message is invalid"
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaNoteReviewError(
                "Ollama response content is invalid"
            )
        content_bytes = content.encode("utf-8")
        evidence = NoteReviewBackendEvidence(
            backend="ollama-loopback",
            model=configuration.model,
            model_digest=configuration.model_digest,
            runtime_version=runtime_version,
            response_sha256=hashlib.sha256(content_bytes).hexdigest(),
            exchange_request_sha256=hashlib.sha256(request_bytes).hexdigest(),
            exchange_response_sha256=hashlib.sha256(
                response_bytes
            ).hexdigest(),
        )
        return NoteReviewBackendCompletion(
            response=content_bytes,
            evidence=evidence,
            exchange_request=request_bytes,
            exchange_response=response_bytes,
        )

    def _validate_runtime(self) -> str:
        configuration = self.configuration_value
        tags = _json_object(
            self.transport.request(
                "/api/tags",
                None,
                timeout_seconds=configuration.timeout_seconds,
            ),
            "model inventory",
        )
        models = tags.get("models")
        if not isinstance(models, list):
            raise OllamaNoteReviewError("Ollama model inventory is invalid")
        matches = [
            item
            for item in models
            if isinstance(item, dict)
            and item.get("name") == configuration.model
        ]
        if len(matches) != 1:
            raise OllamaNoteReviewError("configured model is unavailable")
        if matches[0].get("digest") != configuration.model_digest:
            raise OllamaNoteReviewError(
                "configured model digest does not match"
            )
        version_payload = _json_object(
            self.transport.request(
                "/api/version",
                None,
                timeout_seconds=configuration.timeout_seconds,
            ),
            "runtime version",
        )
        version = version_payload.get("version")
        if not isinstance(version, str):
            raise OllamaNoteReviewError("runtime version is unavailable")
        if _version_tuple(version) < _version_tuple(
            configuration.minimum_runtime_version
        ):
            raise OllamaNoteReviewError("runtime version is below minimum")
        return version


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
        raise OllamaNoteReviewError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise OllamaNoteReviewError(f"{label} must be an object")
    return value


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part) for part in version.split("."))
    except ValueError as error:
        raise OllamaNoteReviewError("runtime version is invalid") from error
    if not parts or any(part < 0 for part in parts):
        raise OllamaNoteReviewError("runtime version is invalid")
    return parts + (0,) * (4 - len(parts))
