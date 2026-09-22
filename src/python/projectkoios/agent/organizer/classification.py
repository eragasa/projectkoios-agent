from __future__ import annotations

import json
from typing import Any, Protocol

from projectkoios.agent.literature_review.ollama import (
    LoopbackOllamaTransport,
    OllamaRuntimeError,
    OllamaTransport,
)
from projectkoios.agent.organizer.models import (
    CategorizationProposal,
    FileObservation,
    LifeDomain,
    ParaCategory,
)


class FileCategorizer(Protocol):
    def propose(
        self, observations: tuple[FileObservation, ...]
    ) -> tuple[CategorizationProposal, ...]: ...


class OllamaMetadataCategorizer:
    """Propose metadata-only organization with exact loopback Ollama."""

    def __init__(
        self,
        *,
        model: str,
        model_digest: str,
        transport: OllamaTransport | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be nonempty")
        if len(model_digest) != 64 or any(
            character not in "0123456789abcdef" for character in model_digest
        ):
            raise ValueError("model_digest must be 64 lowercase hex")
        if not 1 <= timeout_seconds <= 3_600:
            raise ValueError("timeout_seconds is outside its bound")
        self.model = model
        self.model_digest = model_digest
        self.transport = transport or LoopbackOllamaTransport()
        self.timeout_seconds = timeout_seconds

    def propose(
        self, observations: tuple[FileObservation, ...]
    ) -> tuple[CategorizationProposal, ...]:
        if not observations:
            return ()
        if len(observations) > 25:
            raise ValueError(
                "one categorization request is limited to 25 files"
            )
        self._validate_model()
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You categorize personal file metadata. Never claim to "
                        "have read file contents. Return one proposal per "
                        "file. PARA must be project, area, resource, archive, "
                        "or inbox. Domain must be research, teaching, "
                        "software, business, "
                        "personal, administration, finance, health, media, or "
                        "other. Use inbox and low confidence when metadata is "
                        "ambiguous. Suggested groups are short neutral labels."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        [
                            {
                                "file_id": value.file_id,
                                "relative_path": value.relative_path,
                                "extension": value.extension,
                                "byte_size": value.byte_size,
                            }
                            for value in observations
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "format": self._schema(),
            "options": {
                "temperature": 0,
                "seed": 20260922,
                "num_ctx": 16_384,
                "num_predict": 4_096,
            },
        }
        response = self.transport.request(
            "/api/chat",
            (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8"),
            timeout_seconds=self.timeout_seconds,
        )
        envelope = self._object(response, "Ollama response")
        message = envelope.get("message")
        if not isinstance(message, dict):
            raise OllamaRuntimeError("Ollama response message is invalid")
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaRuntimeError("Ollama response content is invalid")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as error:
            raise OllamaRuntimeError(
                "categorization content is invalid JSON"
            ) from error
        if not isinstance(raw, dict) or not isinstance(
            raw.get("proposals"), list
        ):
            raise OllamaRuntimeError("categorization payload is invalid")
        by_id = {value.file_id: value for value in observations}
        proposals: list[CategorizationProposal] = []
        seen: set[str] = set()
        for item in raw["proposals"]:
            if not isinstance(item, dict):
                raise OllamaRuntimeError("categorization proposal is invalid")
            proposal = self._proposal(item)
            if proposal.file_id not in by_id or proposal.file_id in seen:
                raise OllamaRuntimeError(
                    "categorization proposal identity is invalid"
                )
            seen.add(proposal.file_id)
            proposals.append(proposal)
        if seen != set(by_id):
            raise OllamaRuntimeError(
                "categorization response has incomplete coverage"
            )
        return tuple(proposals)

    def _validate_model(self) -> None:
        payload = self._object(
            self.transport.request(
                "/api/tags", None, timeout_seconds=self.timeout_seconds
            ),
            "Ollama model inventory",
        )
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaRuntimeError("Ollama model inventory is invalid")
        matches = [
            value
            for value in models
            if isinstance(value, dict) and value.get("name") == self.model
        ]
        if len(matches) != 1 or matches[0].get("digest") != self.model_digest:
            raise OllamaRuntimeError(
                "configured Ollama model identity is unavailable"
            )

    def _proposal(self, item: dict[str, Any]) -> CategorizationProposal:
        file_id = item.get("file_id")
        para = item.get("para_category")
        domain = item.get("life_domain")
        confidence = item.get("confidence")
        group = item.get("suggested_group")
        rationale = item.get("rationale")
        if not isinstance(file_id, str):
            raise OllamaRuntimeError("proposal file_id is invalid")
        if not isinstance(para, str) or not isinstance(domain, str):
            raise OllamaRuntimeError("proposal category is invalid")
        if not isinstance(confidence, int | float) or isinstance(
            confidence, bool
        ):
            raise OllamaRuntimeError("proposal confidence is invalid")
        if not isinstance(group, str) or not isinstance(rationale, str):
            raise OllamaRuntimeError("proposal text is invalid")
        try:
            return CategorizationProposal(
                file_id=file_id,
                para_category=ParaCategory(para),
                life_domain=LifeDomain(domain),
                confidence=float(confidence),
                suggested_group=group.strip()[:120],
                rationale=rationale.strip()[:500],
                model=self.model,
                model_digest=self.model_digest,
            )
        except ValueError as error:
            raise OllamaRuntimeError(
                "proposal violates its contract"
            ) from error

    @staticmethod
    def _object(payload: bytes, label: str) -> dict[str, Any]:
        try:
            value = json.loads(payload.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OllamaRuntimeError(f"{label} is invalid JSON") from error
        if not isinstance(value, dict):
            raise OllamaRuntimeError(f"{label} must be an object")
        return value

    @staticmethod
    def _schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["proposals"],
            "properties": {
                "proposals": {
                    "type": "array",
                    "maxItems": 25,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "file_id",
                            "para_category",
                            "life_domain",
                            "confidence",
                            "suggested_group",
                            "rationale",
                        ],
                        "properties": {
                            "file_id": {"type": "string"},
                            "para_category": {
                                "type": "string",
                                "enum": [value.value for value in ParaCategory],
                            },
                            "life_domain": {
                                "type": "string",
                                "enum": [value.value for value in LifeDomain],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "suggested_group": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 120,
                            },
                            "rationale": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 500,
                            },
                        },
                    },
                }
            },
        }
