# Review-note MVP boundary

## Status

Slice 1 implements the pure contract, prompt construction, response validation,
and deterministic Markdown rendering. Local artifact publication and a real
model backend remain deferred to later slices.

## Purpose

The review-note MVP reads one local Markdown note and proposes a bounded,
inspectable review artifact. It tests whether a small agent-domain component can
control a model backend, validate structured output, render useful Markdown,
and preserve enough provenance to replay or audit the exchange.

The result is an automated proposal. It is not a scientific, pedagogical, or
editorial acceptance decision.

## Inputs

One request contains:

- a stable request identifier;
- an explicit source path;
- a display title that does not grant filesystem authority;
- a caller-selected artifact root;
- an explicit output path below that root;
- an explicit model-exchange archive directory below that root; and
- a bounded review profile selected from an allowlist.

The application computes the source SHA-256 from the bytes it reads. The first
implementation should impose byte and character limits before prompt
construction. It should reject invalid UTF-8, symbolic-link source or output
paths, an output path that already exists, and artifact paths outside the
caller-selected root.

The source note is untrusted data, never model instructions. Front matter,
embedded HTML, links, and fenced code remain source content.

## Proposed result contract

The model proposes exactly one closed JSON object containing:

- `summary` — a concise account of the note's purpose;
- `strengths` — bounded observations grounded in the supplied note;
- `concerns` — bounded correctness, clarity, or structure concerns;
- `missing_definitions` — terms used without enough local explanation;
- `suggested_revisions` — concrete edits described but not applied; and
- `questions` — matters requiring author judgment.

Every list has an item-count and item-length limit. Unknown fields, duplicate
JSON members, invalid UTF-8, oversized responses, and unexpected value types
are rejected. The application adds an `AUTOMATED_UNREVIEWED` boundary; the
model cannot set or remove it.

The renderer produces Obsidian-compatible Markdown with fixed section order.
It quotes the source identity and request identity but does not copy the full
source note into the result.

## Backend sequence

The first implementation uses a deterministic fake backend in unit tests. The
backend receives the completed prompts and returns response bytes; it does not
receive filesystem paths or write artifacts.

A later loopback Ollama adapter may be added after the pure contract, renderer,
and application service are tested. That adapter should reuse the established
literature-review controls where applicable:

- loopback-only transport;
- pinned model digest and minimum runtime version;
- deterministic model parameters;
- exact request and response archives;
- bounded transport payloads; and
- replay without repeating a completed model call.

Shared backend mechanics should be extracted only after both components expose
the same stable behavior. The first review-note slice may duplicate a small
amount of adapter code rather than generalize prematurely.

## Artifact publication

The application service receives explicit source, output, and archive paths.
It validates the response before publishing the review. Publication uses an
exclusive temporary file, flushes it, and atomically replaces the final path.
It never overwrites an existing review.

A receipt binds at least:

- request schema and request identity;
- source SHA-256;
- prompt/request SHA-256;
- raw response SHA-256;
- rendered review SHA-256;
- model name, digest, and runtime version when a real backend is used; and
- the `AUTOMATED_UNREVIEWED` boundary.

Timestamps may be observational metadata, but they must not determine request,
artifact, or replay identity.

## Workspace API assessment

This MVP does **not** organically require the mothership
`projectkoios.agents` API.

A review is a caller-selected product artifact, not an agent session, mutable
workspace state, coordination decision, or generated handoff.
`AgentWorkspaceAction` would add agent/workspace/session semantics that the
review contract does not need, and its directory routing would weaken the
explicit output boundary. Importing it would also create a target-to-mothership
implementation dependency.

Therefore the first implementation must not import `projectkoios.agents` or
claim to satisfy the independent-consumer gate in
[`projectkoios#9`](https://github.com/eragasa/projectkoios/issues/9). The
workspace prototype remains deferred unless another real use demonstrates its
boundary.

## Non-goals

The MVP does not:

- scan a vault or choose notes;
- modify the source note;
- retrieve external evidence or assess citations;
- accept scientific or pedagogical claims;
- create tasks, decisions, sessions, or handoffs;
- coordinate repositories or invoke a generic workflow engine;
- publish results remotely; or
- define a reusable model-backend framework.

## Implementation slices

### Slice 1: pure contract and rendering

- immutable request and proposal dataclasses;
- closed response parser with bounded fields;
- system and user prompt construction;
- deterministic Markdown renderer; and
- tests for valid output, prompt-injection-shaped note content, duplicate JSON
  members, unknown fields, invalid types, and size limits.

### Slice 2: local application service

- bounded UTF-8 source loading;
- source and artifact identities;
- fake-backend execution;
- atomic no-overwrite review and receipt publication; and
- tests for symlinks, existing outputs, partial writes, and deterministic
  replay.

### Slice 3: pinned loopback backend

- loopback Ollama transport;
- model digest and runtime validation;
- deterministic generation parameters;
- exact exchange archives and replay; and
- an opt-in integration test against a configured local runtime.

Each slice must pass `pytest`, `ruff check .`, and `mypy src/python`. Slice 3
must not begin until slices 1 and 2 establish the backend-independent contract.
