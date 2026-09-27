# Organizer metadata-categorization boundary

Status: bounded candidate; proposals remain automated and unreviewed.

`projectkoios.agent.organizer` owns a small domain contract for proposing PARA
and life-domain categories from caller-supplied file metadata. It provides:

- immutable, bounded file-observation and categorization-proposal values;
- a domain-specific `BaseFileCategorizer` with one abstract `propose` method;
- a deterministic fake implementation in tests through that same is-a
  contract;
- a loopback-only Ollama implementation with pinned model digest and minimum
  runtime version, deterministic generation parameters, and bounded requests;
- strict duplicate-safe, closed, UTF-8 response validation; and
- exact one-proposal-per-input identity coverage.

The base class does not own backend state, transport, lifecycle, persistence, or
shared validation helpers. The Ollama implementation composes with its narrow
transport protocol. This is not a reusable model-backend framework.

## Privacy and trust boundary

The caller supplies only a stable file identifier, relative path, extension,
and byte size. Relative paths and filenames are sensitive, untrusted data and
never model instructions. The package does not read files or verify that the
metadata corresponds to a filesystem object.

The Ollama implementation sends those metadata fields to an explicitly
allowlisted loopback endpoint. Its default transport disables proxies, rejects
redirects, and bounds response bytes. It does not contact a remote service,
archive exchanges, or persist proposals. An injected transport is a caller-owned
test or integration boundary and does not weaken validation of model identity
or returned proposals.

A proposal does not establish file contents, ownership, privacy clearance,
rights, publication approval, or a decision to move or otherwise modify a file.

## Non-goals

This slice does not:

- discover cloud roots, scan directories, or access filesystem metadata;
- read, move, rename, delete, upload, or publish files;
- store observations or proposals in SQLite or another catalog;
- run a daemon, control loop, CLI, scheduler, or process supervisor;
- select files or authorize organization actions;
- coordinate repositories or invoke a generic workflow engine;
- support remote model services; or
- extract a shared backend or transport framework.

Any lifecycle, persistence, filesystem, process, or product-policy behavior
requires a separately demonstrated owner boundary.
