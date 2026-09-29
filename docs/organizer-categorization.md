# Organizer metadata-categorization and observer boundary

Status: bounded owner component; proposals remain automated and unreviewed.

`projectkoios.agent.organizer` owns a local, metadata-only organizer boundary. It
provides:

- immutable, bounded file-observation and categorization-proposal values;
- a domain-specific `BaseFileCategorizer` with one abstract `propose` method;
- a loopback-only Ollama implementation with pinned model digest and minimum
  runtime version, deterministic generation parameters, and bounded requests;
- discovery of conventional macOS cloud-provider roots;
- a read-only filesystem metadata scan that skips symlinks and never opens file
  payloads;
- a private SQLite catalog for roots, observations, proposals, control state,
  and operational events; and
- explicit `on`, `pause`, `off`, and `status` controls for a polling daemon.

The categorizer remains independently usable with caller-supplied bounded
metadata. The daemon composes that contract with local discovery, catalog, and
control components; it does not create a generic workflow or model-backend
framework.

## Privacy and trust boundary

Relative paths and filenames are sensitive, untrusted data and never model
instructions. In direct categorizer use, the caller supplies only a stable file
identifier, relative path, extension, and byte size. In daemon use, the scanner
derives the same categorization fields from local filesystem metadata and also
records root identity, filename, modification time, and cloud-placeholder
availability in the private catalog.

The scanner uses metadata operations only. It traverses directories through
no-follow descriptors so a directory replaced by a symbolic link cannot
redirect a scan outside the root. It skips symbolic links, non-regular files,
inaccessible entries, unavailable directories, and metadata outside the
bounded observation contract. macOS dataless flags are recorded as cloud
placeholders. The scanner does not open source files or request placeholder
downloads. The source tree is never renamed, moved, deleted, or otherwise
modified.

The Ollama implementation sends the bounded categorization fields to an
explicitly allowlisted loopback endpoint. Its default transport disables
proxies, rejects redirects, and bounds response bytes. It does not contact a
remote service. Model output is validated as untrusted input before proposals
are stored.

A proposal does not establish file contents, ownership, privacy clearance,
rights, publication approval, or authorization to organize a file.

## Local operation

The package installs two entry points:

- `koios-organizer {on|pause|off|status}` updates or reports explicit control
  state.
- `koios-organizerd` runs observation and categorization passes while the
  desired mode is `on`.

The catalog defaults to
`~/projectkoios/.koios/store-v1/state/organizer/catalog.sqlite3`. Override it
with an absolute `KOIOS_ORGANIZER_CATALOG` path, or override the data root
with an absolute `KOIOS_DATA_ROOT` path.

The daemon requires both `KOIOS_ORGANIZER_MODEL_DIGEST` and
`KOIOS_ORGANIZER_MINIMUM_RUNTIME_VERSION`. `KOIOS_ORGANIZER_MODEL` defaults to
`qwen3.5:9b`. The digest and minimum runtime version are checked before each
non-empty categorization request.

The catalog directory and database are created with private permissions.
Repeated scans update observed metadata, invalidate a stored proposal when its
source metadata changes, and preserve an event history. Records for paths that
later disappear are not removed automatically, so the catalog is a retained
projection rather than a guaranteed current inventory. It is not an
authorization ledger or a source of file ownership decisions.

## Non-goals

This component does not:

- read file payloads or infer knowledge of their contents;
- move, rename, delete, upload, publish, or otherwise organize source files;
- follow symbolic links or force cloud-placeholder downloads;
- review or accept categorization proposals;
- install a process supervisor, login item, scheduler, or system service;
- coordinate repositories or invoke a generic workflow engine;
- support remote model services; or
- extract a shared backend or transport framework.

Any source mutation, proposal review, organization policy, remote inference,
or service installation requires a separately owned and authorized boundary.
