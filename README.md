# projectkoios-agent

Deferred reusable agent-domain components for Project Koios.

Repository routing is documented in `projectkoios-bootstrap/maps/repositories.md`.

The demonstrated literature-review extraction boundary is documented in
[`docs/literature-review-assessment.md`](docs/literature-review-assessment.md).

## Local organization agent

The `projectkoios.agent.organizer` package provides a private, metadata-only
organization agent. It discovers configured cloud-drive roots, records file
metadata and local/placeholder availability, and asks a pinned local Ollama
model for PARA and life-domain proposals. It does not read file payload bytes,
move, rename, delete, publish, or upload files.

The SQLite catalog exposes bounded proposal projections. Course review starts
with `life_domain="teaching"` proposals and may compare course-code tokens in
private relative paths with the separately reviewed course inventory. These are
candidates only: a proposal does not establish course identity, ownership,
privacy clearance, rights, or publication approval.

The console commands are:

- `koios-organizer on|pause|off|status` for the desired-mode record.
- `koios-organizerd` for the long-running local worker. It requires
  `KOIOS_ORGANIZER_MODEL_DIGEST` and optionally accepts
  `KOIOS_ORGANIZER_MODEL` and `KOIOS_ORGANIZER_CATALOG`.

Process ownership remains outside this package. The Project Koios local
startup/shutdown scripts manage the worker PID directly. GitHubTask and GitHub
Actions are not schedulers, queues, process supervisors, or authorization
mechanisms for this agent.
