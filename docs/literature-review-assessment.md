# Literature-review assessment boundary

Status: extracted candidate; implementation does not accept scientific claims.

`projectkoios.agent.literature_review` owns the bounded model-proposal boundary
proven by a bounded scientific literature-review pilot. It provides:

- immutable claim and evidence-bundle inputs;
- a closed assessment schema with allowlisted statuses;
- duplicate-member, unknown-field, size, citation-label, and conditional-field
  rejection;
- prompting that requires direct support for every material clause of a
  conjunctive claim;
- a loopback-only Ollama adapter with pinned model digest, minimum runtime,
  deterministic parameters, and exact request/response archives; and
- an explicit `AUTOMATED_UNREVIEWED` result boundary.

The package does not retrieve evidence, ingest references, verify equations,
record human dispositions, implement a classifier, choose simulation settings,
or execute scientific calculations. Those concerns remain in their owner
repositories or in separately authorized review workflows.
