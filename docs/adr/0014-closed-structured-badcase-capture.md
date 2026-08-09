# ADR 0014: Closed structured Badcase capture

Status: Accepted (2026-08-09)

## Decision

The active automatic Badcase path consumes only closed runtime facts:
`runtime_failed`, `contract_invalid`, `capability_failed`,
`citation_invalid`, and `action_failed`. The already-selected Agent may also
self-report `answer_status=insufficient_evidence` or
`answer_status=capability_unavailable`; these retain distinct sources so an
operator can tell an Agent-reported gap from a backend failure.

Answer wording, keywords, substrings, and an LLM Judge cannot open a Badcase.
Unknown contract codes remain system observations. Every automatically opened
record starts at `pending`; AI and Darwin may provide suggestions, while the
existing human lifecycle remains authoritative.

Occurrences deduplicate only when trigger code, component, immutable
RuntimeRelease, and a deterministic normalized-question fingerprint all
match. Repeated Traces increment the open record and retain their Trace IDs.
A recurrence after a human terminal decision creates a new pending record and
does not reopen, overwrite, or mutate the historical decision.

## Compatibility

Historical Badcases, actions, statuses, Evaluation cases, Golden Set runs, and
Trace evidence are not rewritten. Golden cases remain explicitly created,
activated, run, and reviewed by an operator. No automatic capture path creates
or runs a Golden case.

