# YIAI Property Repository Instructions

This repository is an AI product manager interview demonstration. Real,
explainable, and verifiable runtime behavior has priority over feature count.

## Mandatory context for runtime work

Before changing chat, Agent, Skill, RAG, MCP/Tool, work-order, Trace,
Evaluation, Badcase, or cost behavior, read:

1. `docs/handover/v1.7.1-runtime-convergence/00-index.md`
2. `docs/handover/v1.7.1-runtime-convergence/08-v1.8-enterprise-runtime-architecture-contract.md`
3. `docs/adr/0002-runtime-release-session-snapshot-and-composite-workflow.md`
4. `docs/releases/v1.8.0.md`

If the handover documents are not present in the current checkout, stop and
read them from the `docs/v1.7.1-runtime-convergence-handover` branch without
checking that branch out into the NAS production working directory.

## Current approved design constraints

These constraints describe the latest approved Living Architecture. They are
not permanently locked. A justified change is allowed after an ADR, an
architecture-document revision, updated acceptance contracts, and user
approval. Implementation must not silently drift from the active revision.

- Agno is the execution runtime; YIAI owns release, authorization, business
  state, and evidence contracts.
- A new session runs against one immutable published configuration snapshot.
- Read-only consultation, controlled state change, and dynamic extension are
  separate workflows and may not take control from one another.
- Models may propose actions but may not authorize, commit, or claim success.
- Every model-triggered write goes through the ActionGateway, persisted
  approval, idempotency protection, and a real receipt.
- Agents receive only tools allowed by the published binding and tool policy.
- Citation text, final citations, UI chunk display, and Trace retrieval evidence
  come from the same immutable EvidenceSet.
- Provider usage that is incomplete or unavailable must not become a
  precise-looking monetary cost.
- Badcases come from evaluation failure, runtime contract violation, user
  feedback, or real operational failure—not merely from a response without RAG.
- Do not keep adding orchestration branches to `app/chat.py`; keep it as a thin
  API/SSE adapter.

## Change governance

Any change to the current approved constraints requires an ADR, a new
architecture-document revision with explicit supersession, and corresponding
contract-test changes before implementation. Proposed revisions are welcome;
unrecorded deviations are not.

Do not describe design, static checks, contract tests, real-model validation,
manual UI acceptance, or deployment as equivalent evidence.

Preserve user data and dirty worktrees. Do not stash, reset, clean, force-push,
delete volumes, persist credentials, or run real model calls without explicit
authorization.
