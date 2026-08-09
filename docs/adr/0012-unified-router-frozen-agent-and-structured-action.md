# ADR 0012: Unified Router, frozen Agent, and structured action boundary

Status: Accepted (2026-08-09)

## Decision

Every new visible chat bubble is appended to its Session first. The runtime
then sends the complete chronological visible message list, with each stored
timestamp, to exactly one Router Provider request. The current bubble has no
special field or priority; it is only the final list item.

The Router sees only each enabled RuntimeRelease Agent's stable ID, name,
description, and structured scope. That same call returns `lane`,
`selected_agent_id`, and a natural-language `reason`. A returns a null Agent;
B and C return one valid same-scope Agent. Invalid output fails transparently:
there is no default, retrying Router, Selector, Resolver, or fallback Agent.

An A result creates one ordinary Handoff in the same turn and short-circuits
all Agent, Skill, RAG, MCP, Tool, Draft, Proposal, and business-write paths.
For B and C, the chosen Agent and Session RuntimeRelease snapshot are frozen
for the turn. Only that Agent's published bindings can be assembled. A failed
retrieval, Tool, or Agent response cannot change the selection.

Only RAG document chunks in the turn's immutable EvidenceSet can become user
citations. Citation validation is an ID/type membership check, not an answer
judge. MCP and ordinary Tool results remain separate execution records.

A work-order flow can begin only from the frozen B Agent's strict
`proposal_request`. The backend validates structured fields. Missing fields
remain a Draft; complete fields create only a pending Proposal. The chat card's
explicit confirm/cancel action uses `proposal_id` and no model or natural
language parser. Confirmation remains Proposal -> Approval -> ActionGateway ->
internal service -> Receipt and is idempotent. MCP is read-only at release,
Gateway, and executor boundaries.

## Compatibility

Historical Router, Handoff subtype, and controlled-action helpers remain only
for reading old records and old code references. They are unreachable from the
public chat stream. Historical Trace, Release, Proposal, Receipt, work-order,
and Provider ledger rows are not rewritten.
