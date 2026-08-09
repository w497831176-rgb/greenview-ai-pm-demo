# ADR 0015: Runtime hot binding and immutable Session snapshots

- Status: Accepted
- Scope: Target 2 runtime orchestration contract
- Date: 2026-08-09

## Context

The platform management surface edits an unpublished runtime draft. Operators
may bind or unbind already-admitted Skill, RAG and read-only MCP/Tool
capabilities on an existing Agent, inspect the resulting Diff, and publish an
immutable `RuntimeRelease`. A draft edit must never alter the current release
or any Session that has already started.

“Hot” means that publishing or rolling back changes the release used by a new
Session without a code change or service restart. It does not mean that an
arbitrary implementation can bypass controlled admission, or that a live
Session can drift to a different configuration.

## Decision

1. Agent capability edits are draft state. The current `RuntimeRelease`
   remains authoritative until an explicit successful publish.
2. Publish compiles Agent bindings and capability content into a new immutable
   graph and atomically advances the current release pointer.
3. The first runtime entry for a Session resolves the current published
   release and persists exactly one `RunConfigSnapshot`. Later bubbles reuse
   that snapshot by `session_id`, regardless of later publish or rollback.
4. Runtime assembly occurs only after an Agent is frozen and reads only that
   Agent's Skill IDs, knowledge document IDs and MCP server bindings from the
   Session snapshot. An unbound capability may remain in the catalog and in
   release history, but it cannot be activated, retrieved, exposed as a Tool,
   or presented as that Agent's capability.
5. Trace records actual activation, retrieval and invocation. A bound but
   unused MCP/Tool is availability, not a fabricated invocation; an unbound
   capability produces no activation, retrieval or Tool invocation record.
6. Rollback re-points production to an already-validated immutable historical
   release. It does not rewrite release content, create a replacement release,
   or alter existing Session snapshots.

## Consequences

- Bind then publish affects only Sessions created after the publish.
- Unbind then publish removes the capability only for subsequently created
  Sessions; Sessions pinned to the bound release retain it.
- Rollback affects subsequently created Sessions while all earlier Session
  hashes and release graphs remain unchanged.
- No model call, process restart, catalog-object creation, or implementation
  mutation is part of runtime binding itself.

## Deterministic verification

`scripts/test_v182_target2_hot_binding_lifecycle.py` uses a temporary SQLite
database and symbolic fixtures to prove draft isolation, bind/publish,
unbind/publish, release Diff, immutable history, old/new Session behavior,
rollback, catalog-count stability, capability assembly boundaries, and truthful
Trace projection. It constructs but never invokes a model, and replaces the
MCP toolkit constructor so no MCP process or network call can occur.
