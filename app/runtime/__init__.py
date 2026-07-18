"""YIAI V1.8 converged runtime.

Agno owns agent/workflow execution.  This package owns the application-level
contracts that Agno must not decide for us: published configuration snapshots,
tool authorization, confirmed writes and the evidence/cost truth ledger.

The package deliberately has no eager imports.  That keeps the isolated V1.7
adapter and V1.8 modules from forming import cycles during FastAPI startup.
"""
