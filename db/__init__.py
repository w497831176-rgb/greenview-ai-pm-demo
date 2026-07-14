"""
Database helpers for AgentOS.

Provides the Postgres-backed agent database used by AgentOS for session
storage, scheduling, and component metadata.
"""

from os import getenv

from agno.db.postgres import PostgresDb


def get_postgres_db() -> PostgresDb:
    """Return the shared PostgresDb instance used by AgentOS."""
    return PostgresDb(
        db_url=getenv(
            "POSTGRES_URL",
            "postgresql+psycopg://postgres:postgres@demo-os-db:5432/agno_demo_os",
        ),
    )
