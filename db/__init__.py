"""
Database helpers for AgentOS.

Provides the Postgres-backed agent database used by AgentOS for session
storage, scheduling, and component metadata.
"""

from os import getenv
from urllib.parse import quote

from agno.db.postgres import PostgresDb


def _build_postgres_url() -> str:
    """Build a SQLAlchemy Postgres URL from the compose-provided DB_* variables.

    Passwords (and other components) are percent-encoded so special characters
    do not break the URL or leak into connection arguments.
    """
    user = getenv("DB_USER", "ai")
    password = getenv("DB_PASS", "ai")
    host = getenv("DB_HOST", "demo-os-db")
    port = getenv("DB_PORT", "5432")
    database = getenv("DB_DATABASE", "ai")

    def _enc(value: str) -> str:
        return quote(str(value), safe="")

    return (
        f"postgresql+psycopg://"
        f"{_enc(user)}:{_enc(password)}@{host}:{port}/{_enc(database)}"
    )


def get_postgres_db() -> PostgresDb:
    """Return the shared PostgresDb instance used by AgentOS."""
    db_url = getenv("POSTGRES_URL") or _build_postgres_url()
    return PostgresDb(db_url=db_url)
