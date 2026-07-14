#!/usr/bin/env sh
set -e

# Wait for the database to be reachable when WAIT_FOR_DB is enabled.
if [ "$WAIT_FOR_DB" = "True" ] || [ "$WAIT_FOR_DB" = "true" ]; then
    echo "Waiting for database at ${DB_HOST:-demo-os-db}:${DB_PORT:-5432}..."
    python - <<PY
import os, socket, time
host = os.environ.get("DB_HOST", "demo-os-db")
port = int(os.environ.get("DB_PORT", "5432"))
while True:
    try:
        socket.create_connection((host, port), timeout=1).close()
        break
    except Exception:
        time.sleep(1)
PY
    echo "Database is reachable."
fi

exec "$@"
