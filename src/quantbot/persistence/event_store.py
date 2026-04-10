from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StoredEvent:
    stream: str
    event_type: str
    payload: dict[str, Any]


class EventStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def append_event(self, *, stream: str, event_type: str, payload: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT INTO events (stream, event_type, payload) VALUES (?, ?, ?)",
            (stream, event_type, json.dumps(payload, separators=(",", ":"), sort_keys=True)),
        )
        self._connection.commit()

    def list_events(self, stream: str) -> list[StoredEvent]:
        rows = self._connection.execute(
            "SELECT stream, event_type, payload FROM events WHERE stream = ? ORDER BY id ASC",
            (stream,),
        ).fetchall()
        return [
            StoredEvent(stream=row[0], event_type=row[1], payload=json.loads(row[2]))
            for row in rows
        ]

