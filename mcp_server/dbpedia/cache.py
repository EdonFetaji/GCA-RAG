"""
A tiny TTL + LRU cache, shared by the SPARQL, Lookup, and Spotlight transports.

Not a general-purpose cache: it exists because DBpedia's public endpoints
rate-limit, and one grounding pass issues many near-identical queries — the same
`search_class("Organization")` for every ORGANIZATION in the graph. Without this
a medium graph gets throttled partway through and the tail of the run silently
degrades to "no candidates".

Thread-safe, because `SyncMCPClient` drives the server from a background thread
and an HTTP server may serve several sessions at once.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """LRU-bounded, time-expiring key→value store."""

    def __init__(self, *, ttl_seconds: float, max_entries: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def get(self, key: str) -> Any | None:
        """The cached value, or None if absent or expired. None is never a cached value."""
        if not self.enabled:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        if not self.enabled or value is None:
            return
        with self._lock:
            self._data[key] = (time.monotonic() + self._ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
