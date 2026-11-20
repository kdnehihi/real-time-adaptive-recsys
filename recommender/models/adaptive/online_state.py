from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


EVENT_ARRAY_FIELDS = {
    "video_id": "recent_video_ids",
    "target_class": "recent_target_classes",
    "engagement_strength": "recent_engagement_strengths",
    "long_view": "recent_long_views",
    "is_like": "recent_likes",
    "is_hate": "recent_hates",
    "is_click": "recent_clicks",
    "time_ms": "recent_time_ms",
}


def _optional_int(value: Any, default: int = 0) -> int:
    """Convert nullable pandas/Arrow scalars to a stable integer value."""
    if value is None:
        return default
    try:
        scalar = np.asarray(value).reshape(-1)[0]
        if np.asarray(value).size == 1 and bool(scalar != scalar):
            return default
    except (TypeError, ValueError):
        pass
    return int(value)


def _optional_float(value: Any, default: float = 0.0) -> float:
    """Convert nullable numeric outcomes to the zero used by Spark aggregates."""
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    """Minimal immutable interaction needed to update adaptive user state."""

    event_id: str
    user_id: int
    video_id: int
    time_ms: int
    session_id: str
    target_class: str
    engagement_strength: float
    long_view: int
    is_like: int
    is_hate: int
    is_click: int

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "InteractionEvent":
        """Build an event from a Gold row while normalizing nullable scalars."""
        event_id = row.get("current_event_id", row.get("event_id", row.get("example_id", "")))
        return cls(
            event_id=str(event_id),
            user_id=_optional_int(row.get("user_id")),
            video_id=_optional_int(row.get("video_id")),
            time_ms=_optional_int(row.get("current_time_ms", row.get("time_ms"))),
            session_id=str(row.get("session_id", "")),
            target_class=str(row.get("target_class", "AMBIGUOUS_WEAK")),
            engagement_strength=_optional_float(row.get("engagement_strength")),
            long_view=_optional_int(row.get("long_view")),
            is_like=_optional_int(row.get("is_like")),
            is_hate=_optional_int(row.get("is_hate")),
            is_click=_optional_int(row.get("is_click")),
        )


class OnlineUserStateReducer:
    """Maintain bounded per-user and per-session state in event-time order.

    Queries at timestamp ``t`` must call :meth:`snapshot` before the timestamp
    batch at ``t`` is passed to :meth:`apply_timestamp_batch`. This reproduces
    the Gold contract that every sequence event is strictly earlier than the
    query, including when multiple interactions share the same millisecond.
    """

    def __init__(self, max_sequence_len: int = 30, max_session_sequence_len: int = 10):
        if max_sequence_len <= 0 or max_session_sequence_len <= 0:
            raise ValueError("sequence lengths must be positive")
        self.max_sequence_len = int(max_sequence_len)
        self.max_session_sequence_len = int(max_session_sequence_len)
        self._users: dict[int, deque[InteractionEvent]] = defaultdict(
            lambda: deque(maxlen=self.max_sequence_len)
        )
        self._sessions: dict[tuple[int, str], deque[InteractionEvent]] = defaultdict(
            lambda: deque(maxlen=self.max_session_sequence_len)
        )
        self._last_committed_time_ms: int | None = None
        self.events_applied = 0

    @staticmethod
    def _event_sort_key(event: InteractionEvent) -> tuple[str, int, int]:
        """Use Gold's event ID tie-breaker, with item/user IDs for total order."""
        return event.event_id, event.video_id, event.user_id

    def apply_timestamp_batch(self, events: Iterable[InteractionEvent]) -> int:
        """Append one same-timestamp batch after all queries at that time run."""
        batch = list(events)
        if not batch:
            return 0
        timestamps = {event.time_ms for event in batch}
        if len(timestamps) != 1:
            raise ValueError("apply_timestamp_batch requires exactly one timestamp")
        timestamp = next(iter(timestamps))
        if self._last_committed_time_ms is not None and timestamp < self._last_committed_time_ms:
            raise ValueError("event timestamp moved backwards")
        for event in sorted(batch, key=self._event_sort_key):
            self._users[event.user_id].append(event)
            self._sessions[(event.user_id, event.session_id)].append(event)
        self._last_committed_time_ms = timestamp
        self.events_applied += len(batch)
        return len(batch)

    def snapshot(self, user_id: int, session_id: Any, query_time_ms: int) -> dict[str, Any]:
        """Return Gold-compatible strict-prior arrays and summary statistics."""
        user_id = int(user_id)
        session_id = str(session_id)
        query_time_ms = int(query_time_ms)
        recent = list(self._users.get(user_id, ()))
        session_recent = list(self._sessions.get((user_id, session_id), ()))
        if any(event.time_ms >= query_time_ms for event in recent + session_recent):
            raise AssertionError("online state contains a non-prior event")
        output = self._sequence_payload(recent, prefix="recent")
        output.update(self._sequence_payload(session_recent, prefix="session_recent"))
        output["seconds_since_last_event"] = self._seconds_since_last(recent, query_time_ms)
        output["session_seconds_since_last_event"] = self._seconds_since_last(
            session_recent, query_time_ms
        )
        return output

    @staticmethod
    def _seconds_since_last(events: list[InteractionEvent], query_time_ms: int) -> float | None:
        """Return elapsed seconds, matching Spark null behavior for empty state."""
        if not events:
            return None
        return (query_time_ms - events[-1].time_ms) / 1000.0

    @staticmethod
    def _sequence_payload(events: list[InteractionEvent], prefix: str) -> dict[str, Any]:
        """Convert an event deque to the array/rate fields used by adaptive Gold."""
        payload: dict[str, Any] = {}
        for attribute, recent_name in EVENT_ARRAY_FIELDS.items():
            name = recent_name if prefix == "recent" else f"session_{recent_name}"
            payload[name] = [getattr(event, attribute) for event in events]
        payload[f"{prefix}_event_count"] = len(events)
        if prefix == "recent":
            denominator = max(len(events), 1)
            payload["recent_long_view_rate"] = sum(event.long_view for event in events) / denominator
            payload["recent_like_rate"] = sum(event.is_like for event in events) / denominator
            payload["recent_hate_rate"] = sum(event.is_hate for event in events) / denominator
            payload["recent_avg_engagement_strength"] = (
                sum(event.engagement_strength for event in events) / denominator
            )
        return payload

    @property
    def active_users(self) -> int:
        """Number of users currently represented in the bounded state store."""
        return len(self._users)

    @property
    def active_sessions(self) -> int:
        """Number of user/session buffers currently represented in memory."""
        return len(self._sessions)
