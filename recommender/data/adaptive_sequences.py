from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


EVENT_VALUE_COLUMNS = (
    "video_id",
    "target_class",
    "engagement_strength",
    "long_view",
    "is_like",
    "is_hate",
    "is_click",
    "time_ms",
)


def _event_struct() -> F.Column:
    """Create a sortable event struct; event_id is the deterministic tie-breaker."""
    return F.struct(
        F.col("event_id").alias("event_id"),
        *[F.col(name).alias(name) for name in EVENT_VALUE_COLUMNS],
    )


def _bounded_prior_events(
    batches: DataFrame,
    partition_columns: tuple[str, ...],
    maximum_events: int,
) -> DataFrame:
    """Flatten enough prior timestamp batches and retain the latest N events.

    Each timestamp batch contains at least one event, so the previous N batches
    always contain enough rows to recover the previous N events. Current-time
    events are excluded before the limit, which backfills older valid history.
    """
    if maximum_events <= 0:
        raise ValueError("maximum_events must be positive")
    window = Window.partitionBy(*partition_columns).orderBy("time_ms")
    lagged_batches = F.array(
        *[F.lag("_event_batch", offset).over(window) for offset in range(maximum_events, 0, -1)]
    )
    return (
        batches.withColumn(
            "_prior_batches", F.filter(lagged_batches, lambda batch: batch.isNotNull())
        )
        .withColumn("_all_prior_events", F.flatten("_prior_batches"))
        .withColumn(
            "_prior_events",
            F.expr(
                f"slice(_all_prior_events, greatest(size(_all_prior_events) - {maximum_events} + 1, 1), {maximum_events})"
            ),
        )
        .drop("_prior_batches", "_all_prior_events")
    )


def _project_event_arrays(frame: DataFrame, prefix: str) -> DataFrame:
    """Project one bounded event-struct array to model-facing aligned arrays."""
    source = "_prior_events"
    names = {
        "video_id": f"{prefix}_video_ids",
        "target_class": f"{prefix}_target_classes",
        "engagement_strength": f"{prefix}_engagement_strengths",
        "long_view": f"{prefix}_long_views",
        "is_like": f"{prefix}_likes",
        "is_hate": f"{prefix}_hates",
        "is_click": f"{prefix}_clicks",
        "time_ms": f"{prefix}_time_ms",
    }
    for attribute, output in names.items():
        frame = frame.withColumn(output, F.transform(source, lambda event: event[attribute]))
    frame = frame.withColumn(f"{prefix}_event_count", F.size(source))
    return frame


def build_strict_prior_global_context(history: DataFrame, maximum_events: int = 30) -> DataFrame:
    """Build one strict-prior global sequence for every user/timestamp batch."""
    batches = (
        history.groupBy("user_id", "time_ms")
        .agg(F.sort_array(F.collect_list(_event_struct())).alias("_event_batch"))
    )
    context = _project_event_arrays(
        _bounded_prior_events(batches, ("user_id",), maximum_events), "recent"
    )
    denominator = F.greatest(F.size("_prior_events"), F.lit(1))
    return (
        context.withColumn(
            "seconds_since_last_event",
            (F.col("time_ms") - F.try_element_at("recent_time_ms", F.lit(-1)))
            / F.lit(1000.0),
        )
        .withColumn(
            "recent_long_view_rate",
            F.aggregate("recent_long_views", F.lit(0.0), lambda total, value: total + F.coalesce(value.cast("double"), F.lit(0.0))) / denominator,
        )
        .withColumn(
            "recent_like_rate",
            F.aggregate("recent_likes", F.lit(0.0), lambda total, value: total + F.coalesce(value.cast("double"), F.lit(0.0))) / denominator,
        )
        .withColumn(
            "recent_hate_rate",
            F.aggregate("recent_hates", F.lit(0.0), lambda total, value: total + F.coalesce(value.cast("double"), F.lit(0.0))) / denominator,
        )
        .withColumn(
            "recent_avg_engagement_strength",
            F.aggregate("recent_engagement_strengths", F.lit(0.0), lambda total, value: total + F.coalesce(value.cast("double"), F.lit(0.0))) / denominator,
        )
        .drop("_event_batch", "_prior_events")
    )


def build_strict_prior_session_context(history: DataFrame, maximum_events: int = 10) -> DataFrame:
    """Build one strict-prior sequence for every user/session/timestamp batch."""
    batches = (
        history.groupBy("user_id", "session_id", "time_ms")
        .agg(F.sort_array(F.collect_list(_event_struct())).alias("_event_batch"))
    )
    context = _project_event_arrays(
        _bounded_prior_events(batches, ("user_id", "session_id"), maximum_events),
        "session_recent",
    )
    return (
        context.withColumn(
            "session_seconds_since_last_event",
            (
                F.col("time_ms")
                - F.try_element_at("session_recent_time_ms", F.lit(-1))
            )
            / F.lit(1000.0),
        )
        .drop("_event_batch", "_prior_events")
    )


def build_strict_prior_sequence_context(
    history: DataFrame,
    maximum_events: int = 30,
    maximum_session_events: int = 10,
) -> tuple[DataFrame, DataFrame]:
    """Return corrected global and session context tables for later query joins."""
    required = {
        "event_id",
        "user_id",
        "video_id",
        "time_ms",
        "session_id",
        "target_class",
        "engagement_strength",
        "long_view",
        "is_like",
        "is_hate",
        "is_click",
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"history is missing required sequence columns: {missing}")
    canonical = history.select(*sorted(required)).dropDuplicates(["event_id"])
    return (
        build_strict_prior_global_context(canonical, maximum_events),
        build_strict_prior_session_context(canonical, maximum_session_events),
    )
