from scripts.build_corrected_test_query_sequences import resolve_exact_query_clocks


def test_exact_query_clocks_use_event_ids_instead_of_truncated_protocol_time(spark):
    queries = spark.createDataFrame(
        [
            ("event-a", 7, 1_650_000_000_000),
            ("event-b", 7, 1_650_000_001_000),
        ],
        "example_id string, user_id long, query_time_ms long",
    )
    history = spark.createDataFrame(
        [
            ("event-a", 7, 1_650_000_000_321),
            ("event-b", 7, 1_650_000_001_987),
            ("event-a", 8, 1_650_000_000_555),
        ],
        "event_id string, user_id long, time_ms long",
    )

    rows = {
        row["example_id"]: row.asDict()
        for row in resolve_exact_query_clocks(queries, history).collect()
    }

    assert set(rows) == {"event-a", "event-b"}
    assert rows["event-a"]["current_time_ms"] == 1_650_000_000_321
    assert rows["event-b"]["current_time_ms"] == 1_650_000_001_987
    assert rows["event-a"]["query_time_ms"] == 1_650_000_000_000
