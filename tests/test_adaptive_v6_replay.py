import numpy as np
import pandas as pd
import pytest

from scripts.run_adaptive_v6_v4_2_replay import validate_replay_order


def test_replay_order_accepts_only_strict_prior_sequence_events():
    features = pd.DataFrame({"recent_time_ms": [[10, 20], [], [25, 35]]})
    protocol = pd.DataFrame({"query_time_ms": [30, 30, 40]})
    result = validate_replay_order(features, protocol)
    assert result == {"queries": 3, "sequence_timestamps_checked": 4, "violations": 0}


def test_replay_order_rejects_out_of_order_queries_and_current_event_leakage():
    with pytest.raises(AssertionError, match="non-decreasing"):
        validate_replay_order(
            pd.DataFrame({"recent_time_ms": [[], []]}),
            pd.DataFrame({"query_time_ms": [20, 10]}),
        )

    with pytest.raises(AssertionError, match="leakage"):
        validate_replay_order(
            pd.DataFrame({"recent_time_ms": [[10, 20]]}),
            pd.DataFrame({"query_time_ms": np.array([20], dtype=np.int64)}),
        )
