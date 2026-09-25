import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    session = get_spark("two-tower-tests", reset=True)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
