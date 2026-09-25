from __future__ import annotations

import os
import sys
from pathlib import Path

from py4j.protocol import Py4JError, Py4JNetworkError
from pyspark import SparkContext
from pyspark.sql import SparkSession


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clear_spark_state() -> None:
    SparkSession._instantiatedSession = None
    SparkSession._activeSession = None
    SparkContext._active_spark_context = None
    SparkContext._gateway = None
    SparkContext._jvm = None


def _session_is_healthy(session: SparkSession | None) -> bool:
    if session is None:
        return False

    try:
        session.range(1).count()
        return True
    except (ConnectionRefusedError, Py4JError, Py4JNetworkError):
        return False


def get_spark(app_name: str = "recommender", reset: bool = False) -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 4g pyspark-shell")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark_local_dir = Path(os.getenv("SPARK_LOCAL_DIR", _project_root() / "data/spark-tmp")).resolve()
    spark_local_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SPARK_LOCAL_DIRS"] = str(spark_local_dir)
    os.environ["LOCAL_DIRS"] = str(spark_local_dir)

    if reset:
        _clear_spark_state()

    existing = SparkSession._activeSession or SparkSession._instantiatedSession
    if _session_is_healthy(existing):
        return existing
    if existing is not None:
        try:
            existing.stop()
        except (ConnectionRefusedError, Py4JError, Py4JNetworkError):
            pass
        _clear_spark_state()

    spark_master = os.getenv("SPARK_MASTER", "local[*]")

    builder = (
        SparkSession.builder.master(spark_master)
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.local.dir", str(spark_local_dir))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
    )

    try:
        session = builder.getOrCreate()
    except (ConnectionRefusedError, Py4JError, Py4JNetworkError):
        _clear_spark_state()
        session = builder.getOrCreate()

    if not _session_is_healthy(session):
        _clear_spark_state()
        session = builder.getOrCreate()

    return session
