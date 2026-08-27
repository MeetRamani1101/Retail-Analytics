"""Small IO helpers.

Parquet is preferred (it preserves dtypes and is far smaller), but the project
must still run on a machine without pyarrow installed, so both writing and
reading transparently fall back to CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # pragma: no cover - depends on the environment
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:  # pragma: no cover
    HAS_PARQUET = False

DATE_COLUMNS = ["order_date", "signup_date", "order_month", "order_week"]


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Write `df` to `path`, downgrading .parquet to .csv if needed."""
    if path.suffix == ".parquet" and not HAS_PARQUET:
        path = path.with_suffix(".csv")
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_table(path: Path) -> pd.DataFrame:
    """Read a table written by `save_table`, restoring datetime columns."""
    if path.suffix == ".parquet" and not path.exists():
        path = path.with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.data_prep` first."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    df = pd.read_csv(path)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
