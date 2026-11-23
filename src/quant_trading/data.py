"""Download and validate one daily US-listed ETF market-data snapshot."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

TICKER = "SPY"
START = "2005-01-01"
END_EXCLUSIVE = "2026-10-31"  # Include 2026-10-30; Yahoo's end is exclusive.
REQUIRED = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def download_spy(start: str = START, end: str = END_EXCLUSIVE) -> pd.DataFrame:
    """Download daily SPY prices, keeping corporate actions and raw OHLC separate."""
    import yfinance as yf  # Imported only when a real download is requested.

    df = yf.download(
        tickers=TICKER,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=False,
        keepna=True,
        multi_level_index=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no SPY observations.")
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected single-level columns; verify yfinance version.")

    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "Date"
    df = df.sort_index()
    for col in ("Dividends", "Stock Splits"):
        if col not in df:
            df[col] = 0.0
    return df


def validate_spy(df: pd.DataFrame, *, check_calendar: bool = True) -> None:
    """Fail loudly for corrupt, incomplete, or misaligned trading-session data."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Market data index must be a DatetimeIndex.")
    if df.empty or len(df) < 70:
        raise ValueError("At least 70 daily rows are needed for 63-day return features.")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Dates must be unique and strictly increasing.")
    if df.index.tz is not None or (df.index != df.index.normalize()).any():
        raise ValueError("Expected timezone-naive US trading dates at midnight.")
    missing_cols = sorted(set(REQUIRED).difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing OHLCV columns: {missing_cols}")
    observed = df[list(REQUIRED)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(observed.to_numpy()).all():
        bad = observed.index[~np.isfinite(observed.to_numpy()).all(axis=1)]
        raise ValueError(f"Invalid/missing numeric price or volume rows: {list(bad[:5])}")
    if (observed[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError("All prices must be strictly positive.")
    if (observed["Volume"] < 0).any():
        raise ValueError("Volume must be nonnegative.")
    tol = 1e-8
    if (observed["High"] + tol < observed[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("High is below Open/Close on at least one session.")
    if (observed["Low"] - tol > observed[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("Low is above Open/Close on at least one session.")
    if (observed["High"] + tol < observed["Low"]).any():
        raise ValueError("High is below Low.")

    if check_calendar:
        # Explicit bounds are essential: exchange_calendars otherwise defaults
        # to a rolling ~20-year window, which can exclude the 2005 start date.
        calendar = xcals.get_calendar(
            "XNYS", start=df.index[0], end=df.index[-1]
        )
        expected = calendar.sessions_in_range(df.index[0], df.index[-1])
        missing = expected.difference(df.index)
        extra = df.index.difference(expected)
        if len(missing) or len(extra):
            raise ValueError(
                "Trading calendar mismatch. "
                f"Missing NYSE sessions: {list(missing[:8])}; "
                f"unexpected dates: {list(extra[:8])}. "
                "Investigate source and market closures; do not silently shift target horizons."
            )


def sha256_file(file: Path) -> str:
    """Return SHA-256 checksum of the exact locally stored file."""
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_metadata(df: pd.DataFrame, path: Path) -> dict:
    return {
        "ticker": TICKER,
        "provider": "Yahoo Finance via yfinance",
        "yfinance_version": version("yfinance"),
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": START,
        "requested_end_exclusive": END_EXCLUSIVE,
        "first_session": str(df.index.min().date()),
        "last_session": str(df.index.max().date()),
        "observations": len(df),
        "download_settings": {"auto_adjust": False, "actions": True, "repair": False},
        "sha256": sha256_file(path),
        "adjusted_price_warning": "Adj Close is an end-of-day retrospectively adjusted series, may later be revised; not point-in-time vendor data.",
    }
