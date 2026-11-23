"""Run Phase 1 once; store a fixed raw snapshot and deterministic ML-ready dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_spy, raw_metadata, sha256_file, validate_spy
from .features import FEATURE_COLUMNS, build_features
from .splits import chronological_splits, make_supervised
from .targets import build_targets


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(root: Path, *, refresh: bool = False) -> dict:
    root = Path(root).resolve()
    raw_path = root / "data" / "raw" / "spy_daily.parquet"
    metadata_path = root / "data" / "metadata" / "spy_daily.json"
    output_dir = root / "data" / "processed"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not refresh:
        spy = pd.read_parquet(raw_path)
        if not metadata_path.is_file():
            raise RuntimeError("Cached raw snapshot has no checksum metadata; inspect it or use --refresh")
        recorded_hash = json.loads(metadata_path.read_text(encoding="utf-8")).get("sha256")
        if not recorded_hash or recorded_hash != sha256_file(raw_path):
            raise RuntimeError("Raw snapshot checksum missing/mismatched; inspect the stored file")
        source = "cached snapshot"
    else:
        spy = download_spy()
        validate_spy(spy, check_calendar=True)
        spy.to_parquet(raw_path)
        write_json(metadata_path, raw_metadata(spy, raw_path))
        source = "new Yahoo Finance download"

    validate_spy(spy, check_calendar=True)
    raw_hash = sha256_file(raw_path)
    features = build_features(spy)
    targets = build_targets(spy)
    supervised = make_supervised(features, targets)
    splits = chronological_splits(supervised)

    features.to_parquet(output_dir / "spy_features.parquet")
    # Dedicated feature-only history for neural sequence construction: no
    # held-out 2022+ rows need to be loaded by Phase 3.
    features.loc[:splits["validation"].index.max()].to_parquet(
        output_dir / "spy_features_through_validation.parquet")
    supervised.to_parquet(output_dir / "spy_supervised.parquet")
    for name, frame in splits.items():
        frame.to_parquet(output_dir / f"spy_{name}.parquet")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset": "SPY",
        "frequency": "1d",
        "source": source,
        "raw_sha256": raw_hash,
        "feature_columns": list(FEATURE_COLUMNS),
        "target": "label = 1[adjusted_open[t+2] / adjusted_open[t+1] - 1 > 0]",
        "prediction_after": "Close(t)",
        "entry_at": "Open(t+1)",
        "economic_exit_at": "Open(t+2)",
        "proxy_label_availability": "Adjusted ratio uses exit-session EOD inputs and vendor revisions; conservative purge requires exit_date < next validation signal date",
        "adjusted_open_proxy": "Open * Adj Close / Close",
        "return_caveat": "Total-return proxy; not a tradable price or a full dividend cash-flow ledger",
        "point_in_time_caveat": "Historical adjusted prices can be retrospectively revised",
        "rows": {name: len(part) for name, part in splits.items()},
        "ranges": {
            name: [str(part.index[0].date()), str(part.index[-1].date())]
            for name, part in splits.items()
        },
        "train_class_rate": float(splits["train"]["label"].mean()),
        "feature_missing_warmup": int((~np.isfinite(features.to_numpy(dtype=float)).all(axis=1)).sum()),
        "usage": "Fit scalers ONLY on training folds. Test set remains held out.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="SPY Phase 1 data + feature + label pipeline")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--refresh", action="store_true", help="Explicitly re-download and replace snapshot")
    args = parser.parse_args()
    summary = run_pipeline(args.root, refresh=args.refresh)
    print(json.dumps({"rows": summary["rows"], "ranges": summary["ranges"]}, indent=2))
    print("Phase 1 completed. Inspect data/processed/manifest.json before ML training.")


if __name__ == "__main__":
    main()
