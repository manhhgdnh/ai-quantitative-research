"""Run the SPY experiments from the project root using the active interpreter."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORECASTING = (
    "quant_trading.pipeline",
    "quant_trading.phase2_baselines",
    "quant_trading.phase2_xgboost",
    "quant_trading.phase2_robustness",
    "quant_trading.phase3_mlp",
    "quant_trading.phase3_lstm",
)
BACKTESTING = ("quant_trading.phase4_backtest", "quant_trading.phase4_robustness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase4-only", action="store_true",
        help="Reuse the processed data and five-model validation predictions for backtesting.",
    )
    args = parser.parse_args()
    modules = BACKTESTING if args.phase4_only else FORECASTING + BACKTESTING
    for module in modules:
        print(f"Running {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
    if not args.phase4_only:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
