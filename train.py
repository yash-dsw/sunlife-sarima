"""
train.py
========
Run full training pipeline:
  1. Load data
  2. Train SARIMA/SARIMAX driver models (core + band-wise HC where available)
  3. Learn per-rate constants from historical data (6 rate files)
  4. Walk-forward validation (Test-2025 and Validation-2026, where available)
  5. Save metrics to models/training_metrics.json
"""

import logging
import json
import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_all, DEPARTMENTS
from src.model_trainer import (
    train_department,
    walk_forward_validate,
    save_metrics,
    learn_series_strategy,
    save_series_strategy,
)
from src.cost_rules import learn_rates, save_rates

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


def _default_data_path(base_dir: Path) -> Path:
    samples = sorted(base_dir.glob("cost_forecast_sample_data*.xlsx"))
    if samples:
        return samples[0]
    return base_dir / "data" / "raw" / "workforce_cost_model_v4.xlsx"


def main(data_path: Path):
    logger.info("Loading data …")
    depts = load_all(data_path)

    all_metrics: dict = {}

    for dept, df in depts.items():
        logger.info(f"\n{'='*55}\nDept: {dept}  ({len(df)} months)\n{'='*55}")

        # ── learn cost rates ──────────────────────────────────────────
        slug = dept.lower().replace(" ", "_").replace("&", "and")
        rates = learn_rates(df, lookback_months=18)
        save_rates(rates, MODELS_DIR / f"per_fte_rates_{slug}.json")
        logger.info(f"  Rates saved → {dept}")

        # ── train SARIMA models ───────────────────────────────────────
        train_department(df, dept, MODELS_DIR)
        logger.info(f"  SARIMA models saved → {dept}")

        # ── walk-forward validation ───────────────────────────────────
        logger.info(f"  Running walk-forward validation …")
        val_metrics = walk_forward_validate(df, dept)
        all_metrics[dept] = val_metrics

        strategy = learn_series_strategy(df, min_lift_wmape=1.0, holdout_months=6)
        save_series_strategy(strategy, MODELS_DIR / f"series_strategy_{slug}.json")
        logger.info(f"  Strategy saved → {dept}")

        for rnd, series_dict in val_metrics.items():
            for series, m in series_dict.items():
                if "error" not in m:
                    logger.info(
                        f"    {rnd} | {series:<20} "
                        f"MAPE={m['mape']:.1f}%  MAE={m['mae']:,.0f}  DirAcc={m['dir_acc']:.0f}%"
                    )
                else:
                    logger.warning(f"    {rnd} | {series}: {m['error']}")

    # ── save aggregate metrics ────────────────────────────────────────
    metrics_path = MODELS_DIR / "training_metrics.json"
    save_metrics(all_metrics, metrics_path)
    logger.info(f"\nTraining complete. Metrics → {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train workforce cost forecasting models.")
    parser.add_argument(
        "--data",
        default=str(_default_data_path(BASE_DIR)),
        help="Path to source Excel workbook.",
    )
    args = parser.parse_args()
    main(Path(args.data))
