from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from src.data_pipeline import FEATURE_SPEC, build_preprocessor, engineer_features, load_raw_data, time_order_train_valid_split
except ModuleNotFoundError:
    from data_pipeline import FEATURE_SPEC, build_preprocessor, engineer_features, load_raw_data, time_order_train_valid_split  # type: ignore


def _make_momentum_table(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.assign(Month=df["OrderDate"].dt.to_period("M").dt.to_timestamp())
        .groupby(["Month", "ProductCategory", "ProductName"], as_index=False)["OrderQuantity"]
        .sum()
        .sort_values(["ProductCategory", "ProductName", "Month"])
    )
    monthly["prev"] = monthly.groupby(["ProductCategory", "ProductName"])["OrderQuantity"].shift(1)
    monthly["growth_rate"] = np.where(monthly["prev"] > 0, (monthly["OrderQuantity"] - monthly["prev"]) / monthly["prev"], np.nan)
    monthly["trend_label"] = pd.cut(
        monthly["growth_rate"],
        bins=[-np.inf, -0.08, 0.08, np.inf],
        labels=["Declining", "Stable", "Rising"],
    ).astype(str)
    return monthly


def train_demand_model(row_limit: int | None = None) -> dict[str, object]:
    raw = load_raw_data(row_limit=row_limit)
    data = engineer_features(raw)
    train_df, valid_df = time_order_train_valid_split(data, valid_fraction=0.2)

    X_train = train_df[FEATURE_SPEC.feature_columns]
    y_train = train_df[FEATURE_SPEC.demand_target]
    X_valid = valid_df[FEATURE_SPEC.feature_columns]
    y_valid = valid_df[FEATURE_SPEC.demand_target]

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=RandomForestRegressor(
                        n_estimators=60,
                        max_depth=14,
                        min_samples_leaf=8,
                        n_jobs=1,
                        random_state=42,
                    ),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_valid), 0.0, None)

    metrics = {
        "mae": float(mean_absolute_error(y_valid, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, preds))),
        "r2": float(r2_score(y_valid, preds)),
    }

    momentum = _make_momentum_table(data)
    latest_month = momentum["Month"].max()
    latest_snapshot = momentum[momentum["Month"] == latest_month].copy()

    return {
        "model": model,
        "metrics": metrics,
        "momentum_table": momentum,
        "latest_snapshot": latest_snapshot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train demand prediction model.")
    parser.add_argument("--row-limit", type=int, default=None, help="Optional row cap for quick runs.")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/demand_model.joblib"),
        help="Where to save model artifact.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/demand_metrics.json"),
        help="Where to save metrics JSON.",
    )
    parser.add_argument(
        "--momentum-path",
        type=Path,
        default=Path("artifacts/demand_momentum.csv"),
        help="Where to save monthly product momentum table.",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("artifacts/latest_product_snapshot.csv"),
        help="Where to save latest product trend snapshot.",
    )
    args = parser.parse_args()

    output = train_demand_model(row_limit=args.row_limit)
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": output["model"],
            "feature_columns": FEATURE_SPEC.feature_columns,
            "demand_target": FEATURE_SPEC.demand_target,
        },
        args.artifact_path,
    )
    args.metrics_path.write_text(json.dumps(output["metrics"], indent=2), encoding="utf-8")
    output["momentum_table"].to_csv(args.momentum_path, index=False)
    output["latest_snapshot"].to_csv(args.snapshot_path, index=False)

    print("Demand model trained.")
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()
