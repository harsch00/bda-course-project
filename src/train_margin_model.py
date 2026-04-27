from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from src.data_pipeline import (
        FEATURE_SPEC,
        build_preprocessor,
        engineer_features,
        generate_data_profile,
        load_raw_data,
        time_order_train_valid_split,
    )
except ModuleNotFoundError:
    from data_pipeline import (  # type: ignore
        FEATURE_SPEC,
        build_preprocessor,
        engineer_features,
        generate_data_profile,
        load_raw_data,
        time_order_train_valid_split,
    )


def train_margin_model(row_limit: int | None = None) -> dict[str, object]:
    raw = load_raw_data(row_limit=row_limit)
    data = engineer_features(raw)
    train_df, valid_df = time_order_train_valid_split(data, valid_fraction=0.2)

    X_train = train_df[FEATURE_SPEC.feature_columns]
    y_train = train_df[FEATURE_SPEC.margin_target]
    X_valid = valid_df[FEATURE_SPEC.feature_columns]
    y_valid = valid_df[FEATURE_SPEC.margin_target]

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=14,
                    min_samples_leaf=8,
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_valid), -5.0, 95.0)

    metrics = {
        "mae": float(mean_absolute_error(y_valid, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, preds))),
        "r2": float(r2_score(y_valid, preds)),
    }
    profile = generate_data_profile(data)
    discount_bounds = {
        "min": float(data["DiscountPercent"].quantile(0.02)),
        "max": float(data["DiscountPercent"].quantile(0.98)),
    }

    return {
        "model": model,
        "metrics": metrics,
        "profile": profile,
        "discount_bounds": discount_bounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train margin prediction model.")
    parser.add_argument("--row-limit", type=int, default=None, help="Optional row cap for quick runs.")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/margin_model.joblib"),
        help="Where to save model artifact.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/margin_metrics.json"),
        help="Where to save metrics JSON.",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=Path("artifacts/data_profile.json"),
        help="Where to save data profile JSON.",
    )
    args = parser.parse_args()

    output = train_margin_model(row_limit=args.row_limit)
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": output["model"],
            "feature_columns": FEATURE_SPEC.feature_columns,
            "margin_target": FEATURE_SPEC.margin_target,
            "discount_bounds": output["discount_bounds"],
        },
        args.artifact_path,
    )
    args.metrics_path.write_text(json.dumps(output["metrics"], indent=2), encoding="utf-8")
    args.profile_path.write_text(json.dumps(output["profile"], indent=2), encoding="utf-8")

    print("Margin model trained.")
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()
