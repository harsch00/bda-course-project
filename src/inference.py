from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_pipeline import DATA_PATH, FEATURE_SPEC, engineer_features, feature_frame_from_records, load_raw_data


@dataclass
class ModelBundle:
    margin_model: Any
    demand_model: Any
    discount_min: float
    discount_max: float


def load_model_bundle(
    margin_path: Path | str = Path("artifacts/margin_model.joblib"),
    demand_path: Path | str = Path("artifacts/demand_model.joblib"),
) -> ModelBundle:
    margin_artifact = joblib.load(margin_path)
    demand_artifact = joblib.load(demand_path)

    bounds = margin_artifact.get("discount_bounds", {"min": 0.0, "max": 30.0})
    return ModelBundle(
        margin_model=margin_artifact["model"],
        demand_model=demand_artifact["model"],
        discount_min=float(bounds["min"]),
        discount_max=float(bounds["max"]),
    )


def build_cost_lookup(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["ProductCategory", "ProductName", "Brand"], as_index=False)["EstimatedUnitCost"]
        .median()
        .rename(columns={"EstimatedUnitCost": "MedianUnitCost"})
    )
    return grouped


def get_default_data(csv_path: Path | str = DATA_PATH) -> pd.DataFrame:
    return engineer_features(load_raw_data(csv_path=csv_path))


def predict_margin(bundle: ModelBundle, feature_record: dict[str, object]) -> float:
    features = feature_frame_from_records([feature_record])[FEATURE_SPEC.feature_columns]
    pred = float(bundle.margin_model.predict(features)[0])
    return float(np.clip(pred, -5.0, 95.0))


def predict_demand(bundle: ModelBundle, feature_record: dict[str, object]) -> float:
    features = feature_frame_from_records([feature_record])[FEATURE_SPEC.feature_columns]
    pred = float(bundle.demand_model.predict(features)[0])
    return float(np.clip(pred, 0.0, None))


def recommend_margin_and_price(
    bundle: ModelBundle,
    base_record: dict[str, object],
    estimated_unit_cost: float,
    search_points: int = 20,
) -> dict[str, float]:
    min_d = max(0.0, bundle.discount_min)
    max_d = min(60.0, max(bundle.discount_max, min_d + 1.0))
    discounts = np.linspace(min_d, max_d, search_points)

    best: dict[str, float] | None = None
    for d in discounts:
        probe = dict(base_record)
        probe["DiscountPercent"] = float(d)
        candidate = evaluate_scenario(bundle=bundle, record=probe, estimated_unit_cost=estimated_unit_cost)
        if best is None or candidate["expected_total_profit"] > best["expected_total_profit"]:
            best = candidate

    assert best is not None
    return best


def evaluate_scenario(bundle: ModelBundle, record: dict[str, object], estimated_unit_cost: float) -> dict[str, float]:
    discount = float(record["DiscountPercent"])
    margin_pct = predict_margin(bundle, record)
    demand_units = predict_demand(bundle, record)
    final_unit_price = float(record["UnitPrice"]) * (1.0 - discount / 100.0)
    unit_profit = final_unit_price * (margin_pct / 100.0)
    expected_profit = demand_units * unit_profit
    return {
        "discount_percent": float(discount),
        "predicted_margin_pct": float(margin_pct),
        "predicted_demand_units": float(demand_units),
        "final_unit_price": float(final_unit_price),
        "expected_unit_profit": float(unit_profit),
        "expected_total_profit": float(expected_profit),
        "estimated_unit_cost": float(estimated_unit_cost),
    }
