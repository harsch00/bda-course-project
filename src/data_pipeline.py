from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


DATA_PATH = Path("ecommerce_dataset.csv")


@dataclass(frozen=True)
class FeatureSpec:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    demand_target: str = "OrderQuantity"
    margin_target: str = "ProfitMargin"


FEATURE_SPEC = FeatureSpec(
    feature_columns=[
        "Company",
        "CustomerAge",
        "CustomerGender",
        "City",
        "State",
        "ProductCategory",
        "ProductName",
        "Brand",
        "UnitPrice",
        "DiscountPercent",
        "ShippingFee",
        "PaymentMode",
        "DeliveryStatus",
        "DeliveryDays",
        "CustomerRating",
        "OrderMonth",
        "OrderDayOfWeek",
        "OrderQuarter",
        "IsWeekend",
    ],
    categorical_columns=[
        "Company",
        "CustomerGender",
        "City",
        "State",
        "ProductCategory",
        "ProductName",
        "Brand",
        "PaymentMode",
        "DeliveryStatus",
    ],
    numeric_columns=[
        "CustomerAge",
        "UnitPrice",
        "DiscountPercent",
        "ShippingFee",
        "DeliveryDays",
        "CustomerRating",
        "OrderMonth",
        "OrderDayOfWeek",
        "OrderQuarter",
        "IsWeekend",
    ],
)


def load_raw_data(csv_path: Path | str = DATA_PATH, row_limit: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(csv_path, nrows=row_limit)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["OrderDate"] = pd.to_datetime(out["OrderDate"], errors="coerce")
    out = out.dropna(subset=["OrderDate", "UnitPrice", "OrderQuantity", "ProfitMargin"])

    out["OrderMonth"] = out["OrderDate"].dt.month.astype(int)
    out["OrderDayOfWeek"] = out["OrderDate"].dt.dayofweek.astype(int)
    out["OrderQuarter"] = out["OrderDate"].dt.quarter.astype(int)
    out["IsWeekend"] = (out["OrderDayOfWeek"] >= 5).astype(int)
    out["OrderYear"] = out["OrderDate"].dt.year.astype(int)

    out["DiscountPercent"] = out["DiscountPercent"].clip(lower=0, upper=95)
    out["ProfitMargin"] = out["ProfitMargin"].clip(lower=-5, upper=95)
    out["DeliveryDays"] = out["DeliveryDays"].clip(lower=0, upper=45)
    out["CustomerRating"] = out["CustomerRating"].clip(lower=1, upper=5)

    out["RealizedProfit"] = out["TotalAmount"] * (out["ProfitMargin"] / 100.0)
    out["FinalUnitPrice"] = out["UnitPrice"] * (1.0 - (out["DiscountPercent"] / 100.0))
    out["LineRevenueBeforeShipping"] = out["FinalUnitPrice"] * out["OrderQuantity"]
    out["EstimatedUnitCost"] = (
        (out["TotalAmount"] - out["RealizedProfit"] - out["ShippingFee"]) / out["OrderQuantity"].replace(0, np.nan)
    )
    out["EstimatedUnitCost"] = out["EstimatedUnitCost"].replace([np.inf, -np.inf], np.nan)
    out["EstimatedUnitCost"] = out.groupby(["ProductCategory", "ProductName"])["EstimatedUnitCost"].transform(
        lambda s: s.fillna(s.median())
    )
    out["EstimatedUnitCost"] = out["EstimatedUnitCost"].fillna(out["EstimatedUnitCost"].median())

    for col in FEATURE_SPEC.categorical_columns:
        out[col] = out[col].astype(str).fillna("Unknown")
    return out


def build_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    numeric_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipe, FEATURE_SPEC.categorical_columns),
            ("numeric", numeric_pipe, FEATURE_SPEC.numeric_columns),
        ],
        remainder="drop",
    )


def time_order_train_valid_split(
    df: pd.DataFrame, valid_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values("OrderDate")
    split_idx = int(len(sorted_df) * (1.0 - valid_fraction))
    return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()


def get_latest_context_defaults(df: pd.DataFrame) -> dict[str, object]:
    latest = df.sort_values("OrderDate").iloc[-1]
    return {
        "Company": latest["Company"],
        "CustomerAge": int(latest["CustomerAge"]),
        "CustomerGender": latest["CustomerGender"],
        "City": latest["City"],
        "State": latest["State"],
        "ProductCategory": latest["ProductCategory"],
        "ProductName": latest["ProductName"],
        "Brand": latest["Brand"],
        "UnitPrice": float(latest["UnitPrice"]),
        "DiscountPercent": float(latest["DiscountPercent"]),
        "ShippingFee": float(latest["ShippingFee"]),
        "PaymentMode": latest["PaymentMode"],
        "DeliveryStatus": latest["DeliveryStatus"],
        "DeliveryDays": float(latest["DeliveryDays"]),
        "CustomerRating": float(latest["CustomerRating"]),
        "OrderMonth": int(latest["OrderMonth"]),
        "OrderDayOfWeek": int(latest["OrderDayOfWeek"]),
        "OrderQuarter": int(latest["OrderQuarter"]),
        "IsWeekend": int(latest["IsWeekend"]),
    }


def feature_frame_from_records(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    missing = [col for col in FEATURE_SPEC.feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return frame[FEATURE_SPEC.feature_columns].copy()


def generate_data_profile(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "date_min": str(df["OrderDate"].min().date()),
        "date_max": str(df["OrderDate"].max().date()),
        "categories": int(df["ProductCategory"].nunique()),
        "products": int(df["ProductName"].nunique()),
        "brands": int(df["Brand"].nunique()),
        "states": int(df["State"].nunique()),
        "profit_margin_mean": float(df["ProfitMargin"].mean()),
        "demand_mean": float(df["OrderQuantity"].mean()),
        "total_revenue": float(df["TotalAmount"].sum()),
    }
