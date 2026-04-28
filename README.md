# 🛒 AI E-Commerce Decision Intelligence Dashboard

A full-featured Streamlit dashboard wrapping the BDA Assignment 04 & 05 ML pipelines.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Open the app in your browser (default: http://localhost:8501)
2. If a saved model exists, choose **📦 Load Saved Model** for instant startup
3. To train on new data, enable retrain, upload CSV, then click **🚀 Run Full Analysis & Save Model**
4. Explore all 13 tabs

## Dataset Columns Expected

The pipeline works best with columns like:
`CustomerID`, `CustomerAge`, `CustomerGender`, `City`, `ProductCategory`,
`UnitPrice`, `DiscountPercent`, `ShippingFee`, `OrderQuantity`, `TotalAmount`,
`ProfitMargin`, `DeliveryDays`, `CustomerRating`, `PaymentMode`, `DeliveryStatus`, etc.

Missing columns are handled gracefully.

## Tabs

| # | Tab | Source |
|---|-----|--------|
| 1 | Dataset Overview | New UI |
| 2 | Data Processing | bda_ass04 |
| 3 | Feature Engineering | bda_ass05 |
| 4 | ML Model Results | bda_ass05 |
| 5 | Visual Analytics | bda_ass05 |
| 6 | Prediction System | bda_ass05 |
| 7 | Revenue Intelligence | **NEW** |
| 8 | Customer Intelligence | bda_ass05 + **NEW CLV** |
| 9 | Purchase Prediction | **NEW** |
| 10 | Discount Optimization | **NEW** |
| 11 | Demand Forecast | **NEW** |
| 12 | Churn Prediction | **NEW** |
| 13 | Decision Support System | **NEW** |

## Notes

- **Zero existing logic modified** — all preprocessing, feature engineering, and ML code is identical to the notebooks.
- New tabs use separate functions that call existing features as inputs only.
- Trained model + pipeline state are persisted to `artifacts/trained_dashboard_state.joblib` for reuse.
