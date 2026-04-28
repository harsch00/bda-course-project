# Ecommerce Seller Intelligence - Project Report

## Introduction

Ecommerce Seller Intelligence is an end-to-end machine learning project designed to support data-driven decisions for online sellers. The system combines predictive analytics and an interactive dashboard to answer three practical business questions:

1. How much demand can be expected for a given product-market scenario?
2. What discount level is likely to maximize expected profit?
3. Which products and categories are currently rising or declining in momentum?

The project is implemented in Python and includes:
- a data engineering and preprocessing pipeline,
- two supervised learning models (margin prediction and demand forecasting),
- an inference/optimization layer for pricing recommendations, and
- a Streamlit dashboard for business-friendly consumption of insights.

This makes the solution usable both as a technical ML workflow and as a lightweight decision-support application for sellers and analysts.

## Literature Survey

This project stands at the intersection of ecommerce analytics, regression modeling, and pricing optimization. The core ideas are aligned with established work in the following areas:

### 1) Demand Forecasting in Retail and Ecommerce
- Demand forecasting literature emphasizes temporal features (month, quarter, day-of-week), price, promotions, and product attributes as key drivers of sales quantity.
- In practical industry settings, robust tree-based regressors are often preferred when data is heterogeneous (mixed categorical + numeric fields) and noisy.

Relevance to this project:
- The demand model uses transaction context, product metadata, customer attributes, and temporal variables to predict `OrderQuantity`.

### 2) Price/Promotion and Margin Analytics
- Revenue management research shows that discount strategy directly impacts both conversion (demand) and unit economics (margin), requiring joint optimization rather than independent decisions.
- A common practical objective is expected profit maximization under uncertain demand.

Relevance to this project:
- The system predicts margin and demand separately for each candidate discount, then selects the discount that maximizes expected profit.

### 3) Ensemble Learning for Tabular Business Data
- Random Forest is a standard baseline-to-strong model family for tabular regression due to nonlinearity capture, low preprocessing burden, and stability against overfitting under tuned depth/leaf constraints.
- Target transformation (for skewed count-like targets) is also well supported in applied ML workflows.

Relevance to this project:
- Margin and demand models are based on Random Forest pipelines.
- Demand prediction uses log-transform/inverse-transform through `TransformedTargetRegressor` (`log1p`/`expm1`) to better model skewed quantity distributions.

### 4) Decision Intelligence Dashboards
- Modern analytics practice favors deployable, interactive dashboards that expose both model outputs and interpretable supporting visuals (trend bars, heatmaps, frontier curves) for human-in-the-loop decision making.

Relevance to this project:
- A Streamlit + Plotly interface operationalizes recommendations and trend analytics for non-technical users.

## Methodology

The project follows a clear ML lifecycle from data ingestion to decision output.

### 1) Data Ingestion and Cleaning
- Source data is read from `ecommerce_dataset.csv`.
- Rows with missing critical fields (`OrderDate`, `UnitPrice`, `OrderQuantity`, `ProfitMargin`) are dropped.
- Date parsing and quality controls are applied.

### 2) Feature Engineering

Derived temporal features:
- `OrderMonth`
- `OrderDayOfWeek`
- `OrderQuarter`
- `IsWeekend`
- `OrderYear` (used for context/profile)

Value clipping / normalization rules:
- `DiscountPercent` clipped to [0, 95]
- `ProfitMargin` clipped to [-5, 95]
- `DeliveryDays` clipped to [0, 45]
- `CustomerRating` clipped to [1, 5]

Additional engineered business variables:
- `RealizedProfit`
- `FinalUnitPrice`
- `LineRevenueBeforeShipping`
- `EstimatedUnitCost` (computed from transaction totals, then median-imputed by product group)

### 3) Preprocessing Strategy
- Categorical features are imputed with most-frequent values and encoded using `OrdinalEncoder` with unknown handling.
- Numeric features are imputed using median strategy.
- A `ColumnTransformer` combines categorical and numeric preprocessing into one reusable pipeline.

### 4) Model Training

#### A) Margin Model
- Target: `ProfitMargin`
- Model: `RandomForestRegressor` inside a sklearn pipeline
- Split: time-ordered train/validation (80/20), avoiding random leakage across time
- Validation metrics: MAE, RMSE, R2
- Output artifacts:
  - `artifacts/margin_model.joblib`
  - `artifacts/margin_metrics.json`
  - `artifacts/data_profile.json`

#### B) Demand Model
- Target: `OrderQuantity`
- Model: `RandomForestRegressor` wrapped with `TransformedTargetRegressor`
- Target transform: `log1p` and `expm1`
- Split: time-ordered train/validation (80/20)
- Validation metrics: MAE, RMSE, R2
- Output artifacts:
  - `artifacts/demand_model.joblib`
  - `artifacts/demand_metrics.json`
  - `artifacts/demand_momentum.csv`
  - `artifacts/latest_product_snapshot.csv`

### 5) Inference and Profit Optimization

For a user-selected scenario, the system:
1. Builds a feature record from UI inputs.
2. Estimates product cost via historical median lookup.
3. Evaluates multiple discount points over a bounded range.
4. For each discount, predicts:
   - margin percentage,
   - demand units,
   - expected unit and total profit.
5. Chooses the discount with maximum expected total profit.

Optimization objective:

`expected_total_profit = predicted_demand_units * final_unit_price * (predicted_margin_pct / 100)`

### 6) Visualization Layer
- Streamlit UI presents recommendation KPIs and interactive plots.
- Plotly charts provide:
  - pricing frontier (profit vs discount),
  - product momentum (hot/declining),
  - demand heatmap by category/month,
  - top demand states,
  - delivery speed vs average demand.

## Features Provided by Application

The application provides the following major features:

1. **Scenario-based pricing recommendation**
   - Suggests discount level expected to maximize profit for a selected product and context.

2. **Dual-model predictive intelligence**
   - Predicts both margin and demand, enabling balanced business decisions.

3. **Pricing frontier analysis**
   - Displays full trade-off curve instead of only one recommendation point.

4. **Product momentum tracking**
   - Identifies rising and declining products based on month-over-month growth rates.

5. **Market and operational insights**
   - Highlights demand concentration across states and relation between delivery speed and demand.

6. **Interactive, business-friendly dashboard**
   - Streamlit interface with selectable inputs for category, product, marketplace, city/state, customer profile, logistics, and time context.

7. **Reusable training/inference pipeline**
   - Separate modules for data processing, model training, and prediction to support maintainability and future upgrades.

8. **Artifact-based deployment pattern**
   - Model and metrics artifacts are persisted in `artifacts/` for reproducible runs.

## Conclusion

This project successfully demonstrates how machine learning can be operationalized for ecommerce decision support. By combining demand forecasting, margin prediction, and discount optimization in one application, it moves beyond static reporting toward actionable recommendations.

The architecture is modular and practical:
- data engineering for reliable feature generation,
- model training scripts for reproducibility,
- inference utilities for optimization logic,
- and a dashboard for stakeholder usability.

Overall, the solution is well-suited as a capstone-style applied ML system that links technical modeling with real business outcomes (profit-focused pricing and trend visibility).

## Future Scope

Potential improvements for next iterations include:

1. **Advanced forecasting models**
   - Compare Random Forest with XGBoost/LightGBM/CatBoost and time-series hybrids.

2. **Elasticity-aware optimization**
   - Explicitly model price elasticity and confidence intervals to support risk-aware recommendations.

3. **Causal promotion analysis**
   - Move from correlation-based prediction to uplift/causal impact estimation for discounts.

4. **Inventory and supply constraints**
   - Include stock limits, procurement lead time, and fulfillment constraints in optimization.

5. **Real-time/near-real-time serving**
   - Build API endpoints and scheduled retraining for production deployment.

6. **Monitoring and MLOps**
   - Add drift detection, model performance monitoring, and automated rollback criteria.

7. **User and business segmentation**
   - Create segment-specific models by region, category, or customer cohorts.

8. **Explainability layer**
   - Add SHAP/permutation importance views so users can interpret recommendation drivers.

## References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.  
   [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.  
   [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

3. Hyndman, R. J., & Athanasopoulos, G. (Forecasting principles). *Forecasting: Principles and Practice*.  
   [https://otexts.com/fpp3/](https://otexts.com/fpp3/)

4. Streamlit Documentation (for interactive ML apps).  
   [https://docs.streamlit.io/](https://docs.streamlit.io/)

5. Plotly Documentation (for analytical visualization).  
   [https://plotly.com/python/](https://plotly.com/python/)

6. Python Software Foundation. Python Documentation.  
   [https://docs.python.org/3/](https://docs.python.org/3/)
