# Ecommerce Seller Intelligence

An end-to-end ML app for ecommerce sellers to:
- estimate demand for a product scenario
- optimize discount/margin setup for expected profit
- monitor product momentum and market trends in a clean Streamlit dashboard

## Project Structure

- `src/data_pipeline.py`: data loading, cleaning, feature engineering
- `src/train_margin_model.py`: margin model training + metrics/profile export
- `src/train_demand_model.py`: demand model training + trend table export
- `src/inference.py`: recommendation and prediction helpers
- `app.py`: Apple-inspired interactive Streamlit dashboard
- `artifacts/`: trained model files and metrics

## Setup

```bash
python -m pip install -r requirements.txt
```

## Train Models

Run full training:

```bash
python -m src.train_margin_model
python -m src.train_demand_model
```

For quick smoke tests:

```bash
python -m src.train_margin_model --row-limit 50000
python -m src.train_demand_model --row-limit 50000
```

## Launch Dashboard

```bash
streamlit run app.py
```

## How Pricing Recommendation Works

1. The margin model predicts expected margin percentage for each tested discount.
2. The demand model predicts expected order quantity for the same scenario.
3. The optimizer searches across discount levels and selects the point with the highest:

`expected_profit = predicted_demand * discounted_unit_price * predicted_margin_pct`

The UI also shows the complete pricing frontier so sellers can trade off risk vs upside.
