# Forecasting Population Health in Sub-Saharan Africa — reproducibility pipeline



This package implements the study’s six-country HALE analyses from 1990–2021 using 28 literature-informed candidate predictors, RF feature importance, Boruta, ARIMAX, Random Forest, XGBoost, a 15-predictor baseline, five expanding-window folds, 25 six-year rolling folds, top-10/top-20/all-28 sensitivity analyses, SHAP, uncertainty calibration, and diagnostic model comparisons.

## Installation

```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py --data /path/to/health.dta --out results --shap --boruta
```


## Designs

- Baseline: country-specific 1990–2020 training and 2021 holdout; 15 predictors.
- Expanding: train through 1995/2000/2005/2010/2015; test 1996/2001/2006/2011/2016.
- Rolling: train 6 consecutive years, test next year, 25 folds from 1996 to 2020.
- Robustness: top 10, top 20 and all 28, historical one-step predictions 2000–2020 and holdout 2021.
- Uncertainty: 21 potential 2000–2020 chronological residual calibration errors per country/model; finite order-statistic 95% intervals for 2021, plus ARIMAX model-based intervals where estimable.
- Interpretation: RF SHAP training values and holdout additivity checks; optional Boruta.



## Main outputs

`all_forecasts.csv`, `country_model_metrics.csv`, `all_sample_sizes.csv`, `failed_folds.csv`, `model_parameters.csv`, `feature_ranking_training_only.csv`, `boruta_training_only.csv` (optional), `table4_residual_intervals_2021.csv`, `interval_calibration_errors.csv`, `ARIMAX_2021_prediction_intervals.csv`, `dm_matched_loss_diagnostics.csv`, `reproduction_audit.csv`, `figures/` and `SHAP_*.csv` (optional).

## Data availability

Source providers include IHME, World Bank WDI and ICRG. Redistribution may be restricted; see `data/README.md`. Use the authors' actual data only where permission allows. 
