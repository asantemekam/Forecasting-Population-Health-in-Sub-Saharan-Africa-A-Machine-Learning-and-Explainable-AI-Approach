# Forecasting Population Health in Sub-Saharan Africa — reproducibility pipeline

**Status: IMPLEMENTATION / AUDIT, NOT YET VERIFIED AGAINST THE AUTHORS’ DATA.**

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

To **attempt** to reproduce the manuscript's exhaustive baseline tuning, append `--baseline-grid`. This is computationally expensive (RF 288, XGBoost 192 configurations, each evaluated in 5 time-series folds). The default baseline uses the manuscript’s stated final fixed hyperparameters. No early stopping is used. Exact original fitted parameters, original data, software versions and optimization details must be checked before asserting exact numerical replication.

## Designs

- Baseline: country-specific 1990–2020 training and 2021 holdout; 15 predictors.
- Expanding: train through 1995/2000/2005/2010/2015; test 1996/2001/2006/2011/2016.
- Rolling: train 6 consecutive years, test next year, 25 folds from 1996 to 2020.
- Robustness: top 10, top 20 and all 28, historical one-step predictions 2000–2020 and holdout 2021.
- Uncertainty: 21 potential 2000–2020 chronological residual calibration errors per country/model; finite order-statistic 95% intervals for 2021, plus ARIMAX model-based intervals where estimable.
- Interpretation: RF SHAP training values and holdout additivity checks; optional Boruta.

## Essential limitations / manuscript reconciliation

1. **Conditional prediction**: Observed same-year predictors are supplied to every holdout and validation forecast. They would not be available for a genuinely prospective forecast. These are *conditional out-of-sample predictions*.
2. **Feature selection**: Top 10/15/20 are frozen from the manuscript’s pooled RF ranking; they are not independently selected inside every historical fold. This creates possible selection look-ahead for early folds. The code also recomputes a 1990–2020 ranking to enable audit, but does not silently replace the article's published feature sets.
3. **Small ARIMAX samples**: With 15 exogenous predictors and six annual observations, ARIMAX is underdetermined. Such folds are logged as failures; do not claim 25 successful ARIMAX folds per country if the audit disagrees.
4. **Missingness**: Complete-case data are used for each feature set; folds with missing annual observations are flagged. Nominal counts in the paper assume complete data.
5. **Model selection**: Manuscript wording describes both exhaustive grid search and repeated fold-local tuning, while fixed-parameter code was requested later. Default code reproduces the fixed configuration; optional baseline tuning does not establish which procedure produced published numbers.
6. **Diebold–Mariano**: `dm_matched_loss_diagnostics.csv` gives exploratory matched-loss differences, NOT validated pooled or country-level DM p-values for manuscript Tables 10–11. A suitable dependent-data inferential design is required to replicate those exact results.
7. **Prediction intervals**: The residual-calibrated intervals are nominal and do not have finite-sample conformal coverage guarantees under temporal dependence. ARIMAX convergence can prevent calibration.
8. **SHAP figures**: Raw SHAP arrays and additivity checks are exported; published beeswarm and waterfall visualizations may require additional styling/plot scripts.
9. **Not all figures/tables are guaranteed exact**: The output includes a reproduction audit and explicit failed folds. Compare all numerical outputs to the paper before publication.

## Main outputs

`all_forecasts.csv`, `country_model_metrics.csv`, `all_sample_sizes.csv`, `failed_folds.csv`, `model_parameters.csv`, `feature_ranking_training_only.csv`, `boruta_training_only.csv` (optional), `table4_residual_intervals_2021.csv`, `interval_calibration_errors.csv`, `ARIMAX_2021_prediction_intervals.csv`, `dm_matched_loss_diagnostics.csv`, `reproduction_audit.csv`, `figures/` and `SHAP_*.csv` (optional).

## Data availability

Source providers include IHME, World Bank WDI and ICRG. Redistribution may be restricted; see `data/README.md`. Use the authors' actual data only where permission allows. No source data or results are fabricated in this repository.
