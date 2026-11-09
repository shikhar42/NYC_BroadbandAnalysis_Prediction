# NYC Broadband Analysis and Prediction

Analysis of broadband availability (fiber, cable and DSL) across the 62
counties of New York State in 2010, and a regression model that predicts
a county's fiber share of housing units.

> Coursework note: this is a coursework project (originally written in
> 2019). It is kept as a portfolio piece; the findings describe the 2010
> data only.

## Problem

Which New York counties and Regional Economic Development Council
(REDC) regions had housing units with access to high-speed fiber
internet in 2010, how did fiber compare with cable and DSL in the same
counties, and can a county's fiber share be predicted from its size and
its cable/DSL availability?

## Data source

- New York State broadband data (managed by the National
  Telecommunications and Information Administration, US Department of
  Commerce), via NY Open Data dataset `sjc6-ftj4`:
  https://data.ny.gov/api/views/sjc6-ftj4/rows.csv?accessType=DOWNLOAD&sorting=true
- **The URL is live and its contents can change.** A pinned copy,
  downloaded on 2026-11-09 (1,635 rows, 24 columns), is saved in this
  repo at `data/broadband_ny_2010.csv`, and the notebook uses that copy
  when it is present. To refresh it, delete the file and re-run the
  notebook (it will download and re-pin a copy) — but results may then
  differ from those reported below. Do not silently replace the pinned
  copy.
- The analysis subset is the 62 records with `Municipality Type` =
  County.

## Files

- `FinalProject.ipynb` — the analysis notebook (EDA, hypothesis tests,
  predictive model).
- `FinalProject.py` — helper module imported by the notebook
  (`csv_import`, `select_cols`, `filter_data`, `LinearRegressionM`, plus
  `vif_table` and `repeated_cv_mae`). The canonical name is
  **FinalProject** — the `.py` file, the notebook's imports and this
  README all use exactly that spelling and case.
- `data/broadband_ny_2010.csv` — pinned copy of the data (see above).
- `requirements.txt` — Python dependencies.

## Methods

- EDA: county and REDC-region summaries and plots of fiber, cable and
  DSL housing units and provider counts.
- Hypothesis tests: one-tailed **Wilcoxon signed-rank tests, paired by
  county** (the same 62 counties supply both values in each comparison;
  a rank-based test also suits these skewed counts). Four comparisons:
  fiber units vs DSL units, fiber units vs cable units, fiber providers
  vs DSL providers, fiber providers vs cable providers.
- Collinearity: variance inflation factors (VIF) for the regression
  predictors.
- Predictive model: linear regression on the **fiber share**
  (`# Hse Units Fiber` / `2010 Muni Housing Units`), not the raw count.
  Count predictions are share × housing units, **clipped to between 0
  and the county's housing-unit count**.
- Evaluation: **repeated K-fold cross-validation** (5 folds × 10
  repeats, fixed seed 42), reporting MAE with its spread. Baseline:
  the training folds' mean fiber share × housing units, scored on the
  same folds.

## Results

(From running the current notebook on the pinned data copy.)

- 32 of the 62 counties record **zero** housing units with fiber access.
  The minimum number of fiber providers is **1**, both overall and
  within those 32 counties.
- Wilcoxon signed-rank tests (paired by county, one-tailed):
  - Fiber units vs DSL units: statistic 3.0, p ≈ 1.000 — not
    significant; fiber units are not shown to be greater.
  - Fiber units vs cable units: statistic 0.0, p ≈ 1.000 — not
    significant.
  - Fiber providers vs DSL providers: statistic 981.5, p = 0.0085 —
    significant at the 0.01 level.
  - Fiber providers vs cable providers: statistic 1412.0,
    p = 2.48e-10 — significant.
- VIF shows extreme collinearity among the size measures: housing units
  ≈ 6,626, cable units ≈ 5,444, DSL units ≈ 955, population ≈ 47.
  Individual regression coefficients should not be interpreted.
- Repeated cross-validation MAE (50 folds): model **38,078 ± 25,657**
  housing units, versus baseline **48,787 ± 28,990**; MAE on the fiber
  share is 0.176 ± 0.038. The model beats the baseline, but the spread
  across folds is large because county sizes are extremely skewed.

The dataset records availability only; it does not contain household
behaviour or marketing data, so the analysis makes no claim about *why*
a county's fiber uptake is high or low.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook FinalProject.ipynb
```

Run the notebook top to bottom. It reads `data/broadband_ny_2010.csv`
if present (see Data source above). Validated by executing the notebook
end-to-end with no errors (Python 3.12, versions in
`requirements.txt`).
