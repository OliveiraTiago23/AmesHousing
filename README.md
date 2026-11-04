# Ames Housing — House Price Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-regression-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF)

End-to-end machine learning project that predicts the **sale price of a house** from its characteristics (surface, quality, neighborhood, age…), built for the Kaggle competition [*House Prices: Advanced Regression Techniques*](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Score of this project : 399/4727 (top 8%, score : 0.11956)

---

## 📊 Results

Models are compared with **5-fold cross-validation** on the training set. The metric is **RMSLE** (RMSE on `log(1 + price)`), the official Kaggle metric — lower is better. As a rule of thumb, an RMSLE of 0.11 means a typical error of about 11 %.

| Model | CV RMSLE |
|---|---|
| Ridge | 0.1141 |
| Gradient Boosting (tuned) | 0.1123 |
| Lasso | 0.1116 |
| ElasticNet | 0.1114 |
| **Ensemble (Ridge + Lasso + ElasticNet + Gradient Boosting)** | **0.1083** |

**How the score improved during the project**

| Step | CV RMSLE |
|---|---|
| Gradient Boosting on 15 hand-picked features | 0.1357 |
| Gradient Boosting on all raw features | 0.1189 |
| Clean preprocessing + feature engineering + regularized linear models | 0.1114 |
| Weighted ensemble of linear models and boosting | **0.1083** |

> Most of the gain came from **better data preparation and feature engineering**, then from **choosing models adapted to the data**, and finally from **ensembling**.

---

## Approach

1. **Exploration** — data types, missing values, duplicates, price distribution (right-skewed → log transform), outliers.
2. **Cleaning**
   - A missing value very often means *"feature absent"* (no pool, no garage…) → filled with the category `"None"` or `0`.
   - `LotFrontage` filled with the median of the same neighborhood.
   - The 2 well-known outliers (huge living area, very low price) are removed from the **training set only**.
3. **Encoding**
   - Ordered quality grades (`Po < Fa < TA < Gd < Ex`…) → ordinal encoding.
   - Unordered categories (e.g. `Neighborhood`) → one-hot encoding; rare categories (< 5 houses) are dropped.
   - Numeric codes (`MSSubClass`, `MoSold`) treated as categories.
4. **Feature engineering** — `TotalSF`, `TotalBath`, `TotalPorchSF`, `HouseAge`, `RemodAge`, `QualSF` (quality × living area), and presence flags (`HasPool`, `HasGarage`, `HasBsmt`…).
5. **Skewness correction** — `log1p` on strongly skewed numeric features; the target is modeled as `log1p(SalePrice)` and converted back with `expm1`.
6. **Modeling** — six models compared on a hold-out set, then with cross-validation.
7. **Tuning** — `GridSearchCV` on Gradient Boosting.
8. **Ensembling** — `VotingRegressor` mixing linear models and trees.
9. **Submission** — final model refit on 100 % of the training data, predictions written to a CSV, with sanity checks (row count, no NaN, no negative price, price distribution).

---

## Models and why they were chosen

| Model | Why it is here |
|---|---|
| **Linear Regression** | Simple baseline; shows the overfitting problem with ~230 columns. |
| **Ridge (L2)** | Stable with many correlated columns (e.g. `GarageCars` / `GarageArea`). |
| **Lasso (L1)** | Sets useless coefficients to exactly 0 → automatic feature selection, interpretable. |
| **ElasticNet** | Mix of both: selection + stability. Best single linear model here. |
| **Random Forest** | Captures non-linearities and interactions; used as a tree-based comparison. |
| **Gradient Boosting** | Usually the most accurate tree model on tabular data; tuned with `GridSearchCV`. |
| **Voting Ensemble** | Averaging models from two different families reduces the error further. |

With only ~1,450 houses, ~230 columns after encoding, many correlated features, and a near-linear relationship once the price is log-transformed, **regularized linear models** are an excellent fit. Trees and boosting add the non-linear effects they cannot capture; the ensemble combines both.

---

## 📏 Evaluation metrics

| Metric | Meaning | Better |
|---|---|---|
| **MAE** | Average absolute error, in dollars | lower |
| **RMSE** | Like MAE but punishes large errors more, in dollars | lower |
| **R²** | Share of the price variance explained by the model | higher |
| **RMSLE** | RMSE on log-prices ≈ typical error in percent (**Kaggle score**) | lower |

---

> The Kaggle data files are **not included** in this repository. Download them from the competition page.

---

## Key lessons

- **Look at what a missing value means** before deleting or imputing it.
- **Never evaluate on the data used for training**; use cross-validation to choose between models.
- **Avoid data leakage**: anything that learns from the data (medians, scalers, encoders, models) must be fit on training data only.
- **Feature engineering and clean preprocessing** beat a fancier algorithm on dirty features.
- **Different model families make different errors** — averaging them helps.
- **Trust cross-validation over the public leaderboard**, which is noisy.

---

## Tech stack

Python · pandas · NumPy · SciPy · scikit-learn · Matplotlib · Seaborn 

