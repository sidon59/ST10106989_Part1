# Medical Aid Charges — Linear Regression (Part 1)

**Author:** Ximiyeto Makhubele

## Overview

A medical aid scheme wants to price member charges based on personal details — age, sex, BMI,
number of children, smoking status and region. This part of the portfolio builds a proof of concept
**linear regression** model that predicts annual charges from those factors, using a public dataset.

## Dataset

- **Source:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (Kaggle)
- **Rows / columns:** 1,338 rows, 7 columns
- **Target:** `charges`
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`

## Repository Contents

| File | Description |
|---|---|
| `Ximiyeto_Makhubele.ipynb` | Main analysis notebook — EDA, feature engineering, model training and evaluation |
| `insurance.csv` | Dataset used to train and test the model |

## How to Run

1. Clone the repository and make sure `insurance.csv` is in the same folder as the notebook.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `Ximiyeto_Makhubele.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run all cells top to bottom.

## Approach

1. **Suitability & data quality** — checked the dataset for missing values, data types and general fit for a regression problem.
2. **EDA** — explored distributions of age, BMI and charges and how charges vary by smoking status, sex and region.
3. **Feature engineering** — encoded categorical fields and added two interaction features (`smoker_age`, `age_bmi`) based on a pattern spotted during EDA: BMI's effect on charges is much larger for smokers than non-smokers.
4. **Model training** — trained two Linear Regression models (80/20 train-test split, `random_state=42`):
   - **Model 1:** original six features + engineered interaction terms
   - **Model 2:** original six features only
5. **Evaluation** — compared both models using MAE, RMSE and R².

## Results

| Metric | Model 1 (engineered features) | Model 2 (original features) |
|---|---|---|
| MAE | $4,199.12 | $4,181.19 |
| RMSE | $5,817.88 | $5,796.28 |
| R² | 0.7820 | 0.7836 |

**Model 2 (the simpler model) performed marginally better** despite Model 1's engineered features
correlating more strongly with charges individually, most likely due to multicollinearity between
the interaction terms and the base features they were built from. Model 2 is the recommended model:
it's slightly more accurate and easier to interpret.

## Limitations & Next Steps

- A likely duplicate record (spotted via `df.nunique()`) was not removed.
- Feature selection was based on correlation strength rather than formal p-value/VIF testing.
- Residual diagnostics (residuals vs. fitted values) were not yet plotted.
