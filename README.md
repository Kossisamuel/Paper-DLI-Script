# Myocardial Infarction Complications

Reproducible machine-learning analysis of the [UCI Myocardial Infarction Complications dataset](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications), prepared as a research portfolio for doctoral work in clinical data science.

## Research question

Can routinely collected information at hospital admission, and additional information available after 72 hours, improve prediction of myocardial infarction complications, including chronic heart failure (`ZSN`)?

## Dataset

- **Source:** UCI Machine Learning Repository, dataset 579
- **Size:** 1,700 observations, 111 predictors, 12 outcomes
- **Missing values:** present in the predictors and handled explicitly in the analysis
- **License:** CC BY 4.0
- **Citation:** Golovenkin, S. et al. (2020), *Myocardial infarction complications*. UCI Machine Learning Repository. DOI: [10.24432/C53P5M](https://doi.org/10.24432/C53P5M)

The raw dataset is not committed to this repository. The notebook downloads it through `ucimlrepo` and falls back to `data/MI.data` when available.

## Repository contents

- `discussiondliV1.ipynb`: main exploratory and predictive analysis
- `discussiondli (2).ipynb`: earlier working version
- `posterdli V1.ipynb`: poster-oriented analysis
- `data/`: local dataset cache, excluded from version control when appropriate

## Reproduce the notebook

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy scikit-learn imbalanced-learn seaborn matplotlib lifelines ucimlrepo
jupyter lab
```

Open `discussiondliV1.ipynb` and run the cells from top to bottom. The first cell downloads dataset 579 through the official UCI Python client and reports the observed shape and missingness before modelling.

## Methodological safeguards to complete before publication

The original notebook contains exploratory result tables and several historical modelling blocks. Before using any metric in a paper, rerun the study with:

1. A train/test split performed before imputation, scaling, PCA, or SMOTE.
2. An `imblearn` pipeline so resampling occurs inside cross-validation only.
3. Stratified repeated cross-validation and confidence intervals.
4. Outcome-specific metrics: ROC-AUC, PR-AUC, sensitivity, specificity, F1, calibration, and confusion matrices.
5. No synthetic or random survival duration. Survival analysis must use a documented follow-up time and event definition.
6. External or temporal validation before clinical interpretation.

These safeguards are essential because this is a small, imbalanced clinical dataset and the models are not clinically validated.

## Status

The project is an active research prototype. Results should be treated as exploratory until the leakage-controlled evaluation and external validation are completed.
