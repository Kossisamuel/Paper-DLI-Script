import ast
import json
from pathlib import Path
from textwrap import dedent

cells = []

def add(kind, source):
    cell = {
        'cell_type': kind,
        'id': f'discussion-{len(cells):02d}',
        'metadata': {},
        'source': dedent(source).strip().splitlines(keepends=True),
    }
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
        ast.parse(''.join(cell['source']))
    cells.append(cell)

add('markdown', '''
# Cluster-specific classification of heart failure complications

This notebook compares support vector machines, random forests, and gradient
boosting for the binary `ZSN` outcome (chronic heart failure) in a K-means subgroup
of the UCI Myocardial Infarction Complications dataset. It retains the original
four-cluster design, selected cluster label `2`, classifier grids, and requested
PCA dimensions. The subgroup is exploratory: cluster numbers are arbitrary labels,
not established clinical phenotypes.

Run the notebook from top to bottom after installing `requirements.txt`. The
shared loader reads `data/MI.data` or obtains the public UCI archive. Set
`DATA_PATH` to a compatible local raw file or CSV to use another copy.

The revised workflow splits patients before fitting any transformation or
resampling. Historical outputs used preprocessing and SMOTE before splitting and
are not valid estimates of held-out performance. They have been cleared; tables
and figures below are calculated from the current run. These results will differ
from the original notebook and do not reproduce its previous cohort.

Dataset documentation: [UCI Myocardial Infarction Complications](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications).
''')
add('markdown', '''
## 1. Imports and configuration

`SMOKE_TEST=True` runs one parameter combination per classifier while keeping all
patients, both scenarios, and the same preprocessing. It checks execution only.
Set it to `False` for the original full grids, which can take substantial time.
Hyperparameters are selected by cross-validation accuracy, as in the original
analysis; the holdout also reports balanced accuracy, precision, recall, and F1.
''')
add('code', '''
import json
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ResamplingPipeline
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = next(
    (
        candidate
        for candidate in [Path.cwd(), *Path.cwd().parents]
        if (candidate / 'src' / 'mi_complications' / 'data.py').is_file()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Open this notebook inside the project directory.')
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from mi_complications.data import feature_columns, load_data

RANDOM_STATE = 42
TEST_SIZE = 0.30
N_CLUSTERS = 4
SELECTED_CLUSTER = 2
CV_FOLDS = 3
SMOKE_TEST = False
N_JOBS = -1
DATA_PATH = None
RESULTS_DIR = PROJECT_ROOT / 'results' / 'discussion'
FIGURES_DIR = PROJECT_ROOT / 'figures' / 'discussion'

plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
''')
add('markdown', '''
## 2. Data and feature availability

The patient identifier and all complication outcomes are excluded from predictors.
The `admission` feature set follows the UCI convention excluding nine repeated
measurements/treatments (raw column positions 93-95 and 100-105, numbered from 1).
The `72h` set includes all 111 input variables. These are dataset-defined feature
sets: the data do not establish a prospective 72-hour landmark population or the
exact onset time of `ZSN`. The comparison therefore does not demonstrate future
risk prediction or treatment effects.
''')
add('code', '''
data = load_data(DATA_PATH)
admission_columns = feature_columns('admission')
scenario_columns = {
    'admission': admission_columns,
    '72h': feature_columns('72h'),
}
target = data['ZSN'].astype(int)
if set(target.unique()) != {0, 1}:
    raise ValueError('ZSN must contain both binary outcome classes, 0 and 1.')

print(f'Patients: {len(data):,}; ZSN-positive cases: {int(target.sum()):,}')
display(pd.DataFrame({
    'Scenario': scenario_columns.keys(),
    'Number of predictors': [len(columns) for columns in scenario_columns.values()],
}))
''')
add('markdown', '''
## 3. Split patients and define the subgroup

A stratified 70/30 split is made before imputation, scaling, or clustering. K-means
and its preprocessing are fitted on admission inputs from the training patients;
held-out patients receive labels from that fitted model. Both feature scenarios
use exactly the same patients. Cluster `2` is retained as a prespecified analysis
setting and is never chosen using held-out performance.

Cross-validation tunes the classifier within this fixed training-defined subgroup.
It does not re-estimate subgroup discovery inside each fold; its scores are
conditional on that subgroup and are used only for tuning. The final test patients
are excluded from both subgroup discovery and classifier fitting.
''')
add('code', '''
train_indices, test_indices = train_test_split(
    data.index,
    test_size=TEST_SIZE,
    stratify=target,
    random_state=RANDOM_STATE,
)
cluster_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
    ('scaler', StandardScaler()),
    ('cluster', KMeans(
        n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE,
    )),
])
train_clusters = pd.Series(
    cluster_pipeline.fit_predict(data.loc[train_indices, admission_columns]),
    index=train_indices,
    name='Cluster',
)
test_clusters = pd.Series(
    cluster_pipeline.predict(data.loc[test_indices, admission_columns]),
    index=test_indices,
    name='Cluster',
)
cluster_counts = pd.DataFrame({
    'Training patients': train_clusters.value_counts(),
    'Test patients': test_clusters.value_counts(),
}).reindex(range(N_CLUSTERS), fill_value=0).fillna(0).astype(int)
display(cluster_counts.rename_axis('Cluster'))

subgroup_train_indices = train_clusters.index[train_clusters == SELECTED_CLUSTER]
subgroup_test_indices = test_clusters.index[test_clusters == SELECTED_CLUSTER]
if len(subgroup_train_indices) == 0 or len(subgroup_test_indices) == 0:
    raise ValueError('The selected cluster must include training and test patients.')
y_train = target.loc[subgroup_train_indices]
y_test = target.loc[subgroup_test_indices]
class_counts = pd.DataFrame({
    'Training patients': y_train.value_counts(),
    'Test patients': y_test.value_counts(),
}).reindex([0, 1], fill_value=0).fillna(0).astype(int)
display(class_counts.rename_axis('ZSN'))
if (class_counts['Training patients'] < CV_FOLDS).any():
    raise ValueError('The selected subgroup has too few cases for stratified CV.')
if (class_counts['Test patients'] == 0).any():
    raise ValueError('Both ZSN classes are required for subgroup holdout evaluation.')
''')
add('markdown', '''
## 4. Reusable model selection and evaluation

Every cross-validation fit contains mean imputation, standardization, SMOTE, PCA,
and the classifier. The test set retains its original class distribution. The
SMOTE neighbor count is reduced only when the smallest training fold cannot
support the original five neighbors; the actual value is recorded.

The original PCA settings are 30 components at admission and 70 components for
72-hour SVM/random forest models, with 90 for gradient boosting. These different
PCA dimensions are retained, so a difference between scenarios cannot be
attributed solely to the additional variables. Invalid dimensions raise a clear
error instead of silently changing the experiment.

Mean imputation, Euclidean K-means, PCA, and ordinary SMOTE treat coded categories
as numeric. This preserves the original modeling design but can create fractional
category values and requires sensitivity analysis with categorical-aware methods.
''')
add('code', '''
def model_specifications(smoke_test=False):
    """Return the original classifiers, parameter grids, and PCA settings."""
    specifications = {
        'SVM': {
            'estimator': SVC(random_state=RANDOM_STATE),
            'grid': {
                'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001],
                'kernel': ['rbf', 'poly', 'sigmoid'],
            },
            'components': {'admission': 30, '72h': 70},
        },
        'Random forest': {
            'estimator': RandomForestClassifier(random_state=RANDOM_STATE),
            'grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'components': {'admission': 30, '72h': 70},
        },
        'Gradient boosting': {
            'estimator': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'components': {'admission': 30, '72h': 90},
        },
    }
    if smoke_test:
        for specification in specifications.values():
            specification['grid'] = {
                name: values[:1] for name, values in specification['grid'].items()
            }
    return specifications


def fit_and_evaluate(model_name, scenario, specification):
    """Tune a training-only pipeline and evaluate untouched subgroup patients."""
    columns = scenario_columns[scenario]
    x_train = data.loc[subgroup_train_indices, columns]
    x_test = data.loc[subgroup_test_indices, columns]
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    folds = list(cross_validation.split(x_train, y_train))
    fold_counts = [
        y_train.iloc[fold_train].value_counts().reindex([0, 1], fill_value=0)
        for fold_train, _ in folds
    ]
    smallest_class = min(int(counts.min()) for counts in fold_counts)
    if smallest_class < 2:
        raise ValueError('SMOTE requires at least two minority cases in each fold.')
    neighbors = min(5, smallest_class - 1)
    components = specification['components'][scenario]
    smallest_resampled_fold = min(2 * int(counts.max()) for counts in fold_counts)
    if components > min(len(columns), smallest_resampled_fold):
        raise ValueError(
            f'{model_name}, {scenario}: {components} PCA components exceed the '
            'available training-fold dimensions. Review the subgroup and PCA '
            'configuration explicitly before continuing.'
        )

    pipeline = ResamplingPipeline([
        ('imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=neighbors)),
        ('pca', PCA(n_components=components, random_state=RANDOM_STATE)),
        ('classifier', specification['estimator']),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={
            f'classifier__{name}': values
            for name, values in specification['grid'].items()
        },
        scoring='accuracy',
        cv=folds,
        n_jobs=N_JOBS,
        refit=True,
        error_score='raise',
    )
    search.fit(x_train, y_train)
    predictions = search.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='macro', zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        'Scenario': scenario,
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Balanced accuracy': balanced_accuracy_score(y_test, predictions),
        'Macro precision': precision,
        'Macro recall': recall,
        'Macro F1': f1,
        'CV accuracy': search.best_score_,
        'Training patients': len(y_train),
        'Test patients': len(y_test),
        'PCA components': components,
        'SMOTE neighbors': neighbors,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'Smoke test': SMOKE_TEST,
    }
    return {
        'search': search,
        'metrics': metrics,
        'confusion_matrix': matrix,
        'report': classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=['ZSN absent', 'ZSN present'], zero_division=0,
        ),
    }
''')
add('code', '''
model_results = {}
for scenario in scenario_columns:
    for model_name, specification in model_specifications(SMOKE_TEST).items():
        print(f'Fitting {model_name}: {scenario}')
        model_results[(scenario, model_name)] = fit_and_evaluate(
            model_name, scenario, specification,
        )
metrics_table = pd.DataFrame(
    result['metrics'] for result in model_results.values()
)
display(metrics_table.round(4))
for (scenario, model_name), result in model_results.items():
    print(f'\\n{model_name} / {scenario}')
    print('Best parameters:', result['search'].best_params_)
    print(result['report'])
''')
add('markdown', '''
## 5. Holdout results and PCA component interpretation

Confusion matrices use the same test patients for all six models and scenarios.
Gradient boosting importances refer to PCA components, each of which mixes many
original variables. Labeling these values with the first original feature names
would be incorrect; they are labeled `PC1`, `PC2`, and so on below. Impurity-based
importance is a descriptive model quantity and does not establish causality.
''')
add('code', '''
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
run_label = 'smoke' if SMOKE_TEST else 'full'

fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
for axis, ((scenario, model_name), result) in zip(axes.flat, model_results.items()):
    ConfusionMatrixDisplay(
        result['confusion_matrix'], display_labels=['ZSN absent', 'ZSN present'],
    ).plot(ax=axis, cmap='Blues', colorbar=False, values_format='d')
    axis.set_title(f'{model_name}: {scenario}')
fig.suptitle(f'Cluster {SELECTED_CLUSTER}: held-out patient classification')
fig.savefig(FIGURES_DIR / f'confusion_matrices_{run_label}.png', bbox_inches='tight')
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
for axis, scenario in zip(axes, scenario_columns):
    fitted_pipeline = model_results[(scenario, 'Gradient boosting')][
        'search'
    ].best_estimator_
    importances = fitted_pipeline.named_steps['classifier'].feature_importances_
    component_importances = pd.Series(
        importances, index=[f'PC{i + 1}' for i in range(len(importances))],
    ).nlargest(10).sort_values()
    component_importances.plot.barh(ax=axis, color='#287D8E')
    axis.set_title(f'Gradient boosting: {scenario}')
    axis.set_xlabel('Impurity-based component importance')
    axis.set_ylabel('Principal component')
fig.savefig(FIGURES_DIR / f'component_importances_{run_label}.png', bbox_inches='tight')
plt.show()
plt.close(fig)
''')
add('code', '''
metrics_path = RESULTS_DIR / f'classification_metrics_{run_label}.csv'
metrics_table.to_csv(metrics_path, index=False)
run_metadata = {
    'random_state': RANDOM_STATE,
    'test_size': TEST_SIZE,
    'n_clusters': N_CLUSTERS,
    'selected_cluster': SELECTED_CLUSTER,
    'cv_folds': CV_FOLDS,
    'smoke_test': SMOKE_TEST,
    'data_path': str(DATA_PATH) if DATA_PATH is not None else 'shared loader default',
    'scenario_columns': scenario_columns,
    'training_patient_ids': data.loc[subgroup_train_indices, 'ID'].astype(int).tolist(),
    'test_patient_ids': data.loc[subgroup_test_indices, 'ID'].astype(int).tolist(),
    'best_parameters': {
        f'{scenario}/{model_name}': result['search'].best_params_
        for (scenario, model_name), result in model_results.items()
    },
    'package_versions': {
        package: version(package)
        for package in ['numpy', 'pandas', 'scikit-learn', 'imbalanced-learn']
    },
}
metadata_path = RESULTS_DIR / f'run_metadata_{run_label}.json'
metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding='utf-8')
print(f'Results saved to {metrics_path.relative_to(PROJECT_ROOT)}')
print(f'Run configuration saved to {metadata_path.relative_to(PROJECT_ROOT)}')
''')
add('markdown', '''
## 6. Survival analysis requires observed follow-up

Earlier cells fitted Cox models and Kaplan-Meier curves using randomly generated
durations, hospital-admission delay (`TIME_B_S`), or differences between symptom
counts. None is observed follow-up from a defined baseline to death or censoring.
Those analyses and their generated figures have been removed from the executable
workflow; their original source remains in Git history.

`ZSN` denotes chronic heart failure, not death. `LET_IS` records discharge outcome
with zero indicating survival and nonzero categories denoting causes of death.
A binary mortality indicator can be derived from `LET_IS`, but it cannot supply
the missing event times. No hazard ratio, survival probability, or survival-test
p-value is reported here. A valid survival study would require observed event or
censoring times, a clearly defined time origin, and covariates available at that
origin.

The classification analysis remains exploratory. Cluster numbering, a single
random train/test split, numeric treatment of categorical predictors, and
uncertain outcome timing limit interpretation. Independent evaluation and
sensitivity analyses are needed before drawing clinical conclusions.
''')
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
Path('discussiondli.ipynb').write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8'
)
print(f'Wrote {len(cells)} cells; all code parses successfully.')
