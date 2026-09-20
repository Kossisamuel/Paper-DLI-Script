# Dataset provenance and local cache

The notebooks use the [Myocardial infarction complications dataset (UCI 579)](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications).
Dataset citation: Golovenkin et al. (2020), UCI Machine Learning Repository, [doi:10.24432/C53P5M](https://doi.org/10.24432/C53P5M). The dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); this is not a license declaration for the project code.

`MI.data` is a headerless, comma-separated file with 1,700 rows and 124 columns: an ID, 111 predictors, and 12 outcomes. `?` denotes a missing observation. The column names and admission/72-hour predictor lists are recorded in `src/mi_complications/data.py` using the UCI documentation.

The loader reads `data/MI.data` first. If it is absent, it fetches dataset 579 with `ucimlrepo`, validates the schema, and caches the data here. For offline use, download the archive from UCI, extract `MI.data` into this directory, and run the notebooks from the repository root. The optional downloaded ZIP is a cache, not an analysis input.

The raw local file audited during cleanup has SHA-256:

```text
d3bf6a8cc162f592a7376c7238b8f92b97a529ac6be6f461095b9071ec1647a4
```

The archive's `MI.data` member has the same digest. A CSV serialized by `ucimlrepo` may have different bytes despite equivalent numeric values. The raw file contains 15,974 missing entries; `ZSN` has 1,306 negative and 394 positive observations. `LET_IS` has 1,429 zero codes and 271 positive cause-of-death codes. These are dataset checks, not model performance results.

Earlier notebook versions read a Kaggle CSV at a machine-specific path. That CSV is not present here, and its saved summaries differ in missingness from the raw UCI file. The cleaned notebooks explicitly use the documented UCI data. Historical model scores are therefore not directly comparable with new runs.

Raw data files remain outside Git; this provenance document and the loader are tracked. To use a different, documented export, call `load_data(path)` with a CSV containing the same named columns, and record its source and preprocessing before interpreting results.
