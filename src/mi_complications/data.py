"""Load the documented UCI dataset and select predictors by observation time.

Column order and observation windows follow UCI dataset 579:
https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "MI.data"

# Explicit names prevent IDs or outcomes from entering a model by position.
ALL_COLUMNS = tuple("""
ID AGE SEX INF_ANAM STENOK_AN FK_STENOK IBS_POST IBS_NASL GB SIM_GIPERT
DLIT_AG ZSN_A nr11 nr01 nr02 nr03 nr04 nr07 nr08 np01 np04 np05 np07 np08
np09 np10 endocr_01 endocr_02 endocr_03 zab_leg_01 zab_leg_02 zab_leg_03
zab_leg_04 zab_leg_06 S_AD_KBRIG D_AD_KBRIG S_AD_ORIT D_AD_ORIT O_L_POST
K_SH_POST MP_TP_POST SVT_POST GT_POST FIB_G_POST ant_im lat_im inf_im post_im
IM_PG_P ritm_ecg_p_01 ritm_ecg_p_02 ritm_ecg_p_04 ritm_ecg_p_06 ritm_ecg_p_07
ritm_ecg_p_08 n_r_ecg_p_01 n_r_ecg_p_02 n_r_ecg_p_03 n_r_ecg_p_04
n_r_ecg_p_05 n_r_ecg_p_06 n_r_ecg_p_08 n_r_ecg_p_09 n_r_ecg_p_10
n_p_ecg_p_01 n_p_ecg_p_03 n_p_ecg_p_04 n_p_ecg_p_05 n_p_ecg_p_06
n_p_ecg_p_07 n_p_ecg_p_08 n_p_ecg_p_09 n_p_ecg_p_10 n_p_ecg_p_11
n_p_ecg_p_12 fibr_ter_01 fibr_ter_02 fibr_ter_03 fibr_ter_05 fibr_ter_06
fibr_ter_07 fibr_ter_08 GIPO_K K_BLOOD GIPER_Na Na_BLOOD ALT_BLOOD AST_BLOOD
KFK_BLOOD L_BLOOD ROE TIME_B_S R_AB_1_n R_AB_2_n R_AB_3_n NA_KB NOT_NA_KB
LID_KB NITR_S NA_R_1_n NA_R_2_n NA_R_3_n NOT_NA_1_n NOT_NA_2_n NOT_NA_3_n
LID_S_n B_BLOK_S_n ANT_CA_S_n GEPAR_S_n ASP_S_n TIKL_S_n TRENT_S_n
FIBR_PREDS PREDS_TAH JELUD_TAH FIBR_JELUD A_V_BLOK OTEK_LANC RAZRIV
DRESSLER ZSN REC_IM P_IM_STEN LET_IS
""".split())
FEATURE_COLUMNS = ALL_COLUMNS[1:112]
TARGET_COLUMNS = ALL_COLUMNS[112:]
POST_ADMISSION_COLUMNS = (
    "R_AB_1_n", "R_AB_2_n", "R_AB_3_n",
    "NA_R_1_n", "NA_R_2_n", "NA_R_3_n",
    "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
)


def feature_columns(scenario: str = "admission") -> list[str]:
    """Return UCI predictors for admission or 72 hours, excluding ID/outcomes.

    These are the repository's documented observation windows, not evidence
    that every complication occurred after the chosen prediction time.
    """
    if scenario == "admission":
        return [name for name in FEATURE_COLUMNS if name not in POST_ADMISSION_COLUMNS]
    if scenario == "72h":
        return list(FEATURE_COLUMNS)
    raise ValueError("scenario must be 'admission' or '72h'.")


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the 124-column schema and codes without imputing observations."""
    if set(data.columns) != set(ALL_COLUMNS) or data.shape[1] != len(ALL_COLUMNS):
        raise ValueError("Expected the documented UCI schema: ID, 111 predictors, 12 outcomes.")
    numeric = data.loc[:, list(ALL_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)[
        numeric.notna().to_numpy()
    ]).all():
        raise ValueError("The dataset must contain rows and no infinite values.")
    if numeric["ID"].isna().any() or numeric["ID"].duplicated().any():
        raise ValueError("Record IDs must be present and unique.")
    for name in TARGET_COLUMNS:
        allowed = range(8) if name == "LET_IS" else (0, 1)
        if not numeric[name].isin(allowed).all():
            raise ValueError(f"Missing or invalid outcome codes in {name}.")
    return numeric


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read raw MI.data or a named CSV; download/cache UCI data if no default exists.

    An explicit missing path raises FileNotFoundError. The default is resolved
    relative to this repository, so notebooks do not depend on a Kaggle path.
    '?' remains missing until an analysis-specific preprocessing step.
    """
    data_path = Path(path).expanduser() if path is not None else DEFAULT_DATA_PATH
    if data_path.exists():
        if data_path.suffix.lower() == ".data":
            data = pd.read_csv(data_path, header=None, na_values="?")
            if data.shape[1] != len(ALL_COLUMNS):
                raise ValueError(f"Expected 124 fields per row in {data_path}.")
            data.columns = ALL_COLUMNS
        else:
            data = pd.read_csv(data_path, na_values="?")
        data = validate_data(data)
        data.attrs["source"] = str(data_path.resolve())
        return data
    if path is not None:
        raise FileNotFoundError(f"Dataset does not exist: {data_path}")

    from ucimlrepo import fetch_ucirepo

    try:
        dataset = fetch_ucirepo(id=579)
    except (ConnectionError, OSError, ValueError) as error:
        raise RuntimeError(
            "UCI download failed. Download dataset 579 from the UCI website, "
            f"extract MI.data to {DEFAULT_DATA_PATH}, and rerun."
        ) from error
    data = validate_data(dataset.data.original)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, header=False, index=False, na_rep="?")
    data.attrs["source"] = "UCI Machine Learning Repository, dataset 579"
    return data
