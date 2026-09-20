"""Regression checks for dataset parsing and predictor boundaries."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mi_complications import data as dataset


class DataTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(0.0, index=range(3), columns=dataset.ALL_COLUMNS)
        self.frame["ID"] = [1, 2, 3]
        self.frame.loc[1, "AGE"] = np.nan
        self.frame["ZSN"] = [0, 1, 0]
        self.frame["LET_IS"] = [0, 1, 7]

    def test_time_windows_exclude_identifiers_and_outcomes(self):
        admission = dataset.feature_columns("admission")
        later = dataset.feature_columns("72h")
        self.assertEqual(len(admission), 102)
        self.assertEqual(len(later), 111)
        self.assertTrue(set(admission) < set(later))
        excluded_positions = [93, 94, 95, 100, 101, 102, 103, 104, 105]
        expected = {dataset.ALL_COLUMNS[position - 1] for position in excluded_positions}
        self.assertEqual(set(later) - set(admission), expected)
        self.assertTrue(set(later).isdisjoint({"ID", *dataset.TARGET_COLUMNS}))
        with self.assertRaises(ValueError):
            dataset.feature_columns("24h")

    def test_raw_and_named_csv_preserve_missingness_and_outcome_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, header in [("MI.data", False), ("export.csv", True)]:
                path = Path(directory) / name
                self.frame.to_csv(path, header=header, index=False, na_rep="?")
                loaded = dataset.load_data(path)
                pd.testing.assert_frame_equal(loaded, self.frame, check_dtype=False)
                self.assertEqual(loaded.isna().sum().sum(), 1)

    def test_invalid_inputs_are_rejected(self):
        for invalid in [
            self.frame.drop(columns="ZSN"),
            self.frame.assign(ID=[1, 1, 3]),
            self.frame.assign(ZSN=[0, 2, 1]),
            self.frame.assign(LET_IS=[0, 8, 1]),
            self.frame.assign(LET_IS=[0, np.nan, 1]),
            self.frame.assign(AGE=np.inf),
        ]:
            with self.subTest(columns=invalid.shape):
                with self.assertRaises(ValueError):
                    dataset.validate_data(invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.data"
            with self.assertRaises(FileNotFoundError):
                dataset.load_data(path)
            path.write_text("1,2,3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                dataset.load_data(path)

    def test_download_is_validated_and_cached_for_offline_reuse(self):
        from types import SimpleNamespace

        remote = SimpleNamespace(data=SimpleNamespace(original=self.frame))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "MI.data"
            with patch.object(dataset, "DEFAULT_DATA_PATH", cache):
                with patch("ucimlrepo.fetch_ucirepo", return_value=remote) as fetch:
                    downloaded = dataset.load_data()
                    cached = dataset.load_data()
                    fetch.assert_called_once_with(id=579)
                    pd.testing.assert_frame_equal(downloaded, cached, check_dtype=False)

    @unittest.skipUnless(dataset.DEFAULT_DATA_PATH.exists(), "Local UCI cache unavailable")
    def test_available_uci_data_matches_documented_counts(self):
        frame = dataset.load_data()
        self.assertEqual(frame.shape, (1700, 124))
        self.assertEqual(int(frame.isna().sum().sum()), 15974)
        self.assertEqual(frame["ZSN"].value_counts().to_dict(), {0: 1306, 1: 394})
        self.assertEqual(int(frame["LET_IS"].gt(0).sum()), 271)


if __name__ == "__main__":
    unittest.main()
