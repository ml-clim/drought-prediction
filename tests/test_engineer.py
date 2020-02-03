import pytest
import pickle
import numpy as np
import xarray as xr
import datetime as dt

from src.engineer import Engineer

from .utils import _make_dataset


def _setup(data_path, add_times=True, static=False):
    # setup
    interim_folder = data_path / "interim"
    interim_folder.mkdir()

    if static:
        interim_folder = interim_folder / "static"
        interim_folder.mkdir()

    expected_output, expected_vars = [], []
    for var in ["a", "b"]:
        (interim_folder / f"{var}_preprocessed").mkdir(exist_ok=True, parents=True)

        # this file should be captured
        data, _, _ = _make_dataset((10, 10), var, const=True, add_times=add_times)
        filename = interim_folder / f"{var}_preprocessed/hello.nc"
        data.to_netcdf(filename)

        expected_output.append(filename)
        expected_vars.append(var)

        # this file should not
        (interim_folder / f"{var}_preprocessed/boo").touch()

    # none of this should be captured
    (interim_folder / "woops").mkdir()
    woops_data, _, _ = _make_dataset((10, 10), "oops")
    woops_data.to_netcdf(interim_folder / "woops/hi.nc")

    return expected_output, expected_vars


class TestEngineer:
    def test_get_preprocessed(self, tmp_path, monkeypatch):

        expected_files, expected_vars = _setup(tmp_path)

        def mock_init(self, data_folder):
            self.name = "dummy"
            self.interim_folder = data_folder / "interim"

        monkeypatch.setattr(Engineer, "__init__", mock_init)

        engineer = Engineer(tmp_path)
        files = engineer._get_preprocessed_files(static=False)

        assert set(expected_files) == set(files), f"Did not retrieve expected files!"

    def test_join(self, tmp_path, monkeypatch):

        expected_files, expected_vars = _setup(tmp_path)

        def mock_init(self, data_folder):
            self.name = "dummy"
            self.interim_folder = data_folder / "interim"

        monkeypatch.setattr(Engineer, "__init__", mock_init)

        engineer = Engineer(tmp_path)
        joined_ds = engineer._make_dataset(static=False)

        dims = ["lon", "lat", "time"]
        output_vars = [var for var in joined_ds.variables if var not in dims]

        assert set(output_vars) == set(
            expected_vars
        ), f"Did not retrieve all the expected variables!"

    def test_init(self, tmp_path):

        with pytest.raises(AssertionError) as e:
            Engineer(tmp_path)
            assert "does not exist. Has the preprocesser been run?" in str(e)

        (tmp_path / "interim").mkdir()

        Engineer(tmp_path)

        assert (tmp_path / "features").exists(), "Features directory not made!"
        assert (
            tmp_path / "features" / "one_month_forecast"
        ).exists(), "\
        one_month_forecast directory not made!"

    def test_static(self, tmp_path):
        _, expected_vars = _setup(tmp_path, add_times=False, static=True)
        engineer = Engineer(tmp_path, process_static=True)

        assert (
            tmp_path / "features/static"
        ).exists(), "Static output folder does not exist!"

        engineer.process_static()

        output_file = tmp_path / "features/static/data.nc"
        assert output_file.exists(), "Static output folder does not exist!"
        static_data = xr.open_dataset(output_file)

        for var in expected_vars:
            assert var in static_data.data_vars

    def test_yearsplit(self, tmp_path):

        _setup(tmp_path)

        dataset, _, _ = _make_dataset(size=(2, 2))

        engineer = Engineer(tmp_path)
        train = engineer._train_test_split(
            dataset,
            years=[2001],
            target_variable="VHI",
            pred_months=11,
            expected_length=11,
        )

        assert (
            train.time.values < np.datetime64("2001-01-01")
        ).all(), "Got years greater than the test year in the training set!"

    def test_engineer(self, tmp_path):

        _setup(tmp_path)

        pred_months = expected_length = 11

        engineer = Engineer(tmp_path)
        engineer.engineer(
            test_year=2001,
            target_variable="a",
            pred_months=pred_months,
            expected_length=expected_length,
        )

        def check_folder(folder_path):
            y = xr.open_dataset(folder_path / "y.nc")
            assert "b" not in set(y.variables), "Got unexpected variables in test set"

            x = xr.open_dataset(folder_path / "x.nc")
            for expected_var in {"a", "b"}:
                assert expected_var in set(
                    x.variables
                ), "Missing variables in testing input dataset"
            assert (
                len(x.time.values) == expected_length
            ), "Wrong number of months in the test x dataset"
            assert len(y.time.values) == 1, "Wrong number of months in test y dataset"

        # check_folder(tmp_path / 'features/one_month_forecast/train/1999_12')
        for month in range(1, 13):
            check_folder(tmp_path / f"features/one_month_forecast/test/2001_{month}")
            check_folder(tmp_path / f"features/one_month_forecast/train/2000_{month}")

        assert (
            len(list((tmp_path / "features/one_month_forecast/train").glob("2001_*")))
            == 0
        ), "Test data in the training data!"

        assert (
            tmp_path / "features/one_month_forecast/normalizing_dict.pkl"
        ).exists(), f"Normalizing dict not saved!"
        with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
            "rb"
        ) as f:
            norm_dict = pickle.load(f)

        for key, val in norm_dict.items():
            assert key in {"a", "b"}, f"Unexpected key!"
            assert norm_dict[key]["mean"] == 1, f"Mean incorrectly calculated!"
            assert norm_dict[key]["std"] == 0, f"Std incorrectly calculated!"

    def test_stratify(self, tmp_path):
        _setup(tmp_path)
        engineer = Engineer(tmp_path)
        ds_target, _, _ = _make_dataset(size=(20, 20))
        ds_predictor, _, _ = _make_dataset(size=(20, 20))
        ds_predictor = ds_predictor.rename({"VHI": "predictor"})
        ds = ds_predictor.merge(ds_target)

        xy_dict, max_train_date = engineer._stratify_xy(
            ds=ds,
            year=2001,
            target_variable="VHI",
            target_month=1,
            pred_months=4,
            expected_length=4,
        )

        assert (
            xy_dict["x"].time.size == 4
        ), f'OneMonthForecast experiment `x`\
        should have 4 times Got: {xy_dict["x"].time.size}'

        assert (
            max_train_date == dt.datetime(2000, 12, 31).date()
        ), f"\
        the max_train_date should be one month before the `target_month`,\
        `year`"
