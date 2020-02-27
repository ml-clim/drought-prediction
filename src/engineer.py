import numpy as np
import calendar
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
import pickle
import xarray as xr
import warnings

from .utils import minus_months

from typing import cast, DefaultDict, Dict, List, Optional, Union, Tuple


class Engineer:
    r"""Takes the output of the processors, and turns it into `NetCDF` files
    ready to be input into the machine learning models.

    Parameters:
    ~~~~~~~~~~~
        ``data_folder (pathlib.Path, optional)``: The location of the data folder.
            Default: ``pathlib.Path("data")``
        ``process_static (bool, optional)``: Defines whether or not to process the static data.
            Default: ``False``
    """

    name: str = "one_month_forecast"

    def __init__(
        self, data_folder: Path = Path("data"), process_static: bool = False
    ) -> None:

        self.data_folder = data_folder
        self._process_static = process_static

        self.interim_folder = data_folder / "interim"
        assert (
            self.interim_folder.exists()
        ), f'{data_folder / "interim"} does not exist. Has the preprocesser been run?'

        try:
            # specific folder for that
            self.output_folder = data_folder / "features" / self.name
            if not self.output_folder.exists():
                self.output_folder.mkdir(parents=True)
        except AttributeError:
            print("Name not defined! No experiment folder set up")

        if self._process_static:
            self.static_output_folder = data_folder / "features/static"
            if not self.static_output_folder.exists():
                self.static_output_folder.mkdir(parents=True)

    def engineer(
        self,
        test_year: Union[int, List[int]],
        target_variable: str = "VHI",
        pred_months: int = 12,
        expected_length: Optional[int] = 12,
    ) -> None:
        r"""Runs the engineer.

        Arguments:
        ~~~~~~~~~~
            ``test_year (int, List[int])``: Years of data to use as test data. Only data from before
                ``min(test_year)`` will be used for training data.
            ``target_variable (str, optional)``: The target variable. Must be in one of the processed
                files. Default: ``"VHI"``.
            ``pred_months (int, optional)``: The number of months to use as input to the model.
                Default: ``12`` (a year's worth of data).
            ``expected_length (int, optional)``: The expected length of the output sequence (e.g. if
                the data was processed to weekly timesteps, then we might expect
                ``expected_length = 4 * pred_months``). If not ``None``, any sequence which does not have this
                length (e.g. due to missing data) will be skipped.
                Default: ``12``.
        """

        self.process_dynamic(test_year, target_variable, pred_months, expected_length)
        if self._process_static:
            self.process_static()

    def process_static(self):

        # this function assumes the static data has only two dimensions,
        # lat and lon

        output_file = self.static_output_folder / "data.nc"
        if output_file.exists():
            warnings.warn("A static data file already exists!")
            return None

        # here, we overwrite the dims because topography (a static variable)
        # uses CDO for regridding, which yields very slightly different
        # coordinates (it seems from rounding)
        static_ds = self._make_dataset(static=True, overwrite_dims=True)

        normalization_values: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        for var in static_ds.data_vars:
            if var.endswith("one_hot"):
                mean = 0
                std = 1
            else:
                mean = float(
                    static_ds[var].mean(dim=["lat", "lon"], skipna=True).values
                )
                std = float(static_ds[var].std(dim=["lat", "lon"], skipna=True).values)

            normalization_values[var]["mean"] = mean
            normalization_values[var]["std"] = std

        static_ds.to_netcdf(self.static_output_folder / "data.nc")
        savepath = self.static_output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_values, f)

    def process_dynamic(
        self,
        test_year: Union[int, List[int]],
        target_variable: str = "VHI",
        pred_months: int = 12,
        expected_length: Optional[int] = 12,
    ) -> None:
        if expected_length is None:
            warnings.warn(
                "** `expected_length` is None. This means that \
            missing data will not be skipped. Are you sure? **"
            )

        # read in all the data from interim/{var}_preprocessed
        data = self._make_dataset(static=False)

        # ensure test_year is List[int]
        if type(test_year) is int:
            test_year = [cast(int, test_year)]

        # save test data (x, y) and return the train_ds (subset of `data`)
        train_ds = self._train_test_split(
            ds=data,
            years=cast(List, test_year),
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        normalization_values = self._calculate_normalization_values(train_ds)

        # split train_ds into x, y for each year-month before `test_year` & save
        self._stratify_training_data(
            train_ds=train_ds,
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        savepath = self.output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_values, f)

    def _get_preprocessed_files(self, static: bool) -> List[Path]:
        processed_files = []
        if static:
            interim_folder = self.interim_folder / "static"
        else:
            interim_folder = self.interim_folder
        for subfolder in interim_folder.iterdir():
            if str(subfolder).endswith("_preprocessed") and subfolder.is_dir():
                processed_files.extend(list(subfolder.glob("*.nc")))
        return processed_files

    def _make_dataset(self, static: bool, overwrite_dims: bool = False) -> xr.Dataset:

        datasets = []
        dims = ["lon", "lat"]
        coords = {}
        for idx, file in enumerate(self._get_preprocessed_files(static)):
            print(f"Processing {file}")
            datasets.append(xr.open_dataset(file))

            if idx == 0:
                for dim in dims:
                    coords[dim] = datasets[idx][dim].values
            else:
                for dim in dims:
                    array_equal = np.array_equal(datasets[idx][dim].values, coords[dim])
                    if (not overwrite_dims) and (not array_equal):
                        # SORT the values first (xarray clever enough to figure out joining)
                        assert np.array_equal(
                            np.sort(datasets[idx][dim].values), np.sort(coords[dim])
                        ), f"{dim} is different! Was this run using the preprocessor?"
                    elif overwrite_dims and (not array_equal):
                        assert len(datasets[idx][dim].values) == len(coords[dim])
                        datasets[idx][dim] = coords[dim]

        # join all preprocessed datasets
        main_dataset = datasets[0]
        for dataset in datasets[1:]:
            # ensure equal timesteps ('inner' join)
            main_dataset = main_dataset.merge(dataset, join="inner")

        return main_dataset

    def _stratify_training_data(
        self,
        train_ds: xr.Dataset,
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
    ) -> None:
        """split `train_ds` into x, y and save the outputs to
        self.output_folder (data/features) """

        min_date = self._get_datetime(train_ds.time.values.min())
        max_date = self._get_datetime(train_ds.time.values.max())

        cur_pred_year, cur_pred_month = max_date.year, max_date.month

        # for every month-year create & save the x, y datasets for training
        cur_min_date = max_date
        while cur_min_date >= min_date:
            # each iteration count down one month (02 -> 01 -> 12 ...)
            arrays, cur_min_date = self._stratify_xy(
                ds=train_ds,
                year=cur_pred_year,
                target_variable=target_variable,
                target_month=cur_pred_month,
                pred_months=pred_months,
                expected_length=expected_length,
            )
            if arrays is not None:
                self._save(
                    arrays,
                    year=cur_pred_year,
                    month=cur_pred_month,
                    dataset_type="train",
                )
            cur_pred_year, cur_pred_month = cur_min_date.year, cur_min_date.month

    def _train_test_split(
        self,
        ds: xr.Dataset,
        years: List[int],
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
    ) -> xr.Dataset:
        """save the test data and return the training dataset"""

        years.sort()

        # for the first `year` Jan calculate the xy_test dictionary and min date
        xy_test, min_test_date = self._stratify_xy(
            ds=ds,
            year=years[0],
            target_variable=target_variable,
            target_month=1,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        # the train_ds MUST BE from before minimum test date
        train_dates = ds.time.values <= np.datetime64(str(min_test_date))
        train_ds = ds.isel(time=train_dates)

        # save the xy_test dictionary
        if xy_test is not None:
            self._save(xy_test, year=years[0], month=1, dataset_type="test")

        # each month in test_year produce an x,y pair for testing
        for year in years:
            for month in range(1, 13):
                if year > years[0] or month > 1:
                    # prevents the initial test set from being recalculated
                    xy_test, _ = self._stratify_xy(
                        ds=ds,
                        year=year,
                        target_variable=target_variable,
                        target_month=month,
                        pred_months=pred_months,
                        expected_length=expected_length,
                    )
                    if xy_test is not None:
                        self._save(xy_test, year=year, month=month, dataset_type="test")
        return train_ds

    @staticmethod
    def _get_datetime(time: np.datetime64) -> date:
        return datetime.strptime(time.astype(str)[:10], "%Y-%m-%d").date()

    def _save(
        self, ds_dict: Dict[str, xr.Dataset], year: int, month: int, dataset_type: str
    ) -> None:

        save_folder = self.output_folder / dataset_type
        save_folder.mkdir(exist_ok=True)

        output_location = save_folder / f"{year}_{month}"
        output_location.mkdir(exist_ok=True)

        for x_or_y, output_ds in ds_dict.items():
            print(f"Saving data to {output_location.as_posix()}/{x_or_y}.nc")
            output_ds.to_netcdf(output_location / f"{x_or_y}.nc")

    def _calculate_normalization_values(
        self, x_data: xr.Dataset
    ) -> DefaultDict[str, Dict[str, float]]:
        normalization_values: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        for var in x_data.data_vars:
            mean = float(
                x_data[var].mean(dim=["lat", "lon", "time"], skipna=True).values
            )
            std = float(x_data[var].std(dim=["lat", "lon", "time"], skipna=True).values)
            normalization_values[var]["mean"] = mean
            normalization_values[var]["std"] = std

        return normalization_values

    @staticmethod
    def _make_fill_value_dataset(
        ds: Union[xr.Dataset, xr.DataArray], fill_value: Union[int, float] = -9999.0
    ) -> Union[xr.Dataset, xr.DataArray]:
        nan_ds = xr.full_like(ds, fill_value)
        return nan_ds

    @staticmethod
    def _stratify_xy(
        ds: xr.Dataset,
        year: int,
        target_variable: str,
        target_month: int,
        pred_months: int = 11,
        expected_length: Optional[int] = 11,
    ) -> Tuple[Optional[Dict[str, xr.Dataset]], date]:
        """
        Note: expected_length should be the same as pred_months when the timesteps
        are monthly, but should be more if the timesteps are at shorter resolution
        than monthly.
        """

        print(f"Generating data for year: {year}, target month: {target_month}")

        max_date = date(year, target_month, calendar.monthrange(year, target_month)[-1])
        mx_year, mx_month, max_train_date = minus_months(
            year, target_month, diff_months=1
        )
        _, _, min_date = minus_months(mx_year, mx_month, diff_months=pred_months)

        # `max_date` is the date to be predicted;
        # `max_train_date` is one timestep before;
        min_date_np = np.datetime64(str(min_date))
        max_date_np = np.datetime64(str(max_date))
        max_train_date_np = np.datetime64(str(max_train_date))

        print(
            f"Max date: {str(max_date)}, max input date: {str(max_train_date)}, "
            f"min input date: {str(min_date)}"
        )

        # boolean array indexing the timestamps to filter `ds`
        x = (ds.time.values > min_date_np) & (ds.time.values <= max_train_date_np)
        y = (ds.time.values > max_train_date_np) & (ds.time.values <= max_date_np)

        # only expect ONE y timestamp
        if sum(y) != 1:
            print(f"Wrong number of y values! Expected 1, got {sum(y)}; returning None")
            return None, cast(date, max_train_date)

        if expected_length is not None:
            if sum(x) != expected_length:
                print(f"Wrong number of x values! Got {sum(x)} Returning None")

                return None, cast(date, max_train_date)

        # filter the dataset
        x_dataset = ds.isel(time=x)
        y_dataset = ds.isel(time=y)[target_variable].to_dataset(name=target_variable)

        if x_dataset.time.size != expected_length:
            # catch the errors as we get closer to the MINIMUM year
            warnings.warn(
                "For the `nowcast` experiment we expect the\
                number of timesteps to be: {pred_months}.\
                Currently: {x_dataset.time.size}"
            )
            return None, cast(date, max_train_date)

        return {"x": x_dataset, "y": y_dataset}, cast(date, max_train_date)
