from pathlib import Path
import xarray as xr
from shutil import rmtree
from typing import Optional

from .base import BasePreProcessor


class GLEAMPreprocessor(BasePreProcessor):
    r"""Preprocess the GLEAM data.

    :param data_folder: The location of the data folder. Default: ``pathlib.Path("data")``
    """
    dataset = "gleam"

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)

    @staticmethod
    def _swap_dims_and_filter(ds: xr.Dataset) -> xr.Dataset:

        ds = ds.drop_dims("bnds")

        expected_dims = ["time", "lat", "lon"]
        print("Transposing arrays")
        return ds.transpose(*expected_dims)

    def _preprocess_single(
        self,
        netcdf_filepath: Path,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[xr.Dataset] = None,
    ) -> None:
        """Run the Preprocessing steps for the GLEAM data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder
        """
        print(f"Starting work on {netcdf_filepath.name}")
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # remove the time bounds variable
        ds = self._swap_dims_and_filter(ds)

        # 2. chop out EastAfrica
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 6. create the filepath and save to that location
        assert (
            netcdf_filepath.name[-3:] == ".nc"
        ), f"filepath name should be a .nc file. Currently: {netcdf_filepath.name}"

        filename = self._create_filename(
            netcdf_filepath.name,
            subset_name=subset_str if subset_str is not None else None,
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"** Done for GLEAM {netcdf_filepath.name} **")

    @staticmethod
    def _create_filename(netcdf_filename: str, subset_name: Optional[str] = None) -> str:
        """
        monthly/E_1980_2018_GLEAM_v3.3a_MO.nc
            =>
        E_1980_2018_GLEAM_v3.3a_MO_kenya.nc
        """
        filename_stem = netcdf_filename[:-3]
        if subset_name is not None:
            new_filename = f"{filename_stem}_{subset_name}.nc"
        else:
            new_filename = f"{filename_stem}.nc"
        return new_filename

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        cleanup: bool = True,
    ) -> None:
        r"""Preprocess all of the GLEAM .nc files to produce
        one subset file.

        Arguments
        ----------
        :param subset_str: The optional subset string used to get a geographical subset of the data.
            Only used to make a more descriptive filename.
        :param regrid: If a Path is passed, the output files will be regridded to have the same
            spatial grid as the dataset at that Path. If None, no regridding happens. Default = ``None``.
        :param resample_time: Defines the time length to which the data will be resampled. If ``None``,
            no time-resampling happens. Default = ``"M"`` (monthly).
        :param upsampling: If true, tells the class the time-sampling will be upsampling. In this case,
            nearest instead of mean is used for the resampling. Default = ``False``.
        :param cleanup: If true, delete interim files created by the class. Default = ``True``.
        """
        print(f"Reading data from {self.raw_folder}. Writing to {self.interim}")

        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        for file in nc_files:
            self._preprocess_single(file, subset_str, regrid)

        # merge all of the timesteps
        self.merge_files(subset_str, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)
