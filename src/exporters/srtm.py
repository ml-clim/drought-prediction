from pathlib import Path

from typing import Tuple

from .base import BaseExporter
from ..utils import Region, region_lookup

elevation = None
gdal = None


class SRTMExporter(BaseExporter):
    r"""Export SRTM elevation data. This exporter leverages
    the elevation package, http://elevation.bopen.eu/en/stable/, to download
    SRTM topography data.
    This exporter requires GDAL and the elevation package to work.

    An additional quirk of this exporter is that the region is defined **here**, instead of
    in the preprocessor.

    :param data_folder: The location of the data folder. Default: ``pathlib.Path("data")``
    """

    dataset = "srtm"

    def __init__(self, data_folder: Path = Path("data")):
        super().__init__(data_folder)

        # try and import gdal
        print(
            "The SRTM exporter requires GDAL and the elevation python package. "
            "The mac conda environment contains them."
            "In addition, a (finnicky) ubuntu environment contains them in "
            "environment.ubuntu.cpu.gdal.yml"
        )
        global gdal
        if gdal is None:
            from osgeo import gdal
        global elevation
        if elevation is None:
            import elevation

    @staticmethod
    def _region_to_tuple(region: Region) -> Tuple[float, float, float, float]:
        return region.lonmin, region.latmin, region.lonmax, region.latmax

    def _tif_to_nc(self, tif_file: Path, nc_file: Path) -> None:
        ds = gdal.Open(tif_file.resolve().as_posix())  # type: ignore
        _ = gdal.Translate(  # type: ignore
            format="NetCDF", srcDS=ds, destName=nc_file.resolve().as_posix()
        )

    def export(
        self,
        region_name: str = "kenya",
        product: str = "SRTM3",
        max_download_tiles: int = 15,
    ) -> None:
        r"""Export SRTm topography data

        :param region_name: Defines a geographical subset of the downloaded data to be used.
            Should be one of the regions defined in src.utils.region_lookup.
            Default = ``"kenya"``.
        :param product: One of ``{"SRTM1", "SRTM3"}``, the product to download the data from.
            Default = ``"SRTM3"``.
        :param max_download_tiles: By default, the elevation package doesn't allow more than 9
            tiles to be downloaded. Kenya is 12 tiles - this increases the limit to allow
            Kenya to be downloaded. Default = ``15``.
        """

        region = region_lookup[region_name]

        output_tif = self.output_folder / f"{region_name}.tif"

        if not output_tif.exists():
            print(f"Downloading tiles. Saving as tif to {output_tif}")
            try:
                elevation.clip(  # type: ignore
                    bounds=self._region_to_tuple(region),
                    output=output_tif.resolve().as_posix(),
                    product=product,
                    max_download_tiles=max_download_tiles,
                    margin="1",
                )
            except Exception as e:
                print(e)

            elevation.clean()  # type: ignore

        output_nc = self.output_folder / f"{region_name}.nc"

        if not output_nc.exists():
            print(f"Converting {output_tif} to NetCDF format ({output_nc})")
            self._tif_to_nc(output_tif, output_nc)
