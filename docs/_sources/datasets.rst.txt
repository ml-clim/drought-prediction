Datasets
========

The following datasets are supported (i.e. an exporter and preprocessor has been implented for it):

Climate Data Store
~~~~~~~~~~~~~~~~~~

From the `Climate Data Store <https://climate.copernicus.eu/climate-data-store>`_ website:

*The C3S Climate Data Store (CDS) is a one-stop shop for information about the climate: past, present and future. It 
provides easy access to a wide range of climate datasets via a searchable catalogue. An online toolbox is available 
that allows users to build workflows and applications suited to their needs.*

The climate data store consists of multiple datasets. The following are supported in this pipeline:

ERA5 monthly averaged data
----------------------------

From the `ERA5 documentation <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=overview>`_:

*ERA5 is the fifth generation ECMWF reanalysis for the global climate and weather for the past 4 to 7 decades. 
Currently data is available from 1979. When complete, ERA5 will contain a detailed record from 1950 onwards. 
ERA5 replaces the ERA-Interim reanalysis.*

.. autoclass:: src.exporters.cds.ERA5Exporter
   :members:

.. autoclass:: src.preprocess.era5.ERA5MonthlyMeanPreprocessor
   :members:
