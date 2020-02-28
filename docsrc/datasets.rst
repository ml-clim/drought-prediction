Datasets
========

The following datasets are supported (i.e. an exporter and preprocessor has been implented for them):

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

CHIRPS Rainfall Estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the `CHIRPS website <https://www.chc.ucsb.edu/data/chirps>`_:

*Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS) is a 35+ year quasi-global rainfall data set. Spanning 50°S-50°N (and all 
longitudes) and ranging from 1981 to near-present, CHIRPS incorporates our in-house climatology, CHPclim, 0.05° resolution satellite imagery, 
and in-situ station data to create gridded rainfall time series for trend analysis and seasonal drought monitoring.*

.. autoclass:: src.exporters.chirps.CHIRPSExporter
   :members:

.. autoclass:: src.preprocess.chirps.CHIRPSPreprocessor
   :members:
