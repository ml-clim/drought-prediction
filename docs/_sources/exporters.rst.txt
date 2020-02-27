Exporters
=========

The exporters are responsible for **downloading** data from external sources, using a common
``exporter.export()`` method for downloading data.

All exporters extend the following base exporter:

~~~~

.. autoclass:: src.exporters.base.BaseExporter
   :members:
