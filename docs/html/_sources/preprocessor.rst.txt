Preprocessor
=============

The Preprocessors work to convert the datasets downloaded by the Exporters into a
**unified data format**. This makes testing and developing models much more straightforward.

As with the Exporters, a unique Preprocessor must be written for each dataset.

All preprocessors extend the following base Preprocessor, which has some useful helper methods:

~~~~

.. autoclass:: src.preprocess.base.BasePreProcessor
   :members:
