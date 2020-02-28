Engineer
=========

The engineer class works to create *train* and *test* data. It reads in processed
data, and writes out ``(X, y)`` pairs. This class allows the nature of the input
and output variables to be defined.

The ``(X, y)`` pairs saved by the engineer are grouped by prediction month (i.e.
a single ``NetCDF`` file is written for each prediction month, both for the train and
test data.

====

.. automodule:: src.engineer
   :members:
