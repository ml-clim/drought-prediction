Models
=======

All models extend the following base model:

.. autoclass:: src.models.base.ModelBase
   :members:


Loading models
~~~~~~~~~~~~~~~

Models have a ``save_model`` function, which saves the model to a pickle object. These
can then be loaded using the ``load_model`` function:

.. autofunction:: src.models.load_model

The following models have been implemented:

Persistence
~~~~~~~~~~~~
.. autoclass:: src.models.parsimonious.Persistence
   :members:

Linear Regression
~~~~~~~~~~~~~~~~~~
.. autoclass:: src.models.regression.LinearRegression
   :members:
