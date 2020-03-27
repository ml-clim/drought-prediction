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

Neural Networks
~~~~~~~~~~~~~~~~

A number of neural networks are implemented.
All are trained using `Smooth L1 Loss <https://pytorch.org/docs/stable/nn.html?highlight=smooth%20l1%20loss#torch.nn.SmoothL1Loss>`_,
with optional early stopping.

All neural network classes extend the following base class:

.. autoclass:: src.models.neural_networks.base.NNBase
   :members:

LSTM
-----
.. autoclass:: src.models.neural_networks.rnn.RecurrentNetwork
   :members:

EA-LSTM
--------
.. autoclass:: src.models.neural_networks.ealstm.EARecurrentNetwork
   :members:

Linear Network
---------------
.. autoclass:: src.models.neural_networks.linear_network.LinearNetwork
   :members:
