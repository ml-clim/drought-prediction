Dataloader
===========

The dataloader class contains most of the experimental flexibility of the models. It reads
in ``NetCDF`` files produced by the Engineer. It is very similar to a
`PyTorch DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

Most of the options in the dataloader are exposed in the models - they're documented here so that
it is explicit where the functionality lives.

.. autoclass:: src.models.data.DataLoader
   :members:
