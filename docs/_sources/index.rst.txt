.. ml_clim documentation master file, created by
   sphinx-quickstart on Thu Feb 27 08:58:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A Machine Learning Pipeline for Climate Science
===============================================

This repository is an end-to-end pipeline for the creation, intercomparison and evaluation of machine learning methods in climate science.

The pipeline carries out a number of tasks to create a unified-data format for training and testing machine learning methods.

These tasks are split into the different classes defined in the `src` folder, and explained further below:

.. image:: ../img/pipeline_overview.png
  :width: 500
  :alt: Pipeline overview

Some basic working knowledge of Python is required to use this pipeline, although it is not too onerous.

For more information, see:

* `A blog post <https://gabrieltseng.github.io/2019/07/10/ML-Climate-Research.html>`_ describing the goals and design of the pipeline
* `An initial presentation <https://www.youtube.com/watch?v=QVFiGERCiYs>`_ of the pipeline

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   install
   develop

.. toctree::
   :maxdepth: 3
   :caption: Pipeline:

   data
   exporters
   preprocessor
   engineer
   datasets

.. * :ref:`search`
.. * :ref:`genindex`
.. :ref:`modindex`
