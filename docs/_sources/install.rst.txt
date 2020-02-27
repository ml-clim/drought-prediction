Installation
===============

`Anaconda <https://www.anaconda.com/download/#macos>`_ running python 3.7 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

.. code-block:: bash

    conda env create -f environment.yml

This will create an environment named `esowc-drought` with all the necessary packages to run the code. To
activate this environment, run

.. code-block:: bash

    conda activate esowc-drought

`Docker <https://www.docker.com/>`_ can also be used to run this code. To do this, first
run the docker app (either `docker desktop <https://www.docker.com/products/docker-desktop>`_
or configure the `docker-machine`):

.. code-block:: bash

    # on macOS
    brew install docker-machine docker

    docker-machine create --driver virtualbox default
    docker-machine env default

See `this link <https://stackoverflow.com/a/33596140/9940782>`_ for help on all machines or `this one <https://stackoverflow.com/a/49719638/9940782>`_
for MacOS.


Then build the docker image:

.. code-block:: bash

    docker build -t ml_drought .


Then, use it to run a container, mounting the data folder to the container:

.. code-block::bash

    docker run -it \
    --mount type=bind,source=<PATH_TO_DATA>,target=/ml_drought/data \
    ml_drought /bin/bash
