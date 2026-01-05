DARLING
------------------------------------
the python **[D]ark** field x-ray microscopy **[A]nalysis** & **[R]econstruction** **[L]ibrary** for rapid data **[IN]spection** & **[G]raphing**

.. image:: https://img.shields.io/badge/platform-cross--platform-brightgreen.svg
   :target: https://www.python.org/
   :alt: cross-platform

.. image:: https://img.shields.io/badge/code-pure%20python-blue.svg
   :target: https://www.python.org/
   :alt: pure python

.. image:: https://github.com/AxelHenningsson/darling/actions/workflows/pytest-linux-py310.yml/badge.svg
   :target: https://github.com/AxelHenningsson/darling/actions/workflows/pytest-linux-py310.yml
   :alt: tests ubuntu-linux

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: code style black

.. image:: https://img.shields.io/badge/docs-sphinx-blue.svg
   :target: https://axelhenningsson.github.io/darling/
   :alt: Sphinx documentation

Authors
------------------------------------
``darling`` is written and maintained by:

`Axel Henningsson <https://github.com/AxelHenningsson>`_,

with important early contributions from:
`Felix Tristan Frankus <https://github.com/adcret>`_ and
`Adam André William Cretton <https://github.com/fetrifra>`_

The core ideas of this library was originally written during a beamtime at ESRF id03D.

Until an associated journal publication is available, if you use this code in your research, we ask that you cite this repository.

If you are interested in collaborating with us on DFXM data analysis, please reach out to us at: naxhe@dtu.dk
and we can discuss the possibilities.

Usecase
------------------------------------------------

================================================
Loading & plotting Data
================================================

Darling collects many useful tools in the ``properties`` and ``transforms`` modules.

For example, it is possible to segment per-pixel peaks and extract features from them. Color-coding the the mean motor position of the first peak per pixel gives a mosaicity map as shown below.


.. code-block:: python

   import darling

   # replace with your own data path
   path_to_data, _, _ = darling.io.assets.domains()
   dset = darling.DataSet(path_to_data, scan_id="1.1")

   # segment per-pixel peaks and extract features
   peakmap = darling.properties.peaks(dset.data, k=4, coordinates=dset.motors)

   # color code the scan based on the first peak-maxima mean motor position
   rgbmap, colorkey, colorgrid = darling.transforms.rgb(
      peakmap.mean, coordinates=dset.motors
   )

   plt.figure()
   plt.imshow(rgbmap)
   plt.show()


.. image:: https://github.com/AxelHenningsson/darling/blob/dev/docs/source/images/domains_mosa.png?raw=true
   :align: center


for more examples see the externally hosted documentation at https://axelhenningsson.github.io/darling/

Installation
------------------------------------------------
From source the key is simply to clone and pip install

.. code-block:: bash

    git clone https://github.com/AxelHenningsson/darling.git
    cd darling
    pip install -e .

In general, you probably want to install in a fresh virtual environment as

.. code-block:: bash

   python3 -m venv .venv_darling
   source .venv_darling/bin/activate
   git clone https://github.com/AxelHenningsson/darling.git
   cd darling
   pip install -e .

use

.. code-block:: bash

   source .venv_darling/bin/activate

whenever you want to activate the environment. To add your env into a jupyter kernel such that
you can use it in an interactive notebook you may add the following two commands:

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name=darling

================================================
Note on jupyter & the ESRF slurm cluster
================================================

In the main ESRF slurm Python jupyter kernel it is possible to do the following hack to get the latest `darling` running.

.. code-block:: bash

   git clone https://github.com/AxelHenningsson/darling.git
   sys.path.insert(0, os.path.abspath('./darling'))
   import darling

This trick is possible since that all dependencies of `darling` are already installed in the big Python jupyter kernel at ESRF.


The following snippet has also been verified to work on the ESRF slurm cluster 19 Dec 2024 in a browser terminal:

.. code-block:: bash

   python3 -m venv .venv_darling
   source .venv_darling/bin/activate
   git clone https://github.com/AxelHenningsson/darling.git
   cd darling
   pip install -e .
   pip install ipykernel
   python -m ipykernel install --user --name=darling

This appraoch should work on other clusters as well, as long as some user permission to install exists.

Documentation
------------------------------------------------
Darling hosts documentation at https://axelhenningsson.github.io/darling/

