.. |PythonMinVersion| replace:: 3.9
.. |NumPyMinVersion| replace:: 1.25.1
.. |SciPyMinVersion| replace:: 1.11.1
.. |MatplotlibMinVersion| replace:: 3.7.2
.. |PandasMinVersion| replace:: 2.0.3

============
Installation
============

.. contents::
   :local:

Dependencies
~~~~~~~~~~~~

Rational-RC requires:

- python (>= |PythonMinVersion|)
- numpy (>= |NumPyMinVersion|)
- scipy (>= |SciPyMinVersion|)
- pandas (>= |PandasMinVersion|)
- matplotlib (>= |MatplotlibMinVersion|)



pip installation
~~~~~~~~~~~~~~~~

The easiest way to install rational-rc is using ``pip``:

.. code:: bash

    pip install -U rational-rc

It is a good practice to use a `virtual environment
<https://docs.python.org/3/tutorial/venv.html>`_ for your project.

From source
~~~~~~~~~~~

If you would like to install the most recent rational-rc under development,
you may install rational-rc from source.

For user mode

.. code:: bash

    git clone https://github.com/ganglix/rational-rc.git
    cd rational-rc
    pip install -r requirements.txt
    pip install .


For `development mode <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_

.. code:: bash

    git clone https://github.com/ganglix/rational-rc.git
    cd rational-rc
    # create a virtual environment (you may also use conda to create)
    python -m venv .venv
    # Activate your environment with:
    #      `source .venv/bin/activate` on Unix/macOS
    # or   `.venv\Scripts\activate` on Windows
    pip install -r requirements_dev.txt
    pip install --editable .
    # Now you have access to your package
    # as if it was installed in .venv
    python -c "import rational-rc"

