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

Optional dependencies for documentation:

- ipython (>= 8.14.0)
- sphinx (== 6.2.1)
- nbsphinx (== 0.9.2)
- nbsphinx-link (== 1.3.0)
- sphinx-rtd-theme (== 1.2.2)
- sphinx-math-dollar (== 1.2.1)
- recommonmark (== 0.7.1)

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

    # install core and optional dependencies for documentation
    pip install --editable ".[doc]"
    
    # Now you have access to your package
    # as if it was installed in .venv
    python -c "import rational_rc"

Testing
~~~~~~~
After the editable installation from the source, configure testing in your IDE or run all tests from the terminal:

.. code:: bash
    
    python -m unittest discover tests
