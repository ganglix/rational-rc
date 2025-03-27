
# Rational RC

[![Documentation Status](https://readthedocs.org/projects/rational-rc/badge/?version=latest)](https://rational-rc.readthedocs.io/en/latest/?badge=latest)

## Overview

Rational RC is a practical life cycle deterioration modeling framework. It utilizes field survey data and provides probabilistic predictions of RC structure deterioration through different stages of the service life cycle. It covers various deterioration mechanisms such as membrane deterioration, concrete carbonation and chloride penetration, corrosion, and cracking.

## Features

- Comprehensive life cycle modeling for RC structure deterioration using field survey data.
- Probabilistic predictions for confident assessment of structural failure risks.
- Modularized design for sequential analysis of different deterioration stages.
- Integrated workflow for evaluating and selecting cost-effective rehabilitation strategies.

## Installation

### Dependencies

Ensure you have the following dependencies installed:

```plaintext
- python (>= 3.9)
- numpy (>= 1.25.1)
- scipy (>= 1.11.1)
- pandas (>= 2.0.3)
- matplotlib (>= 3.7.2)
```

### pip installation

The easiest way to install Rational RC is using pip:

```bash
pip install -U rational-rc
```

It's a good practice to use a virtual environment for your project.

### From source

If you would like to install the most recent version of Rational RC under development, you may install it from the source.

For user mode:

```bash
git clone https://github.com/ganglix/rational-rc.git
cd rational-rc
pip install .
```

For development mode:

```bash
git clone https://github.com/ganglix/rational-rc.git
cd rational-rc
# create a virtual environment (you may also use conda to create)
python -m venv .venv
# Activate your environment with:
#      `source .venv/bin/activate` on Unix/macOS
# or   `.venv\Scripts\activate` on Windows
# On macOS, use quotes for optional dependencies:
pip install --editable ".[doc]"
# On other systems, you may omit the quotes:
# pip install --editable .[doc]
# Now you have access to your package
# as if it was installed in .venv
python -c "import rational_rc"
```

## Testing

After the editable installation from the source, configure testing in your IDE or run all tests from the terminal:

```bash
python -m unittest discover tests
```

## Documentation

Comprehensive documentation for Rational RC is available [here](https://rational-rc.readthedocs.io/en/latest/).

## API Reference

Here are the classes or methods:

- `membrane`
- `carbonation`
- `chloride`
- `corrosion`
- `cracking`
- `math_helper`

## Tutorials

- membrane module example
- carbonation module example
- chloride module example
- corrosion module example
- cracking model example
- case study demo1: dual carbonation
- case study demo2: membrane chloride LCC

## Contributing

We welcome contributions of all kinds—bug fixes, documentation improvements, or new deterioration models. Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Authors

- Gang Li - [ganglix@gmail.com](mailto:ganglix@gmail.com)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE.txt) file for details.
