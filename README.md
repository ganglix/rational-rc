
# Rational RC

[![Documentation Status](https://rational-rc.readthedocs.io/en/latest/?badge=latest)](https://rational-rc.readthedocs.io)

## Overview

Rational RC is a practical life cycle deterioration modeling framework. It utilizes field survey data and provides probabilistic predictions of RC structure deterioration through different stages of the service life cycle. It covers various deterioration mechanisms such as membrane deterioration, concrete carbonation and chloride penetration, corrosion, and cracking.

## Features

- Advanced numerical models for RC structure deterioration
- Integration with field-sensing technology for tailored assessments
- Tools for optimizing maintenance schedules based on service life predictions
- Focus on cost-effectiveness and environmental sustainability

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
pip install -r requirements.txt
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
pip install -r requirements_dev.txt
pip install --editable .
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

Comprehensive documentation for Rational RC is available [here](https://rational-rc-docs.netlify.app).

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

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Create a new Pull Request

## Authors

- Gang Li - [ganglix@gmail.com](mailto:ganglix@gmail.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
