# Contributing to Rational-RC

Thank you for your interest in contributing to **Rational-RC**. We welcome and appreciate all forms of contributions—whether you're fixing bugs, improving documentation, or extending the framework with new deterioration models.

## How to Contribute

To propose changes, please follow these steps:

1. **Fork** the repository  
2. **Create** a new feature branch:  
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes with a clear message:  
   ```bash
   git commit -am "Add a clear and descriptive message"
   ```
4. **Push** your branch:  
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open** a Pull Request describing your changes and their motivation

## Contribution Guidelines

### Code Structure
- Follow the modular design used in Rational-RC. Examples include submodules like `chloride`, `carbonation`, and `corrosion`.
- Name your files, functions, and classes clearly and consistently with existing conventions.

### Testing and Validation
- All new functionality should be covered with appropriate test cases that ensure correctness and reproducibility.
- Contributions must pass the existing test suite before being merged.
- If you're adding a new deterioration model:
  - Base it on validated theoretical formulations or peer-reviewed studies.
  - Clearly cite all sources and provide assumptions.
  - Note: Physical or experimental validation is outside the scope of this repository and should be documented separately (e.g., peer-reviewed publications).

### Documentation
- Include clear docstrings for all functions and classes.
- Update or add documentation (e.g., usage examples, explanations of new models) to help users understand and apply your additions.
- Reference the documentation site where appropriate:  
  [https://rational-rc.readthedocs.io](https://rational-rc.readthedocs.io)

### Collaborating with the Community
We encourage researchers and practitioners to apply Rational-RC in real-world projects. If you're using the package in academic or industrial settings, consider:
- Sharing your case study or results with the community.
- Providing feedback or feature requests via [Issues](https://github.com/ganglix/rational-rc/issues).
- Participating in discussions to shape the future of Rational-RC.

## Questions?

If you have any questions or need help getting started, feel free to open an issue or reach out to the maintainer at [ganglix@gmail.com](mailto:ganglix@gmail.com).

---

Thank you for helping make Rational-RC better and more useful to the broader community!
