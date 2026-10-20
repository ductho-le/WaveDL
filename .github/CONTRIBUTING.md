# Contributing to WaveDL

Thank you for your interest in contributing to WaveDL! This document provides guidelines for contributing to the project.

## Ways to Contribute

- **Bug Reports**: Open an issue describing the bug, including steps to reproduce
- **Feature Requests**: Open an issue describing the proposed feature and its use case
- **Code Contributions**: Submit a pull request with your changes
- **Documentation**: Improve or add documentation

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ductho-le/WaveDL.git
cd WaveDL

# Create virtual environment (or use conda)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (required for contributors)
pre-commit install
```

### Verify Installation

```bash
# Check package imports work
python -c "from wavedl import __version__; print(f'WaveDL v{__version__}')"

# Run tests
pytest

# Run linting
ruff check .
```

## Code Quality

We use **Ruff** for linting and formatting, enforced via pre-commit hooks and CI.

### Automatic Checks (on every commit)

Once you run `pre-commit install`, these checks run automatically before each commit:

| Check | Purpose |
|-------|---------|
| **Ruff linter** | Catches errors, unused imports, style issues |
| **Ruff formatter** | Auto-formats code to consistent style |
| **File hygiene** | Trailing whitespace, end-of-file, YAML/TOML validation |

### Manual Check

To run all checks manually:

```bash
pre-commit run --all-files
```

### Code Style Guidelines

- Follow PEP 8 (enforced by Ruff)
- Use type hints for function signatures
- Include docstrings for public functions and classes
- Line length: 88 characters (Ruff default)

## Running Tests

Run the full unit test suite before submitting any changes:

```bash
# Run all tests (including slow tests)
pytest unit_tests/ -v

# Run fast tests only (only few seconds)
pytest unit_tests/ -v -m "not slow"
```

Investigate new warnings; documented upstream compatibility warnings may remain.
For CPU validation, set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`HF_HUB_OFFLINE=1`, and `ACCELERATE_USE_CPU=true`.

Also validate the shipped distribution and real distributed execution:

```bash
python -m build
python -m twine check --strict dist/*
python tools/check_distributed.py --work-dir /tmp/wavedl-ddp-check
```

Use a new directory for each distributed smoke invocation. The script launches
two CPU workers with a static IPv4 rendezvous and a locally selected free port.
On macOS it defaults Gloo to the loopback interface before launching workers.
When invoked under an existing `torchrun`, it uses that launcher's rendezvous;
the caller must configure networking before starting `torchrun`.

CI installs the wheel and runs `tools/smoke_installed.py` outside the checkout,
then exercises CLI entry
points. The test matrix covers Python 3.11–3.13, minimum/latest PyTorch, and Linux,
Windows, and macOS; the full architecture suite runs separately on Linux.

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Run `pre-commit run --all-files` to ensure code quality
5. Run `pytest unit_tests -v` to ensure tests pass
6. Submit a pull request with a clear description

## Adding New Models

See [`src/wavedl/models/_template.py`](../src/wavedl/models/_template.py) for a template to create new model architectures.

New models should:
- Inherit from `BaseModel`
- Use the `@register_model("model_name")` decorator
- Support 1D, 2D, and 3D input shapes (if possible)
- Declare their output contract: vector models return `(B, out_size)`; dense
  2D models accept `out_shape=(C, H, W)`, set
  `supports_spatial_output = True`, and return `(B, C, H, W)`
- Set `requires_spatial_output = True` when the architecture cannot be used
  with vector targets
- Include docstrings and type hints

## Releasing (Maintainers)

We use automated releases via GitHub Actions. When you push a version tag, a GitHub Release is created automatically with notes from `CHANGELOG.md`.

### Release Process

1. **Update `CHANGELOG.md`** with the new version's changes:
   ```markdown
   ## [1.2.0] - 2025-01-15

   ### Added
   - New feature description

   ### Changed
   - Updated behavior

   ### Fixed
   - Bug fix description
   ```

2. **Bump version** in `src/wavedl/__init__.py` (single source of truth):
   ```python
   __version__ = "1.2.0"
   ```

   > Note: `pyproject.toml` reads the version dynamically from `__init__.py`, so you only need to update it in one place.

3. **Validate the intended tag** before tagging:
   ```bash
   python tools/check_release.py --tag v1.2.0 --notes-file /tmp/wavedl-release-notes.md
   ```

   The tag must match `__version__` and have a nonempty changelog entry. Keep the
   preferred publication citation in `CITATION.cff`; its publication metadata is
   independent of the software release version.

4. **Commit and tag** the reviewed release:
   ```bash
   git add CHANGELOG.md src/wavedl/__init__.py
   git commit -m "Release v1.2.0"
   git tag v1.2.0
   git push origin main v1.2.0
   ```

5. **GitHub Actions** checks the tag, runs lint and the test matrix, builds a wheel
   from the source distribution, validates package metadata, and tests the installed
   wheel. Only after those jobs succeed does it publish the same verified artifacts
   to PyPI and create a GitHub Release. Configure PyPI trusted publishing for the
   `pypi` GitHub environment before releasing.

### Version Guidelines

- Follow [Semantic Versioning](https://semver.org/):
  - **MAJOR** (1.0.0 → 2.0.0): Breaking API changes
  - **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
  - **PATCH** (1.0.0 → 1.0.1): Bug fixes only

## Questions?

Open an issue or start a [GitHub Discussion](https://github.com/ductho-le/WaveDL/discussions).
