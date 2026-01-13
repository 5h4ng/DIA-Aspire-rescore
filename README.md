# DIA-Aspire-Rescore

[![Release](https://img.shields.io/github/v/release/https://github.com/5h4ng/DIA-Aspire-Rescore)](https://img.shields.io/github/v/release/https://github.com/5h4ng/DIA-Aspire-Rescore)
[![Build status](https://img.shields.io/github/actions/workflow/status/https://github.com/5h4ng/DIA-Aspire-Rescore/main.yml?branch=main)](https://github.com/https://github.com/5h4ng/DIA-Aspire-Rescore/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/https://github.com/5h4ng/DIA-Aspire-Rescore/branch/main/graph/badge.svg)](https://codecov.io/gh/https://github.com/5h4ng/DIA-Aspire-Rescore)
[![Commit activity](https://img.shields.io/github/commit-activity/m/https://github.com/5h4ng/DIA-Aspire-Rescore)](https://img.shields.io/github/commit-activity/m/https://github.com/5h4ng/DIA-Aspire-Rescore)
[![License](https://img.shields.io/github/license/https://github.com/5h4ng/DIA-Aspire-Rescore)](https://img.shields.io/github/license/https://github.com/5h4ng/DIA-Aspire-Rescore)

A deep learning–driven rescoring module for DIA-NN identification in the DIA-Aspire pipeline.

- **Github repository**: <https://github.com/https://github.com/5h4ng/DIA-Aspire-Rescore/>
- **Documentation** <https://https://github.com/5h4ng.github.io/DIA-Aspire-Rescore/>

## Installation

### For Users

```bash
pip install git+https://github.com/5h4ng/DIA-Aspire-rescore.git
```

Or install from source:

```bash
git clone https://github.com/5h4ng/DIA-Aspire-rescore.git
cd DIA-Aspire-rescore
pip install .
```

### For Developers

Requires [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/5h4ng/DIA-Aspire-rescore.git
cd DIA-Aspire-rescore
make install  # or: uv sync
```

## Quick Start

### Generate Rescoring Features

```bash
dia-aspire-rescore generate-features \
    --report path/to/diann_report.parquet \
    --ms-file-dir path/to/ms_files/ \
    --ms-file-type mzml \
    --output-dir ./output
```

### Chromatogram Extraction based on Spectral Library

```bash
dia-aspire-rescore extract-xic \
    --report path/to/diann_report.parquet \
    --speclib path/to/speclib.tsv \
    --ms-file-dir path/to/ms_files/ \
    --ms-file-type mzml \
    --output-dir ./output/xic \
```

## For Developers

### Setup Development Environment

Requires [uv](https://docs.astral.sh/uv/):

```bash
make install
```

This will also generate your `uv.lock` file and install pre-commit hooks.

### Development Workflow

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

Run tests:

```bash
make test
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/https://github.com/5h4ng/DIA-Aspire-Rescore/settings/secrets/actions/new).
- Create a [new release](https://github.com/https://github.com/5h4ng/DIA-Aspire-Rescore/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
