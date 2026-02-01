# DIA-Aspire-rescore

[![Release](https://img.shields.io/github/v/release/5h4ng/DIA-Aspire-rescore?style=for-the-badge)](https://github.com/5h4ng/DIA-Aspire-rescore/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/5h4ng/DIA-Aspire-rescore/main.yml?branch=main&style=for-the-badge)](https://github.com/5h4ng/DIA-Aspire-rescore/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/5h4ng/DIA-Aspire-rescore?style=for-the-badge)](https://github.com/5h4ng/DIA-Aspire-rescore/graphs/commit-activity)
[![codecov](https://img.shields.io/codecov/c/github/5h4ng/DIA-Aspire-rescore?style=for-the-badge)](https://codecov.io/gh/5h4ng/DIA-Aspire-rescore)
[![License](https://img.shields.io/github/license/5h4ng/DIA-Aspire-rescore?style=for-the-badge)](https://github.com/5h4ng/DIA-Aspire-rescore/blob/main/LICENSE)

A rescoring module for DIA-Aspire.

- **Github repository**: <https://github.com/5h4ng/DIA-Aspire-rescore/>

## Installation

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/5h4ng/DIA-Aspire-rescore.git
```

## Quick Start

### Generate Rescoring Features

Generate comprehensive features for rescoring DIA-NN identifications:

```bash
dia-aspire-rescore generate-features \
    --report path/to/diann_report.parquet \
    --ms-file-dir path/to/ms_files/ \
    --ms-file-type mzml \
    --output-dir ./output
```

### Chromatogram Extraction

Extract ion chromatograms based on the spectral library:

```bash
dia-aspire-rescore extract-xic \
    --report path/to/diann_report.parquet \
    --speclib path/to/speclib.tsv \
    --ms-file-dir path/to/ms_files/ \
    --ms-file-type mzml \
    --output-dir ./output/xic
```

## For Developers

### Requirements

- Python 3.9–3.13
- Git
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
git clone https://github.com/5h4ng/DIA-Aspire-rescore.git
cd DIA-Aspire-rescore
uv sync
```

### Common Commands

```bash
# Lint / format
make check
# Tests
make test
```

## TODO

- [ ] Generate rescore engine compatible output (PIN file format)
- [ ] Integrate mokapot as built-in rescore engine
- [ ] Feature refinement and expansion

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
