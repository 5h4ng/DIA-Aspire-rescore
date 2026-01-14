# DIA-Aspire-Rescore

[![Release](https://img.shields.io/github/v/release/https://github.com/5h4ng/DIA-Aspire-Rescore)](https://img.shields.io/github/v/release/https://github.com/5h4ng/DIA-Aspire-Rescore)
[![Build status](https://img.shields.io/github/actions/workflow/status/https://github.com/5h4ng/DIA-Aspire-Rescore/main.yml?branch=main)](https://github.com/https://github.com/5h4ng/DIA-Aspire-Rescore/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/https://github.com/5h4ng/DIA-Aspire-Rescore/branch/main/graph/badge.svg)](https://codecov.io/gh/https://github.com/5h4ng/DIA-Aspire-Rescore)
[![Commit activity](https://img.shields.io/github/commit-activity/m/https://github.com/5h4ng/DIA-Aspire-Rescore)](https://img.shields.io/github/commit-activity/m/https://github.com/5h4ng/DIA-Aspire-Rescore)
[![License](https://img.shields.io/github/license/https://github.com/5h4ng/DIA-Aspire-Rescore)](https://img.shields.io/github/license/https://github.com/5h4ng/DIA-Aspire-Rescore)

A rescoring module for DIA-Aspire.

- **Github repository**: <https://github.com/https://github.com/5h4ng/DIA-Aspire-Rescore/>

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

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
