"""Configuration for input/output paths and formats."""

from dataclasses import dataclass
from typing import Optional

from dia_aspire_rescore.config.base import ConfigBase


@dataclass
class IOConfig(ConfigBase):
    """Input/output configuration."""

    # TODO: use psm_reader provider in alphabase to support multiple report formats
    report: str  # DIA-NN parquet file path
    ms_file: str  # file path
    ms_file_format: str = "mzml"
    output_dir: str = "./output"
    raw_name: Optional[str] = None  # if None, process all raws
