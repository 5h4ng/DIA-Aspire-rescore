"""Configuration for the rescoring pipeline."""

from dataclasses import dataclass, field

from dia_aspire_rescore.config.base import ConfigBase
from dia_aspire_rescore.config.finetuning import FineTuneConfig
from dia_aspire_rescore.config.io import IOConfig
from dia_aspire_rescore.config.ms2_match import MS2MatchConfig


@dataclass
class PipelineConfig(ConfigBase):
    """Pipeline configuration with layered sub-configs."""

    io: IOConfig
    ms2_match: MS2MatchConfig = field(default_factory=MS2MatchConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)

    skip_finetuning: bool = False
    feature_generators: list[str] = field(default_factory=lambda: ["basic", "ms2", "rt"])

    # Rescoring (placeholder)
    rescore_method: str = "percolator"
    rescore_config: dict = field(default_factory=dict)
