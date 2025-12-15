"""Pipeline for DIA rescoring workflow."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from alphabase.peptide.precursor import refine_precursor_df
from alpharaw import register_all_readers

from dia_aspire_rescore.config.finetuning import FineTuneConfig
from dia_aspire_rescore.config.io import IOConfig
from dia_aspire_rescore.config.ms2_match import MS2MatchConfig
from dia_aspire_rescore.features import BasicFeatureGenerator, MS2FeatureGenerator, RTFeatureGenerator
from dia_aspire_rescore.finetuning import FineTuner
from dia_aspire_rescore.io import find_ms_files, read_diann2

logger = logging.getLogger(__name__)


class Pipeline:
    """
    DIA rescoring pipeline.

    Workflow:
    1. load_psm()
    2. finetune() (optional)
    3. generate_features()
    4. rescore() (placeholder)
    5. write_report()
    """

    def __init__(
        self,
        io_config: IOConfig,
        ms2_match_config: Optional[MS2MatchConfig] = None,
        finetune_config: Optional[FineTuneConfig] = None,
        feature_generators: Optional[list[str]] = None,
    ):
        self.io_config = io_config
        self.ms2_match_config = ms2_match_config or MS2MatchConfig()
        self.finetune_config = finetune_config or FineTuneConfig()
        self.feature_generators = feature_generators or ["basic", "ms2", "rt"]

        self.psm_df: Optional[pd.DataFrame] = None
        self.finetuner: Optional[FineTuner] = None
        self.ms_files: dict[str, str] = {}

        self._setup()

    def _setup(self):
        register_all_readers()
        Path(self.io_config.output_dir).mkdir(parents=True, exist_ok=True)
        self.ms_files = find_ms_files(self.io_config.ms_file_dir, self.io_config.ms_file_type)
        if not self.ms_files:
            raise ValueError(f"No {self.io_config.ms_file_type} files found in {self.io_config.ms_file_dir}")
        logger.info(f"Found {len(self.ms_files)} MS files")

    def run(self, skip_finetuning: bool = False) -> pd.DataFrame:
        """Full pipeline."""
        self.load_psm()
        if not skip_finetuning:
            self.finetune()
        self.generate_features()
        self.rescore()
        self.write_report()
        return self.psm_df

    def run_feature_generation(self, skip_finetuning: bool = False) -> pd.DataFrame:
        """Run pipeline up to feature generation (no rescoring)."""
        self.load_psm()
        if not skip_finetuning:
            self.finetune()
        self.generate_features()
        self.write_report()
        return self.psm_df

    def load_psm(self):
        """Load PSM data from report file."""
        self.psm_df = read_diann2(self.io_config.report_file)
        self.psm_df = refine_precursor_df(self.psm_df)
        logger.info(f"Loaded {len(self.psm_df)} PSMs")

    def finetune(self):
        """Finetune peptdeep models."""
        self.finetuner = FineTuner(self.finetune_config)
        self.finetuner.load_pretrained("generic")
        self.finetuner.train(
            self.psm_df,
            self.ms_files,
            ms_file_type=self.io_config.ms_file_type,
            ms2_match_config=self.ms2_match_config,
        )
        logger.info("Finetuning completed")

    def generate_features(self):
        """Generate rescoring features."""
        if self.finetuner is None:
            self.finetuner = FineTuner(self.finetune_config)
            self.finetuner.load_pretrained("generic")

        for gen_name in self.feature_generators:
            generator = self._create_generator(gen_name)
            if generator is None:
                logger.warning(f"Unknown generator: {gen_name}")
                continue

            logger.info(f"Generating {gen_name} features...")
            self.psm_df = generator.generate(self.psm_df)
            logger.info(f"Added {len(generator.feature_names)} features")

    def _create_generator(self, name: str):
        if name == "basic":
            return BasicFeatureGenerator()

        if self.finetuner is None:
            raise RuntimeError(f"Finetuner is required for '{name}' generator but was not initialized")

        if name == "ms2":
            return MS2FeatureGenerator(
                model_mgr=self.finetuner.model_manager,
                ms_files=self.ms_files,
                ms_file_type=self.io_config.ms_file_type,
                ms2_match_config=self.ms2_match_config,
            )
        elif name == "rt":
            return RTFeatureGenerator(model_mgr=self.finetuner.model_manager)
        return None

    def rescore(self):
        # TODO: Implement rescoring with percolator/ML
        pass

    def write_report(self):
        output_path = Path(self.io_config.output_dir) / "psm.csv"
        self.psm_df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
