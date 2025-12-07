from pathlib import Path
from typing import Any, Optional

import pandas as pd
from peptdeep.pretrained_models import ModelManager

from dia_aspire_rescore.config import FineTuneConfig


class FineTuner:
    """
    This class provides a simplified interface for fine-tuning MS2 and RT models
    using the AlphaPeptDeep ModelManager. It handles configuration, training,
    prediction, and model persistence.

    Attributes
    ----------
    config : FineTuneConfig
        Configuration object containing all training parameters
    model_manager : ModelManager
        Underlying ModelManager instance
    """

    def __init__(self, config: Optional[FineTuneConfig] = None):
        """
        Initialize the FineTuner.

        Parameters
        ----------
        config : FineTuneConfig, optional
            Configuration object. If None, uses default FineTuneConfig.
        """
        self.config = config if config is not None else FineTuneConfig()
        self._model_manager: Optional[ModelManager] = None
        self._initialize_model_manager()

    def _initialize_model_manager(self) -> None:
        """Initialize the ModelManager with config settings."""
        # Create ModelManager with initial settings
        kwargs: dict[str, Any] = {"mask_modloss": self.config.mask_modloss}
        if self.config.device is not None:
            kwargs["device"] = self.config.device

        self._model_manager = ModelManager(**kwargs)
        self._apply_config()

    def _apply_config(self) -> None:
        """Apply configuration settings to the ModelManager."""
        if self._model_manager is None:
            return

        # Model settings
        self._model_manager.instrument = self.config.instrument
        self._model_manager.nce = self.config.nce

        # MS2 training parameters
        self._model_manager.psm_num_to_train_ms2 = self.config.psm_num_to_train_ms2
        self._model_manager.psm_num_to_test_ms2 = self.config.psm_num_to_test_ms2
        self._model_manager.epoch_to_train_ms2 = self.config.epoch_to_train_ms2
        self._model_manager.warmup_epoch_to_train_ms2 = self.config.warmup_epoch_to_train_ms2
        self._model_manager.batch_size_to_train_ms2 = self.config.batch_size_to_train_ms2
        self._model_manager.lr_to_train_ms2 = self.config.lr_to_train_ms2
        self._model_manager.psm_num_per_mod_to_train_ms2 = self.config.psm_num_per_mod_to_train_ms2
        self._model_manager.top_n_mods_to_train = self.config.top_n_mods_to_train

        # RT training parameters
        self._model_manager.psm_num_to_train_rt_ccs = self.config.psm_num_to_train_rt_ccs
        self._model_manager.epoch_to_train_rt_ccs = self.config.epoch_to_train_rt_ccs
        self._model_manager.warmup_epoch_to_train_rt_ccs = self.config.warmup_epoch_to_train_rt_ccs
        self._model_manager.batch_size_to_train_rt_ccs = self.config.batch_size_to_train_rt_ccs
        self._model_manager.lr_to_train_rt_ccs = self.config.lr_to_train_rt_ccs
        self._model_manager.psm_num_per_mod_to_train_rt_ccs = self.config.psm_num_per_mod_to_train_rt_ccs

        # General settings
        self._model_manager.train_verbose = self.config.train_verbose
        self._model_manager.thread_num = self.config.thread_num

    @property
    def model_manager(self) -> ModelManager:
        """
        Get the underlying ModelManager instance.

        Returns
        -------
        ModelManager
            The ModelManager instance

        Raises
        ------
        RuntimeError
            If ModelManager is not initialized
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager is not initialized")
        return self._model_manager

    def load_pretrained(self, model_type: str = "generic") -> None:
        """
        Load pretrained models.

        Parameters
        ----------
        model_type : str, optional
            Type of pretrained model to load. Options include:
            'generic', 'phospho'. By default 'generic'.

        Raises
        ------
        RuntimeError
            If ModelManager is not initialized
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager is not initialized")

        # TODO: support external models
        self._model_manager.load_installed_models(model_type)

    def train_ms2(
        self,
        psm_df: pd.DataFrame,
        matched_intensity_df: pd.DataFrame,
    ) -> None:
        """
        Fine-tune the MS2 model.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame containing peptide information
        matched_intensity_df : pd.DataFrame
            DataFrame containing matched MS2 intensities

        Raises
        ------
        RuntimeError
            If ModelManager is not initialized or models not loaded
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager is not initialized")

        self._model_manager.train_ms2_model(psm_df, matched_intensity_df)

    def train_rt(self, psm_df: pd.DataFrame) -> None:
        """
        Fine-tune the RT model.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame containing peptide information and RT values

        Raises
        ------
        RuntimeError
            If ModelManager is not initialized or models not loaded
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager is not initialized")

        self._model_manager.train_rt_model(psm_df)

    def predict_ms2(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict MS2 fragment intensities.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame. Must be sorted by `nAA` (https://github.com/MannLabs/alphadia/pull/409).

        Returns
        -------
        pd.DataFrame
            Predicted intensities in wide format. Aligned with `psm_df` via `frag_start_idx`, `frag_stop_idx`.
            Columns are fragment types.

        Raises
        ------
        RuntimeError
            If ModelManager is not initialized or models not loaded

        Warnings
        --------
        Input must be nAA-sorted. Unsorted input causes row misalignment in output.
        See: https://github.com/MannLabs/alphadia/pull/409
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager is not initialized")

        return self._model_manager.predict_ms2(psm_df)

    def predict_rt(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict retention times.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame containing peptide information

        Returns
        -------
        pd.DataFrame
            DataFrame with predicted RT values in 'rt_pred' column

        Raises
        ------
        RuntimeError
            If ModelManager is not initialized or models not loaded
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager is not initialized")

        return self._model_manager.predict_rt(psm_df)

    def save_models(self, save_dir: str) -> None:
        """
        Save fine-tuned models to directory.

        This saves the MS2 and RT models to the specified directory.

        Parameters
        ----------
        save_dir : str
            Directory path to save models to. Will be created if it doesn't exist.

        Raises
        ------
        RuntimeError
            If ModelManager is not initialized
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager is not initialized")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self._model_manager.save_models(str(save_path))

    # TODO: Load models from directory

    def test_rt_model(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Test the RT model on a dataset.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame containing peptide information and RT values

        Returns
        -------
        pd.DataFrame
            DataFrame containing test statistics (R_square, R, slope, intercept, test_num)

        Raises
        ------
        RuntimeError
            If ModelManager is not initialized or models not loaded
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager is not initialized")

        return self._model_manager.rt_model.test(psm_df)
