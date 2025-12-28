import logging
from typing import ClassVar, Optional

import numba
import numpy as np
import pandas as pd
from alphabase.peptide.fragment import create_fragment_mz_dataframe
from alpharaw.match.psm_match import load_ms_data
from tqdm import tqdm

from dia_aspire_rescore.config.ms2_match import MS2MatchConfig
from dia_aspire_rescore.constants.spectrum import PsmDfColsExt
from dia_aspire_rescore.extraction.utils import extract_xic
from dia_aspire_rescore.features.base import BaseFeatureGenerator

logger = logging.getLogger(__name__)


class XICFeatureGenerator(BaseFeatureGenerator):
    """
    Generate co-elution features.
    """

    DEFAULT_FRAG_TYPES: ClassVar[list[str]] = ["b_z1", "b_z2", "y_z1", "y_z2"]
    DEFAULT_PPM_TOLERANCE: ClassVar[float] = 20.0

    def __init__(
        self,
        ms_files: dict[str, str],
        ms_file_type: str = "mzml",
        ppm_tolerance: float = DEFAULT_PPM_TOLERANCE,
        ms2_match_config: Optional[MS2MatchConfig] = None,
    ):
        """
        Initialize XICFeatureGenerator.

        Parameters
        ----------
        ms_files : dict[str, str]
            Dict mapping raw_name to file path.
        ms_file_type : str, optional
            MS file type, by default "mzml".
        ppm_tolerance : float, optional
            m/z tolerance in ppm for XIC extraction, by default 20.0.
        ms2_match_config : Optional[MS2MatchConfig], optional
            MS2 matching config
        """
        self.ms_files = ms_files
        self.ms_file_type = ms_file_type
        self.ppm_tolerance = ppm_tolerance
        self.frag_types = self.DEFAULT_FRAG_TYPES
        self.ms2_match_config = ms2_match_config or MS2MatchConfig()

    @property
    def feature_names(self) -> list[str]:
        return [
            # cosine correlation features
            "cos_corr_top3_median",
            "cos_corr_top3_mean",
            "cos_corr_top6_median",
            "cos_corr_top6_mean",
            "cos_corr_top12_median",
            "cos_corr_top12_mean",
            # pearson correlation features
            "pearson_corr_top3_median",
            "pearson_corr_top3_mean",
            "pearson_corr_top6_median",
            "pearson_corr_top6_mean",
            "pearson_corr_top12_median",
            "pearson_corr_top12_mean",
        ]

    def generate(
        self,
        psm_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate XIC correlation features.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM DataFrame with columns: sequence, charge, mods, mod_sites,
            rt_start, rt_stop, precursor_mz, raw_name.

        Returns
        -------
        pd.DataFrame
            psm_df with added XIC feature columns.
        """

        fragment_mz_df = create_fragment_mz_dataframe(psm_df, self.frag_types)

        for feat_name in self.feature_names:
            psm_df[feat_name] = 0.0

        for raw_name, file_path in self.ms_files.items():
            logger.info(f"Processing XIC features for {raw_name}")

            ms_data = load_ms_data(file_path, self.ms_file_type)
            spectrum_df = ms_data.spectrum_df
            peak_df = ms_data.peak_df

            mask = psm_df.raw_name == raw_name
            psm_indices = psm_df[mask].index.tolist()

            if len(psm_indices) == 0:
                continue

            feature_array = np.zeros((len(psm_indices), len(self.feature_names)), dtype=np.float32)

            for i, idx in enumerate(tqdm(psm_indices, desc=f"Processing {raw_name}")):
                psm = psm_df.loc[idx]
                features = self._calculate_xic_features(
                    psm,
                    fragment_mz_df,
                    spectrum_df,
                    peak_df,
                )
                feature_array[i, :] = features

            psm_df.loc[psm_indices, self.feature_names] = feature_array

        return psm_df

    def _calculate_xic_features(
        self,
        psm: pd.Series,
        fragment_mz_df: pd.DataFrame,
        spectrum_df: pd.DataFrame,
        peak_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Calculate co-elution features for a single PSM.

        Returns
        -------
        np.ndarray
            Array of 12 XIC correlation features (dtype=float32).
        """
        frag_start = int(psm[PsmDfColsExt.FRAG_START_IDX])
        frag_stop = int(psm[PsmDfColsExt.FRAG_STOP_IDX])
        frag_mzs = fragment_mz_df.iloc[frag_start:frag_stop]

        query_mzs = frag_mzs.values.flatten()
        query_mzs = query_mzs[query_mzs > 0]

        if len(query_mzs) < 2:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        try:
            rt_values, intensity_matrix = extract_xic(
                spectrum_df=spectrum_df,
                peak_df=peak_df,
                query_mzs=query_mzs,
                rt_start=psm[PsmDfColsExt.RT_START],
                rt_stop=psm[PsmDfColsExt.RT_STOP],
                precursor_mz=psm[PsmDfColsExt.PRECURSOR_MZ],
                ppm_tolerance=self.ppm_tolerance,
                ms_level=2,
                match_closest=True,
            )
        except Exception:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        if len(rt_values) < 3 or intensity_matrix.shape[1] < 3:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        cos_corr_matrix = _calc_cos_corr_matrix(intensity_matrix)
        pearson_corr_matrix = _calc_pearson_corr_matrix(intensity_matrix)

        cos_features = _extract_top_k_corr_features(cos_corr_matrix)
        pearson_features = _extract_top_k_corr_features(pearson_corr_matrix)

        # Combine features and convert to numpy array
        return np.array(cos_features + pearson_features, dtype=np.float32)


@numba.njit
def _calc_cos_corr_matrix(intensity_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate all cosine correlations between fragments.

    Parameters
    ----------
    intensity_matrix : np.ndarray
        Intensity matrix, shape (n_ions, n_specs).

    Returns
    -------
    np.ndarray
        Cosine correlation matrix, shape (n_ions, n_ions).
    """
    n_ions = intensity_matrix.shape[0]
    corr_matrix = np.zeros((n_ions, n_ions), dtype=np.float32)

    for i in range(n_ions):
        corr_matrix[i, i] = 1.0  # diagonal is always 1
        for j in range(i + 1, n_ions):
            corr = _calc_cos_corr(intensity_matrix[i, :], intensity_matrix[j, :])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


@numba.njit
def _calc_cos_corr(
    array_1: np.ndarray,
    array_2: np.ndarray,
) -> float:
    """
    Calculate cosine correlation between two arrays.
    """
    norm_1 = np.linalg.norm(array_1)
    norm_2 = np.linalg.norm(array_2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    return np.dot(array_1, array_2) / (norm_1 * norm_2)


@numba.njit
def _calc_pearson_corr_matrix(intensity_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate all Pearson correlations between fragments.

    Parameters
    ----------
    intensity_matrix : np.ndarray
        Intensity matrix, shape (n_ions, n_specs).

    Returns
    -------
    np.ndarray
        Pearson correlation matrix, shape (n_ions, n_ions).
    """
    n_ions = intensity_matrix.shape[0]
    corr_matrix = np.zeros((n_ions, n_ions), dtype=np.float32)

    for i in range(n_ions):
        corr_matrix[i, i] = 1.0  # diagonal is always 1
        for j in range(i + 1, n_ions):
            corr = _calc_pearson_corr(intensity_matrix[i, :], intensity_matrix[j, :])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


@numba.njit
def _calc_pearson_corr(
    array_1: np.ndarray,
    array_2: np.ndarray,
) -> float:
    """
    Calculate Pearson correlation between two arrays.
    """
    if np.std(array_1) == 0 or np.std(array_2) == 0:
        return 0.0

    mean_1 = np.mean(array_1)
    mean_2 = np.mean(array_2)
    centered_1 = array_1 - mean_1
    centered_2 = array_2 - mean_2
    numerator = np.sum(centered_1 * centered_2)
    denominator = np.sqrt(np.sum(centered_1**2) * np.sum(centered_2**2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


@numba.njit
def _extract_top_k_corr_features(
    corr_matrix: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Extract top-k correlation features from correlation matrix.

    Takes the upper triangle (excluding diagonal) of the correlation matrix,
    selects the top 3, 6, and 12 values, and calculates median and mean.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix, shape (n_ions, n_ions).

    Returns
    -------
    tuple[float, float, float, float, float, float]
        (top3_median, top3_mean, top6_median, top6_mean,
         top12_median, top12_mean)
    """
    n_ions = corr_matrix.shape[0]

    # excluding diagonal
    upper_tri_values = []
    for i in range(n_ions):
        for j in range(i + 1, n_ions):
            upper_tri_values.append(corr_matrix[i, j])

    if len(upper_tri_values) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    upper_tri_array = np.array(upper_tri_values, dtype=np.float32)

    sorted_corr = np.sort(upper_tri_array)[::-1]

    top3 = sorted_corr[: min(3, len(sorted_corr))]
    top3_median = np.median(top3)
    top3_mean = np.mean(top3)

    top6 = sorted_corr[: min(6, len(sorted_corr))]
    top6_median = np.median(top6)
    top6_mean = np.mean(top6)

    top12 = sorted_corr[: min(12, len(sorted_corr))]
    top12_median = np.median(top12)
    top12_mean = np.mean(top12)

    return (
        float(top3_median),
        float(top3_mean),
        float(top6_median),
        float(top6_mean),
        float(top12_median),
        float(top12_mean),
    )
