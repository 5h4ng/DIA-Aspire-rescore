"""Tests for DIAPeptideSpectrumMatcher."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alphabase.peptide.fragment import flatten_fragments
from alpharaw import register_all_readers

from dia_aspire_rescore.psm.matcher import DIAPeptideSpectrumMatcher

register_all_readers()

TEST_DATA_DIR = Path(__file__).parent / "test_data" / "test_matcher"


@pytest.fixture
def psm_df():
    """Load PSM test data."""
    psm_df_path = TEST_DATA_DIR / "psm_df.tsv"
    df = pd.read_csv(psm_df_path, sep="\t")
    for col in ["mods", "mod_sites"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)
    return df


@pytest.fixture
def expected_frags_0():
    """Load expected fragments for PSM 0 (LPGPGASL)."""
    frags_path = TEST_DATA_DIR / "frags_0.tsv"
    return pd.read_csv(frags_path, sep="\t")


@pytest.fixture
def expected_frags_1():
    """Load expected fragments for PSM 1 (LPGPAESL)."""
    frags_path = TEST_DATA_DIR / "frags_1.tsv"
    return pd.read_csv(frags_path, sep="\t")


@pytest.fixture
def test_hdf5_path():
    """Path to test hdf5 file."""
    return str(TEST_DATA_DIR / "test_data.hdf5")


class TestDIAPeptideSpectrumMatcher:
    """Test suite for DIAPeptideSpectrumMatcher."""

    def test_match_single_psm_0(self, psm_df, expected_frags_0, test_hdf5_path):
        """Test matching single PSM 0 (LPGPGASL)."""
        matcher = DIAPeptideSpectrumMatcher(n_neighbors=0)

        # Process only first PSM
        psm_df_single = psm_df.iloc[0:1].copy()

        psm_df_matched, fragment_mz_df, matched_intensity_df, matched_mz_err_df = matcher.match_ms2_multi_raw(
            psm_df_single,
            {"20200317_QE_HFX2_LC3_DIA_RA957_R01": test_hdf5_path},
            "hdf5",
        )

        psm_df_matched, flat_frag_df = flatten_fragments(
            precursor_df=psm_df_matched,
            fragment_mz_df=fragment_mz_df,
            fragment_intensity_df=matched_intensity_df,
        )

        # Get fragments for PSM 0
        frag_start = psm_df_matched.iloc[0]["flat_frag_start_idx"]
        frag_stop = psm_df_matched.iloc[0]["flat_frag_stop_idx"]
        actual_frags = flat_frag_df.iloc[frag_start:frag_stop].copy()
        self._compare_fragments(actual_frags, expected_frags_0)

    def test_match_single_psm_1(self, psm_df, expected_frags_1, test_hdf5_path):
        """Test matching single PSM 1 (LPGPAESL)."""
        matcher = DIAPeptideSpectrumMatcher(n_neighbors=0)

        # Process only second PSM
        psm_df_single = psm_df.iloc[1:2].copy()

        psm_df_matched, fragment_mz_df, matched_intensity_df, matched_mz_err_df = matcher.match_ms2_multi_raw(
            psm_df_single,
            {"20200317_QE_HFX2_LC3_DIA_RA957_R01": test_hdf5_path},
            "hdf5",
        )

        psm_df_matched, flat_frag_df = flatten_fragments(
            precursor_df=psm_df_matched,
            fragment_mz_df=fragment_mz_df,
            fragment_intensity_df=matched_intensity_df,
        )

        # Get fragments for PSM 1
        frag_start = psm_df_matched.iloc[0]["flat_frag_start_idx"]
        frag_stop = psm_df_matched.iloc[0]["flat_frag_stop_idx"]
        actual_frags = flat_frag_df.iloc[frag_start:frag_stop].copy()
        self._compare_fragments(actual_frags, expected_frags_1)

    def test_match_both_psms(self, psm_df, expected_frags_0, expected_frags_1, test_hdf5_path):
        """Test matching both PSMs together and compare results separately.

        This test verifies that when processing multiple PSMs together,
        each PSM still gets correct fragment matches (tests for indexing bugs).
        """
        matcher = DIAPeptideSpectrumMatcher(n_neighbors=0)

        # Process both PSMs together
        psm_df_both = psm_df.iloc[0:2].copy()

        psm_df_matched, fragment_mz_df, matched_intensity_df, matched_mz_err_df = matcher.match_ms2_multi_raw(
            psm_df_both,
            {"20200317_QE_HFX2_LC3_DIA_RA957_R01": test_hdf5_path},
            "hdf5",
        )

        psm_df_matched, flat_frag_df = flatten_fragments(
            precursor_df=psm_df_matched,
            fragment_mz_df=fragment_mz_df,
            fragment_intensity_df=matched_intensity_df,
        )

        # Find PSM 0 (LPGPGASL) in matched results
        psm_0_idx = None
        for idx in range(len(psm_df_matched)):
            if psm_df_matched.iloc[idx]["sequence"] == "LPGPGASL":
                psm_0_idx = idx
                break

        assert psm_0_idx is not None, "PSM 0 (LPGPGASL) not found in matched results"

        # Get fragments for PSM 0
        frag_start_0 = psm_df_matched.iloc[psm_0_idx]["flat_frag_start_idx"]
        frag_stop_0 = psm_df_matched.iloc[psm_0_idx]["flat_frag_stop_idx"]
        actual_frags_0 = flat_frag_df.iloc[frag_start_0:frag_stop_0].copy()

        # Compare PSM 0 with expected
        self._compare_fragments(actual_frags_0, expected_frags_0)

        # Find PSM 1 (LPGPAESL) in matched results
        psm_1_idx = None
        for idx in range(len(psm_df_matched)):
            if psm_df_matched.iloc[idx]["sequence"] == "LPGPAESL":
                psm_1_idx = idx
                break

        assert psm_1_idx is not None, "PSM 1 (LPGPAESL) not found in matched results"

        # Get fragments for PSM 1
        frag_start_1 = psm_df_matched.iloc[psm_1_idx]["flat_frag_start_idx"]
        frag_stop_1 = psm_df_matched.iloc[psm_1_idx]["flat_frag_stop_idx"]
        actual_frags_1 = flat_frag_df.iloc[frag_start_1:frag_stop_1].copy()

        # Compare PSM 1 with expected
        self._compare_fragments(actual_frags_1, expected_frags_1)

    def test_match_ms2_one_raw(self, psm_df, expected_frags_0, test_hdf5_path):
        """Test match_ms2_one_raw method with single PSM."""
        matcher = DIAPeptideSpectrumMatcher(n_neighbors=0)

        # Process only first PSM using match_ms2_one_raw
        psm_df_single = psm_df.iloc[0:1].copy()

        psm_df_matched, fragment_mz_df, matched_intensity_df, matched_mz_err_df = matcher.match_ms2_one_raw(
            psm_df_single,
            test_hdf5_path,
            "hdf5",
        )

        psm_df_matched, flat_frag_df = flatten_fragments(
            precursor_df=psm_df_matched,
            fragment_mz_df=fragment_mz_df,
            fragment_intensity_df=matched_intensity_df,
        )

        # Get fragments for PSM 0
        frag_start = psm_df_matched.iloc[0]["flat_frag_start_idx"]
        frag_stop = psm_df_matched.iloc[0]["flat_frag_stop_idx"]
        actual_frags = flat_frag_df.iloc[frag_start:frag_stop].copy()
        self._compare_fragments(actual_frags, expected_frags_0)

    def test_match_ms2_one_raw_multiple_raw_files_error(self, psm_df, test_hdf5_path):
        """Test that match_ms2_one_raw raises error with multiple raw files."""
        matcher = DIAPeptideSpectrumMatcher(n_neighbors=0)

        # Create a DataFrame with two different raw files
        psm_df_multi = psm_df.iloc[0:2].copy()
        psm_df_multi.loc[psm_df_multi.index[1], "raw_name"] = "different_raw_file"

        # Should raise ValueError
        with pytest.raises(ValueError, match="should contain only one raw file"):
            matcher.match_ms2_one_raw(
                psm_df_multi,
                test_hdf5_path,
                "hdf5",
            )

    def _compare_fragments(self, actual_frags, expected_frags):
        """Helper method to compare fragment DataFrames."""
        # Sort by mz for comparison
        actual_sorted = actual_frags.sort_values("mz").reset_index(drop=True)
        expected_sorted = expected_frags.sort_values("mz").reset_index(drop=True)

        # Check same number of fragments
        assert len(actual_sorted) == len(expected_sorted), (
            f"Different number of fragments: {len(actual_sorted)} vs {len(expected_sorted)}"
        )

        # Check m/z values using np.isclose (relative tolerance for non-zero, absolute for zero)
        actual_mz = actual_sorted["mz"].values
        expected_mz = expected_sorted["mz"].values
        assert np.isclose(actual_mz, expected_mz, rtol=0.0001, atol=0.0001).all(), (
            f"m/z values differ\n"
            f"Max difference: {np.abs(actual_mz - expected_mz).max()}\n"
            f"Actual:\n{actual_sorted[['mz', 'intensity', 'type', 'number', 'position']]}\n"
            f"Expected:\n{expected_sorted[['mz', 'intensity', 'type', 'number', 'position']]}"
        )

        actual_intensity = actual_sorted["intensity"].values
        expected_intensity = expected_sorted["intensity"].values
        assert np.isclose(actual_intensity, expected_intensity, rtol=0.0001, atol=0.0001).all(), (
            f"Intensity values differ significantly\n"
            f"Max absolute difference: {np.abs(actual_intensity - expected_intensity).max()}\n"
            f"Actual:\n{actual_sorted[['mz', 'intensity', 'type', 'number', 'position']]}\n"
            f"Expected:\n{expected_sorted[['mz', 'intensity', 'type', 'number', 'position']]}"
        )

        # Check fragment types, numbers, and positions
        assert (actual_sorted["type"].values == expected_sorted["type"].values).all(), (
            f"Fragment types differ\nActual: {actual_sorted['type'].values}\nExpected: {expected_sorted['type'].values}"
        )
        assert (actual_sorted["number"].values == expected_sorted["number"].values).all(), (
            f"Fragment numbers differ\n"
            f"Actual: {actual_sorted['number'].values}\n"
            f"Expected: {expected_sorted['number'].values}"
        )
        assert (actual_sorted["position"].values == expected_sorted["position"].values).all(), (
            f"Fragment positions differ\n"
            f"Actual: {actual_sorted['position'].values}\n"
            f"Expected: {expected_sorted['position'].values}"
        )
