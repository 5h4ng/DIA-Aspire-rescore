"""Tests for spec_finder module."""

import numpy as np
import pandas as pd
import pytest

from dia_aspire_rescore.constants.spectrum import SpectrumDfCols
from dia_aspire_rescore.psm.spec_finder import (
    find_batch_DIA_spec_idxes_by_rt,
    find_DIA_spec_idxes_by_rt,
    find_single_DIA_spec_idx_by_rt,
)


@pytest.fixture
def mock_spectrum_df():
    """Create a mock spectrum DataFrame with DIA isolation windows."""
    data = {
        SpectrumDfCols.RT: np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        SpectrumDfCols.ISOLATION_LOWER_MZ: np.array([400.0, 450.0, 500.0, 550.0, 600.0], dtype=np.float32),
        SpectrumDfCols.ISOLATION_UPPER_MZ: np.array([410.0, 460.0, 510.0, 560.0, 610.0], dtype=np.float32),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_spectrum_arrays(mock_spectrum_df):
    """Extract arrays from mock spectrum DataFrame."""
    spec_rts = mock_spectrum_df[SpectrumDfCols.RT].to_numpy()
    spec_isolation_lower_mzs = mock_spectrum_df[SpectrumDfCols.ISOLATION_LOWER_MZ].to_numpy()
    spec_isolation_upper_mzs = mock_spectrum_df[SpectrumDfCols.ISOLATION_UPPER_MZ].to_numpy()
    return spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs


class TestFindSingleDIASpecIdxByRt:
    """Test suite for find_single_DIA_spec_idx_by_rt function."""

    def test_exact_match_within_window(self, mock_spectrum_arrays):
        """Test finding spectrum when RT and m/z match exactly."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        # Query at RT=20.0 with m/z=455.0 (within isolation window [450, 460])
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0,
            query_precursor_mz=455.0,
        )
        assert result == 1

    def test_close_rt_match(self, mock_spectrum_arrays):
        """Test finding spectrum when RT is close but not exact."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        # Query at RT=20.5 (between 20.0 and 30.0) with m/z=505.0 (within [500, 510])
        # Should find spectrum at index 2 (RT=30.0, closer than 20.0)
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.5,
            query_precursor_mz=505.0,
        )
        assert result == 2

    def test_mz_outside_isolation_window(self, mock_spectrum_arrays):
        """Test that spectrum is not found when m/z is outside isolation window."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        # Query at RT=20.0 but m/z=470.0 is outside window [450, 460]
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0,
            query_precursor_mz=470.0,
        )
        assert result == -1

    def test_rt_tolerance_constraint(self, mock_spectrum_arrays):
        """Test RT tolerance parameter."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        # Query at RT=20.1 with m/z=455.0, tolerance of 0.15 (should find spectrum at RT=20.0, diff=0.1)
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.1,
            query_precursor_mz=455.0,
            rt_tolerance=0.15,  # Should find spectrum at RT=20.0
        )
        assert result == 1

    def test_rt_tolerance_exceeded(self, mock_spectrum_arrays):
        """Test that spectrum is not found when RT difference exceeds tolerance."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        # Query at RT=20.2 but tolerance is 0.1 (difference is 0.2 > 0.1)
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.2,
            query_precursor_mz=455.0,
            rt_tolerance=0.1,
        )
        assert result == -1

    def test_query_rt_before_first_spectrum(self, mock_spectrum_arrays):
        """Test query RT before the first spectrum."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        # Query at RT=5.0 (before first spectrum at 10.0)
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=5.0,
            query_precursor_mz=405.0,
        )
        assert result == 0

    def test_query_rt_after_last_spectrum(self, mock_spectrum_arrays):
        """Test query RT after the last spectrum."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        # Query at RT=60.0 (after last spectrum at 50.0)
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=60.0,
            query_precursor_mz=605.0,
        )
        assert result == 4

    def test_single_spectrum(self):
        """Test with only one spectrum."""
        spec_rts = np.array([20.0], dtype=np.float32)
        spec_isolation_lower_mzs = np.array([450.0], dtype=np.float32)
        spec_isolation_upper_mzs = np.array([460.0], dtype=np.float32)

        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0,
            query_precursor_mz=455.0,
        )
        assert result == 0

    def test_single_spectrum_mz_mismatch(self):
        """Test with only one spectrum but m/z doesn't match."""
        spec_rts = np.array([20.0], dtype=np.float32)
        spec_isolation_lower_mzs = np.array([450.0], dtype=np.float32)
        spec_isolation_upper_mzs = np.array([460.0], dtype=np.float32)

        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0,
            query_precursor_mz=470.0,
        )
        assert result == -1

    def test_mz_at_isolation_boundaries(self, mock_spectrum_arrays):
        """Test m/z exactly at isolation window boundaries."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        # Test lower boundary (inclusive)
        result_lower = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0,
            query_precursor_mz=450.0,
        )
        assert result_lower == 1

        # Test upper boundary (inclusive)
        result_upper = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0,
            query_precursor_mz=460.0,
        )
        assert result_upper == 1


class TestFindBatchDIASpecIdxesByRt:
    """Test suite for find_batch_DIA_spec_idxes_by_rt function."""

    def test_batch_multiple_queries(self, mock_spectrum_arrays):
        """Test batch processing with multiple queries."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        query_rts = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        query_precursor_mzs = np.array([405.0, 455.0, 505.0], dtype=np.float32)

        result = find_batch_DIA_spec_idxes_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rts,
            query_precursor_mzs,
        )

        assert len(result) == 3
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 2

    def test_batch_with_failures(self, mock_spectrum_arrays):
        """Test batch processing with some queries failing to find spectra."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        query_rts = np.array([20.0, 25.0, 40.0], dtype=np.float32)
        query_precursor_mzs = np.array([455.0, 470.0, 555.0], dtype=np.float32)

        result = find_batch_DIA_spec_idxes_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rts,
            query_precursor_mzs,
        )

        assert len(result) == 3
        assert result[0] == 1  # Success
        assert result[1] == -1  # Failed: m/z outside window
        assert result[2] == 3  # Success

    def test_batch_empty_queries(self, mock_spectrum_arrays):
        """Test batch processing with empty query arrays."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        query_rts = np.array([], dtype=np.float32)
        query_precursor_mzs = np.array([], dtype=np.float32)

        result = find_batch_DIA_spec_idxes_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rts,
            query_precursor_mzs,
        )

        assert len(result) == 0

    def test_batch_with_rt_tolerance(self, mock_spectrum_arrays):
        """Test batch processing with RT tolerance."""
        spec_rts, spec_isolation_lower_mzs, spec_isolation_upper_mzs = mock_spectrum_arrays

        query_rts = np.array([20.05, 30.08], dtype=np.float32)
        query_precursor_mzs = np.array([455.0, 505.0], dtype=np.float32)

        result = find_batch_DIA_spec_idxes_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rts,
            query_precursor_mzs,
            rt_tolerance=0.1,
        )

        assert len(result) == 2
        assert result[0] == 1
        assert result[1] == 2


class TestFindDIASpecIdxesByRt:
    """Test suite for find_DIA_spec_idxes_by_rt function (high-level API)."""

    def test_find_via_dataframe(self, mock_spectrum_df):
        """Test finding spectra using DataFrame API."""
        query_rts = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        query_precursor_mzs = np.array([405.0, 455.0, 505.0], dtype=np.float32)

        result = find_DIA_spec_idxes_by_rt(
            mock_spectrum_df,
            query_rts,
            query_precursor_mzs,
        )

        assert len(result) == 3
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 2

    def test_find_via_dataframe_with_tolerance(self, mock_spectrum_df):
        """Test finding spectra via DataFrame with RT tolerance."""
        query_rts = np.array([20.05], dtype=np.float32)
        query_precursor_mzs = np.array([455.0], dtype=np.float32)

        result = find_DIA_spec_idxes_by_rt(
            mock_spectrum_df,
            query_rts,
            query_precursor_mzs,
            rt_tolerance=0.1,
        )

        assert len(result) == 1
        assert result[0] == 1

    def test_find_via_dataframe_mixed_results(self, mock_spectrum_df):
        """Test finding spectra with mixed success and failure."""
        query_rts = np.array([10.0, 20.0, 25.0], dtype=np.float32)
        query_precursor_mzs = np.array([405.0, 455.0, 470.0], dtype=np.float32)

        result = find_DIA_spec_idxes_by_rt(
            mock_spectrum_df,
            query_rts,
            query_precursor_mzs,
        )

        assert len(result) == 3
        assert result[0] == 0  # Success
        assert result[1] == 1  # Success
        assert result[2] == -1  # Failed: m/z outside window


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_rt_range(self):
        """Test with a large RT range."""
        spec_rts = np.array([0.1, 100.0, 200.0, 300.0], dtype=np.float32)
        spec_isolation_lower_mzs = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)
        spec_isolation_upper_mzs = np.array([410.0, 510.0, 610.0, 710.0], dtype=np.float32)

        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=150.0,
            query_precursor_mz=505.0,
        )
        assert result == 1

    def test_narrow_isolation_window(self):
        """Test with very narrow isolation windows."""
        spec_rts = np.array([10.0, 20.0], dtype=np.float32)
        spec_isolation_lower_mzs = np.array([500.0, 500.0], dtype=np.float32)
        spec_isolation_upper_mzs = np.array([500.1, 500.1], dtype=np.float32)

        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0,
            query_precursor_mz=500.05,
        )
        assert result == 1

    def test_wide_isolation_window(self):
        """Test with very wide isolation windows."""
        spec_rts = np.array([10.0, 20.0], dtype=np.float32)
        spec_isolation_lower_mzs = np.array([200.0, 300.0], dtype=np.float32)
        spec_isolation_upper_mzs = np.array([1000.0, 1100.0], dtype=np.float32)

        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0,
            query_precursor_mz=999.9,
        )
        assert result == 1

    def test_overlapping_isolation_windows(self):
        """Test with overlapping isolation windows."""
        spec_rts = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        spec_isolation_lower_mzs = np.array([400.0, 450.0, 500.0], dtype=np.float32)
        spec_isolation_upper_mzs = np.array([500.0, 550.0, 600.0], dtype=np.float32)

        # Query between overlapping windows - should match closest RT
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=25.0,
            query_precursor_mz=475.0,
        )
        assert result == 1  # Closest to 25.0 is 20.0

    def test_very_small_rt_tolerance(self):
        """Test with very small RT tolerance."""
        spec_rts = np.array([10.0, 20.0], dtype=np.float32)
        spec_isolation_lower_mzs = np.array([400.0, 450.0], dtype=np.float32)
        spec_isolation_upper_mzs = np.array([410.0, 460.0], dtype=np.float32)

        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=20.0001,
            query_precursor_mz=455.0,
            rt_tolerance=0.00001,
        )
        assert result == -1  # RT difference (0.0001) exceeds tolerance (0.00001)

    def test_negative_rt_tolerance_disables_check(self):
        """Test that negative RT tolerance disables RT checking."""
        spec_rts = np.array([10.0, 20.0], dtype=np.float32)
        spec_isolation_lower_mzs = np.array([400.0, 450.0], dtype=np.float32)
        spec_isolation_upper_mzs = np.array([410.0, 460.0], dtype=np.float32)

        # With negative tolerance, RT check should be disabled
        result = find_single_DIA_spec_idx_by_rt(
            spec_rts,
            spec_isolation_lower_mzs,
            spec_isolation_upper_mzs,
            query_rt=100.0,  # Very far from any spectrum
            query_precursor_mz=455.0,
            rt_tolerance=-1.0,
        )
        # Should still find spectrum if m/z is in window
        assert result == 1
