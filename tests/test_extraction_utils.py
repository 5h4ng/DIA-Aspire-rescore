"""Tests for XIC extraction core functions."""

import numpy as np
import pandas as pd
import pytest

from dia_aspire_rescore.constants.spectrum import SpectrumDfCols
from dia_aspire_rescore.extraction.utils import (
    extract_xic,
    extract_xic_for_mzs,
    filter_spec_idxes_by_isolation,
    filter_spec_idxes_by_ms_level,
    get_spec_idxes_in_rt_window,
)


def test_get_spec_idxes_in_rt_window():
    spec_rts = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

    result = get_spec_idxes_in_rt_window(spec_rts, 15.0, 45.0)
    np.testing.assert_array_equal(result, [1, 2, 3])

    result = get_spec_idxes_in_rt_window(spec_rts, 20.0, 40.0)
    np.testing.assert_array_equal(result, [1, 2])

    result = get_spec_idxes_in_rt_window(spec_rts, 1.0, 5.0)
    assert len(result) == 0

    result = get_spec_idxes_in_rt_window(np.array([], dtype=np.float64), 10.0, 20.0)
    assert len(result) == 0


def test_filter_spec_idxes_by_isolation():
    # TODO: overlap window test
    spec_idxes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    isolation_lower = np.array([400.0, 450.0, 500.0, 550.0, 600.0], dtype=np.float64)
    isolation_upper = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=np.float64)

    result = filter_spec_idxes_by_isolation(spec_idxes, isolation_lower, isolation_upper, 525.0)
    np.testing.assert_array_equal(result, [2])

    result = filter_spec_idxes_by_isolation(spec_idxes, isolation_lower, isolation_upper, 450.0)
    np.testing.assert_array_equal(result, [0, 1])

    result = filter_spec_idxes_by_isolation(spec_idxes, isolation_lower, isolation_upper, 475.0)
    np.testing.assert_array_equal(result, [1])

    result = filter_spec_idxes_by_isolation(spec_idxes, isolation_lower, isolation_upper, 700.0)
    assert len(result) == 0


def test_filter_spec_idxes_by_ms_level():
    spec_idxes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    ms_levels = np.array([1, 2, 1, 2, 2], dtype=np.int32)

    result = filter_spec_idxes_by_ms_level(spec_idxes, ms_levels, 1)
    np.testing.assert_array_equal(result, [0, 2])

    result = filter_spec_idxes_by_ms_level(spec_idxes, ms_levels, 2)
    np.testing.assert_array_equal(result, [1, 3, 4])


@pytest.fixture
def mock_peak_data():
    peak_mzs = np.array([100.0, 200.0, 300.0, 150.0, 250.0, 350.0, 100.0, 200.0, 300.0], dtype=np.float64)
    peak_intensities = np.array(
        [1000.0, 2000.0, 3000.0, 1500.0, 2500.0, 3500.0, 1100.0, 2100.0, 3100.0], dtype=np.float32
    )
    peak_start_idxes = np.array([0, 3, 6], dtype=np.int64)
    peak_stop_idxes = np.array([3, 6, 9], dtype=np.int64)
    return peak_mzs, peak_intensities, peak_start_idxes, peak_stop_idxes


def test_extract_xic_for_mzs(mock_peak_data):
    peak_mzs, peak_intensities, peak_start_idxes, peak_stop_idxes = mock_peak_data

    spec_idxes = np.array([0, 1, 2], dtype=np.int64)
    query_mzs = np.array([200.0], dtype=np.float64)
    query_mz_tols = np.array([0.5], dtype=np.float64)

    result = extract_xic_for_mzs(
        spec_idxes, query_mzs, query_mz_tols, peak_mzs, peak_intensities, peak_start_idxes, peak_stop_idxes
    )
    assert result.shape == (1, 3)
    np.testing.assert_array_almost_equal(result[0], [2000.0, 0.0, 2100.0])

    query_mzs = np.array([100.0, 250.0], dtype=np.float64)
    query_mz_tols = np.array([0.5, 0.5], dtype=np.float64)
    result = extract_xic_for_mzs(
        spec_idxes, query_mzs, query_mz_tols, peak_mzs, peak_intensities, peak_start_idxes, peak_stop_idxes
    )
    assert result.shape == (2, 3)
    np.testing.assert_array_almost_equal(result[0], [1000.0, 0.0, 1100.0])
    np.testing.assert_array_almost_equal(result[1], [0.0, 2500.0, 0.0])

    query_mzs = np.array([500.0], dtype=np.float64)
    query_mz_tols = np.array([0.5], dtype=np.float64)
    result = extract_xic_for_mzs(
        spec_idxes, query_mzs, query_mz_tols, peak_mzs, peak_intensities, peak_start_idxes, peak_stop_idxes
    )
    np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])

    peak_mzs = np.array([99.9, 100.0, 100.1], dtype=np.float64)
    peak_intensities = np.array([500.0, 1000.0, 800.0], dtype=np.float32)
    peak_start_idxes = np.array([0], dtype=np.int64)
    peak_stop_idxes = np.array([3], dtype=np.int64)
    result = extract_xic_for_mzs(
        np.array([0]), np.array([100.0]), np.array([0.2]), peak_mzs, peak_intensities, peak_start_idxes, peak_stop_idxes
    )
    assert result[0, 0] == 1000.0


@pytest.fixture
def mock_spectrum_peak_dfs():
    spectrum_df = pd.DataFrame({
        SpectrumDfCols.RT: [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        SpectrumDfCols.MS_LEVEL: [1, 2, 2, 1, 2, 2],
        SpectrumDfCols.ISOLATION_LOWER_MZ: [-1.0, 400.0, 500.0, -1.0, 400.0, 500.0],
        SpectrumDfCols.ISOLATION_UPPER_MZ: [-1.0, 450.0, 550.0, -1.0, 450.0, 550.0],
        SpectrumDfCols.PEAK_START_IDX: [0, 2, 4, 6, 8, 10],
        SpectrumDfCols.PEAK_STOP_IDX: [2, 4, 6, 8, 10, 12],
    })
    peak_df = pd.DataFrame({
        "mz": [420.0, 520.0, 200.0, 300.0, 200.0, 300.0, 420.0, 520.0, 200.0, 300.0, 200.0, 300.0],
        "intensity": [1000.0, 2000.0, 1100.0, 2100.0, 1200.0, 2200.0, 1300.0, 2300.0, 1400.0, 2400.0, 1500.0, 2500.0],
    })
    return spectrum_df, peak_df


def test_extract_xic_ms1(mock_spectrum_peak_dfs):
    spectrum_df, peak_df = mock_spectrum_peak_dfs
    rt_values, intensities = extract_xic(spectrum_df, peak_df, np.array([420.0]), 9.0, 21.0, None, 20.0, 1)
    assert len(rt_values) == 2
    np.testing.assert_array_almost_equal(rt_values, [10.0, 16.0])
    np.testing.assert_array_almost_equal(intensities[0], [1000.0, 1300.0])


def test_extract_xic_ms2(mock_spectrum_peak_dfs):
    spectrum_df, peak_df = mock_spectrum_peak_dfs
    rt_values, intensities = extract_xic(spectrum_df, peak_df, np.array([200.0, 300.0]), 9.0, 21.0, 425.0, 20.0, 2)
    assert len(rt_values) == 2
    np.testing.assert_array_almost_equal(rt_values, [12.0, 18.0])
    np.testing.assert_array_almost_equal(intensities[0], [1100.0, 1400.0])
    np.testing.assert_array_almost_equal(intensities[1], [2100.0, 2400.0])

    rt_values, intensities = extract_xic(spectrum_df, peak_df, np.array([200.0]), 9.0, 21.0, 525.0, 20.0, 2)
    np.testing.assert_array_almost_equal(rt_values, [14.0, 20.0])
    np.testing.assert_array_almost_equal(intensities[0], [1200.0, 1500.0])


def test_extract_xic_edge_cases(mock_spectrum_peak_dfs):
    spectrum_df, peak_df = mock_spectrum_peak_dfs

    rt_values, intensities = extract_xic(spectrum_df, peak_df, np.array([420.0]), 100.0, 110.0, None, 20.0, 1)
    assert len(rt_values) == 0

    rt_values, intensities = extract_xic(spectrum_df, peak_df, np.array([200.0]), 9.0, 21.0, 700.0, 20.0, 2)
    assert len(rt_values) == 0

    with pytest.raises(ValueError, match="precursor_mz is required"):
        extract_xic(spectrum_df, peak_df, np.array([200.0]), 9.0, 21.0, None, 20.0, 2)
