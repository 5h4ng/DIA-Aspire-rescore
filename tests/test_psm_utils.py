"""Tests for PSM utils functions."""

import numpy as np
import pandas as pd
import pytest

from dia_aspire_rescore.psm.utils import refine_matcher_results


@pytest.fixture
def psm_data():
    """
    Create test data: 3 PSMs with their fragments.
    - PSM 0: nAA=8, 7 fragments (idx 0-7)
    - PSM 1: nAA=10, 9 fragments (idx 7-16)
    - PSM 2: nAA=6, 5 fragments (idx 16-21)
    """
    psm_df = pd.DataFrame({
        "sequence": ["ABCDEFGH", "ABCDEFGHIJ", "ABCDEF"],
        "nAA": [8, 10, 6],
        "mods": ["", "", ""],
        "mod_sites": ["", "", ""],
        "charge": [2, 2, 3],
        "frag_start_idx": [0, 7, 16],
        "frag_stop_idx": [7, 16, 21],
    })

    # Each PSM has unique fragment values (total 21 rows)
    frag_cols = ["b_z1", "y_z1"]

    # PSM 0: mz=100-106, intensity=1000-1060
    # PSM 1: mz=300-308, intensity=3000-3080
    # PSM 2: mz=500-504, intensity=5000-5040
    mz_data = (
        [[100 + i, 200 + i] for i in range(7)]
        + [[300 + i, 400 + i] for i in range(9)]
        + [[500 + i, 600 + i] for i in range(5)]
    )

    intensity_data = (
        [[1000 + i * 10, 2000 + i * 10] for i in range(7)]
        + [[3000 + i * 10, 4000 + i * 10] for i in range(9)]
        + [[5000 + i * 10, 6000 + i * 10] for i in range(5)]
    )

    mz_err_data = (
        [[0.01 + i * 0.001, 0.02 + i * 0.001] for i in range(7)]
        + [[0.05 + i * 0.001, 0.06 + i * 0.001] for i in range(9)]
        + [[0.11 + i * 0.001, 0.12 + i * 0.001] for i in range(5)]
    )

    fragment_mz_df = pd.DataFrame(mz_data, columns=frag_cols)
    matched_intensity_df = pd.DataFrame(intensity_data, columns=frag_cols)
    matched_mz_err_df = pd.DataFrame(mz_err_data, columns=frag_cols)

    return psm_df, fragment_mz_df, matched_intensity_df, matched_mz_err_df


def test_refine_after_shuffle(psm_data):
    psm_df, frag_mz, frag_intensity, frag_mz_err = psm_data

    original = {}
    for _, row in psm_df.iterrows():
        seq = row["sequence"]
        start, stop = row["frag_start_idx"], row["frag_stop_idx"]
        original[seq] = {
            "mz": frag_mz.iloc[start:stop].values.copy(),
            "intensity": frag_intensity.iloc[start:stop].values.copy(),
            "mz_err": frag_mz_err.iloc[start:stop].values.copy(),
        }

    # Shuffle: [PSM2, PSM1, PSM0]
    shuffled_psm = psm_df.iloc[[2, 1, 0]].reset_index(drop=True)

    refined_psm, refined_mz, refined_intensity, refined_mz_err = refine_matcher_results(
        shuffled_psm.copy(), frag_mz.copy(), frag_intensity.copy(), frag_mz_err.copy()
    )

    for _, row in refined_psm.iterrows():
        seq = row["sequence"]
        start, stop = row["frag_start_idx"], row["frag_stop_idx"]

        refined_mz_vals = refined_mz.iloc[start:stop].values
        refined_int_vals = refined_intensity.iloc[start:stop].values
        refined_err_vals = refined_mz_err.iloc[start:stop].values

        np.testing.assert_array_almost_equal(refined_mz_vals, original[seq]["mz"], decimal=6)
        np.testing.assert_array_almost_equal(refined_int_vals, original[seq]["intensity"], decimal=6)
        np.testing.assert_array_almost_equal(refined_err_vals, original[seq]["mz_err"], decimal=6)
