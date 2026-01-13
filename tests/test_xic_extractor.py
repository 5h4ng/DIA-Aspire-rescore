"""Tests for XIC extractor."""

import pandas as pd
import pytest

from dia_aspire_rescore.extraction.extractor import XICExtractor


@pytest.fixture
def precursor_df():
    return pd.DataFrame({
        "sequence": ["PEPTIDE", "ANOTHER", "MODIFIED"],
        "mods": ["", "", "Oxidation@M"],
        "mod_sites": ["", "", "4"],
        "charge": [2, 3, 2],
        "precursor_id": ["PEPTIDE2", "ANOTHER3", "MODIFIED2"],
        "transition_start_idx": [0, 5, 10],
        "transition_stop_idx": [5, 10, 15],
    })


@pytest.fixture
def transition_df():
    return pd.DataFrame({
        "mz": [200.0, 300.0, 400.0, 500.0, 600.0] * 3,
        "charge": [1] * 15,
        "type": ["y3", "y4", "y5", "b3", "b4"] * 3,
    })


@pytest.fixture
def psm_df():
    return pd.DataFrame({
        "sequence": ["PEPTIDE", "ANOTHER", "UNKNOWN"],
        "mods": ["", "", ""],
        "mod_sites": ["", "", ""],
        "charge": [2, 3, 2],
        "precursor_mz": [400.0, 300.0, 350.0],
        "rt": [10.0, 20.0, 25.0],
        "rt_start": [9.5, 19.5, 24.5],
        "rt_stop": [10.5, 20.5, 25.5],
        "raw_name": ["raw1", "raw1", "raw1"],
    })


def test_match_psm_to_speclib(precursor_df, transition_df, psm_df):
    """Test PSM matching - should match 2 out of 3."""
    extractor = XICExtractor(precursor_df, transition_df)
    matched = extractor.match_psm_to_speclib(psm_df)
    assert len(matched) == 2
    assert "precursor_id" in matched.columns
    assert matched[matched["sequence"] == "PEPTIDE"]["precursor_id"].iloc[0] == "PEPTIDE2"
    assert matched[matched["sequence"] == "ANOTHER"]["precursor_id"].iloc[0] == "ANOTHER3"
