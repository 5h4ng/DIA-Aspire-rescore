"""Tests for BasicFeatureGenerator."""

import pandas as pd
import pytest

from dia_aspire_rescore.features.basic import BasicFeatureGenerator


@pytest.fixture
def basic_generator():
    """Create a BasicFeatureGenerator instance."""
    return BasicFeatureGenerator()


class TestBasicFeatureGenerator:
    """Test suite for BasicFeatureGenerator."""

    def test_feature_names(self, basic_generator):
        """Test that feature_names property returns correct list."""
        expected_names = [
            "charge_1",
            "charge_2",
            "charge_3",
            "charge_4",
            "charge_5",
            "charge_6",
            "charge_gt_6",
            "mod_num",
            "pep_length",
            "precursor_mz",
        ]
        assert basic_generator.feature_names == expected_names

    def test_charge_one_hot_encoding(self, basic_generator):
        """Test charge one-hot encoding for charges 1-6 and >6."""
        test_df = pd.DataFrame({
            "sequence": ["P1", "P2", "P3", "P6", "P7"],
            "charge": [1, 2, 3, 6, 7],
            "mods": [""] * 5,
            "precursor_mz": [100.0, 200.0, 300.0, 600.0, 700.0],
        })

        result_df = basic_generator.generate(test_df.copy())

        # Check charge 1
        assert result_df.loc[0, "charge_1"] == 1
        assert result_df.loc[0, "charge_gt_6"] == 0

        # Check charge 2
        assert result_df.loc[1, "charge_2"] == 1
        assert result_df.loc[1, "charge_1"] == 0

        # Check charge 3
        assert result_df.loc[2, "charge_3"] == 1

        # Check charge 6
        assert result_df.loc[3, "charge_6"] == 1
        assert result_df.loc[3, "charge_gt_6"] == 0

        # Check charge > 6
        assert result_df.loc[4, "charge_gt_6"] == 1
        assert result_df.loc[4, "charge_6"] == 0

    def test_mod_count_no_mods(self, basic_generator):
        """Test modification counting with no modifications."""
        test_df = pd.DataFrame({
            "sequence": ["PEPTIDE"],
            "charge": [2],
            "mods": [""],
            "precursor_mz": [400.0],
        })
        result_df = basic_generator.generate(test_df.copy())
        assert result_df.loc[0, "mod_num"] == 0

    def test_mod_count_excludes_carbamidomethyl(self, basic_generator):
        """Test that Carbamidomethyl@C is not counted."""
        test_df = pd.DataFrame({
            "sequence": ["PEPTIDE"],
            "charge": [2],
            "mods": ["Carbamidomethyl@C"],
            "precursor_mz": [400.0],
        })
        result_df = basic_generator.generate(test_df.copy())
        assert result_df.loc[0, "mod_num"] == 0

    def test_mod_count_with_mods(self, basic_generator):
        """Test modification counting with single and multiple modifications."""
        test_df = pd.DataFrame({
            "sequence": ["P1", "P2"],
            "charge": [2, 2],
            "mods": ["Oxidation@M", "Oxidation@M;Phospho@S;Acetyl@K"],
            "precursor_mz": [300.0, 350.0],
        })
        result_df = basic_generator.generate(test_df.copy())

        # Single mod
        assert result_df.loc[0, "mod_num"] == 1

        # Multiple mods
        assert result_df.loc[1, "mod_num"] == 3

    def test_mod_count_mixed(self, basic_generator):
        """Test modification counting with Carbamidomethyl@C mixed with other mods."""
        test_df = pd.DataFrame({
            "sequence": ["PEPTIDE"],
            "charge": [2],
            "mods": ["Oxidation@M;Carbamidomethyl@C;Phospho@S"],
            "precursor_mz": [450.0],
        })
        result_df = basic_generator.generate(test_df.copy())

        # Should count only Oxidation@M and Phospho@S, not Carbamidomethyl@C
        assert result_df.loc[0, "mod_num"] == 2

    def test_pep_length(self, basic_generator):
        """Test peptide length calculation."""
        test_df = pd.DataFrame({
            "sequence": ["PEPTIDE", "PEPTIDEKK"],
            "charge": [2, 2],
            "mods": ["", ""],
            "precursor_mz": [400.0, 500.0],
        })
        result_df = basic_generator.generate(test_df.copy())
        assert result_df.loc[0, "pep_length"] == 7
        assert result_df.loc[1, "pep_length"] == 9

    def test_precursor_mz(self, basic_generator):
        """Test precursor m/z calculation."""
        test_df = pd.DataFrame({
            "sequence": ["PEPTIDE", "PEPTIDEKK"],
            "charge": [2, 2],
            "mods": ["", ""],
            "precursor_mz": [100.0, 500.0],
        })
        result_df = basic_generator.generate(test_df.copy())
        assert result_df.loc[0, "precursor_mz"] == 100.0
        assert result_df.loc[1, "precursor_mz"] == 500.0
