"""Tests for BasicFeatureGenerator."""

import pandas as pd
import pytest

from dia_aspire_rescore.features.basic import BasicFeatureGenerator


@pytest.fixture
def basic_generator():
    """Create a BasicFeatureGenerator instance."""
    return BasicFeatureGenerator()


@pytest.fixture
def sample_psm_df():
    """Create a sample PSM DataFrame for testing."""
    return pd.DataFrame({
        "sequence": ["PEPTIDE", "SEQUENCE", "AAAAAA", "TESTPEPTIDE"],
        "charge": [2, 3, 6, 7],
        "mods": [
            "",
            "Oxidation@M",
            "Carbamidomethyl@C",
            "Oxidation@M;Carbamidomethyl@C;Phospho@S",
        ],
    })


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
        ]
        assert basic_generator.feature_names == expected_names

    def test_charge_one_hot_encoding(self, basic_generator, sample_psm_df):
        """Test charge one-hot encoding for charges 1-6 and >6."""
        result_df = basic_generator.generate(sample_psm_df.copy())

        # Check charge 2 (second row)
        assert result_df.loc[0, "charge_1"] == 0
        assert result_df.loc[0, "charge_2"] == 1
        assert result_df.loc[0, "charge_3"] == 0
        assert result_df.loc[0, "charge_4"] == 0
        assert result_df.loc[0, "charge_5"] == 0
        assert result_df.loc[0, "charge_6"] == 0
        assert result_df.loc[0, "charge_gt_6"] == 0

        # Check charge 3 (second row)
        assert result_df.loc[1, "charge_1"] == 0
        assert result_df.loc[1, "charge_2"] == 0
        assert result_df.loc[1, "charge_3"] == 1
        assert result_df.loc[1, "charge_4"] == 0
        assert result_df.loc[1, "charge_5"] == 0
        assert result_df.loc[1, "charge_6"] == 0
        assert result_df.loc[1, "charge_gt_6"] == 0

        # Check charge 6 (third row)
        assert result_df.loc[2, "charge_1"] == 0
        assert result_df.loc[2, "charge_2"] == 0
        assert result_df.loc[2, "charge_3"] == 0
        assert result_df.loc[2, "charge_4"] == 0
        assert result_df.loc[2, "charge_5"] == 0
        assert result_df.loc[2, "charge_6"] == 1
        assert result_df.loc[2, "charge_gt_6"] == 0

        # Check charge > 6 (fourth row)
        assert result_df.loc[3, "charge_1"] == 0
        assert result_df.loc[3, "charge_2"] == 0
        assert result_df.loc[3, "charge_3"] == 0
        assert result_df.loc[3, "charge_4"] == 0
        assert result_df.loc[3, "charge_5"] == 0
        assert result_df.loc[3, "charge_6"] == 0
        assert result_df.loc[3, "charge_gt_6"] == 1

    def test_mod_count_empty(self, basic_generator):
        """Test modification counting with no modifications."""
        test_df = pd.DataFrame({"sequence": ["PEPTIDE"], "charge": [2], "mods": [""]})

        result_df = basic_generator.generate(test_df.copy())
        assert result_df.loc[0, "mod_num"] == 0

    def test_mod_count_excludes_carbamidomethyl(self, basic_generator):
        """Test that Carbamidomethyl@C is not counted."""
        test_df = pd.DataFrame({
            "sequence": ["PEPTIDE"],
            "charge": [2],
            "mods": ["Carbamidomethyl@C"],
        })

        result_df = basic_generator.generate(test_df.copy())
        assert result_df.loc[0, "mod_num"] == 0

    def test_mod_count_single_mod(self, basic_generator):
        """Test modification counting with single modification."""
        test_df = pd.DataFrame({"sequence": ["PEPTIDE"], "charge": [2], "mods": ["Oxidation@M"]})

        result_df = basic_generator.generate(test_df.copy())
        assert result_df.loc[0, "mod_num"] == 1

    def test_mod_count_multiple_mods(self, basic_generator):
        """Test modification counting with multiple modifications."""
        test_df = pd.DataFrame({
            "sequence": ["PEPTIDE"],
            "charge": [2],
            "mods": ["Oxidation@M;Phospho@S;Acetyl@K"],
        })

        result_df = basic_generator.generate(test_df.copy())
        assert result_df.loc[0, "mod_num"] == 3

    def test_mod_count_mixed_with_carbamidomethyl(self, basic_generator):
        """Test modification counting with Carbamidomethyl@C mixed with other mods."""
        test_df = pd.DataFrame({
            "sequence": ["PEPTIDE"],
            "charge": [2],
            "mods": ["Oxidation@M;Carbamidomethyl@C;Phospho@S"],
        })

        result_df = basic_generator.generate(test_df.copy())
        # Should count only Oxidation@M and Phospho@S, not Carbamidomethyl@C
        assert result_df.loc[0, "mod_num"] == 2

    def test_all_features_present(self, basic_generator, sample_psm_df):
        """Test that all expected features are present in output."""
        result_df = basic_generator.generate(sample_psm_df.copy())

        expected_features = basic_generator.feature_names
        for feature in expected_features:
            assert feature in result_df.columns, f"Feature {feature} not found in output"

    def test_original_columns_preserved(self, basic_generator, sample_psm_df):
        """Test that original columns are preserved in output."""
        result_df = basic_generator.generate(sample_psm_df.copy())

        for col in sample_psm_df.columns:
            assert col in result_df.columns, f"Original column {col} not preserved"
