"""Tests for RTFeatureGenerator."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from dia_aspire_rescore.features.rt import RTFeatureGenerator


@pytest.fixture
def mock_model_mgr():
    """Create a mock ModelManager that simulates RT prediction."""

    def mock_predict_rt(psm_df):
        """Mock predict_rt that adds rt_pred column with controlled values."""
        psm_df["rt_pred"] = psm_df["rt_norm"] * 1.1
        return psm_df

    model_mgr = Mock()
    model_mgr.predict_rt.side_effect = mock_predict_rt
    return model_mgr


@pytest.fixture
def rt_generator(mock_model_mgr):
    """Create an RTFeatureGenerator instance with mocked model."""
    return RTFeatureGenerator(mock_model_mgr)


@pytest.fixture
def sample_psm_df_with_rt():
    """Create a sample PSM DataFrame with rt_norm column."""
    return pd.DataFrame({
        "sequence": ["PEPTIDE", "SEQUENCE", "AAAAAA", "TESTPEPTIDE"],
        "charge": [2, 3, 4, 2],
        "rt_norm": [0.5, 0.3, 0.8, 0.6],
    })


@pytest.fixture
def sample_psm_df_without_rt():
    """Create a sample PSM DataFrame without rt_norm column."""
    return pd.DataFrame({
        "sequence": ["PEPTIDE", "SEQUENCE", "AAAAAA", "TESTPEPTIDE"],
        "charge": [2, 3, 4, 2],
    })


class TestRTFeatureGenerator:
    """Test suite for RTFeatureGenerator."""

    def test_feature_names(self, rt_generator):
        """Test that feature_names property returns correct list."""
        expected_names = ["rt_pred", "delta_rt", "abs_rt_delta", "rt_ratio"]
        assert rt_generator.feature_names == expected_names

    def test_generate_with_rt_norm(self, rt_generator, sample_psm_df_with_rt, mock_model_mgr):
        """Test RT feature generation with rt_norm present."""
        result_df = rt_generator.generate(sample_psm_df_with_rt.copy())

        # Verify predict_rt was called
        assert mock_model_mgr.predict_rt.called

        # Verify rt_pred values (mocked to be rt_norm * 1.1)
        expected_rt_pred = sample_psm_df_with_rt["rt_norm"] * 1.1
        np.testing.assert_array_almost_equal(result_df["rt_pred"].values, expected_rt_pred.values)

        # Verify delta_rt = rt_pred - rt_norm
        expected_delta_rt = sample_psm_df_with_rt["rt_norm"] * 0.1
        np.testing.assert_array_almost_equal(result_df["delta_rt"].values, expected_delta_rt.values, decimal=6)

        # Verify abs_rt_delta = abs(delta_rt)
        np.testing.assert_array_almost_equal(
            result_df["abs_rt_delta"].values, np.abs(result_df["delta_rt"].values), decimal=6
        )

        # Verify rt_ratio = min/max and is in [0, 1]
        rt_pred = result_df["rt_pred"].values
        rt_norm = result_df["rt_norm"].values
        expected_ratio = np.minimum(rt_pred, rt_norm) / np.maximum(rt_pred, rt_norm)
        np.testing.assert_array_almost_equal(result_df["rt_ratio"].values, expected_ratio, decimal=6)
        assert (result_df["rt_ratio"] >= 0).all() and (result_df["rt_ratio"] <= 1).all()

    def test_generate_without_rt_norm(self, rt_generator, sample_psm_df_without_rt, mock_model_mgr):
        """Test RT feature generation when rt_norm is NOT present (fallback case)."""
        result_df = rt_generator.generate(sample_psm_df_without_rt.copy())

        # Verify predict_rt was NOT called (no rt_norm column)
        assert not mock_model_mgr.predict_rt.called

        # All values should be 0
        assert (result_df["rt_pred"] == 0).all()
        assert (result_df["delta_rt"] == 0).all()
        assert (result_df["abs_rt_delta"] == 0).all()
        assert (result_df["rt_ratio"] == 0).all()

    def test_negative_delta_rt(self):
        """Test that delta_rt can be negative when rt_pred < rt_norm."""

        def mock_predict_lower(psm_df):
            """Mock that predicts lower than observed."""
            psm_df["rt_pred"] = psm_df["rt_norm"] * 0.8
            return psm_df

        model_mgr = Mock()
        model_mgr.predict_rt.side_effect = mock_predict_lower
        rt_gen = RTFeatureGenerator(model_mgr)

        test_df = pd.DataFrame({"sequence": ["P1", "P2"], "charge": [2, 3], "rt_norm": [0.5, 0.8]})
        result_df = rt_gen.generate(test_df.copy())

        # delta_rt should be negative
        assert (result_df["delta_rt"] < 0).all()

        # abs_rt_delta should be positive
        assert (result_df["abs_rt_delta"] > 0).all()

        # Verify abs_rt_delta equals abs(delta_rt)
        np.testing.assert_array_almost_equal(
            result_df["abs_rt_delta"].values, np.abs(result_df["delta_rt"].values), decimal=6
        )
