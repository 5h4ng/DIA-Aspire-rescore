"""Tests for FineTuner class."""

import json
from pathlib import Path

import pandas as pd
import pytest
from peptdeep.model.ms2 import calc_ms2_similarity

from dia_aspire_rescore.config import FineTuneConfig
from dia_aspire_rescore.finetuning import FineTuner

TEST_DATA_DIR = Path(__file__).parent / "test_data" / "finetuning"


@pytest.fixture
def baseline_metrics():
    metrics_path = TEST_DATA_DIR / "baseline_metrics.json"
    with open(metrics_path) as f:
        return json.load(f)


@pytest.fixture
def psm_df():
    """Load PSM test data."""
    psm_df_path = TEST_DATA_DIR / "psm_df.parquet"
    df = pd.read_parquet(psm_df_path)
    df = df.sort_values(by="nAA", ascending=True).reset_index(drop=True)
    return df


@pytest.fixture
def matched_intensity_df():
    """Load matched intensity test data."""
    intensity_path = TEST_DATA_DIR / "matched_intensity.parquet"
    return pd.read_parquet(intensity_path)


@pytest.fixture
def config():
    """Load FineTuneConfig from test data."""
    config_path = TEST_DATA_DIR / "config.yaml"
    return FineTuneConfig.from_yaml(str(config_path))


@pytest.fixture
def finetuner(config):
    """Create FineTuner instance."""
    finetuner = FineTuner(config)
    finetuner.load_pretrained("generic")
    return finetuner


@pytest.fixture
def psm_df_rt(psm_df):
    """Prepare RT evaluation data by grouping by peptide sequence."""
    return psm_df.groupby(["sequence", "mods", "mod_sites"])[["rt_norm"]].median().reset_index()


def test_load_pretrained_models(config):
    """Test that pretrained models can be loaded."""
    finetuner = FineTuner(config)
    finetuner.load_pretrained("generic")

    assert finetuner.model_manager is not None
    assert finetuner.model_manager.ms2_model is not None
    assert finetuner.model_manager.rt_model is not None


def test_predict_rt_before_finetuning(finetuner, psm_df_rt, baseline_metrics):
    """Test RT prediction before fine-tuning matches baseline."""
    rt_test_result = finetuner.test_rt_model(psm_df_rt)
    r_square = rt_test_result["R_square"].values[0]

    expected_r_square = baseline_metrics["rt_before"]["r_square"]

    assert abs(r_square - expected_r_square) < 0.01, (
        f"RT R_square {r_square:.4f} differs from baseline {expected_r_square:.4f}"
    )

    psm_df_rt_pred = finetuner.predict_rt(psm_df_rt.copy())
    mae = (psm_df_rt_pred["rt_pred"] - psm_df_rt_pred["rt_norm"]).abs().mean()

    expected_mae = baseline_metrics["rt_before"]["mae"]

    assert abs(mae - expected_mae) < 0.01, f"RT MAE {mae:.4f} differs from baseline {expected_mae:.4f}"


def test_train_rt_model(finetuner, psm_df):
    """Test RT model training completes without errors."""
    finetuner.train_rt(psm_df)

    assert finetuner.model_manager.rt_model is not None


def test_predict_rt_after_finetuning(finetuner, psm_df, psm_df_rt, baseline_metrics):
    """Test RT prediction after fine-tuning shows improvement."""
    finetuner.train_rt(psm_df)

    rt_test_result = finetuner.test_rt_model(psm_df_rt)
    r_square = rt_test_result["R_square"].values[0]

    expected_r_square = baseline_metrics["rt_after"]["r_square"]

    assert abs(r_square - expected_r_square) < 0.01, (
        f"RT R_square {r_square:.4f} differs from expected {expected_r_square:.4f}"
    )

    psm_df_rt_pred = finetuner.predict_rt(psm_df_rt.copy())
    mae = (psm_df_rt_pred["rt_pred"] - psm_df_rt_pred["rt_norm"]).abs().mean()

    expected_mae = baseline_metrics["rt_after"]["mae"]

    assert abs(mae - expected_mae) < 0.01, f"RT MAE {mae:.4f} differs from expected {expected_mae:.4f}"


def test_predict_ms2_before_finetuning(finetuner, psm_df, matched_intensity_df, baseline_metrics):
    """Test MS2 prediction before fine-tuning matches baseline."""
    psm_df_ms2 = psm_df.copy()
    intensity_pred = finetuner.predict_ms2(psm_df_ms2)

    psm_df_ms2, metrics = calc_ms2_similarity(
        psm_df_ms2, intensity_pred, matched_intensity_df, metrics=["PCC", "COS", "SA", "SPC"]
    )

    target_psms = psm_df_ms2[psm_df_ms2["decoy"] == 0]

    expected_metrics = baseline_metrics["ms2_before"]

    for metric in ["PCC", "COS", "SA", "SPC"]:
        observed = target_psms[metric].mean()
        expected = expected_metrics[metric]

        assert abs(observed - expected) < 0.05, f"MS2 {metric} {observed:.4f} differs from baseline {expected:.4f}"


def test_train_ms2_model(finetuner, psm_df, matched_intensity_df):
    """Test MS2 model training completes without errors."""
    finetuner.train_ms2(psm_df, matched_intensity_df)

    assert finetuner.model_manager.ms2_model is not None


def test_predict_ms2_after_finetuning(finetuner, psm_df, matched_intensity_df, baseline_metrics):
    """Test MS2 prediction after fine-tuning shows improvement."""
    psm_df_ms2_before = psm_df.copy()
    intensity_before = finetuner.predict_ms2(psm_df_ms2_before)
    psm_df_ms2_before, _ = calc_ms2_similarity(
        psm_df_ms2_before, intensity_before, matched_intensity_df, metrics=["PCC", "COS", "SA", "SPC"]
    )
    pcc_before = psm_df_ms2_before[psm_df_ms2_before["decoy"] == 0]["PCC"].mean()
    cos_before = psm_df_ms2_before[psm_df_ms2_before["decoy"] == 0]["COS"].mean()

    finetuner.train_ms2(psm_df, matched_intensity_df)

    psm_df_ms2_after = psm_df.copy()
    intensity_pred = finetuner.predict_ms2(psm_df_ms2_after)

    psm_df_ms2_after, metrics = calc_ms2_similarity(
        psm_df_ms2_after, intensity_pred, matched_intensity_df, metrics=["PCC", "COS", "SA", "SPC"]
    )

    target_psms = psm_df_ms2_after[psm_df_ms2_after["decoy"] == 0]

    pcc_after = target_psms["PCC"].mean()
    cos_after = target_psms["COS"].mean()

    assert pcc_after >= pcc_before, (
        f"MS2 PCC should improve or stay same after fine-tuning: {pcc_before:.4f} -> {pcc_after:.4f}"
    )

    assert cos_after >= cos_before, (
        f"MS2 COS should improve or stay same after fine-tuning: {cos_before:.4f} -> {cos_after:.4f}"
    )


def test_complete_finetuning_workflow(config, psm_df, psm_df_rt, matched_intensity_df, baseline_metrics):
    """Test complete fine-tuning workflow from start to finish."""
    finetuner = FineTuner(config)
    finetuner.load_pretrained("generic")

    rt_before = (finetuner.predict_rt(psm_df_rt)["rt_pred"] - finetuner.predict_rt(psm_df_rt)["rt_norm"]).abs().mean()

    psm_df_ms2_before = psm_df.copy()
    intensity_before = finetuner.predict_ms2(psm_df_ms2_before)
    psm_df_ms2_before, _ = calc_ms2_similarity(
        psm_df_ms2_before, intensity_before, matched_intensity_df, metrics=["PCC", "COS", "SA", "SPC"]
    )
    pcc_before = psm_df_ms2_before[psm_df_ms2_before["decoy"] == 0]["PCC"].mean()

    finetuner.train_rt(psm_df)
    finetuner.train_ms2(psm_df, matched_intensity_df)

    rt_after = (finetuner.predict_rt(psm_df_rt)["rt_pred"] - finetuner.predict_rt(psm_df_rt)["rt_norm"]).abs().mean()

    psm_df_ms2_after = psm_df.copy()
    intensity_after = finetuner.predict_ms2(psm_df_ms2_after)
    psm_df_ms2_after, _ = calc_ms2_similarity(
        psm_df_ms2_after, intensity_after, matched_intensity_df, metrics=["PCC", "COS", "SA", "SPC"]
    )
    pcc_after = psm_df_ms2_after[psm_df_ms2_after["decoy"] == 0]["PCC"].mean()

    assert rt_after < rt_before, f"RT MAE should improve after fine-tuning: {rt_before:.4f} -> {rt_after:.4f}"

    assert pcc_after > pcc_before, f"MS2 PCC should improve after fine-tuning: {pcc_before:.4f} -> {pcc_after:.4f}"
