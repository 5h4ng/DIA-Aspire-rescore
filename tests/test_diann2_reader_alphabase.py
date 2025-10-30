"""Tests for diann2_reader using alphabase backend."""

import pandas as pd
import pytest

from dia_aspire_rescore.io import read_diann2

# Expected values from the test data
EXPECTED_PSM = {
    "raw_name": "20200317_QE_HFX2_LC3_DIA_RA957_R02",
    "sequence": "AAAAVVEFQR",
    "charge": 2,
    "rt": 90.303909,
    "rt_start": 90.211235,
    "rt_stop": 90.489799,
    "mobility": 0.0,
    "proteins": "2/sp|O00231|PSD11_HUMAN/sp|O00231-2|PSD11_HUMAN",
    "fdr": 0.000599,
    "precursor_intensity": 1196415.875,
    "precursor_id": "(UniMod:1)AAAAVVEFQR2",
    "decoy": 0,
    "nAA": 10,
    "precursor_mz": 552.29583,
}

EXPECTED_COLUMNS = {
    "raw_name",
    "sequence",
    "charge",
    "rt",
    "rt_start",
    "rt_stop",
    "mobility",
    "proteins",
    "uniprot_ids",
    "genes",
    "fdr",
    "intensity",
    "precursor_intensity",
    "precursor_id",
    "gene_intensity",
    "decoy",
    "fdr1_search1",
    "fdr2_search1",
    "fdr1_search2",
    "fdr2_search2",
    "mods",
    "mod_sites",
    "nAA",
    "rt_norm",
    "precursor_mz",
    "ccs",
}


class TestDIANN2ReaderAlphabase:
    """Test suite for the DIANN2 reader using alphabase backend."""

    def test_read_diann2_returns_dataframe(self):
        """Test that read_diann2 returns a pandas DataFrame."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert isinstance(df, pd.DataFrame)

    def test_read_diann2_shape(self):
        """Test that the returned DataFrame has the correct shape."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.shape[0] == 1  # One row in test data
        assert df.shape[1] == 26  # Expected number of columns

    def test_read_diann2_columns(self):
        """Test that the returned DataFrame has the expected columns."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert set(df.columns) == EXPECTED_COLUMNS

    def test_read_diann2_decoy_column_type(self):
        """Test that the decoy column is int8."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df["decoy"].dtype == "int8"

    def test_read_diann2_charge_column_type(self):
        """Test that the charge column is int64."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df["charge"].dtype == "int64"

    def test_read_diann2_sequence_values(self):
        """Test that the sequence column has the expected value."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.loc[0, "sequence"] == EXPECTED_PSM["sequence"]

    def test_read_diann2_charge_values(self):
        """Test that the charge column has the expected value."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.loc[0, "charge"] == EXPECTED_PSM["charge"]

    def test_read_diann2_raw_name(self):
        """Test that the raw_name column has the expected value."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.loc[0, "raw_name"] == EXPECTED_PSM["raw_name"]

    def test_read_diann2_proteins(self):
        """Test that the proteins column has the expected value."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.loc[0, "proteins"] == EXPECTED_PSM["proteins"]

    def test_read_diann2_decoy_value(self):
        """Test that the decoy column has the expected value."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.loc[0, "decoy"] == EXPECTED_PSM["decoy"]

    def test_read_diann2_fdr_value(self):
        """Test that the fdr column has the expected value (with tolerance)."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.loc[0, "fdr"] == pytest.approx(EXPECTED_PSM["fdr"], rel=1e-3)

    def test_read_diann2_rt_value(self):
        """Test that the rt column has the expected value (with tolerance)."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.loc[0, "rt"] == pytest.approx(EXPECTED_PSM["rt"], rel=1e-5)

    def test_read_diann2_precursor_mz_value(self):
        """Test that the precursor_mz column has the expected value (with tolerance)."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert df.loc[0, "precursor_mz"] == pytest.approx(EXPECTED_PSM["precursor_mz"], rel=1e-5)

    def test_read_diann2_with_fdr_filter(self):
        """Test read_diann2 with FDR filtering."""
        # Read with strict FDR threshold
        df_strict = read_diann2("./tests/test_data/test_diann2.parquet", fdr=0.0001)
        # Read without FDR threshold
        df_loose = read_diann2("./tests/test_data/test_diann2.parquet", fdr=1.0)

        # With strict FDR, we might filter out rows
        assert df_strict.shape[0] <= df_loose.shape[0]

    def test_read_diann2_with_decoy_filter(self):
        """Test read_diann2 with decoy filtering."""
        # Read with decoy
        df_with_decoy = read_diann2("./tests/test_data/test_diann2.parquet", keep_decoy=True)
        # Read without decoy
        df_no_decoy = read_diann2("./tests/test_data/test_diann2.parquet", keep_decoy=False)

        # Without decoy, we might filter out rows
        assert df_no_decoy.shape[0] <= df_with_decoy.shape[0]

    def test_read_diann2_float_columns_dtype(self):
        """Test that floating-point columns have the correct dtype."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")

        float_columns = {
            "rt",
            "rt_start",
            "rt_stop",
            "mobility",
            "fdr",
            "intensity",
            "precursor_intensity",
            "gene_intensity",
            "fdr1_search1",
            "fdr2_search1",
            "fdr1_search2",
            "fdr2_search2",
            "rt_norm",
            "ccs",
        }

        for col in float_columns:
            if col in df.columns and not df[col].isna().all():
                # Check if dtype is float-like
                assert df[col].dtype in ["float32", "float64"], f"Column {col} has dtype {df[col].dtype}"

    def test_read_diann2_all_rows_have_sequence(self):
        """Test that all rows have a non-null sequence."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert not df["sequence"].isna().any()

    def test_read_diann2_all_rows_have_charge(self):
        """Test that all rows have a valid charge."""
        df = read_diann2("./tests/test_data/test_diann2.parquet")
        assert not df["charge"].isna().any()
        assert (df["charge"] > 0).all()
