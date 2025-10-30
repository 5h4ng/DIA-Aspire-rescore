"""Tests for psm_utils.io.diann."""

import pytest
from psm_utils.psm import PSM

from dia_aspire_rescore.io import DIANN2ParquetReader

test_psm = [
    PSM(
        peptidoform="[UNIMOD:1]-AAAAVVEFQR/2",
        spectrum_id="NaN",
        run="20200317_QE_HFX2_LC3_DIA_RA957_R02",
        collection=None,
        spectrum=None,
        is_decoy=False,
        score=None,
        qvalue=0.00059930905,
        pep=0.006649293,
        precursor_mz=552.2958,
        retention_time=90.30391,
        ion_mobility=0.0,
        protein_list=["sp|O00231|PSD11_HUMAN", "sp|O00231-2|PSD11_HUMAN"],
        rank=None,
        source="diann2",
        metadata={},
        rescoring_features={
            "Precursor.Charge": 2,
            "RT": 90.30391,
            "Predicted.RT": 91.15889,
            "iRT": 75.98548,
            "Predicted.iRT": 75.289116,
            "Ms1.Profile.Corr": 0.8117441,
            "Ms1.Area": 3822604.5,
            "Ms1.Normalised": 4063848.5,
            "IM": 0.0,
            "iIM": 0.0,
            "Predicted.IM": 0.0,
            "Predicted.iIM": 0.0,
            "Evidence": 2.9326496,
            "Mass.Evidence": 0.0,
            "Channel.Evidence": 0.200295,
            "Quantity.Quality": 0.7839234,
            "Empirical.Quality": 0.0,
            "Normalisation.Noise": 0.0069298875,
            "FWHM": 0.16186678,
            "Ms1.Apex.Area": 3281368.8,
            "Ms1.Apex.Mz.Delta": 0.002319336,
            "Ms1.Total.Signal.Before": 5654317600.0,
            "Ms1.Total.Signal.After": 5440430000.0,
            "RT.Start": 90.211235,
            "RT.Stop": 90.4898,
        },
    ),
]


def approx_equal_psm(psm1, psm2, rel_tol=1e-6):
    """
    Compare two PSM objects with tolerance for floating-point values,
    to avoid the floating-point precision issues.
    """
    # Compare non-floating point attributes directly
    assert psm1.peptidoform == psm2.peptidoform
    assert psm1.spectrum_id == psm2.spectrum_id
    assert psm1.run == psm2.run
    assert psm1.is_decoy == psm2.is_decoy
    assert psm1.protein_list == psm2.protein_list
    assert psm1.source == psm2.source

    # Compare floating-point attributes with tolerance
    assert psm1.qvalue == pytest.approx(psm2.qvalue, rel=rel_tol)
    assert psm1.pep == pytest.approx(psm2.pep, rel=rel_tol)
    assert psm1.precursor_mz == pytest.approx(psm2.precursor_mz, rel=rel_tol)
    assert psm1.retention_time == pytest.approx(psm2.retention_time, rel=rel_tol)
    assert psm1.ion_mobility == pytest.approx(psm2.ion_mobility, rel=rel_tol)

    # Compare rescoring features with tolerance
    assert len(psm1.rescoring_features) == len(psm2.rescoring_features)
    for key in psm1.rescoring_features:
        assert psm1.rescoring_features[key] == pytest.approx(psm2.rescoring_features[key], rel=rel_tol)


class TestDIANN2ParquetReader:
    def test_iter(self):
        with DIANN2ParquetReader("./tests/test_data/test_diann2.parquet") as reader:
            for i, psm in enumerate(reader):
                psm.provenance_data = {}
                approx_equal_psm(psm, test_psm[i])

    def test__parse_peptidoform(self):
        test_cases = [
            (("ACDE", "4"), "ACDE/4"),
            (("AC(UniMod:1)DE", "4"), "AC[UNIMOD:1]DE/4"),
            (("(UniMod:4)ACDE", "4"), "[UNIMOD:4]-ACDE/4"),
        ]

        reader = DIANN2ParquetReader("./tests/test_data/test_diann2.parquet")
        for (peptide, charge), expected in test_cases:
            assert reader._parse_peptidoform(peptide, charge) == expected

    def test__get_protein_list(self):
        test_cases = [
            ("2/sp|O00231|PSD11_HUMAN/sp|O00231-2|PSD11_HUMAN", ["sp|O00231|PSD11_HUMAN", "sp|O00231-2|PSD11_HUMAN"]),
            ("1/sp|P12345|TEST_HUMAN", ["sp|P12345|TEST_HUMAN"]),
            ("3/protein1/protein2/protein3", ["protein1", "protein2", "protein3"]),
        ]

        reader = DIANN2ParquetReader("./tests/test_data/test_diann2.parquet")
        for protein_ids_raw, expected in test_cases:
            assert reader._get_protein_list(protein_ids_raw) == expected
