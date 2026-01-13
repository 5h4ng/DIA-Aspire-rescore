"""Tests for spectral library reader."""

import io

import pytest

from dia_aspire_rescore.io.speclib_reader import SpecLibReader, read_speclib

TEST_SPECLIB = """PrecursorMz	ProductMz	LibraryIntensity	Protein_name	ModifiedPeptide	StrippedPeptide	FragmentType	FragmentNumber	PrecursorCharge	FragmentCharge	uniprot_id	shared	decoy	ions	NormalizedRetentionTime
407.2507	671.4225	5079.5	P26599	AAAAGRIAI	AAAAGRIAI	y7	7.0	2	1.0	P26599	False	False	AAAAGRIAI2	46.06
407.2507	600.3851	4177.1	P26599	AAAAGRIAI	AAAAGRIAI	y6	6.0	2	1.0	P26599	False	False	AAAAGRIAI2	46.06
407.2507	540.3287	651.1	P26599	AAAAGRIAI	AAAAGRIAI	m2:7		2	1.0	P26599	False	False	AAAAGRIAI2	46.06
407.2507	469.2908	408.3	P26599	AAAAGRIAI	AAAAGRIAI	m3:7		2	1.0	P26599	False	False	AAAAGRIAI2	46.06
429.7614	293.1782	515.7	Q9BSJ5	AAAAGRKTL	AAAAGRKTL	m3:8		2	2.0	Q9BSJ5	False	False	AAAAGRKTL2	17.34
429.7614	716.4445	4773.6	Q9BSJ5	AAAAGRKTL	AAAAGRKTL	y7	7.0	2	1.0	Q9BSJ5	False	False	AAAAGRKTL2	17.34
509.3043	833.4845	1744.2	Q9NZJ5	AIM(UniMod:35)DIVIKV	AIM[147]DIVIKV	y7	7.0	2	1.0	Q9NZJ5	False	False	AIM(UniMod:35)DIVIKV2	68.80
509.3043	376.1552	1670.1	Q9NZJ5	AIM(UniMod:35)DIVIKV	AIM[147]DIVIKV	m3:5		2	1.0	Q9NZJ5	False	False	AIM(UniMod:35)DIVIKV2	68.80
509.3043	475.2239	1421.4	Q9NZJ5	AIM(UniMod:35)DIVIKV	AIM[147]DIVIKV	m3:6		2	1.0	Q9NZJ5	False	False	AIM(UniMod:35)DIVIKV2	68.80
474.7502	673.3484	2959.6	P55036	AIRNAM(UniMod:35)GSL	AIRNAM[147]GSL	b6	6.0	2	1.0	P55036	False	False	AIRNAM(UniMod:35)GSL2	31.83
474.7502	219.135	2619.7	P55036	AIRNAM(UniMod:35)GSL	AIRNAM[147]GSL	y2	2.0	2	1.0	P55036	False	False	AIRNAM(UniMod:35)GSL2	31.83
"""


@pytest.fixture
def test_file():
    return io.StringIO(TEST_SPECLIB)


class TestSpecLibReader:
    def test_fetch_transitions_by_index(self, test_file):
        reader = SpecLibReader()
        precursor_df, transition_df = reader.import_file(test_file)

        expected_data = [
            (407.2507, [671.4225, 600.3851, 540.3287, 469.2908], ["y7", "y6", "m2:7", "m3:7"]),
            (429.7614, [293.1782, 716.4445], ["m3:8", "y7"]),
            (509.3043, [833.4845, 376.1552, 475.2239], ["y7", "m3:5", "m3:6"]),
            (474.7502, [673.3484, 219.135], ["b6", "y2"]),
        ]

        assert len(precursor_df) == 4
        assert len(transition_df) == 11

        for i, (expected_mz, expected_trans_mzs, expected_types) in enumerate(expected_data):
            precursor = precursor_df.iloc[i]
            start = int(precursor["transition_start_idx"])
            stop = int(precursor["transition_stop_idx"])
            transitions = transition_df.iloc[start:stop]

            assert precursor["precursor_mz"] == pytest.approx(expected_mz, abs=1e-4)
            assert len(transitions) == len(expected_trans_mzs)

            for actual_mz, expected_trans_mz in zip(transitions["mz"], expected_trans_mzs):
                assert actual_mz == pytest.approx(expected_trans_mz, abs=1e-3)

            assert list(transitions["type"]) == expected_types

        assert precursor_df.iloc[-1]["transition_stop_idx"] == len(transition_df)

    def test_modifications(self, test_file):
        reader = SpecLibReader()
        precursor_df, _ = reader.import_file(test_file)

        unmodified = precursor_df[precursor_df["sequence"] == "AAAAGRIAI"].iloc[0]
        assert unmodified["mods"] == ""

        modified = precursor_df[precursor_df["sequence"] == "AIMDIVIKV"].iloc[0]
        assert "Oxidation@M" in modified["mods"]
        assert modified["mod_sites"] == "3"

        assert len(precursor_df[precursor_df["mods"] != ""]) == 2

    def test_internal_ions(self, test_file):
        reader = SpecLibReader()
        _, transition_df = reader.import_file(test_file)

        internal_ions = transition_df[transition_df["type"].str.startswith("m")]
        assert len(internal_ions) == 5
        assert set(internal_ions["type"]) == {"m2:7", "m3:7", "m3:8", "m3:5", "m3:6"}

    def test_convenience_function(self, test_file):
        precursor_df, transition_df = read_speclib(test_file)
        assert len(precursor_df) == 4
        assert len(transition_df) == 11

    def test_empty_file(self):
        empty = io.StringIO(
            "PrecursorMz\tProductMz\tLibraryIntensity\tProtein_name\tModifiedPeptide\t"
            "StrippedPeptide\tFragmentType\tFragmentNumber\tPrecursorCharge\t"
            "FragmentCharge\tuniprot_id\tshared\tdecoy\tions\tNormalizedRetentionTime\n"
        )
        reader = SpecLibReader()
        reader.import_file(empty)
        assert len(reader.precursor_df) == 0
        assert len(reader.transition_df) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
