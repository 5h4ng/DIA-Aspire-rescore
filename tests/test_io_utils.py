"""Tests for io/utils.py - MS file discovery utilities."""

from dia_aspire_rescore.io.utils import find_ms_files, ms_files_to_dict


class TestMsFilesToDict:
    """Test ms_files_to_dict function."""

    def test_mzml_files_to_dict(self):
        """Test converting mzML file paths to raw_name dict."""
        ms_files = [
            "/data/sample1.mzML",
            "/data/sample2.mzml",
        ]
        result = ms_files_to_dict(ms_files)

        assert result == {
            "sample1": "/data/sample1.mzML",
            "sample2": "/data/sample2.mzml",
        }

    def test_double_extension_removal(self):
        """Test that double extensions like .mzML.hdf5 are handled correctly."""
        ms_files = ["/data/sample.mzML.hdf5"]
        result = ms_files_to_dict(ms_files)

        assert result == {"sample": "/data/sample.mzML.hdf5"}


class TestFindMsFiles:
    """Test find_ms_files function."""

    def test_find_mzml_files(self, tmp_path):
        """Test finding .mzML files in a directory."""
        # Create test mzML files
        (tmp_path / "sample1.mzML").touch()
        (tmp_path / "sample2.mzml").touch()
        (tmp_path / "ignored.txt").touch()  # Should be ignored

        result = find_ms_files(str(tmp_path), "mzml")

        assert len(result) == 2
        assert "sample1" in result
        assert "sample2" in result
        assert "ignored" not in result

    def test_find_mzml_empty_directory(self, tmp_path):
        """Test finding mzML files in an empty directory."""
        result = find_ms_files(str(tmp_path), "mzml")

        assert result == {}
