"""Column name constants for DIA data."""

from alphabase.psm_reader.keys import PsmDfCols


class SpectrumDfCols:
    """Column names for DIA spectrum DataFrame."""

    RT = PsmDfCols.RT
    PRECURSOR_MZ = PsmDfCols.PRECURSOR_MZ
    ISOLATION_LOWER_MZ = "isolation_lower_mz"
    ISOLATION_UPPER_MZ = "isolation_upper_mz"
