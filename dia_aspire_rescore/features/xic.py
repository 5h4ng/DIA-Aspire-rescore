import pandas as pd

from dia_aspire_rescore.features.base import BaseFeatureGenerator


# TODO: implement this by using DIA-NN's XIC parquet file
class XICFeatureGenerator(BaseFeatureGenerator):
    """
    Generate XIC features.
    """

    def __init__(self, xic_df: pd.DataFrame):
        """
        Initialize XICFeatureGenerator.
        """
        self.xic_df = xic_df

    def generate(self, psm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate XIC features.
        """
        return psm_df
