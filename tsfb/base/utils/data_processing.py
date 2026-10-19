import pandas as pd

from tsfb.base.utils.logging import get_logger

logger = get_logger(__name__)


def split_time(data, index):
    """
    Split a pandas DataFrame or Series at a given index.

    Parameters:
        data (pd.DataFrame or pd.Series): The input data to split.
        index (int): The index at which to split the data.

    Returns:
        tuple: Two DataFrames or Series,
        corresponding to data before and after the split index.
    """
    if isinstance(data, pd.Series):
        return data.iloc[:index], data.iloc[index:]
    return data.iloc[:index, :], data.iloc[index:, :]


def _process_index_and_dates(pdf: pd.DataFrame, src) -> pd.DataFrame:
    """Helper to set index and parse dates for DataFrame."""
    if isinstance(pdf, pd.DataFrame):
        if src.index_col and src.index_col in pdf.columns:
            pdf = pdf.set_index(src.index_col)
        if (
            src.index_col
            and src.parse_dates
            and not pd.api.types.is_datetime64_any_dtype(pdf.index)
        ):
            try:
                pdf.index = pd.to_datetime(pdf.index)
            except Exception:
                logger.warning("Could not parse index %s to datetime", src.index_col)
        if src.index_col and pd.api.types.is_datetime64_any_dtype(pdf.index):
            pdf = pdf.sort_index()
    return pdf
