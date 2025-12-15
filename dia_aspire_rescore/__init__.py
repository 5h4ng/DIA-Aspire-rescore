import logging
import warnings


def _simple_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}"


warnings.formatwarning = _simple_warning_format

warnings.filterwarnings(
    "ignore",
    message="Dotnet-based dependencies could not be loaded",
    category=UserWarning,
    module="alpharaw",
)

# TODO: doesn't work
warnings.filterwarnings(
    "ignore",
    message=r"mask_modloss is deprecated",
    category=UserWarning,
)
logging.captureWarnings(True)
