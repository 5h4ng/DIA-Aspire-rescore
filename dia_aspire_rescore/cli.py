"""Command-line interface for DIA-Aspire-Rescore."""

import logging
import sys
import warnings

import click


def setup_warnings() -> None:
    """Configure warning filters before importing heavy dependencies."""
    warnings.filterwarnings(
        "ignore",
        message="mask_modloss is deprecated",
        category=UserWarning,
        module="peptdeep.model.ms2",
    )

    warnings.filterwarnings(
        "ignore",
        message="Dotnet-based dependencies could not be loaded",
        category=UserWarning,
        module="alpharaw",  # only involve mzml files in this project
    )


setup_warnings()

from dia_aspire_rescore.config import FineTuneConfig, IOConfig  # noqa: E402
from dia_aspire_rescore.pipeline import Pipeline, find_ms_files  # noqa: E402

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli():
    """DIA-Aspire-Rescore"""
    pass


@cli.command()
@click.option("--report", "-r", required=True, type=click.Path(exists=True), help="DIA-NN report parquet file")
@click.option(
    "--ms-file-dir",
    "-m",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing MS files",
)
@click.option(
    "--ms-file-type", type=click.Choice(["mzml", "hdf5"]), default="mzml", help="MS file extension to search for"
)
@click.option("--output-dir", "-o", default="./output", help="Output directory")
@click.option("--skip-finetuning", is_flag=True, help="Skip model finetuning")
@click.option("--finetune-fdr-threshold", type=float, default=0.01, help="FDR cutoff for selecting PSMs for finetuning")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def generate_features(
    report: str,
    ms_file_dir: str,
    ms_file_type: str,
    output_dir: str,
    skip_finetuning: bool,
    finetune_fdr_threshold: float,
    verbose: bool,
):
    """
    Generate rescoring features from DIA-NN results.
    """
    setup_logging(verbose)

    ms_files = find_ms_files(ms_file_dir, ms_file_type)
    if not ms_files:
        logger.error(f"No {ms_file_type} files found in {ms_file_dir}")
        raise SystemExit(1)

    io_config = IOConfig(
        report_file=report,
        ms_file_dir=ms_file_dir,
        ms_file_type=ms_file_type,
        output_dir=output_dir,
    )
    finetune_config = FineTuneConfig(fdr_threshold=finetune_fdr_threshold)

    pipeline = Pipeline(
        io_config=io_config,
        finetune_config=finetune_config,
    )
    pipeline.run_feature_generation(skip_finetuning=skip_finetuning)

    logger.info("Feature generation completed!")


@cli.command()
@click.option("--report", "-r", required=True, type=click.Path(exists=True), help="DIA-NN report parquet file")
@click.option(
    "--ms-file-dir",
    "-m",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing MS files",
)
@click.option(
    "--ms-file-type", type=click.Choice(["mzml", "hdf5"]), default="mzml", help="MS file extension to search for"
)
@click.option("--output-dir", "-o", default="./output", help="Output directory")
@click.option("--skip-finetuning", is_flag=True, help="Skip model finetuning")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def rescore(
    report: str,
    ms_file_dir: str,
    ms_file_type: str,
    output_dir: str,
    skip_finetuning: bool,
    verbose: bool,
):
    """
    Not implemented yet.
    """
    setup_logging(verbose)

    logger.warning("Rescoring not yet implemented. Running feature generation only.")

    ms_files = find_ms_files(ms_file_dir, ms_file_type)
    if not ms_files:
        click.echo(f"ERROR: No {ms_file_type} files found in {ms_file_dir}", err=True)
        raise SystemExit(1)

    io_config = IOConfig(
        report_file=report,
        ms_file_dir=ms_file_dir,
        ms_file_type=ms_file_type,
        output_dir=output_dir,
    )

    pipeline = Pipeline(io_config=io_config)
    pipeline.run(skip_finetuning=skip_finetuning)

    logger.info("Rescoring completed!")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML config file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(config: str, verbose: bool):
    """
    Run pipeline from YAML config file. Not implemented yet.
    """
    setup_logging(verbose)

    # TODO: Implement config file loading
    logger.error("Config file loading not yet implemented.")
    raise SystemExit(1)


@cli.command()
@click.option("--report", "-r", required=True, type=click.Path(exists=True), help="DIA-NN report parquet file")
@click.option("--speclib", "-s", required=True, type=click.Path(exists=True), help="Spectral library TSV file")
@click.option(
    "--ms-file-dir",
    "-m",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing MS files",
)
@click.option(
    "--ms-file-type", type=click.Choice(["mzml", "hdf5"]), default="mzml", help="MS file extension to search for"
)
@click.option("--output-dir", "-o", default="./output", help="Output directory")
@click.option("--output-format", type=click.Choice(["parquet", "csv"]), default="parquet", help="Output file format")
@click.option("--n-isotopes", type=int, default=3, help="Number of MS1 isotopes to extract")
@click.option("--rt-extension", type=float, default=30.0, help="RT window extension in seconds")
@click.option("--ppm-tolerance", type=float, default=15.0, help="m/z tolerance in ppm")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def extract_xic(
    report: str,
    speclib: str,
    ms_file_dir: str,
    ms_file_type: str,
    output_dir: str,
    output_format: str,
    n_isotopes: int,
    rt_extension: float,
    ppm_tolerance: float,
    verbose: bool,
):
    """
    Extract XIC traces for MS1 isotopes and MS2 transitions.
    Outputs one file per raw file with columns: precursor_id, level, mz, type, charge, rt, intensity.
    """
    setup_logging(verbose)

    from alpharaw import register_all_readers

    from dia_aspire_rescore.extraction import XICExtractor
    from dia_aspire_rescore.io import find_ms_files, read_diann2, read_speclib

    register_all_readers()

    ms_files = find_ms_files(ms_file_dir, ms_file_type)
    if not ms_files:
        logger.error(f"No {ms_file_type} files found in {ms_file_dir}")
        raise SystemExit(1)
    logger.info(f"Found {len(ms_files)} MS files")

    logger.info(f"Loading spectral library from {speclib}")
    precursor_df, transition_df = read_speclib(speclib)
    logger.info(f"Loaded {len(precursor_df)} precursors, {len(transition_df)} transitions")

    logger.info(f"Loading DIA-NN report from {report}")
    psm_df = read_diann2(report)
    logger.info(f"Loaded {len(psm_df)} PSMs")

    extractor = XICExtractor(
        precursor_df=precursor_df,
        transition_df=transition_df,
        n_isotopes=n_isotopes,
        rt_extension=rt_extension,
        ppm_tolerance=ppm_tolerance,
    )
    extractor.extract_all(
        psm_df=psm_df,
        ms_files=ms_files,
        ms_file_type=ms_file_type,
        output_dir=output_dir,
        output_format=output_format,
    )
    logger.info("XIC extraction completed.")


def main() -> None:
    """Entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def setup_logging(verbose: bool) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    verbose : bool
        If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    logging.getLogger("alphabase").setLevel(logging.ERROR)
    logging.getLogger("peptdeep").setLevel(logging.ERROR)
    logging.getLogger("alpharaw").setLevel(logging.ERROR)
    # Re-apply warning filters after logging setup (in case logging reset them)
    setup_warnings()


if __name__ == "__main__":
    main()
