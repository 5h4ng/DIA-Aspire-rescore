"""Command-line interface for DIA-Aspire-Rescore."""

import logging
import sys

import click

from dia_aspire_rescore.config import FineTuneConfig, IOConfig
from dia_aspire_rescore.pipeline import Pipeline, find_ms_files

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


def main():
    cli()


def setup_logging(verbose: bool):
    # Default to INFO level to show pipeline progress
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    logging.getLogger("alphabase").setLevel(logging.WARNING)
    logging.getLogger("peptdeep").setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
