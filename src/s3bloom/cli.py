"""Typer-based command-line interface for the s3bloom pipeline.

Defines two commands:

* ``s3bloom run`` — executes the full pipeline (search → download →
  per-pass processing → compositing).
* ``s3bloom list-presets`` — prints the available bounding-box and
  masking presets.

The CLI is intentionally a *thin* layer: it parses arguments, builds a
:class:`s3bloom.config.PipelineConfig`, and orchestrates calls into the
search / download / processing / compositing modules. Anything that is
reusable beyond the CLI lives in those modules, not here.

Heavy imports (satpy, matplotlib, …) are deferred to inside ``run`` so
that ``s3bloom --help`` and ``s3bloom list-presets`` stay fast.
"""

from __future__ import annotations

import gc
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Load CDSE_USERNAME / CDSE_PASSWORD before any submodule that might
# need them is imported.
load_dotenv()

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from s3bloom import __version__
from s3bloom.config import (
    BoundingBox,
    MaskingConfig,
    OutputConfig,
    PipelineConfig,
    TimePeriod,
)
from s3bloom.defaults import (
    BBOX_PRESETS,
    DEFAULT_DATASETS,
    DEFAULT_MASKING,
    MASKING_PRESETS,
    PRODUCT_FLAGS,
    get_masking_flags,
)

app = typer.Typer(
    name="s3bloom",
    help="Sentinel-3 OLCI L2 phytoplankton bloom pipeline.",
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    """Eager ``--version`` callback: print and exit before anything else runs."""
    if value:
        console.print(f"s3bloom {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Sentinel-3 OLCI L2 phytoplankton bloom pipeline."""


@app.command()
def run(
    start_date: str = typer.Option(
        ...,
        "--start-date",
        "-s",
        help="Start date (YYYY-MM-DD).",
    ),
    end_date: str = typer.Option(
        ...,
        "--end-date",
        "-e",
        help="End date (YYYY-MM-DD).",
    ),
    bbox: str = typer.Option(
        ...,
        "--bbox",
        "-b",
        help=(
            "Bounding box: preset name or 'lon_min,lat_min,lon_max,lat_max'. "
            f"Presets: {', '.join(BBOX_PRESETS)}."
        ),
    ),
    masking: str = typer.Option(
        DEFAULT_MASKING,
        "--masking",
        "-m",
        help=f"Masking strictness: {', '.join(MASKING_PRESETS)}.",
    ),
    datasets: str = typer.Option(
        ",".join(DEFAULT_DATASETS),
        "--datasets",
        "-d",
        help="Comma-separated dataset names (e.g. chl_nn,chl_oc4me,tsm_nn).",
    ),
    output_dir: Path = typer.Option(
        Path("data"),
        "--output-dir",
        "-o",
        help="Base output directory.",
    ),
    projection: str = typer.Option(
        "EPSG:3035",
        "--projection",
        help="Target projection (EPSG code).",
    ),
    resolution: int = typer.Option(
        300,
        "--resolution",
        help="Target resolution in metres.",
    ),
    composite_window: int = typer.Option(
        3,
        "--composite-window",
        help="Composite window size in days.",
    ),
    formats: str = typer.Option(
        "geotiff,netcdf,png",
        "--formats",
        help="Comma-separated output formats.",
    ),
    no_composites: bool = typer.Option(
        False,
        "--no-composites",
        help="Skip composite generation.",
    ),
    delete_raw: bool = typer.Option(
        False,
        "--delete-raw",
        help="Delete .SEN3 directories after successful processing to free disk space.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-process and overwrite outputs even if they exist and are up to date.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Run the full bloom processing pipeline.

    Search → download → per-pass processing → optional compositing. Each
    stage prints a numbered banner so a long run can be followed in the
    terminal. Failures in one pass are logged and the next pass is
    attempted; a run completes successfully as long as at least one
    pass was processed or was skipped because its outputs were already
    current.

    Compositing always runs over every pass of the period, including
    the skipped ones, which are read back from their per-pass NetCDFs.
    A resumed run therefore builds the same composites as a single
    fresh run, and composites that are already up to date are left
    alone. This needs ``netcdf`` among the output formats; without it
    skipped passes cannot be read back and a warning is printed.
    """
    _setup_logging(verbose)

    try:
        config = _build_config(
            start_date=start_date,
            end_date=end_date,
            bbox_str=bbox,
            masking=masking,
            datasets=datasets,
            output_dir=output_dir,
            projection=projection,
            resolution=resolution,
            composite_window=composite_window,
            formats=formats,
            delete_raw=delete_raw,
            overwrite=overwrite,
        )
    except Exception as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_config_summary(config)
    config.ensure_directories()

    # Heavy modules (satpy, matplotlib via the export package) are
    # imported here so `--help` and `list-presets` stay fast.
    from s3bloom.compositing.temporal import create_composites
    from s3bloom.discovery.download import download_products
    from s3bloom.discovery.search import search_products
    from s3bloom.processing.pipeline import process_single_pass

    console.print("\n[bold blue]Step 1/4: Searching for products...[/bold blue]")
    products = search_products(config.bbox, config.time_period)

    if not products:
        console.print("[yellow]No products found for the given parameters.[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"Found [green]{len(products)}[/green] products.")

    console.print("\n[bold blue]Step 2/4: Downloading products...[/bold blue]")
    product_paths = download_products(products, config.output.raw_dir, config)

    if not product_paths:
        console.print("[red]No products were downloaded successfully.[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold blue]Step 3/4: Processing {len(product_paths)} passes...[/bold blue]"
    )
    # Processed and skipped passes alike: both take part in compositing.
    pass_results = []
    for i, product_path in enumerate(product_paths, 1):
        console.print(
            f"  [{i}/{len(product_paths)}] {product_path.name}"
        )
        try:
            result = process_single_pass(product_path, config)
            pass_results.append(result)
            if result.skipped:
                console.print("  [dim]Skipped (outputs exist)[/dim]")
            elif config.output.delete_raw:
                # Safe right away: the result reads from the per-pass
                # outputs (or holds the array), never from the product.
                shutil.rmtree(product_path)
                logging.getLogger(__name__).debug(
                    "Deleted raw product: %s", product_path.name
                )
        except Exception as exc:
            # One bad product must not abort the run — log and continue
            # so the user gets results from every product that worked.
            console.print(f"  [red]Failed: {exc}[/red]")
            logging.getLogger(__name__).exception(
                "Failed to process %s", product_path.name
            )
        finally:
            # The satpy scene of a pass (file handles, resampling
            # caches) is no longer referenced once the pass returns;
            # collect it between products to keep peak RSS low on long
            # runs.
            gc.collect()

    if not pass_results:
        console.print("[red]No passes were processed successfully.[/red]")
        raise typer.Exit(code=1)

    composite_files: list[Path] = []
    if no_composites:
        console.print("\n[dim]Skipping composite generation (--no-composites)[/dim]")
    else:
        console.print(
            f"\n[bold blue]Step 4/4: Creating {config.output.composite_window_days}-day "
            f"composites...[/bold blue]"
        )
        _warn_if_skipped_passes_lack_data(pass_results)
        composite_files = create_composites(pass_results, config)

    _print_summary(pass_results, composite_files, config)


@app.command()
def list_presets() -> None:
    """Print the bounding-box and per-product masking presets.

    Useful as a quick reference and as a sanity check that the masking
    layer is producing the flag list you expect for a given preset and
    dataset.
    """
    table = Table(title="Bounding Box Presets")
    table.add_column("Name", style="green")
    table.add_column("lon_min")
    table.add_column("lat_min")
    table.add_column("lon_max")
    table.add_column("lat_max")

    for name, coords in sorted(BBOX_PRESETS.items()):
        table.add_row(name, *[str(c) for c in coords])

    console.print(table)
    console.print()

    table2 = Table(title="Masking Presets (product-aware)")
    table2.add_column("Preset", style="green")
    table2.add_column("Product", style="cyan")
    table2.add_column("Flags")

    for preset in MASKING_PRESETS:
        for product in sorted(PRODUCT_FLAGS):
            flags = get_masking_flags(preset, product)
            table2.add_row(preset, product, ", ".join(flags))

    console.print(table2)


def _build_config(
    *,
    start_date: str,
    end_date: str,
    bbox_str: str,
    masking: str,
    datasets: str,
    output_dir: Path,
    projection: str,
    resolution: int,
    composite_window: int,
    formats: str,
    delete_raw: bool = False,
    overwrite: bool = False,
) -> PipelineConfig:
    """Translate raw CLI strings into a validated :class:`PipelineConfig`.

    All argument validation is delegated to the pydantic models (see
    :mod:`s3bloom.config`). This function's only job is shape-conversion
    (string → date, comma-list → list, etc.).
    """
    sd = date.fromisoformat(start_date)
    ed = date.fromisoformat(end_date)

    return PipelineConfig(
        bbox=BoundingBox.from_string(bbox_str),
        time_period=TimePeriod(start_date=sd, end_date=ed),
        datasets=[d.strip() for d in datasets.split(",")],
        masking=MaskingConfig(preset=masking),
        output=OutputConfig(
            base_dir=output_dir,
            formats=[f.strip() for f in formats.split(",")],
            projection=projection,
            resolution_m=resolution,
            composite_window_days=composite_window,
            delete_raw=delete_raw,
            overwrite=overwrite,
        ),
    )


def _print_config_summary(config: PipelineConfig) -> None:
    """Print the resolved configuration so the user can verify it."""
    console.print("\n[bold]Pipeline Configuration[/bold]")
    console.print(f"  Period: {config.time_period.start_date} to {config.time_period.end_date}")
    console.print(
        f"  BBox: ({config.bbox.lon_min}, {config.bbox.lat_min}) to "
        f"({config.bbox.lon_max}, {config.bbox.lat_max})"
    )
    console.print(f"  Datasets: {', '.join(config.datasets)}")
    for ds in config.datasets:
        ds_flags = config.masking.flags_for_product(ds)
        console.print(f"  Masking ({ds}): {config.masking.preset} ({len(ds_flags)} flags)")
    console.print(f"  Projection: {config.output.projection}")
    console.print(f"  Resolution: {config.output.resolution_m}m")
    console.print(f"  Composites: {config.output.composite_window_days}-day window")
    console.print(f"  Output: {config.output.base_dir}")


def _warn_if_skipped_passes_lack_data(pass_results: list) -> None:
    """Warn when skipped passes cannot take part in compositing.

    A skipped pass is read back from its per-pass NetCDF. Without
    ``netcdf`` among the output formats there is nothing to read:
    existing composites for those dates are kept and new ones are built
    from the passes processed in this run only.
    """
    without_data = sum(1 for pr in pass_results if pr.skipped and not pr.datasets)
    if not without_data:
        return
    console.print(
        f"[yellow]Warning: {without_data} skipped pass(es) have no per-pass "
        "NetCDF to read back. Existing composites for their dates are kept; "
        "new ones cover only the passes processed in this run. Add netcdf to "
        "--formats for resumable compositing.[/yellow]"
    )


def _print_summary(
    pass_results: list,
    composite_files: list[Path],
    config: PipelineConfig,
) -> None:
    """Print the final tally of passes and outputs at the end of a run.

    *pass_results* holds processed and skipped passes; they are counted
    separately. The file counts are the files written by this run, so
    skipped passes and composites that were already up to date do not
    add to them.
    """
    skipped_count = sum(1 for pr in pass_results if pr.skipped)
    total_pass_files = sum(len(pr.output_files) for pr in pass_results)
    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print(f"  Passes processed: {len(pass_results) - skipped_count}")
    if skipped_count:
        console.print(f"  Passes skipped (already existed): {skipped_count}")
    console.print(f"  Pass output files written: {total_pass_files}")
    console.print(f"  Composite files written: {len(composite_files)}")
    console.print(f"  Output directory: {config.output.base_dir}")


def _setup_logging(verbose: bool) -> None:
    """Configure rich-formatted logging for the CLI.

    In non-verbose mode, satpy/pyresample/trollsift are clamped to
    WARNING — they're chatty at INFO and drown out the pipeline's own
    progress output.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    if not verbose:
        logging.getLogger("satpy").setLevel(logging.WARNING)
        logging.getLogger("pyresample").setLevel(logging.WARNING)
        logging.getLogger("trollsift").setLevel(logging.WARNING)
