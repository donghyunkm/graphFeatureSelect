#!/usr/bin/env python
"""
Data Processing Script - Concatenate h5ad files

This script concatenates all h5ad files from data/raw/ into two files:
- 638850-metadata.h5ad (from all metadata/*.h5ad files)
- 638850-sp-self.h5ad (from all sp-self/*.h5ad files)

Usage:
    python 00_dataproc.py --test      # Test with 5 files per category
    python 00_dataproc.py             # Process all files
"""

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import anndata as ad

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Allow writing nullable strings in anndata
ad.settings.allow_write_nullable_strings = True


def setup_logging(log_dir: Path, test_mode: bool = False) -> logging.Logger:
    """
    Setup logging with both file and console handlers.

    Args:
        log_dir: Directory to save log files
        test_mode: If True, append '-test' to log filename

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"dataproc_{timestamp}" + ("_test" if test_mode else "")
    log_file = log_dir / f"{log_name}.log"

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - less verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")
    logger.debug(f"Logger initialized at {datetime.now()}")

    return logger


def find_h5ad_files(base_dir: Path, pattern: str, logger: logging.Logger) -> List[Path]:
    """
    Recursively find all h5ad files matching the pattern.

    Args:
        base_dir: Base directory to search
        pattern: Pattern to match (e.g., "**/metadata/*.h5ad")
        logger: Logger instance

    Returns:
        Sorted list of paths to h5ad files
    """
    logger.debug(f"Searching for files: {base_dir}/{pattern}")
    start_time = time.time()
    files = sorted(base_dir.glob(pattern))
    elapsed = time.time() - start_time
    logger.debug(f"Found {len(files)} files in {elapsed:.2f}s")

    if files:
        logger.debug(f"First file: {files[0]}")
        logger.debug(f"Last file: {files[-1]}")

    return files


def concatenate_h5ad_files(files: List[Path], output_path: Path, label: str, logger: logging.Logger):
    """
    Concatenate multiple h5ad files into a single file.

    Args:
        files: List of paths to h5ad files
        output_path: Path to save concatenated file
        label: Label for logging (e.g., "metadata" or "sp-self")
        logger: Logger instance
    """
    if not files:
        logger.warning(f"No files found for {label}")
        return

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing {label}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Found {len(files)} files")
    logger.debug(f"First file: {files[0]}")
    logger.debug(f"Last file: {files[-1]}")

    # Read and concatenate all files
    adatas = []
    failed_files = []
    start_time = time.time()

    for i, file in enumerate(files, 1):
        if i % 100 == 0 or i == len(files):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            logger.info(f"Reading file {i}/{len(files)}: {file.name} ({rate:.1f} files/s)")

        logger.debug(f"Reading: {file}")
        try:
            file_start = time.time()
            adata = ad.read_h5ad(file)
            file_elapsed = time.time() - file_start

            logger.debug(f"  Shape: {adata.shape}, Time: {file_elapsed:.2f}s")
            logger.debug(f"  obs keys: {list(adata.obs.keys())[:5]}")
            logger.debug(f"  var keys: {list(adata.var.keys())[:5] if hasattr(adata, 'var') else []}")

            adatas.append(adata)

        except Exception as e:
            logger.error(f"Error reading {file.name}: {type(e).__name__}: {e}")
            logger.debug(f"Full path: {file}", exc_info=True)
            failed_files.append((file, str(e)))
            continue

    read_time = time.time() - start_time
    logger.info(f"Reading complete: {len(adatas)}/{len(files)} files in {read_time:.2f}s")

    if failed_files:
        logger.warning(f"Failed to read {len(failed_files)} files:")
        for failed_file, error in failed_files[:10]:  # Log first 10 failures
            logger.warning(f"  {failed_file.name}: {error}")
        if len(failed_files) > 10:
            logger.warning(f"  ... and {len(failed_files) - 10} more")

    if not adatas:
        logger.error(f"No valid files could be read for {label}")
        return

    # Log shapes before concatenation
    logger.debug("Individual AnnData shapes:")
    for i, adata in enumerate(adatas[:5]):
        logger.debug(f"  [{i}] {adata.shape}")
    if len(adatas) > 5:
        logger.debug(f"  ... and {len(adatas) - 5} more")

    logger.info(f"\nConcatenating {len(adatas)} AnnData objects...")
    concat_start = time.time()

    # Concatenate along observation axis (cells)
    try:
        adata_combined = ad.concat(adatas, axis=0, join="outer", merge="unique")
        concat_time = time.time() - concat_start
        logger.info(f"Concatenation complete in {concat_time:.2f}s")
    except Exception as e:
        logger.error(f"Error during concatenation: {type(e).__name__}: {e}", exc_info=True)
        return

    logger.info(f"Combined shape: {adata_combined.shape}")
    logger.info(f"  n_obs (cells): {adata_combined.n_obs}")
    logger.info(f"  n_vars (genes): {adata_combined.n_vars}")

    # Log combined data info
    logger.debug(f"Combined obs columns: {list(adata_combined.obs.columns)}")
    logger.debug(f"Combined var columns: {list(adata_combined.var.columns) if hasattr(adata_combined, 'var') else []}")

    # Memory usage estimate
    try:
        memory_mb = adata_combined.X.nbytes / 1024**2 if hasattr(adata_combined.X, "nbytes") else 0
        logger.debug(f"Estimated memory usage: {memory_mb:.2f} MB")
    except:
        pass

    # Save to output file
    logger.info(f"\nSaving to {output_path}...")
    save_start = time.time()

    try:
        adata_combined.write_h5ad(output_path)
        save_time = time.time() - save_start
        file_size_mb = output_path.stat().st_size / 1024**2
        logger.info(f"Successfully saved in {save_time:.2f}s!")
    except Exception as e:
        logger.error(f"Error saving file: {type(e).__name__}: {e}", exc_info=True)
        return

    # Print summary statistics
    total_time = time.time() - start_time
    logger.info(f"\nSummary:")
    logger.info(f"  Total files processed: {len(adatas)}/{len(files)}")
    logger.info(f"  Failed files: {len(failed_files)}")
    logger.info(f"  Output file: {output_path}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Read time: {read_time:.2f}s")
    logger.info(f"  Concat time: {concat_time:.2f}s")
    logger.info(f"  Save time: {save_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Concatenate h5ad files")
    parser.add_argument("--test", action="store_true", help="Test mode: only process first 5 files per category")
    parser.add_argument(
        "--raw-dir", type=str, default="../data/raw", help="Path to raw data directory (default: ../data/raw)"
    )
    parser.add_argument("--output-dir", type=str, default="../data", help="Path to output directory (default: ../data)")
    parser.add_argument(
        "--log-dir", type=str, default="../data/logs", help="Path to log directory (default: ../data/logs)"
    )
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    raw_dir = (script_dir / args.raw_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    log_dir = (script_dir / args.log_dir).resolve()

    # Setup logging
    logger = setup_logging(log_dir, test_mode=args.test)

    logger.info("=" * 60)
    logger.info("Data Processing Script - h5ad Concatenation")
    logger.info("=" * 60)
    logger.info(f"Script started at: {datetime.now()}")
    logger.info(f"Command line args: {' '.join(sys.argv[1:])}")
    logger.info(f"Test mode: {args.test}")
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"AnnData version: {ad.__version__}")

    logger.info(f"\nPaths:")
    logger.info(f"  Raw data directory: {raw_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Log directory: {log_dir}")

    if not raw_dir.exists():
        logger.error(f"Raw data directory does not exist: {raw_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all h5ad files
    logger.info("\nSearching for h5ad files...")
    metadata_files = find_h5ad_files(raw_dir, "**/metadata/*.h5ad", logger)
    sp_self_files = find_h5ad_files(raw_dir, "**/sp-self/*.h5ad", logger)

    logger.info(f"\nFound {len(metadata_files)} metadata files")
    logger.info(f"Found {len(sp_self_files)} sp-self files")

    # Test mode: use only first 5 files
    if args.test:
        logger.info("\n" + "=" * 60)
        logger.info("TEST MODE: Processing only first 5 files per category")
        logger.info("=" * 60)
        metadata_files = metadata_files[:5]
        sp_self_files = sp_self_files[:5]
        logger.debug(f"Metadata files: {[f.name for f in metadata_files]}")
        logger.debug(f"Sp-self files: {[f.name for f in sp_self_files]}")

    # Overall timing
    script_start = time.time()

    # Process metadata files
    if metadata_files:
        output_path = output_dir / "638850-metadata.h5ad"
        if args.test:
            output_path = output_dir / "638850-metadata-test.h5ad"
        logger.debug(f"Output path for metadata: {output_path}")
        concatenate_h5ad_files(metadata_files, output_path, "metadata", logger)
    else:
        logger.warning("No metadata files found to process")

    # Process sp-self files
    if sp_self_files:
        output_path = output_dir / "638850-sp-self.h5ad"
        if args.test:
            output_path = output_dir / "638850-sp-self-test.h5ad"
        logger.debug(f"Output path for sp-self: {output_path}")
        concatenate_h5ad_files(sp_self_files, output_path, "sp-self", logger)
    else:
        logger.warning("No sp-self files found to process")

    total_time = time.time() - script_start

    logger.info("\n" + "=" * 60)
    logger.info("Processing complete!")
    logger.info("=" * 60)
    logger.info(f"Total script time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    logger.info(f"Script ended at: {datetime.now()}")


if __name__ == "__main__":
    main()
