"""
Main runner for data update scripts.

Usage:
    uv run python -m data.scripts [--only gpus|models|quants] [--dry-run] [-v]

Examples:
    uv run python -m data.scripts              # Update all data files
    uv run python -m data.scripts --only gpus  # Update only gpus.json
    uv run python -m data.scripts --dry-run    # Parse without writing files
    uv run python -m data.scripts -v           # Verbose logging
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .update_gpus import main as update_gpus
from .update_models import main as update_models
from .update_quants import main as update_quants

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_update(
    name: str,
    update_fn: Any,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run a single update function with error handling.

    Args:
        name: Human-readable name for logging
        update_fn: Update function to call
        dry_run: If True, parse but don't write

    Returns:
        Result dict from update function
    """
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Updating {name}...")

    try:
        result = update_fn(dry_run=dry_run)
        if result["success"]:
            logger.info(f"✓ {name}: {result['count']} entries")
        else:
            logger.error(f"✗ {name} failed: {result['errors']}")
        return result
    except Exception as e:
        logger.exception(f"✗ {name} crashed: {e}")
        return {
            "success": False,
            "count": 0,
            "errors": [str(e)],
            "warnings": [],
        }


def main() -> int:
    """Main entry point for the data update runner."""
    parser = argparse.ArgumentParser(
        description="Regenerate data JSON files from authoritative sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m data.scripts              Update all data files
  uv run python -m data.scripts --only gpus  Update only gpus.json
  uv run python -m data.scripts --dry-run    Parse without writing
  uv run python -m data.scripts -v           Verbose debug logging
        """,
    )

    parser.add_argument(
        "--only",
        choices=["gpus", "models", "quants"],
        help="Update only a specific data file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse data without writing files",
    )

    parser.add_argument(
        "--output-matching-report",
        type=Path,
        metavar="PATH",
        help="Path to write benchmark matching report (models only)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("can-I-run Data Update Pipeline")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be written")

    # Define update tasks
    updates = {
        "gpus": ("GPUs", update_gpus),
        "models": ("Models", update_models),
        "quants": ("Quantizations", update_quants),
    }

    # Filter to specific update if requested
    if args.only:
        updates = {args.only: updates[args.only]}
        logger.info(f"Running only: {args.only}")

    # Run updates
    results: dict[str, dict[str, Any]] = {}
    all_success = True

    for key, (name, update_fn) in updates.items():
        result = run_update(name, update_fn, dry_run=args.dry_run)
        results[key] = result
        if not result["success"]:
            all_success = False

    # Handle matching report output
    if args.output_matching_report and "models" in results:
        report_path = DATA_DIR / "benchmark_matching_report.json"
        if report_path.exists():
            try:
                # Copy to requested location
                with open(report_path) as f:
                    report_data = json.load(f)
                with open(args.output_matching_report, "w") as f:
                    json.dump(report_data, f, indent=2)
                logger.info(f"Matching report copied to: {args.output_matching_report}")
            except Exception as e:
                logger.warning(f"Failed to copy matching report: {e}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    fallbacks_used = []
    for key, result in results.items():
        status = "✓" if result["success"] else "✗"
        count = result["count"]
        errors = len(result.get("errors", []))
        warnings = len(result.get("warnings", []))
        used_fallback = result.get("used_fallback", False)

        status_parts = [f"{count} entries"]
        if errors:
            status_parts.append(f"{errors} errors")
        if warnings:
            status_parts.append(f"{warnings} warnings")
        if used_fallback:
            status_parts.append("FALLBACK USED")
            fallbacks_used.append(key)

        logger.info(f"  {status} {key}: {', '.join(status_parts)}")

    # Warn loudly about fallback usage
    if fallbacks_used:
        logger.warning("")
        logger.warning("=" * 60)
        logger.warning("⚠️  FALLBACK DATA USED - REVIEW REQUIRED")
        logger.warning("=" * 60)
        for key in fallbacks_used:
            logger.warning(f"  - {key}: Using hardcoded defaults instead of live data")
            for warning in results[key].get("warnings", []):
                logger.warning(f"    {warning}")
        logger.warning("=" * 60)

    if all_success:
        logger.info("")
        if fallbacks_used:
            logger.warning("Updates completed but with fallback data used!")
        else:
            logger.info("All updates completed successfully!")
        return 0
    else:
        logger.error("")
        logger.error("Some updates failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
