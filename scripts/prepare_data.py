#!/usr/bin/env python3
"""
Data preparation script for Atlas.

Extracts and prepares training data from various sources.
Currently supports Wikipedia SimpleEnglish dataset from Kaggle.

Usage:
    python scripts/prepare_data.py --input data/raw/archive.zip
    python scripts/prepare_data.py --input data/raw/archive.zip --output data/processed/wikipedia
    python scripts/prepare_data.py --list  # List available datasets
"""

import argparse
import logging
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """
    Extract a zip file to the specified output directory.
    
    Args:
        zip_path: Path to the zip file
        output_dir: Directory to extract to
    """
    logger.info(f"[EXTRACT] Extracting {zip_path.name}...")
    logger.info(f"          Output: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total size for progress
        total_files = len(zip_ref.namelist())
        logger.info(f"          Files: {total_files}")
        
        # Extract all files
        zip_ref.extractall(output_dir)
    
    logger.info(f"[SUCCESS] Extraction complete!")


def collect_text_files(source_dir: Path) -> List[Path]:
    """
    Recursively collect all .txt files from a directory.
    
    Args:
        source_dir: Directory to search
        
    Returns:
        List of paths to .txt files
    """
    txt_files = list(source_dir.rglob("*.txt"))
    logger.info(f"[SCAN] Found {len(txt_files)} text files")
    return txt_files


def prepare_wikipedia_simple(input_path: Path, output_dir: Path) -> None:
    """
    Prepare Wikipedia SimpleEnglish dataset for training.
    
    Args:
        input_path: Path to archive.zip
        output_dir: Directory to place processed data
    """
    logger.info("=" * 80)
    logger.info("PREPARING WIKIPEDIA SIMPLEENGLISH DATASET")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract the zip
    temp_extract = output_dir / "temp_extract"
    extract_zip(input_path, temp_extract)
    
    # Collect all text files
    txt_files = collect_text_files(temp_extract)
    
    if not txt_files:
        logger.error("[ERROR] No .txt files found in archive!")
        return
    
    # Move text files to output directory
    logger.info(f"[ORGANIZE] Organizing files...")
    for i, txt_file in enumerate(txt_files):
        # Create a clean filename
        dest_file = output_dir / f"wiki_{i:05d}.txt"
        shutil.move(str(txt_file), str(dest_file))
    
    # Clean up temp directory
    logger.info("[CLEANUP] Cleaning up temporary files...")
    shutil.rmtree(temp_extract)
    
    # Display statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"[INFO] Output directory: {output_dir}")
    logger.info(f"[INFO] Text files: {len(txt_files)}")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.txt"))
    size_mb = total_size / (1024 * 1024)
    logger.info(f"[INFO] Total size: {size_mb:.1f} MB")
    
    logger.info("")
    logger.info("[SUCCESS] Ready for training!")
    logger.info(f"          Use: python scripts/train.py --data {output_dir}")


def list_available_datasets(data_dir: Path) -> None:
    """
    List available datasets in the data directory.
    
    Args:
        data_dir: Root data directory
    """
    logger.info("=" * 80)
    logger.info("AVAILABLE DATASETS")
    logger.info("=" * 80)
    
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Check raw data
    logger.info("\n[RAW] Raw data (data/raw/):")
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*"))
        raw_files = [f for f in raw_files if f.is_file() and f.name != ".gitkeep"]
        
        if raw_files:
            for f in raw_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"      - {f.name} ({size_mb:.1f} MB)")
        else:
            logger.info("      (none found)")
    else:
        logger.info("      (directory not found)")
    
    # Check processed data
    logger.info("\n[PROCESSED] Processed data (data/processed/):")
    if processed_dir.exists():
        processed_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and d.name != ".gitkeep"]
        
        if processed_dirs:
            for d in processed_dirs:
                txt_files = list(d.glob("*.txt"))
                if txt_files:
                    total_size = sum(f.stat().st_size for f in txt_files)
                    size_mb = total_size / (1024 * 1024)
                    logger.info(f"            - {d.name}/ ({len(txt_files)} files, {size_mb:.1f} MB)")
        else:
            logger.info("            (none found)")
    else:
        logger.info("            (directory not found)")
    
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare Wikipedia SimpleEnglish dataset
  python scripts/prepare_data.py --input data/raw/archive.zip
  
  # Specify custom output directory
  python scripts/prepare_data.py --input data/raw/archive.zip --output data/processed/my_wiki
  
  # List available datasets
  python scripts/prepare_data.py --list
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file (e.g., data/raw/archive.zip)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: data/processed/wikipedia)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    # Handle --list flag
    if args.list:
        list_available_datasets(data_dir)
        return
    
    # Validate input
    if not args.input:
        parser.error("--input is required (or use --list to see available datasets)")
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"[ERROR] Input file not found: {input_path}")
        logger.info(f"\n[TIP] Place your dataset in data/raw/ first")
        return
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Auto-detect based on input filename
        if "wikipedia" in input_path.name.lower() or "archive" in input_path.name.lower():
            output_dir = data_dir / "processed" / "wikipedia"
        else:
            output_dir = data_dir / "processed" / input_path.stem
    
    # Process based on file type
    if input_path.suffix == ".zip":
        prepare_wikipedia_simple(input_path, output_dir)
    else:
        logger.error(f"[ERROR] Unsupported file type: {input_path.suffix}")
        logger.info("        Supported: .zip")
        return


if __name__ == "__main__":
    main()
