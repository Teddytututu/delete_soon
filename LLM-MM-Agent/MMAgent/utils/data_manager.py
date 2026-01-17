r"""
Data Directory Manager - Step 0 Fix for CSV Path Chaos

PROBLEM: LLM generates code with full Windows paths like:
  pd.read_csv(r'C:\Users\...\data.csv')
  pd.read_csv('FULL_PATH_TO_DATA_FILE')

SOLUTION: Lock all CSV paths to a single DATA_DIR.
All code uses just filenames, system handles path resolution.

Author: Engineering Fix
Date: 2026-01-15
"""

import os
from pathlib import Path
from typing import List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """
    Centralized data file path manager.

    Enforces single DATA_DIR policy:
    - All CSV files live in DATA_DIR
    - Code generation only uses filenames
    - System resolves full paths at execution time
    """

    def __init__(self, dataset_dir: Optional[str] = None):
        """
        Initialize data manager with dataset directory.

        Args:
            dataset_dir: Path to dataset directory (absolute or relative)
        """
        if dataset_dir:
            self.data_dir = Path(dataset_dir).resolve()
        else:
            # Default: current directory
            self.data_dir = Path.cwd()

        logger.info(f"[DataManager] Initialized with DATA_DIR: {self.data_dir}")

        # Verify directory exists
        if not self.data_dir.exists():
            logger.warning(f"[DataManager] DATA_DIR does not exist: {self.data_dir}")

    def get_full_path(self, filename: str) -> Path:
        """
        Get full path for a data file.

        Args:
            filename: Just the filename (e.g., 'data.csv')

        Returns:
            Full path: DATA_DIR/filename
        """
        return self.data_dir / filename

    def file_exists(self, filename: str) -> bool:
        """
        Check if data file exists.

        Args:
            filename: Just the filename (e.g., 'data.csv')

        Returns:
            True if file exists in DATA_DIR
        """
        return self.get_full_path(filename).exists()

    def list_csv_files(self, fileset_filter: str = 'clean_only') -> List[str]:
        """
        List CSV files in DATA_DIR with optional filtering.

        P0-A FIX: Prevents mixing of three file sets (clean_*, cleaned_*, clean_summerOly_*).

        Args:
            fileset_filter: Which file set to return
                - 'clean_only': Only clean_*.csv files (DEFAULT - prevents confusion)
                - 'all': All CSV files (for backward compatibility)

        Returns:
            List of filenames (not full paths)
        """
        if not self.data_dir.exists():
            return []

        csv_files = []
        for f in self.data_dir.iterdir():
            if f.suffix == '.csv':
                # P0-A FIX: Filter to only use one file set
                if fileset_filter == 'clean_only':
                    # Only allow clean_*.csv, reject cleaned_* and clean_summerOly_*
                    if f.name.startswith('clean_') and not f.name.startswith('cleaned_') and not f.name.startswith('clean_summerOly_'):
                        csv_files.append(f.name)
                else:
                    # Return all CSV files
                    csv_files.append(f.name)

        return sorted(csv_files)

    def validate_files_exist(self, required_files: List[str]) -> tuple[bool, List[str]]:
        """
        Validate that required CSV files exist.

        Args:
            required_files: List of filenames to check

        Returns:
            (all_exist, missing_files)
        """
        missing = []
        for filename in required_files:
            if not self.file_exists(filename):
                missing.append(filename)

        return len(missing) == 0, missing

    def get_read_csv_code(self, filename: str, encoding: str = 'latin-1') -> str:
        """
        Generate pandas read_csv code for this file.

        IMPORTANT: Returns code that uses ONLY filename, not full path.
        The actual path resolution happens via sys.path or working directory.

        Args:
            filename: Just the filename
            encoding: CSV encoding (default: latin-1 for Windows compatibility)

        Returns:
            Python code string: pd.read_csv('filename', encoding='...')
        """
        return f"pd.read_csv('{filename}', encoding='{encoding}')"

    def sanitize_llm_code(self, code: str) -> str:
        r"""
        Sanitize LLM-generated code to remove absolute paths.

        Finds patterns like:
          pd.read_csv(r'C:\Users\...\file.csv')
          pd.read_csv('C:/Users/.../file.csv')

        Replaces with:
          pd.read_csv('file.csv')

        Args:
            code: LLM-generated Python code

        Returns:
            Sanitized code with only filenames
        """
        import re

        # Pattern 1: pd.read_csv(r'C:\Users\...\file.csv')
        # Extract filename and replace
        pattern = r"pd\.read_csv\(['\"]r?[\"']?[A-Za-z]:[\\\\/][^\"']+[/\\]?([^\"'\\]+)[\"']?['\"]\)"
        code = re.sub(pattern, lambda m: f"pd.read_csv('{m.group(1)}')", code)

        # Pattern 2: pd.read_csv('FULL_PATH_TO_xxx')
        code = re.sub(r"pd\.read_csv\(['\"]FULL_PATH_TO_[^'\"]+['\"]\)", "pd.read_csv('DATA_FILE')", code)

        return code

    def get_available_columns(self, filename: str, nrows: int = 0) -> Optional[List[str]]:
        """
        Get column names from a CSV file.

        Args:
            filename: Just the filename
            nrows: Number of rows to read (0 = just header)

        Returns:
            List of column names, or None if file doesn't exist
        """
        if not self.file_exists(filename):
            logger.warning(f"[DataManager] File not found: {filename}")
            return None

        try:
            import pandas as pd
            df = pd.read_csv(self.get_full_path(filename), encoding='latin-1', nrows=nrows)
            return list(df.columns)
        except Exception as e:
            logger.error(f"[DataManager] Failed to read {filename}: {e}")
            return None

    def preflight_check(self, required_files: Optional[List[str]] = None) -> dict:
        """
        Perform pre-flight check on data directory.

        Checks:
        1. DATA_DIR exists
        2. Required CSV files exist
        3. Can read headers from CSV files

        Args:
            required_files: Optional list of required filenames

        Returns:
            Dictionary with check results
        """
        result = {
            'data_dir_exists': self.data_dir.exists(),
            'data_dir_path': str(self.data_dir),
            'csv_files': [],
            'missing_files': [],
            'readable_files': [],
            'unreadable_files': []
        }

        if not result['data_dir_exists']:
            return result

        # List all CSV files
        result['csv_files'] = self.list_csv_files()

        # Check required files
        if required_files:
            all_exist, missing = self.validate_files_exist(required_files)
            result['missing_files'] = missing

        # Test readability
        for filename in result['csv_files']:
            columns = self.get_available_columns(filename, nrows=0)
            if columns is not None:
                result['readable_files'].append({
                    'filename': filename,
                    'columns': columns,
                    'column_count': len(columns)
                })
            else:
                result['unreadable_files'].append(filename)

        return result


# Global instance (will be initialized in main.py)
_data_manager: Optional[DataManager] = None


def init_data_manager(dataset_dir: Optional[str] = None) -> DataManager:
    """
    Initialize global data manager.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        DataManager instance
    """
    global _data_manager
    _data_manager = DataManager(dataset_dir)
    return _data_manager


def get_data_manager() -> Optional[DataManager]:
    """
    Get global data manager instance.

    Returns:
        DataManager instance or None if not initialized
    """
    return _data_manager

# ============================================================================
# P0-A FIX: Unified Column Name Standardization
# ============================================================================
# PROBLEM: Column names are inconsistent across CSV files:
#   - 'Gold' vs 'GOLD' vs 'gold' (case mismatch)
#   - 'Year' vs 'YEAR' vs 'ï»¿Year' vs 'Ï»¿YEAR' (BOM + case)
#   - 'Event' vs 'EVENT' (case mismatch)
# This causes 90% of KeyError errors in Task1/Task2/Task3.
#
# SOLUTION: Force all column names through standardization:
#   1. Remove BOM characters (ï»¿, Ï»¿)
#   2. Strip whitespace
#   3. Convert to UPPERCASE
#   4. Result: All code uses YEAR, GOLD, SILVER, BRONZE, TOTAL, HOST, etc.
# ============================================================================

def standardize_column_names(df):
    """
    P0-A FIX: Standardize DataFrame column names to prevent KeyError.

    This function:
    1. Removes BOM (Byte Order Mark) characters from column names
    2. Strips leading/trailing whitespace
    3. Converts all column names to UPPERCASE
    4. Returns the modified DataFrame

    Args:
        df: pandas DataFrame with potentially messy column names

    Returns:
        Modified DataFrame with standardized column names

    Example:
        BEFORE: ['Year', 'Gold', 'Silver', 'Bronze', 'Total']
        AFTER:  ['YEAR', 'GOLD', 'SILVER', 'BRONZE', 'TOTAL']

        BEFORE: ['ï»¿Year', 'Ï»¿YEAR', ' Event ', 'Event']
        AFTER:  ['YEAR', 'YEAR', 'EVENT', 'EVENT']
    """
    import pandas as pd

    # BOM characters that appear in CSV files (UTF-8, UTF-16, etc.)
    BOM_MARKERS = [
        '\ufeff',      # UTF-8 BOM (Zero Width No-Break Space)
        '\ufffe',      # UTF-16 BOM
        '\ufeff',      # UTF-8 BOM (alternative representation)
        'ï»¿',         # UTF-8 BOM (mojibake representation)
        'Ï»¿',         # UTF-8 BOM (uppercase mojibake)
        '\xfe\xff',     # UTF-16 BE BOM
        '\xff\xfe',     # UTF-16 LE BOM
    ]

    def clean_column_name(col_name):
        """
        Clean a single column name.

        Steps:
        1. Convert to string
        2. Remove all BOM markers
        3. Strip whitespace
        4. Convert to UPPERCASE
        """
        # Convert to string (in case col_name is not a string)
        col_str = str(col_name)

        # Remove BOM markers
        for bom in BOM_MARKERS:
            col_str = col_str.replace(bom, '')

        # Strip whitespace and convert to uppercase
        col_str = col_str.strip().upper()

        return col_str

    # Apply standardization to all columns
    old_columns = list(df.columns)
    new_columns = [clean_column_name(col) for col in df.columns]

    # Rename columns
    df.columns = new_columns

    # Log the transformation (if there were changes)
    if old_columns != new_columns:
        logger_col = logging.getLogger(__name__)
        logger_col.info(f"[P0-A FIX] Column name standardization applied:")
        for old, new in zip(old_columns, new_columns):
            if old != new:
                logger_col.info(f"  '{old}' → '{new}'")

    return df


def load_csv_with_standardized_columns(filepath, **kwargs):
    """
    P0-A FIX: Load CSV file with automatic column name standardization.

    This is a drop-in replacement for pd.read_csv() that automatically
    standardizes column names after loading.

    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        DataFrame with standardized column names

    Example:
        # OLD (causes KeyError):
        df = pd.read_csv('file.csv')
        # Returns: ['Year', 'Gold', 'Silver']  # Inconsistent case

        # NEW (prevents KeyError):
        df = load_csv_with_standardized_columns('file.csv')
        # Returns: ['YEAR', 'GOLD', 'SILVER']  # Consistent UPPERCASE
    """
    import pandas as pd

    # Use utf-8-sig encoding to auto-remove BOM during reading
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8-sig'

    # Read CSV
    df = pd.read_csv(filepath, **kwargs)

    # Standardize column names
    df = standardize_column_names(df)

    return df


# End of P0-A FIX
# ============================================================================
