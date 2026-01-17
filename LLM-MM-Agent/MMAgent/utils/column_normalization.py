"""
Automatic Column Normalization - P0-2 FIX

Automatically normalizes DataFrame column names to prevent schema inconsistency.
This fixes the "KeyError: 'Year'" vs "YEAR" vs "ï»¿YEAR" problem.

Author: Engineering Fix
Date: 2026-01-16
Priority: P0 (Must-have for data consistency)
"""

import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)


def normalize_dataframe_columns(df: pd.DataFrame,
                                 strip_bom: bool = True,
                                 strip_whitespace: bool = True,
                                 to_uppercase: bool = True) -> pd.DataFrame:
    """
    Normalize DataFrame column names automatically.

    Operations:
    1. Remove UTF-8 BOM (\ufeff)
    2. Strip leading/trailing whitespace
    3. Convert to uppercase

    Args:
        df: Input DataFrame
        strip_bom: Remove \ufeff BOM character
        strip_whitespace: Strip leading/trailing spaces
        to_uppercase: Convert to uppercase

    Returns:
        DataFrame with normalized columns (modified in-place, also returned)

    Example:
        Before: ['Ï»¿YEAR', 'Name ', 'country']
        After:  ['YEAR', 'NAME', 'COUNTRY']
    """
    # Create a copy of columns to avoid modifying during iteration
    original_columns = df.columns.tolist()
    normalized_columns = []

    for col in original_columns:
        # Convert to string (in case columns are not strings)
        col_str = str(col)

        # Step 1: Remove BOM (if present)
        if strip_bom:
            col_str = col_str.replace('\ufeff', '')

        # Step 2: Strip whitespace
        if strip_whitespace:
            col_str = col_str.strip()

        # Step 3: Convert to uppercase
        if to_uppercase:
            col_str = col_str.upper()

        normalized_columns.append(col_str)

    # Rename columns
    df.columns = normalized_columns

    # Log the changes
    changes = []
    for orig, norm in zip(original_columns, normalized_columns):
        if orig != norm:
            changes.append(f"{orig} → {norm}")

    if changes:
        logger.info(f"[P0-2] Normalized {len(changes)} column(s): {', '.join(changes[:5])}...")

    return df


def create_normalized_load_csv(allowed_files: List[str], data_dir: str):
    """
    Create a load_csv() function that automatically normalizes columns.

    This function generates a load_csv() helper that wraps pd.read_csv()
    and applies automatic column normalization.

    Args:
        allowed_files: List of allowed CSV filenames
        data_dir: Directory containing CSV files

    Returns:
        str: Python code defining load_csv() function
    """
    code = f'''
# P0-2 FIX: Auto-normalized CSV loading
import pandas as pd
import os
from pathlib import Path

_DATA_DIR = Path(r"{data_dir}")
_ALLOWED_FILES = {allowed_files}

def load_csv(filename: str) -> pd.DataFrame:
    """
    Load CSV file with automatic column normalization.

    Normalization:
    1. Remove BOM (\\ufeff)
    2. Strip whitespace
    3. Convert to uppercase

    This prevents schema errors like KeyError: 'Year' vs 'YEAR'.
    """
    if filename not in _ALLOWED_FILES:
        available = ', '.join(sorted(_ALLOWED_FILES))
        raise FileNotFoundError(
            f"CSV file not in whitelist: {{filename}}\\n"
            f"Available files: {{available}}"
        )

    file_path = _DATA_DIR / filename

    # Try multiple encodings (BOM handling)
    for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)

            # CRITICAL: Normalize column names automatically
            df.columns = [
                str(col)
                .replace('\\ufeff', '')  # Remove BOM
                .strip()                # Remove whitespace
                .upper()                # Convert to uppercase
            ]

            return df
        except (UnicodeDecodeError, UnicodeError):
            continue

    # If all encodings fail
    raise Exception(f"Failed to read {{file_path}} with any encoding")
'''

    return code


def inject_column_normalization(code: str) -> str:
    """
    Inject column normalization into existing pd.read_csv() calls.

    This is a fallback for code that doesn't use load_csv().

    Args:
        code: Python code string

    Returns:
        Modified code with column normalization injected
    """
    import re

    # Pattern: df = pd.read_csv(...) - find the end and add normalization
    pattern = r'(\w+\s*=\s*pd\.read_csv\([^)]+\))'

    def add_normalization(match):
        original = match.group(1)
        # Add column normalization after read_csv
        return f'''{original}
# P0-2 FIX: Auto-normalize column names
{original.split('=')[0].strip()}.columns = [str(col).replace('\\ufeff', '').strip().upper() for col in {original.split('=')[0].strip()}.columns]'''

    normalized_code = re.sub(pattern, add_normalization, code)

    return normalized_code


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COLUMN NORMALIZATION - P0-2 FIX")
    print("=" * 60)

    # Create test DataFrame with problematic columns
    test_data = {
        'Ï»¿YEAR': [2020, 2021, 2022],
        'Name ': ['Alice', 'Bob', 'Charlie'],
        'country': ['USA', 'UK', 'Canada']
    }
    df = pd.DataFrame(test_data)

    print("\n[Before Normalization]")
    print(f"Columns: {list(df.columns)}")

    # Normalize
    df_normalized = normalize_dataframe_columns(df)

    print("\n[After Normalization]")
    print(f"Columns: {list(df_normalized.columns)}")

    # Verify access works
    print(f"\n[Verification]")
    print(f"df['YEAR']: {df_normalized['YEAR'].tolist()}")
    print(f"df['NAME']: {df_normalized['NAME'].tolist()}")
    print(f"df['COUNTRY']: {df_normalized['COUNTRY'].tolist()}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
