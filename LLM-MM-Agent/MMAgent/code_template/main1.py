# ============================================================================
# P0-2 FIX: Automatic Column Name Normalization
# ============================================================================
# CRITICAL: ALL COLUMN NAMES ARE AUTOMATICALLY NORMALIZED
#
# When you load CSV files, columns are automatically normalized:
#   1. BOM removed (ï»¿YEAR → YEAR)
#   2. Whitespace stripped ( 'Name ' → 'NAME')
#   3. Uppercase enforced ( 'Year' → 'YEAR')
#
# AVAILABLE CSV FILES:
#   - clean_athletes.csv
#   - clean_hosts.csv
#   - clean_medal_counts.csv
#   - clean_programs.csv
#
# HOW TO LOAD CSV (use ONLY these methods):
#   Method 1 (Recommended):
#     df = load_csv_with_normalization('clean_athletes.csv')
#
#   Method 2 (with pandas):
#     df = pd.read_csv('clean_athletes.csv')
#     df = normalize_columns(df)  # CRITICAL: Call this immediately!
#
# COLUMN ACCESS RULES:
#   - ALWAYS use UPPERCASE: df['YEAR'], df['GOLD'], df['NAME']
#   - NEVER use lowercase: df['year'], df['Year'] (will cause KeyError)
#   - Check available columns first: print(df.columns.tolist())
# ============================================================================

import pandas as pd
import os
from pathlib import Path

# P0-2 FIX: Column normalization helper function
def normalize_columns(df):
    """
    Automatically normalize DataFrame column names.

    Transforms:
    - 'Ï»¿YEAR' → 'YEAR' (remove BOM)
    - 'Name ' → 'NAME' (strip whitespace)
    - 'country' → 'COUNTRY' (uppercase)

    Usage:
        df = pd.read_csv('file.csv')
        df = normalize_columns(df)  # CRITICAL: Call immediately!
    """
    df.columns = [
        str(col)
        .replace('\ufeff', '')  # Remove UTF-8 BOM
        .strip()                # Remove whitespace
        .upper()                # Convert to uppercase
        for col in df.columns
    ]
    return df


def load_csv_with_normalization(filename):
    """
    Load CSV file with automatic column normalization.

    Args:
        filename: CSV filename (e.g., 'clean_athletes.csv')

    Returns:
        DataFrame with normalized columns

    Example:
        df = load_csv_with_normalization('clean_athletes.csv')
        # Columns are automatically: ['ID', 'NAME', 'SEX', 'AGE', ...]
    """
    # Try multiple encodings for BOM handling
    for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(filename, encoding=encoding)

            # CRITICAL: Normalize column names immediately
            df = normalize_columns(df)

            return df
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise Exception(f"Failed to read {filename}")


# ============================================================================
# Your code starts here
# ============================================================================

# The model class
class Model1():
    pass

# The function to complete the current Task
def task1():
    # ...
    # print(result) or save generated image to ./task1.png
    # Note: The output must print the necessary explanations.
    return

if __name__ == '__main__':
    # complete task
    task1()
