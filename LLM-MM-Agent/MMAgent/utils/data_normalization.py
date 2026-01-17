"""
数据标准化工具模块 (P1 FIX #4)

处理CSV读取后的列名标准化:
- 移除BOM字符
- 去除空格
- 转大写
- 替换空格为下划线

Created: 2026-01-15
Purpose: Fix BOM issues like 'Ï»¿YEAR' → 'YEAR'
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def normalize_column_name(col: str) -> str:
    """
    标准化单个列名

    Args:
        col: 原始列名

    Returns:
        标准化后的列名
    """
    # Step 1: Remove BOM (UTF-8 BOM = \ufeff)
    col_clean = col.replace('\ufeff', '')

    # Step 2: Strip leading/trailing whitespace
    col_clean = col_clean.strip()

    # Step 3: Convert to uppercase
    col_clean = col_clean.upper()

    # Step 4: Replace internal spaces with underscores
    col_clean = col_clean.replace(' ', '_')

    return col_clean


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化DataFrame的所有列名

    Args:
        df: 输入DataFrame

    Returns:
        列名标准化后的DataFrame (in-place修改)
    """
    original_columns = df.columns.tolist()
    normalized_columns = [normalize_column_name(col) for col in original_columns]

    # Log changes
    changes = []
    for orig, norm in zip(original_columns, normalized_columns):
        if orig != norm:
            changes.append(f"  '{orig}' → '{norm}'")

    if changes:
        logger.info("Column name normalization:")
        for change in changes:
            logger.info(change)

    # Apply normalization
    df.columns = normalized_columns

    return df


def read_csv_with_normalized_columns(
    file_path,
    encoding='utf-8-sig',  # P0-ENCODING FIX: Changed default to utf-8-sig
    **kwargs
) -> pd.DataFrame:
    """
    读取CSV文件并自动标准化列名

    P0-ENCODING FIX: Now uses utf-8-sig by default to properly handle:
    - UTF-8 files with BOM (ï»¿ characters)
    - UTF-8 files without BOM
    - Falls back to latin1 if UTF-8 decoding fails

    Args:
        file_path: CSV文件路径
        encoding: 文件编码 (默认utf-8-sig，不再使用latin-1作为默认值)
        **kwargs: 传递给pd.read_csv的其他参数

    Returns:
        列名已标准化的DataFrame

    Raises:
        FileNotFoundError: 文件不存在
        Exception: 读取失败
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    logger.debug(f"Reading CSV: {file_path}")

    # P0-ENCODING FIX: Try utf-8-sig first (handles BOM), fallback to latin1
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', **kwargs)
        logger.debug(f"Successfully read with utf-8-sig encoding")
    except UnicodeDecodeError:
        # Fallback to latin1 for files that aren't UTF-8
        logger.debug(f"utf-8-sig failed, trying latin1 encoding")
        df = pd.read_csv(file_path, encoding='latin1', **kwargs)

    # Normalize columns
    df = normalize_dataframe_columns(df)

    logger.debug(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    logger.debug(f"Columns: {df.columns.tolist()}")

    return df


def check_bom_in_file(file_path: Path) -> bool:
    """
    检查文件是否有UTF-8 BOM

    Args:
        file_path: 文件路径

    Returns:
        True如果有BOM
    """
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(4)
            return first_bytes.startswith(b'\xef\xbb\xbf')
    except Exception:
        return False


def scan_directory_for_bom(directory: Path) -> dict:
    """
    扫描目录中所有CSV文件的BOM情况

    Args:
        directory: 要扫描的目录

    Returns:
        字典: {filename: has_bom}
    """
    results = {}
    csv_files = list(directory.glob("*.csv"))

    logger.info(f"Scanning {len(csv_files)} CSV files for BOM...")

    for csv_file in csv_files:
        has_bom = check_bom_in_file(csv_file)
        results[csv_file.name] = has_bom

        if has_bom:
            logger.warning(f"BOM detected: {csv_file.name}")
        else:
            logger.debug(f"No BOM: {csv_file.name}")

    bom_count = sum(1 for has_bom in results.values() if has_bom)
    logger.info(f"Found {bom_count} files with BOM out of {len(csv_files)}")

    return results


# 测试代码
if __name__ == "__main__":
    import sys

    # Test 1: 单个列名标准化
    print("Test 1: Column name normalization")
    test_columns = [
        "YEAR",  # Already clean
        "Ï»¿YEAR",  # BOM
        " Year ",  # Spaces
        "Event ",  # Trailing space
        "NOC Code",  # Space in middle
    ]

    for col in test_columns:
        normalized = normalize_column_name(col)
        print(f"  '{col}' → '{normalized}'")

    # Test 2: DataFrame标准化
    print("\nTest 2: DataFrame normalization")
    import numpy as np
    df = pd.DataFrame({
        'Ï»¿YEAR': [2000, 2004],
        ' Event ': ['Swimming', 'Athletics'],
        'NOC Code': ['USA', 'CHN']
    })
    print("Before:", df.columns.tolist())
    df = normalize_dataframe_columns(df)
    print("After:", df.columns.tolist())

    print("\n[P1 FIX #4] BOM cleaning module loaded successfully!")
