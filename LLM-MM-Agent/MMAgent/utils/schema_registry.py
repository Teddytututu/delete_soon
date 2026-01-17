"""
Schema Registry - P1-5 Enhancement for Dataset Column Tracking

This module implements a dynamic schema registry system that:
1. Scans CSV files and extracts column schemas (name, type, sample values)
2. Maintains a centralized registry of all available datasets
3. Validates code against available schemas before execution
4. Provides schema information to LLM prompts to prevent column name errors

Author: Engineering Enhancement
Date: 2026-01-15
Priority: P1-5 (Medium complexity, high impact on column validation)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetSchema:
    """
    Represents the schema of a single dataset file.
    """

    def __init__(self, filepath: str, filename: str, source: str = 'unknown'):
        """
        Initialize dataset schema.

        Args:
            filepath: Full path to the CSV file
            filename: Name of the file (basename)
            source: Source type ('input_data', 'task_output', 'unknown')
        """
        self.filepath = filepath
        self.filename = filename
        self.source = source
        self.columns = []  # List of column names
        self.column_types = {}  # column_name -> dtype
        self.column_samples = {}  # column_name -> [sample_values]
        self.shape = None  # (rows, columns)
        self.file_size = 0  # bytes
        self.last_modified = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            'filepath': self.filepath,
            'filename': self.filename,
            'source': self.source,
            'columns': self.columns,
            'column_types': self.column_types,
            'column_samples': self.column_samples,
            'shape': self.shape,
            'file_size': self.file_size,
            'last_modified': self.last_modified
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSchema':
        """Create schema from dictionary."""
        schema = cls(data['filepath'], data['filename'], data.get('source', 'unknown'))
        schema.columns = data.get('columns', [])
        schema.column_types = data.get('column_types', {})
        schema.column_samples = data.get('column_samples', {})
        schema.shape = data.get('shape')
        schema.file_size = data.get('file_size', 0)
        schema.last_modified = data.get('last_modified')
        return schema

    def get_column_info(self, column_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific column.

        Args:
            column_name: Name of the column

        Returns:
            Dict with column info or None if column doesn't exist
        """
        if column_name not in self.columns:
            return None

        return {
            'name': column_name,
            'type': str(self.column_types.get(column_name, 'unknown')),
            'samples': self.column_samples.get(column_name, [])
        }

    def has_column(self, column_name: str) -> bool:
        """Check if schema contains a specific column."""
        return column_name in self.columns

    def get_similar_columns(self, column_name: str, threshold: int = 2) -> List[str]:
        """
        Find columns with similar names (for typo detection).

        Args:
            column_name: Column name to find matches for
            threshold: Maximum edit distance

        Returns:
            List of similar column names
        """
        from difflib import get_close_matches

        matches = get_close_matches(column_name, self.columns, n=5, cutoff=0.6)
        return matches


class SchemaRegistry:
    """
    Centralized registry for all dataset schemas.

    Features:
    1. Dynamic schema extraction from CSV files
    2. Schema persistence (save/load from JSON)
    3. Column validation with helpful error messages
    4. Schema information formatted for LLM prompts
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize schema registry.

        Args:
            registry_path: Path to save/load registry JSON file
        """
        self.schemas: Dict[str, DatasetSchema] = {}  # filename -> schema
        self.registry_path = registry_path
        self.scan_timestamp = None

    def scan_directory(self, directory: str, pattern: str = '*.csv',
                      source: str = 'input_data', max_rows: int = 100) -> int:
        """
        Scan directory for CSV files and extract schemas.

        Args:
            directory: Directory to scan
            pattern: File pattern to match (default: *.csv)
            source: Source type ('input_data', 'task_output', 'unknown')
            max_rows: Maximum rows to read for sample values

        Returns:
            Number of schemas extracted
        """
        count = 0
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return 0

        for csv_file in dir_path.glob(pattern):
            try:
                schema = self.extract_schema(str(csv_file), source, max_rows)
                if schema:
                    self.schemas[schema.filename] = schema
                    count += 1
                    logger.info(f"Extracted schema: {schema.filename} ({len(schema.columns)} columns)")
            except Exception as e:
                logger.error(f"Failed to extract schema from {csv_file}: {e}")

        self.scan_timestamp = datetime.now().isoformat()
        return count

    def extract_schema(self, filepath: str, source: str = 'unknown',
                      max_rows: int = 100) -> Optional[DatasetSchema]:
        """
        Extract schema from a single CSV file.

        Args:
            filepath: Path to CSV file
            source: Source type
            max_rows: Maximum rows to read for samples

        Returns:
            DatasetSchema object or None if extraction fails
        """
        path = Path(filepath)
        if not path.exists():
            logger.error(f"File does not exist: {filepath}")
            return None

        filename = path.name
        schema = DatasetSchema(filepath, filename, source)

        try:
            # Read CSV with error handling
            df = pd.read_csv(filepath, nrows=max_rows, encoding='latin-1')

            # Basic info
            schema.shape = df.shape
            schema.columns = list(df.columns)
            schema.file_size = path.stat().st_size
            schema.last_modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

            # Column types and samples
            for col in df.columns:
                schema.column_types[col] = str(df[col].dtype)

                # Get sample values (non-null, up to 5 values)
                samples = df[col].dropna().head(5).tolist()
                schema.column_samples[col] = [str(v)[:50] for v in samples]  # Limit string length

            logger.debug(f"Extracted schema from {filename}: {schema.shape}")
            return schema

        except Exception as e:
            logger.error(f"Failed to read CSV {filepath}: {e}")
            return None

    def scan_files(self, file_paths: List[str], source: str = 'task_output',
                  max_rows: int = 100) -> int:
        """
        Extract schemas from a list of file paths.

        Args:
            file_paths: List of file paths
            source: Source type
            max_rows: Maximum rows to read for samples

        Returns:
            Number of schemas extracted
        """
        count = 0
        for filepath in file_paths:
            try:
                schema = self.extract_schema(filepath, source, max_rows)
                if schema:
                    self.schemas[schema.filename] = schema
                    count += 1
            except Exception as e:
                logger.error(f"Failed to extract schema from {filepath}: {e}")

        self.scan_timestamp = datetime.now().isoformat()
        return count

    def get_schema(self, filename: str) -> Optional[DatasetSchema]:
        """
        Get schema for a specific file.

        Args:
            filename: Name of the file

        Returns:
            DatasetSchema or None if not found
        """
        return self.schemas.get(filename)

    def get_all_columns(self, filename: str) -> List[str]:
        """
        Get all column names for a file.

        Args:
            filename: Name of the file

        Returns:
            List of column names or empty list if file not found
        """
        schema = self.get_schema(filename)
        return schema.columns if schema else []

    def validate_column_usage(self, code: str, filename: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate that code only uses existing columns.

        Args:
            code: Python code to validate
            filename: Specific file to check against (None = check all files)

        Returns:
            (is_valid, list_of_invalid_columns)
        """
        import re

        # Pattern to match df['column_name'] or df.column_name
        bracket_pattern = r"df\[['\"]([^'\"]+)['\"]\]"
        dot_pattern = r"df\.(\w+)"

        invalid_columns = []

        # Extract column names from code
        bracket_matches = re.findall(bracket_pattern, code)
        dot_matches = re.findall(dot_pattern, code)
        all_columns = set(bracket_matches + dot_matches)

        # Get allowed columns
        if filename:
            allowed_columns = set(self.get_all_columns(filename))
        else:
            # Check against all files
            allowed_columns = set()
            for schema in self.schemas.values():
                allowed_columns.update(schema.columns)

        # Check each column
        for col in all_columns:
            if col not in allowed_columns:
                # Check if it's a DataFrame method (not a column)
                if col not in ['head', 'tail', 'describe', 'info', 'shape', 'columns']:
                    invalid_columns.append(col)

        return len(invalid_columns) == 0, invalid_columns

    def format_for_prompt(self, max_files: int = 10, max_columns: int = 20) -> str:
        """
        Format schema information for LLM prompt.

        Args:
            max_files: Maximum number of files to include
            max_columns: Maximum columns per file to show

        Returns:
            Formatted string for prompt
        """
        if not self.schemas:
            return "# No dataset schemas available"

        lines = ["## Available Dataset Schemas"]
        lines.append("")
        lines.append("The following datasets are available with their column information:")
        lines.append("")

        # Group by source
        input_files = [s for s in self.schemas.values() if s.source == 'input_data']
        task_files = [s for s in self.schemas.values() if s.source == 'task_output']

        if input_files:
            lines.append("### Input Datasets (from problem):")
            for schema in input_files[:max_files]:
                lines.append(f"\n**File**: `{schema.filename}`")
                lines.append(f"- **Columns** ({len(schema.columns)} total): {', '.join(schema.columns[:max_columns])}")
                if len(schema.columns) > max_columns:
                    lines.append(f"  ... and {len(schema.columns) - max_columns} more columns")
                lines.append(f"- **Shape**: {schema.shape[0]}+ rows × {schema.shape[1]} columns")

                # Show sample values for first few columns
                lines.append("- **Sample Values**:")
                for col in schema.columns[:5]:
                    samples = schema.column_samples.get(col, [])
                    if samples:
                        lines.append(f"  - `{col}`: {', '.join(samples[:3])}")

        if task_files:
            lines.append("\n### Task-Generated Datasets:")
            for schema in task_files[:max_files]:
                lines.append(f"\n**File**: `{schema.filename}`")
                lines.append(f"- **Columns** ({len(schema.columns)} total): {', '.join(schema.columns[:max_columns])}")
                if len(schema.columns) > max_columns:
                    lines.append(f"  ... and {len(schema.columns) - max_columns} more columns")
                lines.append(f"- **Shape**: {schema.shape[0]}+ rows × {schema.shape[1]} columns")

        lines.append("\n### CRITICAL INSTRUCTIONS:")
        lines.append("1. Use ONLY the column names listed above")
        lines.append("2. Column names are case-sensitive: 'Event' != 'event'")
        lines.append("3. Do NOT invent or guess column names")
        lines.append("4. Access columns with: `df['column_name']`")

        return "\n".join(lines)

    def save(self, filepath: Optional[str] = None) -> bool:
        """
        Save registry to JSON file.

        Args:
            filepath: Path to save (default: self.registry_path)

        Returns:
            True if successful
        """
        save_path = filepath or self.registry_path
        if not save_path:
            logger.warning("No registry path specified, skipping save")
            return False

        try:
            data = {
                'scan_timestamp': self.scan_timestamp,
                'schemas': {name: schema.to_dict() for name, schema in self.schemas.items()},
                'total_files': len(self.schemas),
                'total_columns': sum(len(s.columns) for s in self.schemas.values())
            }

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved schema registry to {save_path} ({len(self.schemas)} files)")
            return True

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            return False

    def load(self, filepath: Optional[str] = None) -> bool:
        """
        Load registry from JSON file.

        Args:
            filepath: Path to load from (default: self.registry_path)

        Returns:
            True if successful
        """
        load_path = filepath or self.registry_path
        if not load_path:
            logger.warning("No registry path specified, skipping load")
            return False

        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.scan_timestamp = data.get('scan_timestamp')
            self.schemas = {
                name: DatasetSchema.from_dict(schema_data)
                for name, schema_data in data.get('schemas', {}).items()
            }

            logger.info(f"Loaded schema registry from {load_path} ({len(self.schemas)} files)")
            return True

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with statistics
        """
        return {
            'total_files': len(self.schemas),
            'input_files': sum(1 for s in self.schemas.values() if s.source == 'input_data'),
            'task_files': sum(1 for s in self.schemas.values() if s.source == 'task_output'),
            'total_columns': sum(len(s.columns) for s in self.schemas.values()),
            'scan_timestamp': self.scan_timestamp
        }
