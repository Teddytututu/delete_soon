"""
Column Schema Manager - Step 1 Fix for KeyError Issues

PROBLEM: LLM generates code with df['EVENT'] but data has:
  - 'Event' (different case)
  - 'event' (lowercase)
  - 'EVENT_NAME' (different name)
  - ' event ' (whitespace)

SOLUTION: Column name normalization + alias mapping
All column names are stripped and mapped to canonical names.

Author: Engineering Fix
Date: 2026-01-15
"""

import re
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Column name schema manager.

    Handles:
    1. Column name normalization (strip whitespace, standardize case)
    2. Alias mapping (Event/event/EVENT_NAME → EVENT)
    3. Case-insensitive column lookup
    """

    def __init__(self):
        """Initialize schema manager with common aliases."""
        # Common column name aliases for mathematical modeling
        self.aliases = {
            # Time/Year related
            'year': 'Year',
            'YEAR': 'Year',
            'time': 'Time',
            'TIME': 'Time',
            'date': 'Date',
            'DATE': 'Date',

            # ID/Identifier
            'id': 'ID',
            'ID': 'ID',
            'identifier': 'ID',
            'key': 'Key',

            # Event/Competition
            'event': 'Event',
            'EVENT': 'Event',
            'event_name': 'Event',
            'EVENT_NAME': 'Event',
            'competition': 'Event',
            'discipline': 'Event',

            # Country/Region
            'country': 'Country',
            'COUNTRY': 'Country',
            'nation': 'Country',
            'region': 'Region',
            'noc': 'Country',  # National Olympic Committee

            # Medal/Count
            'gold': 'Gold',
            'GOLD': 'Gold',
            'silver': 'Silver',
            'SILVER': 'Silver',
            'bronze': 'Bronze',
            'BRONZE': 'Bronze',
            'total': 'Total',
            'TOTAL': 'Total',
            'medal': 'Medal',
            'MEDAL': 'Medal',

            # Rank/Position
            'rank': 'Rank',
            'RANK': 'Rank',
            'position': 'Position',
            'POSITION': 'Position',
            'place': 'Rank',

            # Score/Value
            'score': 'Score',
            'SCORE': 'Score',
            'value': 'Value',
            'VALUE': 'Value',
            'amount': 'Amount',

            # Name/Label
            'name': 'Name',
            'NAME': 'Name',
            'label': 'Label',
            'LABEL': 'Label',
            'title': 'Title',
        }

        # DataFrame-specific column cache
        self._df_columns: Dict[str, Dict[str, str]] = {}

    def normalize_column_name(self, col_name: str) -> str:
        """
        Normalize a column name.

        1. Strip whitespace
        2. Apply alias mapping
        3. Return canonical name

        Args:
            col_name: Raw column name from CSV

        Returns:
            Normalized column name
        """
        if not isinstance(col_name, str):
            return str(col_name)

        # Step 1: Strip whitespace
        normalized = col_name.strip()

        # Step 2: Apply alias mapping (case-insensitive)
        lower = normalized.lower()
        if lower in self.aliases:
            normalized = self.aliases[lower]

        return normalized

    def get_canonical_columns(self, df_columns: List[str]) -> Dict[str, str]:
        """
        Get mapping of normalized → actual column names.

        Args:
            df_columns: List of actual column names from DataFrame

        Returns:
            Dict mapping canonical names → actual names
            Example: {'Year': 'year', 'Event': 'EVENT'}
        """
        mapping = {}

        for actual_col in df_columns:
            canonical = self.normalize_column_name(actual_col)
            mapping[canonical] = actual_col

        return mapping

    def find_column(self, df_columns: List[str], target: str, fuzzy: bool = False) -> Optional[str]:
        """
        Find a column in DataFrame, with alias lookup and fuzzy matching.

        Args:
            df_columns: List of actual column names from DataFrame
            target: Target column name (what LLM is looking for)
            fuzzy: If True, try fuzzy matching for similar names

        Returns:
            Actual column name if found, None otherwise
        """
        # Normalize target
        target_normalized = self.normalize_column_name(target)

        # Build mapping
        mapping = self.get_canonical_columns(df_columns)

        # Direct lookup
        if target_normalized in mapping:
            return mapping[target_normalized]

        # Case-insensitive lookup
        for actual_col in df_columns:
            if actual_col.strip().lower() == target.lower():
                return actual_col

        # Fuzzy matching (if enabled)
        if fuzzy:
            # Try substring match
            for actual_col in df_columns:
                if target.lower() in actual_col.lower() or actual_col.lower() in target.lower():
                    logger.debug(f"[Schema] Fuzzy match: '{target}' → '{actual_col}'")
                    return actual_col

        return None

    def validate_code_columns(self, code: str, df_columns: List[str]) -> tuple[bool, List[str]]:
        """
        Validate that all column names in code exist in DataFrame.

        Args:
            code: Python code generated by LLM
            df_columns: List of actual column names from DataFrame

        Returns:
            (all_valid, invalid_columns)
        """
        # Extract column names from df['col'] and df.col patterns
        used_columns = set()

        # Pattern 1: df['column_name']
        pattern1 = r"df\['([^']+)'\]"
        for match in re.finditer(pattern1, code):
            used_columns.add(match.group(1))

        # Pattern 2: df["column_name"]
        pattern2 = r'df\["([^"]+)"\]'
        for match in re.finditer(pattern2, code):
            used_columns.add(match.group(1))

        # Pattern 3: df.column_name (excluding methods like df.groupby)
        pattern3 = r"df\.(\w+)\("
        for match in re.finditer(pattern3, code):
            col = match.group(1)
            # Exclude common methods
            if col not in ['groupby', 'agg', 'apply', 'merge', 'sort_values',
                          'fillna', 'dropna', 'describe', 'head', 'tail']:
                used_columns.add(col)

        # Validate each column
        invalid = []
        for col in used_columns:
            if not self.find_column(df_columns, col, fuzzy=False):
                invalid.append(col)

        return len(invalid) == 0, list(invalid)

    def get_column_suggestion(self, df_columns: List[str], invalid_col: str) -> Optional[str]:
        """
        Suggest correct column name for invalid column.

        Args:
            df_columns: List of actual column names
            invalid_col: Invalid column name from LLM code

        Returns:
            Suggested column name or None
        """
        return self.find_column(df_columns, invalid_col, fuzzy=True)

    def generate_column_mapping_code(self, df_columns: List[str]) -> str:
        """
        Generate code that creates column mapping for LLM.

        This helps LLM understand which columns are available.

        Args:
            df_columns: List of actual column names from DataFrame

        Returns:
            Python code string that prints column info
        """
        mapping = self.get_canonical_columns(df_columns)

        lines = [
            "# Available columns (canonical → actual):",
            "column_map = {"
        ]

        for canonical, actual in sorted(mapping.items()):
            lines.append(f"    '{canonical}': '{actual}',")

        lines.append("}")
        lines.append("")
        lines.append("print('[INFO] Available columns:')")
        lines.append("for canonical, actual in column_map.items():")
        lines.append("    print(f'  {canonical}: df[\\'{actual}\\']')")

        return "\n".join(lines)


# Global instance
_schema_manager: Optional[SchemaManager] = None


def get_schema_manager() -> SchemaManager:
    """Get global schema manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = SchemaManager()
    return _schema_manager
