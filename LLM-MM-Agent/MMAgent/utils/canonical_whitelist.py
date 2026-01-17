"""
Canonical CSV Whitelist Manager - P0-1 FIX

Enforces a single canonical set of CSV files.
Prevents confusion between clean_/cleaned_/clean_summerOly_ versions.

Author: Engineering Fix
Date: 2026-01-16
Priority: P0 (Chart generation stability)
"""

import os
import logging
from pathlib import Path
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# CANONICAL CSV FILE SET
# ============================================================================

# ONLY these 4 files are allowed to be exposed to LLM
CANONICAL_CSV_FILES = {
    'clean_athletes.csv',
    'clean_hosts.csv',
    'clean_medal_counts.csv',
    'clean_programs.csv'
}

# These patterns should be filtered out from available files
FILTERED_PATTERNS = [
    'cleaned_',        # Old version (before BOM fix)
    'clean_summerOly_', # Summer-only version
    'clean_winterOly_', # Winter-only version
    'summerOly_',      # Alternative naming
    'winterOly_',      # Alternative naming
]


# ============================================================================
# WHITELIST MANAGER
# ============================================================================

class CSVWhitelistManager:
    """
    Manages the canonical whitelist of CSV files.

    Ensures only the 4 canonical files are exposed to LLM:
    - clean_athletes.csv
    - clean_hosts.csv
    - clean_medal_counts.csv
    - clean_programs.csv

    All other versions (cleaned_, clean_summerOly_, etc.) are filtered out.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize whitelist manager.

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.canonical_files = CANONICAL_CSV_FILES.copy()

    def get_canonical_files(self) -> Set[str]:
        """
        Get the canonical whitelist.

        Returns:
            Set of 4 canonical CSV filenames
        """
        return self.canonical_files.copy()

    def filter_available_files(self, all_files: List[str]) -> List[str]:
        """
        Filter available files to only canonical ones.

        Removes:
        - Files with filtered patterns (cleaned_, clean_summerOly_)
        - Files not in canonical set

        Args:
            all_files: List of all available CSV files

        Returns:
            Filtered list containing only canonical files
        """
        canonical_only = []

        for filename in all_files:
            basename = os.path.basename(filename)

            # Check if file matches canonical name
            if basename in self.canonical_files:
                canonical_only.append(filename)
            else:
                # Check if it should be filtered
                should_filter = any(
                    pattern in basename.lower()
                    for pattern in FILTERED_PATTERNS
                )

                if should_filter:
                    logger.debug(f"[P0-1] Filtered out non-canonical file: {basename}")
                else:
                    logger.warning(f"[P0-1] Unknown file not in canonical set: {basename}")

        return canonical_only

    def validate_file_access(self, filename: str) -> tuple[bool, str]:
        """
        Validate that a file is in the canonical whitelist.

        Args:
            filename: CSV filename to validate

        Returns:
            (is_allowed, error_message)
        """
        basename = os.path.basename(filename)

        if basename not in self.canonical_files:
            available = ', '.join(sorted(self.canonical_files))
            return False, (
                f"File not in canonical whitelist: {basename}\n"
                f"Available canonical files: {available}\n"
                f"Note: Versions like 'cleaned_' or 'clean_summerOly_' are not allowed."
            )

        return True, ""

    def get_file_summary(self) -> str:
        """
        Get a summary of canonical files for LLM prompts.

        Returns:
            Formatted string listing canonical files
        """
        files = sorted(self.canonical_files)
        return (
            "Available CSV Files (Canonical Set Only):\n"
            f"  {files[0]}\n"
            f"  {files[1]}\n"
            f"  {files[2]}\n"
            f"  {files[3]}\n"
            "\n"
            "IMPORTANT: Only use these exact filenames.\n"
            "DO NOT use versions like 'cleaned_' or 'clean_summerOly_'."
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_canonical_whitelist(data_dir: str) -> CSVWhitelistManager:
    """
    Setup canonical whitelist for a data directory.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        CSVWhitelistManager instance
    """
    manager = CSVWhitelistManager(data_dir)

    # Log the canonical set
    logger.info(f"[P0-1] Canonical CSV Whitelist Initialized")
    logger.info(f"[P0-1] Allowed files: {sorted(manager.canonical_files)}")

    return manager


def expose_only_canonical_to_llm(all_csv_files: List[str]) -> List[str]:
    """
    Filter CSV files to expose only canonical ones to LLM.

    This is the main function to use when preparing file lists for LLM prompts.

    Args:
        all_csv_files: List of all CSV files found in dataset directory

    Returns:
        Filtered list with only canonical files
    """
    manager = CSVWhitelistManager()
    canonical_only = manager.filter_available_files(all_csv_files)

    logger.info(f"[P0-1] Filtered CSV files: {len(all_csv_files)} -> {len(canonical_only)} canonical")
    logger.info(f"[P0-1] Canonical files: {canonical_only}")

    return canonical_only


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CANONICAL CSV WHITELIST MANAGER - P0-1 FIX")
    print("=" * 70)

    # Test file filtering
    all_files = [
        'clean_athletes.csv',
        'clean_hosts.csv',
        'clean_medal_counts.csv',
        'clean_programs.csv',
        'cleaned_athletes.csv',      # Filtered out
        'clean_summerOly_hosts.csv',  # Filtered out
        'data.csv',                    # Filtered out
    ]

    print("\n[Test] File Filtering")
    print(f"Input: {len(all_files)} files")
    print(f"Files: {all_files}")

    canonical_only = expose_only_canonical_to_llm(all_files)

    print(f"\nOutput: {len(canonical_only)} canonical files")
    print(f"Files: {canonical_only}")

    # Test validation
    print("\n[Test] File Validation")
    manager = CSVWhitelistManager()

    # Valid file
    is_allowed, msg = manager.validate_file_access('clean_athletes.csv')
    print(f"clean_athletes.csv: {'ALLOWED' if is_allowed else 'REJECTED'}")
    if not is_allowed:
        print(f"  Reason: {msg}")

    # Invalid file (cleaned_ version)
    is_allowed, msg = manager.validate_file_access('cleaned_athletes.csv')
    print(f"cleaned_athletes.csv: {'ALLOWED' if is_allowed else 'REJECTED'}")
    if not is_allowed:
        print(f"  Reason: {msg}")

    # Test file summary
    print("\n[Test] File Summary for LLM")
    print(manager.get_file_summary())

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
