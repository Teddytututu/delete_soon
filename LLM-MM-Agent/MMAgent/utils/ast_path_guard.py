"""
AST Path Normalization Guard - P0-1 FIX

Critical fix to prevent LLM from deciding file paths.
Uses AST rewriting to FORCE all CSV paths to use only filenames (no hardcoded paths).

Author: Engineering Fix
Date: 2026-01-16
Priority: P0 (Must-have for system stability)
"""

import ast
import os
import re
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PathNormalizer(ast.NodeTransformer):
    """
    AST Node Transformer that normalizes all file paths in CSV reading calls.

    Transforms:
    - load_csv("C:\\Users\\...\\file.csv") → load_csv("file.csv")
    - load_csv("dataset/2025_C/file.csv") → load_csv("file.csv")
    - pd.read_csv("path/to/file.csv") → pd.read_csv("file.csv")
    """

    def __init__(self, allowed_files: Optional[List[str]] = None):
        """
        Initialize with whitelist of allowed filenames.

        Args:
            allowed_files: List of allowed CSV filenames (e.g., ["clean_athletes.csv"])
        """
        self.allowed_files = set(allowed_files) if allowed_files else set()
        self.violations = []  # Track violations

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """
        Transform function calls that load CSV files.

        Handles:
        - load_csv("path")
        - pd.read_csv("path")
        """
        # Check if this is a CSV loading call
        func_name = None

        # Case 1: load_csv("path")
        if isinstance(node.func, ast.Name):
            if node.func.id == 'load_csv':
                func_name = 'load_csv'

        # Case 2: pd.read_csv("path")
        elif isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'pd' and
                node.func.attr == 'read_csv'):
                func_name = 'pd.read_csv'

        if func_name and node.args:
            # Get the first argument (file path)
            first_arg = node.args[0]

            # Case 1: String literal
            if isinstance(first_arg, ast.Constant):
                original_path = first_arg.value
                if isinstance(original_path, str) and self._is_path_like(original_path):
                    normalized = self._normalize_path(original_path)
                    if normalized:
                        # Replace with normalized path
                        node.args[0] = ast.Constant(value=normalized)
                        logger.info(f"[P0-1] Normalized path: {original_path} → {normalized}")

                        # Check if file is allowed
                        if self.allowed_files and normalized not in self.allowed_files:
                            violation = {
                                'type': 'disallowed_file',
                                'file': normalized,
                                'allowed': list(self.allowed_files)
                            }
                            self.violations.append(violation)

            # Case 2: Variable (skip - we can't normalize at static time)
            elif isinstance(first_arg, ast.Name):
                logger.debug(f"[P0-1] Skipping variable path: {first_arg.id}")

        # Continue traversing
        self.generic_visit(node)
        return node

    def _is_path_like(self, s: str) -> bool:
        """Check if string looks like a file path."""
        # Contains path separators or file extension
        return ('.' in s and
                (s.endswith('.csv') or s.endswith('.json') or
                 '/' in s or '\\' in s or
                 s.startswith('C:') or s.startswith('/') or
                 s.startswith('./') or s.startswith('../')))

    def _normalize_path(self, path: str) -> Optional[str]:
        """
        Normalize path to basename only.

        Examples:
        - "C:\\Users\\...\\clean_athletes.csv" → "clean_athletes.csv"
        - "dataset/2025_C/file.csv" → "file.csv"
        - "./data.csv" → "data.csv"
        """
        # Extract basename
        basename = os.path.basename(path)

        # Remove any remaining directory components
        if basename != path:
            return basename

        return path


class ASTPathGuard:
    """
    P0-1 FIX: AST-based path normalization guard.

    Prevents LLM from deciding file paths by:
    1. Parsing generated code as AST
    2. Finding all CSV loading calls
    3. Normalizing paths to basename only
    4. Validating against whitelist
    5. Rewriting code with normalized paths
    """

    def __init__(self, allowed_files: Optional[List[str]] = None):
        """
        Initialize path guard with whitelist.

        Args:
            allowed_files: List of allowed CSV filenames
        """
        self.allowed_files = allowed_files or []

    def normalize_code_paths(self, code: str) -> Tuple[str, List[Dict]]:
        """
        Normalize all file paths in generated code.

        Args:
            code: Generated Python code

        Returns:
            (normalized_code, violations) - Modified code and list of violations
        """
        try:
            # Parse code to AST
            tree = ast.parse(code)

            # Create transformer
            normalizer = PathNormalizer(allowed_files=self.allowed_files)

            # Transform AST
            modified_tree = normalizer.visit(tree)

            # Fix line numbers (required for code generation)
            ast.fix_missing_locations(modified_tree)

            # Generate code from modified AST
            normalized_code = ast.unparse(modified_tree)

            return normalized_code, normalizer.violations

        except SyntaxError as e:
            # Code has syntax errors - can't normalize
            logger.warning(f"[P0-1] Cannot normalize code with syntax errors: {e}")
            return code, [{
                'type': 'syntax_error',
                'message': str(e)
            }]
        except Exception as e:
            # Unexpected error
            logger.error(f"[P0-1] Error normalizing paths: {e}")
            return code, [{
                'type': 'unexpected_error',
                'message': str(e)
            }]


def verify_no_hardcoded_paths(code: str) -> Tuple[bool, List[str]]:
    """
    Quick check for hardcoded paths using regex (fallback).

    Args:
        code: Python code string

    Returns:
        (is_clean, violations) - True if no hardcoded paths found
    """
    violations = []

    # Patterns for hardcoded paths
    patterns = [
        r'[\'"]C:\\\\Users\\\\',  # Windows absolute path
        r'[\'"][A-Z]:\\\\',       # Windows drive letter
        r'[\'"]\.?\.?/',         # Relative path (./ or ../)
        r'[\'"]dataset/',         # Hardcoded relative path
        r'[\'"]data/',            # Hardcoded data dir
        r'[\'"]MMBench/',         # Hardcoded project path
    ]

    for pattern in patterns:
        matches = re.findall(pattern, code)
        if matches:
            violations.extend(matches)

    return len(violations) == 0, violations


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AST PATH GUARD - P0-1 FIX")
    print("=" * 60)

    # Example allowed files
    allowed = ["clean_athletes.csv", "clean_hosts.csv", "clean_medal_counts.csv"]

    # Example code with hardcoded paths
    bad_code = '''
import pandas as pd
from utils import load_csv

# Hardcoded Windows path
df1 = load_csv("C:\\Users\\Teddy\\OneDrive\\study\\2026\\mcm\\clean_athletes.csv")

# Hardcoded relative path
df2 = pd.read_csv("dataset/2025_C/clean_hosts.csv")

# Good path (should remain unchanged)
df3 = load_csv("clean_medal_counts.csv")
'''

    print("\n[Original Code]")
    print(bad_code)

    # Normalize paths
    guard = ASTPathGuard(allowed_files=allowed)
    normalized_code, violations = guard.normalize_code_paths(bad_code)

    print("\n[Normalized Code]")
    print(normalized_code)

    print("\n[Violations]")
    for v in violations:
        print(f"  - {v}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
