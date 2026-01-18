"""
Import Whitelist Guard - P0-3 FIX

Prevents LLM from introducing new dependencies that may not exist in the environment.
Uses AST scanning to validate all imports against a whitelist before code execution.

Author: Engineering Fix
Date: 2026-01-16
Priority: P0 (Prevents dependency death spiral)
"""

import ast
import logging
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


# Default import whitelist for Task 1 (data engineering, not ML)
DEFAULT_IMPORT_WHITELIST = {
    # Standard library (always allowed)
    'os', 'sys', 're', 'json', 'pathlib', 'datetime', 'collections',
    'itertools', 'math', 'statistics', 'random',

    # [FIX 2026-01-18] Added standard library modules commonly used in data processing
    'io', 'warnings', 'traceback',

    # Data manipulation (core)
    'pandas', 'numpy',

    # Visualization (core)
    'matplotlib', 'matplotlib.pyplot', 'seaborn',

    # Scikit-learn (SELECTIVE - only basic data processing)
    'sklearn.model_selection',
    'sklearn.linear_model',
    'sklearn.preprocessing',
    'sklearn.metrics',

    # [FIX 2026-01-18] Added statsmodels and scipy for statistical analysis
    'statsmodels', 'statsmodels.api', 'statsmodels.formula.api',
    'scipy', 'scipy.stats',

    # FORBIDDEN in Task 1 (complex ML - causes dependency issues)
    # 'sklearn.decomposition',  # No PCA
    # 'torch', 'tensorflow',      # No deep learning
}

# Forbidden imports that should trigger immediate rejection
FORBIDDEN_IMPORTS = {
    'torch', 'tensorflow', 'keras', 'mxnet', 'caffe',
    'sklearn.decomposition',  # No PCA in Task 1
    'sklearn.cluster',         # No clustering in Task 1
    'sklearn.manifold',        # No manifold learning in Task 1
}


class ImportGuard(ast.NodeVisitor):
    """
    AST visitor that extracts all import statements from code.

    Detects:
    - import x
    - import x as y
    - from x import y
    - from x import y as z
    """

    def __init__(self):
        self.imports = []

    def visit_Import(self, node: ast.Import):
        """Handle: import x"""
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'type': 'import'
            })

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle: from x import y"""
        module = node.module or ''
        for alias in node.names:
            self.imports.append({
                'module': module,
                'name': alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'type': 'from'
            })


class ImportWhitelistGuard:
    """
    P0-3 FIX: Import whitelist guard.

    Validates all imports in generated code against a whitelist.
    Prevents LLM from introducing dependencies that don't exist.
    """

    def __init__(self,
                 whitelist: Optional[Set[str]] = None,
                 forbidden: Optional[Set[str]] = None):
        """
        Initialize import guard.

        Args:
            whitelist: Set of allowed import modules
            forbidden: Set of explicitly forbidden imports
        """
        self.whitelist = whitelist or DEFAULT_IMPORT_WHITELIST
        self.forbidden = forbidden or FORBIDDEN_IMPORTS

    def check_imports(self, code: str) -> tuple[bool, List[Dict]]:
        """
        Check all imports in code against whitelist.

        Args:
            code: Python code string

        Returns:
            (is_allowed, violations) - True if all imports are allowed
        """
        try:
            tree = ast.parse(code)

            # Extract all imports
            guard = ImportGuard()
            guard.visit(tree)

            violations = []

            for imp in guard.imports:
                module = imp['module']
                full_import = f"{module}.{imp['name']}" if imp.get('name') else module

                # Check 1: Explicitly forbidden
                if any(forbidden in full_import for forbidden in self.forbidden):
                    violations.append({
                        'type': 'forbidden_import',
                        'import': full_import,
                        'line': imp['line'],
                        'reason': f"Import '{full_import}' is not allowed in Task 1 (data engineering only)"
                    })
                    continue

                # Check 2: Not in whitelist
                # Check both full module and parent module
                allowed = False
                for allowed_module in self.whitelist:
                    if full_import.startswith(allowed_module) or module.startswith(allowed_module):
                        allowed = True
                        break

                if not allowed:
                    violations.append({
                        'type': 'unknown_import',
                        'import': full_import,
                        'line': imp['line'],
                        'reason': f"Import '{full_import}' is not in whitelist. Allowed: {sorted(self.whitelist)[:5]}..."
                    })

            return len(violations) == 0, violations

        except SyntaxError as e:
            # Code has syntax errors - can't check imports
            return False, [{
                'type': 'syntax_error',
                'reason': f'Cannot check imports due to syntax error: {e}'
            }]


def fix_import_dependencies(code: str, error_message: str) -> str:
    """
    Attempt to fix import-related errors by removing problematic imports.

    This is a defensive fallback when imports cause failures.

    Args:
        code: Python code with problematic imports
        error_message: Error message from execution

    Returns:
        Fixed code with problematic imports removed/commented
    """
    import re

    # Detect common import errors
    import_errors = [
        (r"ImportError: cannot import name '(\w+)' from 'sklearn.preprocessing'", 'sklearn.preprocessing'),
        (r"NameError: name '(\w+)' is not defined", None),
        (r"ModuleNotFoundError: No module named '([\w.]+)'", None),
    ]

    fixed_code = code

    for pattern, module in import_errors:
        if re.search(pattern, error_message):
            if module:
                # Comment out the problematic import
                fixed_code = re.sub(
                    rf'(from {re.escape(module)} import.*?$)',
                    r'# \1  # DISABLED: Import error',
                    fixed_code,
                    flags=re.MULTILINE
                )
                logger.info(f"[P0-3] Disabled problematic import: {module}")

    return fixed_code


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IMPORT WHITELIST GUARD - P0-3 FIX")
    print("=" * 60)

    # Test code with problematic imports
    bad_code = '''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA  # FORBIDDEN in Task 1
from sklearn.preprocessing import StandardScaler
import torch  # FORBIDDEN - deep learning
'''

    print("\n[Test Code]")
    print(bad_code)

    # Check imports
    guard = ImportWhitelistGuard()
    is_allowed, violations = guard.check_imports(bad_code)

    print("\n[Result]")
    print(f"Allowed: {is_allowed}")

    if violations:
        print(f"\n[Violations ({len(violations)})]")
        for v in violations:
            print(f"  Line {v.get('line', '?')}: {v['reason']}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
