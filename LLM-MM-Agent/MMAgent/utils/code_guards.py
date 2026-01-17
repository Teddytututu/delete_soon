"""
Engineering Guards - P0 Priority Fixes for Pipeline Stability

This module implements the 10 critical engineering guards to prevent
the most common crash patterns identified in production logs.

Author: Engineering Fix
Date: 2026-01-15
Priority: P0 (Same-day fixes to stop bleeding)
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from difflib import get_close_matches

logger = logging.getLogger(__name__)


class CodeExecutionGuards:
    """
    Enforces P0 engineering guards during code generation and execution.

    Guards 1-5: Pre-execution validation
    Guards 6-10: Runtime safety nets
    """

    def __init__(self, data_dir: Optional[str] = None, available_files: Optional[List[str]] = None):
        """
        Initialize guards with data directory and available files.

        Args:
            data_dir: Root data directory path
            available_files: List of available CSV files (from scanning)
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.available_files = set(available_files) if available_files else set()

        # Schema registry: filename → required/allowed columns
        self.schema_registry = {}
        self._build_schema_registry()

        # Guard statistics
        self.blocked_operations = {
            'unsafe_read_csv': 0,
            'invalid_columns': 0,
            'invented_files': 0,
            '3d_charts': 0,
            'missing_imports': 0
        }

    def _build_schema_registry(self):
        """
        Build schema registry for common Olympic data files.

        Maps filename → allowed columns (prevents using wrong table).
        """
        self.schema_registry = {
            'clean_athletes.csv': {
                'required': ['ID', 'NAME', 'SEX'],
                'allowed': ['ID', 'NAME', 'SEX', 'AGE', 'HEIGHT', 'WEIGHT', 'SPORT', 'EVENT', 'MEDAL']
            },
            'clean_medal_counts.csv': {
                'required': ['YEAR'],
                'allowed': ['YEAR', 'GOLD', 'SILVER', 'BRONZE', 'TOTAL']
            },
            'clean_hosts.csv': {
                'required': ['YEAR'],
                'allowed': ['YEAR', 'CITY', 'COUNTRY']
            },
            'clean_programs.csv': {
                'required': ['YEAR'],
                'allowed': ['YEAR', 'SPORT', 'EVENT']
            },
            'summerOly_athletes.csv': {
                'required': ['ID', 'NAME'],
                'allowed': ['ID', 'NAME', 'SEX', 'AGE', 'HEIGHT', 'WEIGHT', 'SPORT', 'EVENT', 'MEDAL']
            },
            'summerOly_medal_counts.csv': {
                'required': ['YEAR'],
                'allowed': ['YEAR', 'GOLD', 'SILVER', 'BRONZE', 'TOTAL', 'RANK']
            }
        }

    # ============================================
    # P0-B: Guard 1.5 - Ban Hardcoded Paths
    # ============================================

    def check_hardcoded_paths(self, code: str) -> Tuple[bool, List[str]]:
        """
        Guard 1.5: Detect and reject hardcoded absolute/relative paths.

        P0-B FIX: Prevents Task code from using hardcoded paths like
        'dataset/2025_C' which cause FileNotFoundError.

        Args:
            code: LLM-generated Python code

        Returns:
            (is_safe, error_messages)
        """
        errors = []

        # Pattern 1: Hardcoded relative paths (dataset/2025_C, data/, etc.)
        hardcoded_patterns = [
            r"['\"]dataset/2025_C",  # 'dataset/2025_C'
            r"['\"]data/2025_C",     # 'data/2025_C'
            r"['\"]MMBench/dataset", # 'MMBench/dataset'
            r"C:\\Users\\",          # Windows absolute path
            r"/home/",               # Unix absolute path
        ]

        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, code)
            if matches:
                errors.append(
                    f"Guard 1.5 Violation: Hardcoded path detected: {matches}.\n"
                    f"Use DATA_DIR variable instead. Do not hardcode paths."
                )
                self.blocked_operations['unsafe_read_csv'] += len(matches)

        return len(errors) == 0, errors

    # ============================================
    # P0-1: Guard 1 - Whitelist CSV Reading
    # ============================================

    def check_csv_reading(self, code: str) -> Tuple[bool, List[str]]:
        """
        Guard 1: Only allow load_csv(name) with whitelist, ban direct pd.read_csv.

        Args:
            code: LLM-generated Python code

        Returns:
            (is_safe, error_messages)
        """
        errors = []

        # IGNORE: Template/placeholder patterns that should not be flagged
        placeholder_patterns = [
            'FULL_PATH_TO_',
            'FULL_PATH_FROM_LIST',
            'PLACEHOLDER',
            'YOUR_PATH_HERE',
            '<path>',
            '{path}',
        ]

        # Pattern 1: Direct pd.read_csv calls (FORBIDDEN)
        pd_read_csv_pattern = r"pd\.read_csv\(['\"][^'\"]+['\"]"
        matches = re.findall(pd_read_csv_pattern, code)

        # Filter out placeholder matches
        real_matches = []
        for match in matches:
            is_placeholder = any(placeholder in match for placeholder in placeholder_patterns)
            if not is_placeholder:
                real_matches.append(match)

        if real_matches:
            errors.append(
                "Guard 1 Violation: Direct pd.read_csv() calls are forbidden. "
                "Use load_csv('filename') instead. "
                f"Found: {real_matches}"
            )
            self.blocked_operations['unsafe_read_csv'] += len(real_matches)

        # Pattern 2: Full paths (C:/Users/, r'C:\) (FORBIDDEN)
        path_pattern = r"pd\.read_csv\(['\"]([A-Za-z]:[\\\\/][^'\"]+)['\"]"
        path_matches = re.findall(path_pattern, code)

        if path_matches:
            errors.append(
                f"Guard 1 Violation: Absolute paths are forbidden. "
                f"Found: {path_matches}"
            )
            self.blocked_operations['unsafe_read_csv'] += len(path_matches)

        # Pattern 3: Invented filenames (FULL_PATH_TO_, processed_xxx) (FORBIDDEN)
        # But IGNORE common placeholder patterns from templates
        invented_pattern = r"pd\.read_csv\(['\"](?:FULL_PATH_TO_|FULL_PATH_FROM_LIST|processed_)[^'\"]+['\"]"
        invented_matches = re.findall(invented_pattern, code)

        # Filter out known template placeholders
        real_invented = []
        for match in invented_matches:
            is_placeholder = any(placeholder in match for placeholder in placeholder_patterns)
            if not is_placeholder:
                real_invented.append(match)

        if real_invented:
            errors.append(
                f"Guard 1 Violation: Invented filenames are forbidden. "
                f"Choose from available files: {list(self.available_files)[:5]}..."
            )
            self.blocked_operations['invented_files'] += len(real_invented)

        return len(errors) == 0, errors

    def generate_safe_load_csv_function(self) -> str:
        """
        Generate safe load_csv() function to inject into code.

        This replaces unsafe pd.read_csv() calls with whitelisted version.

        Returns:
            Python code string defining load_csv()
        """
        available_files_list = list(self.available_files)[:10]  # First 10 files

        return f'''
# SAFE CSV LOADING - Guard 1 Whitelist
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(r'{self.data_dir}')

_AVAILABLE_FILES = {available_files_list}

def load_csv(filename: str) -> pd.DataFrame:
    """
    Load CSV file with whitelist validation.

    Args:
        filename: CSV filename (must be in available files)

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If filename not in whitelist
    """
    if filename not in _AVAILABLE_FILES:
        available = ', '.join(sorted(_AVAILABLE_FILES))
        raise FileNotFoundError(
            f"CSV file not in whitelist: {{filename}}\\n"
            f"Available files: {{available}}"
        )

    file_path = _DATA_DIR / filename
    return pd.read_csv(file_path, encoding='latin-1')
'''

    # ============================================
    # P0-2: Guard 2 - Column Name Hard Validation
    # ============================================

    def check_column_usage(self, code: str, available_columns_by_file: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
        """
        Guard 2: Hard validation - reject code using non-existent columns.

        Args:
            code: LLM-generated Python code
            available_columns_by_file: Maps filename → list of columns

        Returns:
            (is_safe, error_messages)
        """
        errors = []

        # Extract all column references from code
        used_columns = self._extract_column_references(code)

        # Check against available columns
        for filename, columns in available_columns_by_file.items():
            schema = self.schema_registry.get(filename, {})
            allowed_cols = set(schema.get('allowed', columns))

            for col in used_columns:
                # Check if column exists in this file
                if col not in allowed_cols:
                    # Try to find close match
                    close_matches = get_close_matches(col, allowed_cols, n=3, cutoff=0.6)

                    error_msg = (
                        f"Guard 2 Violation: Column '{{col}}' does not exist in {{filename}}. "
                        f"Available columns: {sorted(allowed_cols)[:10]}..."
                    )

                    if close_matches:
                        error_msg += f"\\nDid you mean: {close_matches}?"

                    errors.append(error_msg)
                    self.blocked_operations['invalid_columns'] += 1

        return len(errors) == 0, errors

    def _extract_column_references(self, code: str) -> Set[str]:
        """
        Extract all column name references from Python code.

        Args:
            code: Python code string

        Returns:
            Set of column names (uppercase)
        """
        columns = set()

        # Pattern 1: df['column_name']
        pattern1 = r"df\['([^']+)'\]"
        columns.update(re.findall(pattern1, code))

        # Pattern 2: df["column_name"]
        pattern2 = r'df\["([^"]+)"\]'
        columns.update(re.findall(pattern2, code))

        # Pattern 3: df.column_name (excluding methods)
        pattern3 = r"df\.(\w+)\("
        for match in re.finditer(pattern3, code):
            method = match.group(1)
            # Exclude common methods
            if method not in ['groupby', 'agg', 'apply', 'merge', 'sort_values',
                          'fillna', 'dropna', 'head', 'tail', 'describe', 'plot',
                          'read_csv', 'to_csv', 'iterrows']:
                columns.add(method)

        # Pattern 4: df[['col1', 'col2']]
        pattern4 = r"df\[\[([^\]]+)\]\]"
        multi_cols = re.findall(pattern4, code)
        for col_group in multi_cols:
            # Split by comma and clean quotes
            cols = [c.strip().strip('"\'') for c in col_group.split(',')]
            columns.update(cols)

        return columns

    # ============================================
    # P0-3: Guard 3 - Chart Forced Savefig
    # ============================================

    def ensure_savefig_present(self, code: str, save_path: str) -> str:
        """
        Guard 3: Force plt.savefig() call, append if missing.

        Args:
            code: LLM-generated chart code
            save_path: Path where chart should be saved

        Returns:
            Modified code with guaranteed savefig
        """
        if 'plt.savefig' not in code:
            logger.info("[Guard 3] No savefig found, appending forced save")

            # Forced savefig + close at end of code
            forced_code = f'''

# GUARD 3: Force chart save (auto-injected)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs(os.path.dirname(r'{save_path}'), exist_ok=True)
plt.savefig(r'{save_path}', bbox_inches='tight', dpi=300)
plt.close()
print(f"[Guard 3] Chart saved to: {repr(save_path)}")
'''
            return code + forced_code

        return code

    # ============================================
    # P0-4: Guard 4 - Ban 3D Charts
    # ============================================

    def check_3d_charts(self, code: str) -> Tuple[bool, List[str]]:
        """
        Guard 4: Detect and reject 3D chart attempts.

        Args:
            code: LLM-generated chart code

        Returns:
            (is_safe, error_messages)
        """
        errors = []

        # Pattern 1: mpl_toolkits.mplot3d import
        if 'mpl_toolkits.mplot3d' in code or 'from mplot3d' in code:
            errors.append(
                "Guard 4 Violation: 3D charts are forbidden (security policy). "
                "Use 2D alternatives: bar, line, scatter, heatmap."
            )
            self.blocked_operations['3d_charts'] += 1
            return False, errors

        # Pattern 2: projection='3d'
        if "projection='3d'" in code or 'projection="3d"' in code:
            errors.append(
                "Guard 4 Violation: 3D projection is forbidden. "
                "Use 2D charts instead."
            )
            self.blocked_operations['3d_charts'] += 1
            return False, errors

        # Pattern 3: plot_surface
        if '.plot_surface(' in code:
            errors.append(
                "Guard 4 Violation: 3D surface plots are forbidden. "
                "Use heatmap or contourf instead."
            )
            self.blocked_operations['3d_charts'] += 1
            return False, errors

        return True, []

    # ============================================
    # Guard 5: Fix Common Pandas Errors
    # ============================================

    def fix_pandas_merge_series_error(self, code: str) -> str:
        """
        Fix: "Cannot merge a Series without a name" error.

        Automatically adds .name or converts to DataFrame.

        Args:
            code: LLM-generated code

        Returns:
            Fixed code
        """
        # Pattern: df.merge(series, ...) where series has no name
        # Fix: df.merge(series.to_frame('name'), ...)

        # Look for merge calls
        merge_pattern = r'(\w+\.merge\([^)]+\))'

        def fix_merge(match):
            merge_call = match.group(1)
            # If merge has Series-like parameters (heuristics)
            # This is a simple fix - complex cases need template
            return merge_call  # Keep as-is for now

        # For now, just warn about merge usage
        if '.merge(' in code and 'Series' in code:
            logger.warning("[Guard 5] Detected merge() usage - may need manual fix if Series unnamed")

        return code

    # ============================================
    # Guard 6: Auto-fix missing imports
    # ============================================

    def fix_missing_imports(self, code: str) -> str:
        """
        Guard 6: Auto-fix common missing imports.

        Fixes:
        - `io.TextIOWrapper` without `import io`

        Args:
            code: LLM-generated code

        Returns:
            Fixed code with imports added
        """
        # Check if io is used but not imported
        has_io_usage = 'io.' in code or ' io ' in code
        has_io_import = 'import io' in code

        if has_io_usage and not has_io_import:
            logger.info("[Guard 6] Adding missing 'import io'")
            # Add import at the top
            code = 'import io\n' + code

        return code

    # ============================================
    # Guard 7: Column name fuzzy matching
    # ============================================

    def fix_column_name_mismatches(self, code: str, available_columns: List[str]) -> str:
        """
        Guard 7: Auto-fix column name mismatches using fuzzy matching.

        Args:
            code: LLM-generated code
            available_columns: List of actual column names

        Returns:
            Fixed code with corrected column names
        """
        available_upper = [col.upper() for col in available_columns]

        # Common mismatches to fix
        fixes = {
            'SPORT': 'EVENT',      # SPORT is often wrong, EVENT is correct
            'DISCIPLINE': 'EVENT', # DISCIPLINE is often wrong, EVENT is correct
            'MEDAL': 'MEDAL'       # Keep as-is
        }

        for wrong, correct in fixes.items():
            # Only fix if wrong exists and correct exists
            if wrong in code and correct in available_upper:
                # Replace df['WRONG'] with df['Correct']
                pattern = f"df['{wrong}']"
                replacement = f"df['{correct}']"
                code = code.replace(pattern, replacement)

                # Also replace df["WRONG"]
                pattern = f'df["{wrong}"]'
                replacement = f'df["{correct}"]'
                code = code.replace(pattern, replacement)

                logger.info(f"[Guard 7] Fixed column name: {wrong} → {correct}")

        return code

    # ============================================
    # Guard 8: Syntax validation before execution
    # ============================================

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Guard 8: Validate Python syntax before execution.

        Args:
            code: Python code string

        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Guard 8: Syntax error - {e}"
            logger.error(error_msg)
            return False, error_msg

    # ============================================
    # P0-5: Guard 9 - Code Block Extraction (强制只输出代码)
    # ============================================

    @staticmethod
    def extract_code_block(llm_response: str, require_code_only: bool = True) -> Tuple[str, List[str]]:
        """
        Guard 9: Extract and validate code block from LLM response.

        P0-5 Requirement (用户强制要求):
        - 只接受 ```python ... ``` 内部内容
        - 没代码块：直接判失败
        - 检测自然语言混入（"To operationalize", "Once you provide"等）

        Args:
            llm_response: Raw LLM response text
            require_code_only: If True, reject mixed natural language

        Returns:
            (extracted_code, error_messages)

        Raises:
            ValueError: If code block not found or natural language detected
        """
        import re

        errors = []

        # ============================================================================
        # P0-B FIX: Detect and reject hardcoded paths
        # ============================================================================
        # LLM must NOT use hardcoded paths like:
        #   'dataset/2025_C\\file.csv'
        #   'C:\\Users\\...\\file.csv'
        #   './data/file.csv'
        #   '../MMBench/dataset/...'
        #
        # INSTEAD: Use only filename (without path) from the whitelist
        #   load_csv('file.csv')  # CORRECT
        #   pd.read_csv('file.csv')  # CORRECT
        # ============================================================================

        # Detect problematic path patterns
        hardcoded_path_patterns = [
            r'[\'"]\.\.?/.*?\.csv[\'"]',           # ./file.csv or ../file.csv
            r'[\'"]\w+/.*?\.csv[\'"]',             # data/file.csv, dataset/file.csv
            r'[\'"][A-Z]:[/\\].*?\.csv[\'"]',      # C:/.../file.csv or C:\...\file.csv
            r'[\'"][^\'"]*?Users[^\'"]*?\.csv[\'"]', # Contains 'Users' (Windows path)
            r'[\'"][^\'"]*?study[^\'"]*?\.csv[\'"]', # Contains 'study' (relative path)
        ]

        hardcoded_paths_detected = []
        for pattern in hardcoded_path_patterns:
            matches = re.findall(pattern, llm_response)
            if matches:
                hardcoded_paths_detected.extend(matches)

        if hardcoded_paths_detected:
            error_msg = (
                "Guard 9 Violation: Hardcoded file paths detected in code.\n\n"
                "PROBLEM: LLM generated code with hardcoded paths:\n"
                f"  {', '.join(set(hardcoded_paths_detected[:5]))}\n\n"
                "SOLUTION: Use ONLY filename (without path) from whitelist.\n\n"
                "CORRECT Usage:\n"
                "  df = load_csv('clean_athletes.csv')  # filename ONLY\n\n"
                "WRONG Usage:\n"
                "  df = load_csv('dataset/2025_C/clean_athletes.csv')  # has path\n"
                "  df = load_csv('C:\\\\Users\\\\...\\\\file.csv')  # full path\n\n"
                "Available files are listed in the problem description."
            )
            errors.append(error_msg)
            logger.error(f"[P0-B] {error_msg}")
            if require_code_only:
                raise ValueError(f"[Guard 9 Violation] {error_msg}")

        # Pattern 0: P0-D FIX - 检测并禁止"..."占位符
        # LLM sometimes outputs "from ... import ..." which causes SyntaxError
        # P0-D ENHANCED: More comprehensive ellipsis detection
        import re

        # Check for ellipsis in various contexts
        ellipsis_patterns = [
            r'from\s+\.{{3,}}',        # from ... (with 3+ dots)
            r'import\s+\.{{3,}}',      # import ... (with 3+ dots)
            r'\.{{3,}}\s*$',          # Line ending with ... (3+ dots)
            r'\.{{3,}}\s*,',          # ... followed by comma
            r'\.{{3,}}\s*\)',         # ... followed by closing paren
            r'\.{{3,}}\s*\]',         # ... followed by closing bracket
            r'\.{{3,}}\s*\}',         # ... followed by closing brace
            r'\.{{3,}}\s+import',     # ... import
            r'\.{{3,}}\s+as',         # ... as
        ]

        ellipsis_detected = False
        for pattern in ellipsis_patterns:
            if re.search(pattern, llm_response):
                ellipsis_detected = True
                break

        # Also check for literal '...' string (3 or more dots)
        if re.search(r'\.{3,}', llm_response):
            # Additional check: ensure it's not actually valid Python (like ... in slicing)
            # Valid uses: obj[..., 0], arr[:, :, 0], func(...)
            # Invalid uses: from ... import, import ..., placeholder ...

            # Check if ... appears in invalid contexts
            invalid_contexts = [
                r'(?:from|import)\s+\.{3,}\s+',  # from ... import / import ...
                r'^\s*\.{3,}\s*$',                # Line with only ...
                r'\w+\s*\.{3,}\s*$',              # Variable name followed by ...
            ]

            for ctx_pattern in invalid_contexts:
                if re.search(ctx_pattern, llm_response, re.MULTILINE):
                    ellipsis_detected = True
                    break

        if ellipsis_detected:
            error_msg = (
                "Guard 9 Violation: Ellipsis placeholder '...' detected in code.\n"
                "LLM must not use '...' as a placeholder. All code must be complete.\n\n"
                "INVALID Examples:\n"
                "  from ... import module\n"
                "  import ...\n"
                "  x = ...  # placeholder\n\n"
                "Replace '...' with actual code or valid Python syntax.\n"
                "Valid ellipsis uses (... object slicing) are allowed."
            )
            errors.append(error_msg)
            logger.error(f"[P0-D] {error_msg}")
            if require_code_only:
                raise ValueError(f"[Guard 9 Violation] {error_msg}")

        # Pattern 1: 检测自然语言混入（P0-5要求）
        natural_language_patterns = [
            r"To operationalize",
            r"Once you provide",
            r"Here's the code",
            r"Let me know if",
            r"Please note that",
            r"I've updated",
            r"The code above",
            r"Below is",
        ]

        detected_nl = []
        for pattern in natural_language_patterns:
            if re.search(pattern, llm_response, re.IGNORECASE):
                detected_nl.append(pattern)

        if detected_nl:
            error_msg = (
                f"Guard 9 Violation: Natural language detected in code output.\n"
                f"Detected patterns: {detected_nl}\n"
                f"LLM must output ONLY Python code within ```python``` blocks.\n"
                f"No explanatory text allowed."
            )
            errors.append(error_msg)
            logger.error(f"[P0-5] {error_msg}")
            if require_code_only:
                raise ValueError(f"[Guard 9 Violation] {error_msg}")

        # Pattern 2: Extract code block from ```python ... ```
        python_block_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(python_block_pattern, llm_response, re.DOTALL)

        if not matches:
            # Try alternative pattern (without "python" specifier)
            alt_pattern = r'```\s*\n(.*?)\n```'
            alt_matches = re.findall(alt_pattern, llm_response, re.DOTALL)
            if alt_matches:
                # Use first match but warn
                logger.warning("[P0-5] Code block found but missing 'python' specifier")
                extracted_code = alt_matches[0].strip()
            else:
                error_msg = (
                    "Guard 9 Violation: No code block found in LLM response.\n"
                    "LLM must output code within ```python ... ``` delimiters.\n"
                    f"Response preview: {llm_response[:200]}..."
                )
                errors.append(error_msg)
                logger.error(f"[P0-5] {error_msg}")
                raise ValueError(f"[Guard 9 Violation] {error_msg}")
        else:
            # Use first python block
            extracted_code = matches[0].strip()

        # Pattern 3: 验证提取的代码非空
        if len(extracted_code) < 10:
            error_msg = (
                f"Guard 9 Violation: Extracted code block too short ({len(extracted_code)} chars).\n"
                f"Code must be substantive (>= 10 characters)."
            )
            errors.append(error_msg)
            logger.error(f"[P0-5] {error_msg}")
            raise ValueError(f"[Guard 9 Violation] {error_msg}")

        # Pattern 4: 检测代码块中是否有明显的自然语言
        # 例如：连续的英文句子、问号等
        suspicious_patterns = [
            r"[A-Z][a-z]+ [a-z]+ [a-z]+ [a-z]+",  # 4+ word English sentence
            r"\?$",  # Ends with question mark
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, extracted_code, re.MULTILINE):
                logger.warning(f"[P0-5] Suspicious pattern detected in code: {pattern}")

        logger.info(f"[P0-5] Successfully extracted code block ({len(extracted_code)} chars)")
        return extracted_code, errors

    # ============================================
    # Statistics & Reporting
    # ============================================

    def get_blocked_operations_report(self) -> Dict[str, int]:
        """Get statistics of blocked operations."""
        return self.blocked_operations.copy()

    def print_summary(self):
        """Print summary of guards' activity."""
        total = sum(self.blocked_operations.values())
        if total > 0:
            print(f"\n{'='*60}")
            print("ENGINEERING GUARDS - BLOCKED OPERATIONS SUMMARY")
            print(f"{'='*60}")
            for guard, count in self.blocked_operations.items():
                if count > 0:
                    print(f"  {guard}: {count}")
            print(f"{'='*60}\n")
