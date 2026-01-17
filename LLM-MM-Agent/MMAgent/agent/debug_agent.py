
"""
DebugAgent - Self-Healing Feedback Loop for MM-Agent

CRITICAL FIX: Implement state-aware retry mechanism that:
1. Captures actual error information (traceback, missing columns, etc.)
2. Extracts system state (available data, actual columns, etc.)
3. Provides SPECIFIC debugging information to LLM
4. Enables intelligent self-correction instead of blind retry

This replaces the "fake" self-correction that just says "please fix this error".
"""

import re
import traceback
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from pathlib import Path


class DebugAgent:
    """
    DebugAgent analyzes execution failures and provides intelligent fix suggestions.

    Unlike the naive fallback_strategy that just repeats "fix the error",
    DebugAgent extracts concrete information about what went wrong.
    """

    @staticmethod
    def diagnose_code_error(
        error_message: str,
        code: str,
        available_data_files: Optional[List[str]] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze code execution error and provide detailed diagnosis.

        Args:
            error_message: The error message
            code: The code that failed
            available_data_files: List of available data files (for context)
            stdout: Standard output from execution
            stderr: Standard error from execution

        Returns:
            Dictionary with diagnosis:
            - error_type: Type of error (KeyError, FileNotFoundError, etc.)
            - root_cause: Human-readable explanation
            - suggested_fix: Specific fix instructions
            - context_info: Additional context (available columns, etc.)
        """
        diagnosis = {
            'error_type': 'Unknown',
            'root_cause': error_message,
            'suggested_fix': 'Review the error message and code logic',
            'context_info': {}
        }

        # CRITICAL: Detect KeyError and extract missing column name
        if 'KeyError' in error_message:
            diagnosis['error_type'] = 'KeyError'

            # Extract column name from error
            match = re.search(r"KeyError:\s*['\"]([^'\"]+)['\"]", error_message)
            if match:
                missing_column = match.group(1)

                # Search for the column usage in code
                column_usage = DebugAgent._find_column_usage(code, missing_column)

                diagnosis['root_cause'] = (
                    f"Code references column '{missing_column}' which does not exist in the dataset. "
                    f"Found {len(column_usage)} usage(s) in code."
                )
                diagnosis['suggested_fix'] = (
                    f"Replace ALL occurrences of df['{missing_column}'] or df.{missing_column} "
                    f"with actual column names from the dataset."
                )
                diagnosis['context_info'] = {
                    'missing_column': missing_column,
                    'usage_locations': column_usage
                }

                # CRITICAL: Extract available columns from data files
                if available_data_files:
                    available_columns = DebugAgent._extract_available_columns(available_data_files)
                    if available_columns:
                        diagnosis['context_info']['available_columns'] = available_columns

                        # Suggest similar columns using fuzzy matching
                        similar = DebugAgent._find_similar_columns(missing_column, available_columns)
                        if similar:
                            diagnosis['context_info']['suggested_alternatives'] = similar
                            diagnosis['suggested_fix'] += (
                                f"\n\nDid you mean: {similar}? "
                                f"Available columns: {available_columns}"
                            )

        return diagnosis

    @staticmethod
    def _find_column_usage(code: str, column_name: str) -> List[Dict[str, int]]:
        """Find all usages of a column name in code."""
        usages = []
        lines = code.split('\n')

        for i, line in enumerate(lines, start=1):
            if f"['{column_name}']" in line or f'["{column_name}"]' in line:
                usages.append({'line': i, 'type': 'bracket', 'snippet': line.strip()})
            if re.search(rf'df\.{column_name}', line):
                usages.append({'line': i, 'type': 'dot', 'snippet': line.strip()})

        return usages

    @staticmethod
    def _extract_available_columns(data_files: List[str]) -> List[str]:
        """Extract all available column names from data files."""
        all_columns = set()

        for data_file in data_files:
            try:
                file_path = Path(data_file)
                if file_path.exists():
                    df = pd.read_csv(data_file, nrows=0, encoding='latin-1')
                    all_columns.update(df.columns.tolist())
            except Exception as e:
                continue

        return sorted(list(all_columns))

    @staticmethod
    def _find_similar_columns(target: str, available: List[str], n: int = 3) -> List[str]:
        """Find similar column names using fuzzy matching."""
        import difflib
        return difflib.get_close_matches(target, available, n=n, cutoff=0.3)

    @staticmethod
    def generate_fix_prompt(
        original_code: str,
        diagnosis: Dict[str, Any],
        task_context: Optional[str] = None
    ) -> str:
        """Generate an intelligent fix prompt based on diagnosis."""
        prompt = f"""# DEBUG REQUEST: Fix Code Error

## Error Analysis
**Error Type**: {diagnosis['error_type']}
**Root Cause**: {diagnosis['root_cause']}

## Suggested Fix
{diagnosis['suggested_fix']}
"""

        if diagnosis.get('context_info'):
            context = diagnosis['context_info']
            if 'available_columns' in context:
                prompt += f"""

## Available Data Columns
```
{context['available_columns']}
```
"""
                prompt += "**CRITICAL**: Use ONLY these column names.\n"

        prompt += f"""
## Original Code
```python
{original_code}
```
"""

        prompt += """
## Instructions
Fix the error based on the analysis above.
"""

        return prompt
