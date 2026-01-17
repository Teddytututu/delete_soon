"""
Schema Normalization Utilities - P0 Priority Fixes

This module provides normalization functions for schema stability and crash prevention.
Implements fixes for E1, E3, E4, and E7 from the comprehensive error analysis.

Author: Engineering Fix
Date: 2026-01-16
Priority: P0 (Critical crash prevention)
"""

import ast
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================
# E1 FIX: normalize_chart() - Prevents KeyError crashes
# ============================================

def normalize_chart(chart: Any) -> Dict[str, Any]:
    """
    CRITICAL FIX E1: Normalize chart metadata to prevent KeyError crashes.

    Root cause: chart objects can be dict, list, or custom objects,
    causing chart.get('success') to fail with KeyError.

    This function ALWAYS returns a dict with safe defaults for all required fields.

    FIX 2026-01-16 (问题1): Added 'code' field to Chart Success Contract.
    Without 'code' field, LaTeX stage considers the chart as "missing" even if PNG exists.

    FIX 2026-01-16 (问题5): Added 'status' field for three-state chart classification.
    - SUCCESS: PNG exists + code is non-empty
    - PARTIAL: PNG exists + code is empty (LaTeX can still use PNG)
    - FAILED: PNG doesn't exist

    Args:
        chart: Chart metadata (can be dict, list, custom object, or None)

    Returns:
        Dict with guaranteed keys: success, code, path, title, desc, error, image_path, status

    Chart Success Contract (REQUIRED for LaTeX):
    - success: bool - Whether image was created successfully
    - code: str - Complete executable matplotlib code (may be empty if failed)
    - path: str - Path to generated PNG file
    - error: str - Error message if failed (empty if success)
    - status: str - Three-state classification (SUCCESS/PARTIAL/FAILED)
    """
    import os

    def compute_status(success: bool, code: str, image_path: str) -> str:
        """问题5修复：计算三态状态"""
        # Check if PNG exists
        png_exists = image_path and os.path.exists(str(image_path))

        if not png_exists:
            return "FAILED"  # PNG不存在 → 完全失败

        # PNG存在
        if success and code and len(code.strip()) > 10:
            return "SUCCESS"  # PNG存在 + code完整 → 完全成功
        else:
            return "PARTIAL"  # PNG存在 + code缺失/有问题 → 部分成功

    # Case 1: None or missing
    if chart is None:
        return {
            "success": False,
            "code": "",  # PROBLEM 1 FIX: Always include code field
            "error": "chart is None",
            "path": "",
            "title": "",
            "desc": "",
            "image_path": "",
            "status": "FAILED"  # 问题5修复：三态状态
        }

    # Case 2: Already a dict - just add defaults
    if isinstance(chart, dict):
        chart.setdefault("success", False)
        chart.setdefault("code", "")  # PROBLEM 1 FIX: Always include code field
        chart.setdefault("error", "")
        chart.setdefault("path", "")
        chart.setdefault("title", "")
        chart.setdefault("desc", "")
        chart.setdefault("image_path", "")

        # 问题5修复：计算三态状态
        if "status" not in chart:
            chart["status"] = compute_status(
                chart["success"],
                chart.get("code", ""),
                chart.get("image_path", "")
            )

        return chart

    # Case 3: List - try to extract first dict
    if isinstance(chart, list):
        # Look for first dict element
        first_dict = next((item for item in chart if isinstance(item, dict)), None)
        if first_dict:
            return normalize_chart(first_dict)  # Recurse
        # No dict found - wrap as error
        return {
            "success": False,
            "code": "",  # PROBLEM 1 FIX: Always include code field
            "error": f"chart is list with {len(chart)} non-dict items",
            "path": "",
            "title": "",
            "desc": "",
            "image_path": "",
            "status": "FAILED"  # 问题5修复：三态状态
        }

    # Case 4: Custom object - try to convert to dict
    if hasattr(chart, "__dict__"):
        try:
            chart_dict = dict(chart.__dict__)
            return normalize_chart(chart_dict)  # Recurse to add defaults
        except Exception as e:
            logger.warning(f"Failed to convert chart object to dict: {e}")
            return {
                "success": False,
                "code": "",  # PROBLEM 1 FIX: Always include code field
                "error": f"chart object conversion failed: {e}",
                "path": "",
                "title": "",
                "desc": "",
                "image_path": ""
            }

    # Case 5: Try dict() conversion (for dict-like objects)
    try:
        chart_dict = dict(chart)
        return normalize_chart(chart_dict)  # Recurse to add defaults
    except Exception as e:
        logger.error(f"Cannot normalize chart of type {type(chart)}: {e}")
        return {
            "success": False,
            "code": "",  # PROBLEM 1 FIX: Always include code field
            "error": f"unsupported chart type: {type(chart).__name__}",
            "path": "",
            "title": "",
            "desc": "",
            "image_path": ""
        }


# ============================================
# E3 FIX: normalize_solution() - Prevents evaluation crashes
# ============================================

def normalize_solution(solution: Any) -> Dict[str, Any]:
    """
    CRITICAL FIX E3: Normalize solution JSON to prevent evaluation crashes.

    Root cause: solution.json sometimes has fields as list instead of dict,
    causing .get() calls to fail with AttributeError.

    This function ensures:
    - 'tasks' is always a list of dicts
    - All required top-level fields exist with defaults
    - Each task has required fields with defaults

    Args:
        solution: Solution data (from JSON)

    Returns:
        Normalized solution dict
    """
    # Case 1: None or invalid
    if solution is None:
        return {
            "tasks": [],
            "background": "",
            "problem_requirement": [],
            "status": "ERROR",
            "error": "solution is None"
        }

    # Case 2: Not a dict - try to convert
    if not isinstance(solution, dict):
        try:
            solution = dict(solution)
        except Exception:
            return {
                "tasks": [],
                "background": "",
                "problem_requirement": [],
                "status": "ERROR",
                "error": f"solution is not dict: {type(solution)}"
            }

    # Ensure top-level fields exist
    solution.setdefault("tasks", [])
    solution.setdefault("background", "")
    solution.setdefault("problem_requirement", [])
    solution.setdefault("status", "UNKNOWN")

    # Normalize tasks to list of dicts
    if not isinstance(solution["tasks"], list):
        # Single task as dict? Convert to list
        if isinstance(solution["tasks"], dict):
            solution["tasks"] = [solution["tasks"]]
        else:
            solution["tasks"] = []

    # Normalize each task
    normalized_tasks = []
    for i, task in enumerate(solution["tasks"]):
        if not isinstance(task, dict):
            # Skip non-dict tasks or convert to dict
            try:
                task = dict(task) if hasattr(task, "__iter__") else {"value": task}
            except Exception:
                task = {"error": f"task {i} is not dict"}

        # Ensure task has required fields
        task.setdefault("task_description", "")
        task.setdefault("task_analysis", "")
        task.setdefault("modeling_formulas", "")
        task.setdefault("mathematical_modeling_process", "")
        task.setdefault("task_code", "")
        task.setdefault("is_pass", False)
        task.setdefault("execution_result", "")
        task.setdefault("solution_interpretation", "")
        task.setdefault("subtask_outcome_analysis", "")
        task.setdefault("charts", [])

        # Normalize charts in task
        if isinstance(task.get("charts"), list):
            normalized_charts = [normalize_chart(c) for c in task["charts"]]
            task["charts"] = normalized_charts

        normalized_tasks.append(task)

    solution["tasks"] = normalized_tasks
    return solution


# ============================================
# E4 FIX: assert_syntax_ok() - Pre-execution validation
# ============================================

def assert_syntax_ok(code: str) -> Tuple[bool, Optional[str]]:
    """
    CRITICAL FIX E4: Validate Python syntax before execution.

    Prevents Guard 8 (invalid syntax) crashes by checking syntax
    BEFORE attempting to execute code.

    This is a lightweight wrapper around ast.parse() with better error messages.

    Args:
        code: Python code string to validate

    Returns:
        (is_valid, error_message) - True if valid, False with error msg if invalid
    """
    if not code or not isinstance(code, str):
        return False, "Code is empty or not a string"

    # Quick check for minimum length
    if len(code.strip()) < 10:
        return False, f"Code too short ({len(code)} chars)"

    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        # Provide detailed error message
        error_msg = (
            f"Syntax error at line {e.lineno}, offset {e.offset}:\n"
            f"  {e.msg}\n"
            f"  {e.text}"
            if e.text else f"Syntax error: {e.msg}"
        )
        return False, error_msg
    except Exception as e:
        return False, f"Parsing error: {e}"


# ============================================
# E7 FIX: safe_read_csv() - BOM and encoding handling
# ============================================

def safe_read_csv(
    filepath: str,
    *,
    encoding_hint: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    CRITICAL FIX E7: Read CSV with automatic BOM removal and encoding fallback.

    Root cause: Files with UTF-8 BOM show corrupted column names like 'Ï»¿YEAR'.
    Different platforms save files with different encodings.

    Encoding fallback order:
    1. User-provided hint (if specified)
    2. utf-8-sig (removes BOM automatically)
    3. utf-8 (standard)
    4. latin-1 / cp1252 (Windows CSV files)
    5. iso-8859-1 (very permissive)

    Args:
        filepath: Path to CSV file
        encoding_hint: Optional preferred encoding (skipped if None)
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        pandas.DataFrame with normalized column names (BOM removed, stripped)

    Raises:
        Exception: If all encodings fail
    """
    # Build encoding list
    encodings = []

    if encoding_hint:
        encodings.append(encoding_hint)

    # Standard encoding list (from the fix document)
    encodings += ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]

    # Remove duplicates while preserving order
    seen = set()
    unique_encodings = []
    for enc in encodings:
        if enc not in seen:
            seen.add(enc)
            unique_encodings.append(enc)

    last_error = None

    for encoding in unique_encodings:
        try:
            logger.debug(f"Attempting to read {filepath} with encoding={encoding}")
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)

            # CRITICAL: Normalize column names (remove BOM, strip whitespace)
            # This handles cases where BOM wasn't caught by encoding
            df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

            logger.info(f"Successfully read {filepath} with encoding={encoding}")
            return df

        except (UnicodeDecodeError, UnicodeError) as e:
            # Try next encoding
            logger.debug(f"Failed with encoding={encoding}: {e}")
            last_error = e
            continue
        except Exception as e:
            # Non-encoding error - fail immediately
            logger.error(f"Non-encoding error reading {filepath}: {e}")
            raise

    # All encodings failed
    raise Exception(
        f"Failed to read CSV file {filepath} with any encoding "
        f"(tried: {unique_encodings}). Last error: {last_error}"
    )


# ============================================
# Utility: normalize_dataframe_columns()
# ============================================

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names for consistency.

    Operations:
    1. Remove UTF-8 BOM (\ufeff)
    2. Strip leading/trailing whitespace
    3. Convert to string

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with normalized column names (modified in place, also returned)
    """
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


# ============================================
# Testing/Verification
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("SCHEMA NORMALIZATION UTILITIES - SELF-TEST")
    print("=" * 60)

    # Test 1: normalize_chart()
    print("\n[Test 1] normalize_chart() - E1 Fix")
    test_cases = [
        None,
        {"success": True, "path": "test.png"},
        [{"success": False}, {"success": True}],
        "invalid",
    ]

    for i, case in enumerate(test_cases, 1):
        result = normalize_chart(case)
        print(f"  Case {i}: {type(case).__name__} -> success={result['success']}")

    # Test 2: assert_syntax_ok()
    print("\n[Test 2] assert_syntax_ok() - E4 Fix")
    code_tests = [
        ("valid code", "x = 1\nprint(x)"),
        ("invalid syntax", "x = 1\nprint(x"),  # Missing closing paren
        ("empty", ""),
    ]

    for name, code in code_tests:
        is_valid, error = assert_syntax_ok(code)
        print(f"  {name}: {'OK' if is_valid else 'FAIL - ' + str(error)[:50]}")

    # Test 3: normalize_solution()
    print("\n[Test 3] normalize_solution() - E3 Fix")
    solution_tests = [
        None,
        {"tasks": [{"task_description": "test"}]},
        {"tasks": ["not a dict"]},
        {"tasks": [{"invalid": True}]},
    ]

    for i, sol in enumerate(solution_tests, 1):
        result = normalize_solution(sol)
        print(f"  Solution {i}: {len(result['tasks'])} tasks, status={result.get('status', 'N/A')}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
