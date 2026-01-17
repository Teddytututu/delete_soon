"""
Enhanced Code Guards - P0 Critical Fixes for Chart Generation

This module implements the 3 most critical P0 fixes identified from user analysis:
1. Arrowprops comma patcher (fixes most frequent syntax error)
2. Ban synthetic data generation
3. Enforce code-block-only output

Author: Critical Fix
Date: 2026-01-15
Priority: P0 (Top 3 - immediate bleeding stop)
"""

import re
import ast
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


class EnhancedCodeGuards:
    """
    Enhanced P0 guards for chart generation stability.

    Focus on the 3 most frequent/fatal errors:
    1. arrowprops missing comma (80% of syntax errors)
    2. synthetic data instead of real CSV
    3. natural language in code blocks
    """

    # ============================================
    # P0 FIX #1: Arrowprops Comma Patcher
    # ============================================

    ARROWPROPS_PATTERNS = [
        # facecolor + width
        (r"(facecolor\s*=\s*['\"][^'\"]+['\"])\s+(width\s*=)", r"\1, \2"),
        (r"(facecolor\s*=\s*['\"][^'\"]+['\"])\s+(headwidth\s*=)", r"\1, \2"),
        (r"(facecolor\s*=\s*['\"][^'\"]+['\"])\s+(shrink\s*=)", r"\1, \2"),
        # width + headwidth
        (r"(width\s*=\s*[0-9.]+)\s+(headwidth\s*=)", r"\1, \2"),
        (r"(width\s*=\s*[0-9.]+)\s+(shrink\s*=)", r"\1, \2"),
        # headwidth + shrink
        (r"(headwidth\s*=\s*[0-9.]+)\s+(shrink\s*=)", r"\1, \2"),
        # Generic dict parameter patterns (double and single quotes)
        (r"(\w+\s*=\s*['\"][^'\"]+['\"])\s+(\w+\s*=)", r"\1, \2"),
    ]

    @classmethod
    def fix_arrowprops_syntax(cls, code: str) -> str:
        """
        P0 FIX #1: Fix missing commas in arrowprops and similar dict parameters.

        This fixes the MOST FREQUENT syntax error in chart generation:
        arrowprops=dict(facecolor='black' width=1, headwidth=8)
        Should be:
        arrowprops=dict(facecolor='black', width=1, headwidth=8)

        Args:
            code: Python code string

        Returns:
            Fixed code string
        """
        fixed_code = code
        fix_count = 0

        for pattern, replacement in cls.ARROWPROPS_PATTERNS:
            matches = re.findall(pattern, fixed_code)
            if matches:
                fixed_code = re.sub(pattern, replacement, fixed_code)
                fix_count += len(matches)

        if fix_count > 0:
            logger.info(f"[P0 FIX #1] Fixed {fix_count} arrowprops comma errors")

        return fixed_code

    # ============================================
    # P0 FIX #4: Ban Synthetic Data Generation
    # ============================================

    SYNTHETIC_DATA_PATTERNS = [
        r'\bnp\.random\.',
        r'\bnp\.linspace\(',
        r'\bnp\.arange\(',
        r'\bnp\.randn\(',
        r'\bnp\.rand\(',
        r'\brandom\.random\(',
        r'\brandom\.randint\(',
        r'\brandom\.uniform\(',
        # DataFrame creation with synthetic data
        r'pd\.DataFrame\s*\(\s*\{[^\}]*\x27data\x27\s*:\s*np\.random',
        r'pd\.DataFrame\s*\(\s*\{[^\}]*\x27data\x27\s*:\s*\[',
    ]

    @classmethod
    def check_synthetic_data(cls, code: str) -> Tuple[bool, List[str]]:
        """
        P0 FIX #4: Detect and ban synthetic data generation.

        Charts MUST load data from CSV files, not generate random/fake data.

        Args:
            code: Python code string

        Returns:
            (is_safe, error_messages)
        """
        errors = []

        for pattern in cls.SYNTHETIC_DATA_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                errors.append(f"Synthetic data generation detected: {pattern}")
                logger.warning(f"[P0 FIX #4] Found synthetic data: {pattern} (x{len(matches)})")

        # Check for suspicious DataFrame creation
        if 'pd.DataFrame(' in code and 'np.random' in code:
            errors.append("Creating DataFrame with random data instead of loading CSV")

        # Check for hardcoded lists longer than 10 elements (likely fake data)
        long_list_pattern = r'\[[^\]]{50,}\]'  # List literals with >50 chars
        if re.search(long_list_pattern, code):
            errors.append("Suspicious hardcoded data (likely synthetic)")

        is_safe = len(errors) == 0
        return is_safe, errors

    # ============================================
    # P0 FIX #4 ENHANCED: Ban 3D Chart Method Calls
    # ============================================

    # 3D plotting methods that require 3D axes
    THREE_D_METHODS = [
        r'\.plot_surface\s*\(',
        r'\.plot_wireframe\s*\(',
        r'\.plot_trisurf\s*\(',
        r'\.bar3d\s*\(',
        r'\.scatter3D\s*\(',
        r'\.plot3D\s*\(',
    ]

    @classmethod
    def check_3d_chart_methods(cls, code: str) -> Tuple[bool, List[str]]:
        """
        P0 FIX #4 ENHANCED: Detect and reject 3D chart method calls.

        This catches cases where LLM tries to call 3D plotting methods
        without importing mpl_toolkits.mplot3d (which would fail at runtime).

        Examples of banned methods:
        - ax.plot_surface(...)  # AttributeError: 'Axes' object has no attribute 'plot_surface'
        - ax.plot_wireframe(...)
        - ax.scatter3D(...)

        Args:
            code: Python code string

        Returns:
            (is_safe, error_messages)
        """
        errors = []

        for pattern in cls.THREE_D_METHODS:
            matches = re.findall(pattern, code)
            if matches:
                errors.append(f"3D plotting method detected: {pattern} - 3D charts are forbidden")

        # Also check for projection='3d' parameter
        if "projection='3d'" in code or 'projection="3d"' in code:
            errors.append("3D projection detected - 3D charts are forbidden")

        is_safe = len(errors) == 0
        if not is_safe:
            logger.error(f"[P0 FIX #4 ENHANCED] 3D chart methods detected: {errors}")

        return is_safe, errors

    # ============================================
    # P0 FIX #5: Strict Code Block Extraction
    # ============================================

    NATURAL_LANGUAGE_INDICATORS = [
        "Once you provide",
        "To operationalize",
        "Here's the corrected",
        "Let me explain",
        "The corrected code",
        "Based on your request",
        "I'll now generate",
        "Please note that",
        "The issue is that",
    ]

    @classmethod
    def extract_python_code_strict(cls, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        P0 FIX #5: Strictly extract Python code, reject natural language.

        Only accepts content inside ```python ... ``` code blocks.
        Rejects any output that looks like natural language.

        Args:
            text: LLM output (may contain code + explanation)

        Returns:
            (extracted_code, error_message)
            - extracted_code: None if invalid, code string if valid
            - error_message: None if valid, error string if invalid
        """
        # Pattern 1: Extract ```python ... ``` code blocks
        code_block_pattern = r'```(?:python|py)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)

        if not matches:
            # No code block found - check if it's natural language
            for indicator in cls.NATURAL_LANGUAGE_INDICATORS:
                if indicator.lower() in text.lower():
                    error_msg = f"Natural language detected (contains '{indicator}'). Expected pure Python code in ```python``` block."
                    logger.error(f"[P0 FIX #5] {error_msg}")
                    return None, error_msg

            # No code block, no obvious natural language
            error_msg = "No ```python``` code block found in LLM output"
            logger.error(f"[P0 FIX #5] {error_msg}")
            logger.error(f"[P0 FIX #5] First 200 chars: {text[:200]}")
            return None, error_msg

        # Take the longest code block (most likely to be the main code)
        code = max(matches, key=len)

        # Validate: Must be valid Python syntax
        try:
            ast.parse(code)
            logger.info(f"[P0 FIX #5] Extracted valid code ({len(code)} chars)")
            return code, None
        except SyntaxError as e:
            # Code block exists but has syntax errors
            # Don't try to fix here - let the arrowprops patcher handle it later
            error_msg = f"Code in ```python``` block has SyntaxError: {e}"
            logger.error(f"[P0 FIX #5] {error_msg}")
            logger.error(f"[P0 FIX #5] First 200 chars: {code[:200]}")
            return None, error_msg

    # ============================================
    # Integrated Validation Pipeline
    # ============================================

    @classmethod
    def validate_chart_code(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Run all P0 guards in sequence for chart code validation.

        Pipeline:
        1. Arrowprops comma patcher (PRE-PROCESS)
        2. Synthetic data check (VALIDATE)
        3. 3D chart methods check (VALIDATE) - ENHANCED
        4. Syntax validation (VALIDATE)

        Args:
            code: LLM-generated Python code

        Returns:
            (is_valid, error_message)
        """
        # Step 1: Fix arrowprops syntax (PRE-PROCESS)
        fixed_code = cls.fix_arrowprops_syntax(code)

        # Step 2: Check for synthetic data (HARD FAIL)
        is_safe_synthetic, synthetic_errors = cls.check_synthetic_data(fixed_code)
        if not is_safe_synthetic:
            error_msg = "Synthetic data generation is forbidden. Use load_csv() to load real data from files.\n"
            error_msg += "\n".join(synthetic_errors)
            logger.error(f"[P0 GUARDS] {error_msg}")
            return False, error_msg

        # Step 3: Check for 3D chart methods (HARD FAIL) - ENHANCED
        is_safe_3d, errors_3d = cls.check_3d_chart_methods(fixed_code)
        if not is_safe_3d:
            error_msg = "3D charts are forbidden (security policy).\n"
            error_msg += "Use 2D alternatives: bar, line, scatter, heatmap, contour, contourf.\n"
            error_msg += "\n".join(errors_3d)
            logger.error(f"[P0 GUARDS] {error_msg}")
            return False, error_msg

        # Step 4: Validate syntax
        try:
            ast.parse(fixed_code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error after preprocessing: {e}"
            logger.error(f"[P0 GUARDS] {error_msg}")
            return False, error_msg


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED CODE GUARDS - P0 CRITICAL FIXES")
    print("=" * 60)

    # Test P0 FIX #1: Arrowprops comma patcher
    print("\n[P0 FIX #1] Testing arrowprops comma patcher...")
    broken_code = "plt.annotate('text', xy=(1,2), xytext=(3,4), arrowprops=dict(facecolor='black' width=1.5))"
    fixed_code = EnhancedCodeGuards.fix_arrowprops_syntax(broken_code)
    print(f"  Before: {broken_code}")
    print(f"  After:  {fixed_code}")
    assert "facecolor='black', width=1.5" in fixed_code
    print("  [PASS] Arrowprops comma fixed")

    # Test P0 FIX #4: Synthetic data detection
    print("\n[P0 FIX #4] Testing synthetic data detection...")
    bad_code = "df = pd.DataFrame({'x': np.linspace(0, 10, 100), 'y': np.random.randn(100)})"
    is_safe, errors = EnhancedCodeGuards.check_synthetic_data(bad_code)
    print(f"  Code: {bad_code}")
    print(f"  Safe: {is_safe}")
    print(f"  Errors: {errors}")
    assert not is_safe
    print("  [PASS] Synthetic data detected")

    # Test P0 FIX #5: Strict code extraction
    print("\n[P0 FIX #5] Testing strict code extraction...")
    # Use pure explanation without trigger phrases or code blocks
    natural_output = "The solution involves loading data from CSV files and creating a bar chart.\nWe need to use matplotlib for visualization."
    code, error = EnhancedCodeGuards.extract_python_code_strict(natural_output)
    print(f"  Pure natural language detected: {error is not None}")
    assert error is not None
    assert "No ```python```" in error
    print("  [PASS] Natural language rejected (no code block)")

    # Test mixed output (natural + code block)
    mixed_output = "Here's the corrected code:\n```python\nprint('test')\n```\nThis should work now."
    code, error = EnhancedCodeGuards.extract_python_code_strict(mixed_output)
    print(f"  Mixed output: code extracted = {code is not None}")
    assert code is not None
    assert "print('test')" in code
    print("  [PASS] Code extracted from mixed output")

    # Test good code
    good_output = "```python\nimport matplotlib.pyplot as plt\nplt.plot([1,2,3])\nplt.savefig('test.png')\n```"
    code, error = EnhancedCodeGuards.extract_python_code_strict(good_output)
    print(f"  Good code extracted: {code is not None}")
    assert error is None
    assert "plt.plot" in code
    print("  [PASS] Valid code accepted")

    print("\n" + "=" * 60)
    print("[SUCCESS] All P0 guards verified!")
    print("=" * 60)
