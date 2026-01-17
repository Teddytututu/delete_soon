"""
P0-GUARD8: Automatic Syntax Error Fixer for LLM-Generated Code

This module attempts to automatically fix common syntax errors in LLM-generated
matplotlib code before failing the chart generation.

Author: P0 Fix
Date: 2026-01-15
Priority: P0 (Critical - prevents chart generation failures)
P0-3 FIX: Added pre_syntax_sanitize() for syntax garbage like "=,"
"""

import re
import logging
import ast

logger = logging.getLogger(__name__)


# ============================================================================
# P0-3: PRE-SYNTAX SANITIZE (Fix syntax garbage before AST parsing)
# ============================================================================

def pre_syntax_sanitize(code: str) -> tuple[str, list[str]]:
    """
    P0-3 FIX: Sanitize code by removing syntax garbage BEFORE validation.

    Fixes LLM-generated syntax garbage like:
    - `df =, pd.read_csv(...)`  →  `df = pd.read_csv(...)`
    - `x ==, y`                 →  `x == y`
    - `,,`                       →  `,`
    - Multiple trailing commas

    Args:
        code: Raw LLM-generated code that may contain syntax garbage

    Returns:
        (sanitized_code, list_of_fixes_applied)
    """
    original_code = code
    fixes_applied = []

    # FIX 1: Remove comma after equals sign (most common pattern from user's logs)
    # Pattern: `= ,` or `=,` followed by space or expression
    pattern_comma_after_equals = r'=\s*,\s*'
    if re.search(pattern_comma_after_equals, code):
        code = re.sub(pattern_comma_after_equals, '= ', code)
        fixes_applied.append("  - Removed comma after equals sign (= ,)")

    # FIX 2: Remove double commas (`,,`)
    pattern_double_comma = r',\s*,'
    if re.search(pattern_double_comma, code):
        code = re.sub(pattern_double_comma, ', ', code)
        fixes_applied.append("  - Removed double commas (,,)")

    # FIX 3: Remove trailing commas at end of lines (before newline)
    pattern_trailing_comma = r',(\s*\n)'
    if re.search(pattern_trailing_comma, code):
        code = re.sub(pattern_trailing_comma, r'\1', code)
        fixes_applied.append("  - Removed trailing commas at end of lines")

    # FIX 4: Fix spaces around equals in assignments (for consistency)
    # This doesn't fix syntax errors but improves code quality
    pattern_spaces_equals = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*'
    # (Not applied - just for documentation, may be too aggressive)

    # FIX 5: Remove commas before closing brackets/parentheses
    pattern_comma_before_close = r',\s*([\]\}\)])'
    if re.search(pattern_comma_before_close, code):
        code = re.sub(pattern_comma_before_close, r'\1', code)
        fixes_applied.append("  - Removed commas before closing brackets")

    # FIX 6: Remove empty parentheses in function calls (caused by LLM hallucination)
    # Pattern: `func_name(, )` or `func_name(,)`
    pattern_empty_paren_comma = r'\(\s*,\s*\)'
    if re.search(pattern_empty_paren_comma, code):
        code = re.sub(pattern_empty_paren_comma, '()', code)
        fixes_applied.append("  - Fixed empty parentheses with commas")

    if fixes_applied:
        logger.info(f"[P0-3] pre_syntax_sanitize: Applied {len(fixes_applied)} fixes")
        for fix in fixes_applied:
            logger.debug(f"[P0-3] {fix}")
    else:
        logger.debug("[P0-3] No syntax garbage found")

    return code, fixes_applied


class SyntaxAutoFixer:
    """
    Automatically fix common syntax errors in LLM-generated code.

    This is NOT a replacement for proper code generation - it's a safety net
    to prevent pipeline crashes when LLM makes predictable mistakes.
    """

    # Common syntax patterns that LLMs get wrong
    FIX_PATTERNS = [
        # Pattern 1: Assignment in if/while conditions (should be ==)
        (
            r'\bif\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^)]+)\s*\)',
            r'if (\1 == \2)',
            "Assignment in if condition (changed = to ==)"
        ),
        (
            r'\bwhile\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^)]+)\s*\)',
            r'while (\1 == \2)',
            "Assignment in while condition (changed = to ==)"
        ),

        # Pattern 2: Assignment without comparison in condition
        (
            r'\bif\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^:\s]+)\s*:',
            r'if \1 == \2:',
            "Assignment in if statement (changed = to ==)"
        ),

        # Pattern 3: Walrus operator misuse (:= should be = in simple assignment)
        (
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:=\s*([^,\n]+)\s*$',  # At end of line
            r'\1 = \2',
            "Walrus operator in simple assignment (changed := to =)"
        ),

        # Pattern 4: Missing colons after if/elif/else/for/while/def/class
        (
            r'\b(if|elif|else|for|while|def|class)\s*([^\n:]*)(?<!:)\s*\n',
            r'\1 \2:\n',
            "Missing colon after control structure"
        ),

        # Pattern 5: Double equals in assignment (should be single =)
        (
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*==\s*([^,\n]+)\s*$',  # At end of line, not in if/while
            r'\1 = \2',
            "Comparison in assignment (changed == to =)"
        ),
    ]

    @staticmethod
    def fix_common_syntax_errors(code: str) -> tuple[str, list[str]]:
        """
        Attempt to fix common syntax errors in LLM-generated code.

        Args:
            code: Python code string that may contain syntax errors

        Returns:
            (fixed_code, list_of_fixes_applied)
        """
        original_code = code
        fixes_applied = []

        for pattern, replacement, description in SyntaxAutoFixer.FIX_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                code = re.sub(pattern, replacement, code)
                fixes_applied.append(f"  - {description} ({len(matches)} occurrence(s))")
                logger.info(f"[P0-GUARD8] Auto-fixed: {description}")

        if fixes_applied:
            logger.info(f"[P0-GUARD8] Applied {len(fixes_applied)} syntax fixes")
            # Re-validate after fixes
            try:
                ast.parse(code)
                logger.info("[P0-GUARD8) Fixed code is now syntactically valid")
                return code, fixes_applied
            except SyntaxError as e:
                logger.warning(f"[P0-GUARD8] Fixes applied but syntax still invalid: {e}")
                return code, fixes_applied
        else:
            logger.debug("[P0-GUARD8] No common syntax errors found to fix")

        return code, fixes_applied

    @staticmethod
    def validate_and_fix(code: str) -> tuple[bool, str, str]:
        """
        Validate syntax and attempt to fix errors.

        Args:
            code: Python code string

        Returns:
            (is_valid, fixed_code, error_message)
            - is_valid: True if code is syntactically valid
            - fixed_code: The (possibly fixed) code
            - error_message: Error message if still invalid
        """
        # First, try to parse as-is
        try:
            ast.parse(code)
            return True, code, None
        except SyntaxError as e:
            initial_error = str(e)
            logger.warning(f"[P0-GUARD8] Initial syntax error: {initial_error}")

        # Try to fix common errors
        fixed_code, fixes = SyntaxAutoFixer.fix_common_syntax_errors(code)

        # Re-validate
        try:
            ast.parse(fixed_code)
            if fixes:
                logger.info("[P0-GUARD8] Successfully fixed syntax errors")
                return True, fixed_code, None
            else:
                # No fixes applied, still has errors
                return False, fixed_code, initial_error
        except SyntaxError as e:
            final_error = str(e)
            logger.error(f"[P0-GUARD8] Syntax error after auto-fix: {final_error}")
            return False, fixed_code, final_error


def apply_syntax_auto_fix(code: str) -> tuple[bool, str, str, list[str]]:
    """
    Convenience function to apply syntax auto-fixing.

    Args:
        code: Python code string

    Returns:
        (success, fixed_code, error_message, fixes_applied)
    """
    is_valid, fixed_code, error_msg = SyntaxAutoFixer.validate_and_fix(code)

    if is_valid:
        return True, fixed_code, None, []
    else:
        # Extract what fixes were attempted
        _, fixes = SyntaxAutoFixer.fix_common_syntax_errors(code)
        return False, fixed_code, error_msg, fixes
