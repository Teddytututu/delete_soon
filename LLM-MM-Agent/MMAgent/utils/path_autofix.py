"""
路径AutoFix模块 - 问题3修复

统一Chart和Computational阶段的路径自动修复逻辑。

核心功能：
- 自动检测并修复代码中的硬编码路径
- 只保留文件名（basename），移除所有路径部分
- 支持pd.read_csv和load_csv两种模式
- 白名单验证：只处理允许的文件

问题3修复：
- 将autofix逻辑从create_charts.py提取到公共模块
- Chart和Computational共用同一个AutoFix函数
- 确保两个阶段的路径处理一致

Author: Claude Code (AI Assistant)
Date: 2026-01-16
Priority: P0 (Critical - fixes path Guard violations)
"""

import os
import re
import logging
from typing import Tuple, List, Set

logger = logging.getLogger(__name__)


def autofix_hardcoded_paths(code_str: str, allowed_files: Set[str]) -> Tuple[str, List[str]]:
    """
    Automatically rewrite hardcoded paths to basename.

    This is a unified AutoFix function used by BOTH Chart and Computational stages.

    Examples:
        pd.read_csv("C:/Users/.../file.csv") -> pd.read_csv("file.csv")
        load_csv('C:\\Users\\...\\file.csv') -> load_csv("file.csv")

    Args:
        code_str: Code string that may contain hardcoded paths
        allowed_files: Set of allowed filenames (whitelist)

    Returns:
        (fixed_code, list_of_fixes_applied)

    问题3修复历史：
    - 2026-01-16: 从create_charts.py提取到公共模块
    - 确保Chart和Computational使用相同逻辑
    """
    fixes_applied = []

    # Pattern 1: pd.read_csv("path") or pd.read_csv('path')
    def replace_read_csv_path(match):
        full_path = match.group(1)
        basename = os.path.basename(full_path.replace("\\", "/"))
        if basename in allowed_files:
            if basename != full_path:
                fixes_applied.append(f"Rewrote path: {full_path} → {basename}")
            return f'pd.read_csv("{basename}")'
        return match.group(0)

    code_str = re.sub(
        r'pd\.read_csv\(\s*[rR]?["\']([^"\']+)["\']\s*\)',
        replace_read_csv_path,
        code_str
    )

    # Pattern 2: load_csv("path")
    def replace_load_csv_path(match):
        full_path = match.group(1)
        basename = os.path.basename(full_path.replace("\\", "/"))
        if basename in allowed_files:
            if basename != full_path:
                fixes_applied.append(f"Rewrote path: {full_path} → {basename}")
            return f'load_csv("{basename}")'
        return match.group(0)

    code_str = re.sub(
        r'load_csv\(\s*[rR]?["\']([^"\']+)["\']\s*\)',
        replace_load_csv_path,
        code_str
    )

    # Pattern 3: Generic file paths (backup pattern)
    # Matches things like: "C:/Users/.../file.csv" or r'C:\Users\...\file.csv'
    def replace_generic_path(match):
        quote_char = match.group(1)  # " or '
        prefix = match.group(2) if match.group(2) else ''  # r or R prefix
        full_path = match.group(3)
        basename = os.path.basename(full_path.replace("\\", "/"))

        if basename in allowed_files and basename != full_path:
            fixes_applied.append(f"Rewrote generic path: {full_path} → {basename}")
            return f'{quote_char}{basename}{quote_char}'

        return match.group(0)

    # Pattern: quote + optional r/R + path with :/ or :\ (Windows paths)
    code_str = re.sub(
        r'(["\'])([rR]?)?([A-Za-z]:[/\\][^"\']+)\1',
        replace_generic_path,
        code_str
    )

    if fixes_applied:
        logger.info(f"[问题3-AutoFix] Applied {len(fixes_applied)} path fix(es)")
        for fix in fixes_applied:
            logger.debug(f"[问题3-AutoFix] {fix}")
    else:
        logger.debug("[问题3-AutoFix] No hardcoded paths found")

    return code_str, fixes_applied


def check_for_remaining_paths(code_str: str, allowed_files: Set[str]) -> List[str]:
    """
    Check if code still contains hardcoded paths after AutoFix.

    Args:
        code_str: Code string to check
        allowed_files: Set of allowed filenames (whitelist)

    Returns:
        List of error messages for remaining violations
    """
    errors = []

    # Pattern: Windows paths (C:/, C:\, etc.)
    windows_path_pattern = r'["\'][A-Za-z]:[/\\][^"\']*["\']'
    if re.search(windows_path_pattern, code_str):
        errors.append("Windows absolute paths detected (e.g., C:/Users/...)")

    # Pattern: Relative paths with directory separators
    relative_path_pattern = r'["\'][^"\']*[/\\][^"\']*["\']'
    matches = re.findall(relative_path_pattern, code_str)
    for match in matches:
        # Extract the path
        path = match.strip('"\'').strip()

        # Skip if it's just a filename (no directory separators)
        if '/' not in path and '\\' not in path:
            continue

        # Check if it's in allowed list
        basename = os.path.basename(path.replace("\\", "/"))
        if basename not in allowed_files:
            errors.append(f"Path not in whitelist: {path}")

    # Pattern: Path() constructor with paths
    path_constructor_pattern = r'Path\(["\'][^"\']+["\']\)'
    if re.search(path_constructor_pattern, code_str):
        errors.append("Path() constructor with string argument (use filename only)")

    return errors


def apply_path_autofix_with_validation(code_str: str, allowed_files: Set[str]) -> Tuple[str, bool, List[str]]:
    """
    Apply AutoFix and validate the result.

    This is the main entry point for path AutoFix in both Chart and Computational stages.

    Args:
        code_str: Code string that may contain hardcoded paths
        allowed_files: Set of allowed filenames (whitelist)

    Returns:
        (fixed_code, success, error_messages)

    Example:
        code, success, errors = apply_path_autofix_with_validation(
            raw_code,
            {'clean_athletes.csv', 'clean_hosts.csv'}
        )
        if not success:
            print(f"Path errors: {errors}")
    """
    # Step 1: Apply AutoFix
    fixed_code, fixes = autofix_hardcoded_paths(code_str, allowed_files)

    # Step 2: Check for remaining violations
    remaining_errors = check_for_remaining_paths(fixed_code, allowed_files)

    if remaining_errors:
        # AutoFix couldn't resolve all issues
        return fixed_code, False, remaining_errors

    return fixed_code, True, []


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_autofix(code: str, allowed_files: List[str]) -> str:
    """
    Quick autofix without validation (use when you just want the fixed code).

    Args:
        code: Code string
        allowed_files: List of allowed filenames

    Returns:
        Fixed code string
    """
    allowed_set = set(allowed_files)
    fixed_code, _ = autofix_hardcoded_paths(code, allowed_set)
    return fixed_code


# ============================================================================
# 问题3修复历史
# ============================================================================

"""
修复历史：
- 2026-01-16: 从create_charts.py提取autofix_hardcoded_paths函数到公共模块
- 确保Chart和Computational阶段使用相同的路径处理逻辑
- 添加check_for_remaining_paths验证函数
- 添加apply_path_autofix_with_validation统一入口

相关文件：
- MMAgent/agent/create_charts.py: 原始autofix实现（已移除）
- MMAgent/utils/computational_solving.py: 使用公共autofix（待集成）
- MMAgent/utils/path_autofix.py: 本文件（公共模块）
"""
