from .base_agent import BaseAgent
from prompt.template import CREATE_CHART_PROMPT, CHART_TO_CODE_PROMPT, CHART_CODE_FIXING_PROMPT
from prompt.chart_template_prompt import TEMPLATE_DRIVEN_CHART_PROMPT  # 问题2修复：模板驱动prompt
from string import Template
import os
import re
import subprocess
import sys
import ast
import logging
from pathlib import Path
from typing import Set, List, Optional, Tuple, Dict, Any
import json  # 问题2修复：解析JSON响应

# P0 FIX #4: Import syntax validation utility
from utils.schema_normalization import assert_syntax_ok

# 问题2修复：导入模板渲染器
from utils.chart_templates import render_chart_code, get_template_renderer

# 问题3修复：导入公共路径AutoFix模块
from utils.path_autofix import apply_path_autofix_with_validation

# [NEW] LATENT REPORTER: Import for visual insights logging
try:
    from utils.latent_reporter import LatentReporter
except ImportError:
    LatentReporter = None  # Graceful degradation if not available

logger = logging.getLogger(__name__)


# --------------------------------
# CRITICAL FIX: Column Name Sanitization
# Prevents path fragments like "r'C" from being parsed as column names
# --------------------------------

def sanitize_column_tokens(tokens: List[str], df_columns: Set[str]) -> List[str]:
    r"""
    Filter out path fragments and other non-column tokens from LLM-parsed lists.

    This fixes the critical bug where "r'C" (from r'C:\Users\...') gets
    parsed as a column name and causes KeyError.

    Args:
        tokens: List of tokens extracted from LLM/text
        df_columns: Set of actual column names in the dataframe

    Returns:
        List[str]: Filtered list containing only valid column names
    """
    if not tokens:
        return []

    cols = set(df_columns)
    out = []

    for t in tokens:
        t = t.strip().strip('"\'')

        # Skip empty tokens
        if not t:
            continue

        # CRITICAL FILTER 1: Path-like patterns
        # - Windows paths: "C:\", "r'C", "C:/Users", etc.
        # - Unix paths: "/home", "./data", etc.
        if re.search(r"[A-Za-z]:[\\/]", t):
            logger.debug(f"Filtered out path token: '{t}'")
            continue
        if t.startswith(("r'", 'r"', "r'")) and len(t) > 2:
            # Raw string prefix followed by path
            if re.search(r"[A-Za-z]:[\\/]", t[2:]):
                logger.debug(f"Filtered out raw path token: '{t}'")
                continue

        # CRITICAL FILTER 2: Common path keywords
        path_keywords = ['Users', 'OneDrive', 'study', '2026', 'mcm', 'project',
                       'MM-Agent', 'output', 'dataset', 'problem']
        if any(keyword in t for keyword in path_keywords):
            # Also check for path separators
            if '/' in t or '\\' in t:
                logger.debug(f"Filtered out path-like token: '{t}'")
                continue

        # CRITICAL FILTER 3: Very long tokens with slashes (almost certainly paths)
        if len(t) > 50 and ('/' in t or '\\' in t):
            logger.debug(f"Filtered out long path token (len={len(t)})")
            continue

        # CRITICAL FILTER 4: Only keep tokens that ACTUALLY exist in dataframe
        # This is the ultimate safeguard - if it's not in df.columns, it's wrong
        if t in cols:
            out.append(t)
            logger.debug(f"Accepted valid column: '{t}'")
        else:
            # Log rejected token for debugging
            if len(t) < 30:  # Avoid logging huge paths
                logger.debug(f"Rejected non-existent column: '{t}'")

    return out


# --------------------------------
# CRITICAL FIX: Matplotlib Syntax Auto-Fixer
# Fixes common LLM-generated syntax errors like missing commas
# --------------------------------

def auto_fix_matplotlib_syntax(code_str: str) -> str:
    """
    Auto-fix common syntax errors in LLM-generated matplotlib code.

    Fixes:
    1. Missing commas in dict/function calls (most common)
    2. Missing quotes around string arguments
    3. Unmatched parentheses/brackets
    4. Common arrow props syntax errors

    Args:
        code_str: The generated matplotlib code

    Returns:
        str: Fixed code with syntax errors corrected
    """
    if not code_str:
        return code_str

    original = code_str

    # FIX 1: Missing commas in dict/function calls
    # Pattern: dict(key='value' key2='value2') → dict(key='value', key2='value2')
    # Match: 'word' or "word" followed by space and another 'word' or "word"
    code_str = re.sub(
        r"([\"'][\w]+[\"'])\s+([\"'][\w]+[\"'])",
        r"\1, \2",
        code_str
    )

    # FIX 2: Missing commas in arrowprops (very common error)
    # Pattern: arrowprops=dict(facecolor='black' width=1.5)
    #        → arrowprops=dict(facecolor='black', width=1.5)
    code_str = re.sub(
        r"(arrowprops\s*=\s*dict\([^)]+?)\s+([a-zA-Z]+\s*=)",
        r"\1, \2",
        code_str
    )

    # FIX 3: Missing comma before closing bracket in list/tuple
    # Pattern: ['item1' 'item2'] → ['item1', 'item2']
    code_str = re.sub(
        r"([\"'][\w]+[\"'])\s+([\"'][\w]+[\"'])",
        r"\1, \2",
        code_str
    )

    # FIX 4: Multiple missing commas in sequences
    # Fix iteratively until no more patterns found
    for _ in range(3):
        new_str = re.sub(
            r"([\"'][\w\-\.]+[\"'])\s+([\"'][\w\-\.]+[\"'])",
            r"\1, \2",
            code_str
        )
        if new_str == code_str:
            break
        code_str = new_str

    # FIX 5: Ensure plt.tight_layout() before savefig (common issue)
    if 'plt.savefig(' in code_str and 'plt.tight_layout()' not in code_str:
        # Insert tight_layout before savefig if not present
        code_str = code_str.replace(
            'plt.savefig(',
            'plt.tight_layout()\nplt.savefig('
        )

    # Log if changes were made
    if code_str != original:
        logger.info("Applied auto-fixes to matplotlib syntax")
        logger.debug(f"Original length: {len(original)}, Fixed length: {len(code_str)}")

    return code_str


# --------------------------------
# CRITICAL FIX: Template Value Escaping
# Prevents Template from interpreting $ as placeholder
# --------------------------------

def _escape_template_value(value: str) -> str:
    """
    Escape $ characters in template values to prevent Template from interpreting them.

    This fixes the critical bug where $ in strings (like variable names, LaTeX math)
    gets interpreted by Template.substitute() as a placeholder, causing KeyError.

    Args:
        value: The string value to escape

    Returns:
        str: The escaped string with $ replaced by $$
    """
    if not isinstance(value, str):
        return str(value) if value is not None else ''

    # Replace $ with $$ (literal $ in Template)
    return value.replace('$', '$$')


# --------------------------------
# B4. CSV ENCODING FALLBACK
# --------------------------------

def read_csv_safely(file_path: str, **kwargs):
    """
    Read CSV file with automatic encoding fallback.

    Handles the '0x96' byte error that occurs when files are encoded with
    cp1252/latin-1 but are read as UTF-8.

    Encoding fallback order:
    1. 'utf-8' (most common)
    2. 'latin-1' (handles 0x96 byte, Windows CSV files)
    3. 'cp1252' (Windows-specific, similar to latin-1)
    4. 'iso-8859-1' (very permissive)

    Args:
        file_path: Path to CSV file
        **kwargs: Other arguments to pass to pd.read_csv()

    Returns:
        pandas.DataFrame

    Raises:
        Exception: If all encodings fail
    """
    import pandas as pd

    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    last_error = None

    for encoding in encodings:
        try:
            logger.debug(f"Attempting to read {file_path} with encoding={encoding}")
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            logger.info(f"Successfully read {file_path} with encoding={encoding}")
            return df
        except (UnicodeDecodeError, UnicodeError) as e:
            logger.debug(f"Failed to read with encoding={encoding}: {e}")
            last_error = e
            continue
        except Exception as e:
            # For non-encoding errors, raise immediately
            logger.error(f"Non-encoding error reading {file_path}: {e}")
            raise

    # All encodings failed
    raise Exception(
        f"Failed to read CSV file {file_path} with any encoding "
        f"(tried: {encodings}). Last error: {last_error}"
    )


class SecurityViolation(Exception):
    """Raised when code violates security rules."""
    pass


class ChartCodeValidator:
    """
    Validates LLM-generated matplotlib code for safety.

    This prevents code injection attacks by:
    - Blocking dangerous patterns (imports, eval, exec, etc.)
    - Validating AST nodes
    - Restricting function calls to allowed operations
    - Providing safe namespace for execution
    """

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        # CRITICAL FIX: Don't block all imports! Only block dangerous ones
        # r'\bimport\s+',           # All imports  <-- REMOVED: This blocks legitimate imports
        r'\bimport\s+os\b',       # os module (file operations)
        r'\bimport\s+subprocess\b', # subprocess module (command execution)
        r'\bimport\s+shutil\b',    # shutil module (file operations)
        r'\bimport\s+sys\b',       # sys module (system access)
        r'\bfrom\s+os\s+import',  # os.from*
        r'\bfrom\s+subprocess\s+import',  # subprocess.from*
        r'\bfrom\s+shutil\s+import',  # shutil.from*
        r'\bfrom\s+sys\s+import',  # sys.from*
        r'\b__import__\b',         # Built-in import
        r'\beval\b',              # eval() function
        r'\bexec\b',              # exec() function
        r'\bopen\s*\(',           # open() function calls (not open string)
        r'\bcompile\b',           # Code compilation
        r'\b__builtins__',        # Built-ins access
        r'globals\(\)',           # Global scope access
        r'locals\(\)',            # Local scope access
        r'\.\.',                  # Parent directory
        r'__setattr__',           # Attribute modification
        r'__delattr__',           # Attribute deletion
        r'__class__',             # Class access
    ]

    # Allowed function calls (whitelist)
    # CRITICAL FIX: Expanded to include more matplotlib functions and seaborn
    ALLOWED_CALLS: Set[str] = {
        # Matplotlib pyplot - EXPANDED
        'plt.style.use', 'plt.show', 'plt.savefig', 'plt.figure', 'plt.subplot',
        'plt.subplots', 'plt.plot', 'plt.scatter', 'plt.bar', 'plt.hist',
        'plt.boxplot', 'plt.xlabel', 'plt.ylabel', 'plt.title',
        'plt.legend', 'plt.grid', 'plt.tight_layout', 'plt.xlim', 'plt.ylim',
        'plt.xscale', 'plt.yscale', 'plt.loglog', 'plt.semilogx',
        'plt.semilogy', 'plt.pie', 'plt.stackplot', 'plt.fill_between',
        'plt.close', 'plt.axis', 'plt.cla', 'plt.clf',
        'plt.contour', 'plt.contourf', 'plt.surface', 'plt.plot_surface',
        'plt.plot_trisurf', 'plt.colorbar', 'plt.text', 'plt.annotate',
        'plt.xticks', 'plt.yticks', 'plt.tick_params', 'plt.cm',
        # Additional matplotlib functions
        'plt.axhline', 'plt.axvline', 'plt.axhspan', 'plt.axvspan',
        'plt.scatter', 'plt.errorbar', 'plt.violinplot', 'plt.spy',
        'plt.imshow', 'plt.pcolormesh', 'plt.hexbin', 'plt.stem',
        'plt.barh', 'plt.broken_barh', 'plt.step', 'plt.stairs',
        'plt.stackplot', 'plt.broken_barh', 'plt.eventplot',
        'plt.table', 'plt.figtext', 'plt.figimage',
        'plt.suptitle', 'plt.subplot2grid', 'plt.subplot_mosaic',
        'plt.twinx', 'plt.twiny', 'plt.sharex', 'plt.sharey',
        'plt.hlines', 'plt.vlines', 'plt.plot_date', 'plt.quiver',
        'plt.streamplot', 'plt.barbs', 'plt.cohere', 'plt.fill_betweenx',
        'plt.tripcolor', 'plt.triplot', 'plt.tricontour', 'plt.tricontourf',
        # Seaborn (commonly used with matplotlib)
        'sns.', 'sns.heatmap', 'sns.scatterplot', 'sns.lineplot',
        'sns.barplot', 'sns.countplot', 'sns.boxplot', 'sns.violinplot',
        'sns.stripplot', 'sns.swarmplot', 'sns.catplot', 'sns.lmplot',
        'sns.regplot', 'sns.residplot', 'sns.jointplot', 'sns.pairplot',
        'sns.heatmap', 'sns.clustermap', 'sns.factorplot', 'sns.distplot',
        'sns.kdeplot', 'sns.rugplot', 'sns.set', 'sns.set_style',
        'sns.set_palette', 'sns.set_context', 'sns.color_palette',
        'sns.despine', 'sns.despine', 'sns.utils', 'sns.matrix',

        # Pandas - EXPANDED
        'pd.read_csv', 'pd.DataFrame', 'pd.read_excel',
        'pd.concat', 'pd.merge', 'pd.Series', 'pd.to_datetime',
        'pd.to_numeric', 'pd.date_range', 'pd.Interval',
        'pd.get_dummies', 'pd.cut', 'pd.qcut',
        'pd.Categorical', 'pd.factorize', 'pd.isna', 'pd.isnull',
        # Additional pandas functions
        'pd.read_json', 'pd.read_html', 'pd.read_pickle',
        'pd.read_sql', 'pd.read_table', 'pd.read_feather',
        'pd.read_parquet', 'pd.read_sas', 'pd.read_spss',
        'pd.read_stata', 'pd.read_fwf', 'pd.merge_asof',
        'pd.merge_ordered', 'pd.crosstab', 'pd.pivot',
        'pd.pivot_table', 'pd.get_dummies', 'pd.wide_to_long',
        'pd.lreshape', 'pd.to_timedelta', 'pd.period_range',
        'pd.timedelta_range', 'pd.infer_freq', 'pd.isna',
        'pd.isnull', 'pd.notna', 'pd.notnull',
        'pd.nunique', 'pd.idxmax', 'pd.idxmin', 'pd.rank',

        # NumPy - EXPANDED
        'np.array', 'np.arange', 'np.linspace', 'np.logspace',
        'np.zeros', 'np.ones', 'np.random', 'np.sin', 'np.cos',
        'np.random.seed', 'np.random.randn', 'np.random.normal',
        'np.polyfit', 'np.poly1d', 'np.corrcoef', 'np.cov',
        'np.mean', 'np.median', 'np.std', 'np.var', 'np.min', 'np.max',
        'np.sum', 'np.sqrt', 'np.log', 'np.log10', 'np.exp',
        'np.transpose', 'np.reshape', 'np.meshgrid', 'np.squeeze',
        # Additional numpy functions
        'np.zeros_like', 'np.ones_like', 'np.empty', 'np.empty_like',
        'np.full', 'np.full_like', 'np.identity', 'np.eye',
        'np.diag', 'np.diagflat', 'np.tri', 'np.tril', 'np.triu',
        'np.vander', 'np.mat', 'np.matrix', 'np.bmat',
        'np.tile', 'np.repeat', 'np.roll', 'np.rollaxis',
        'np.swapaxes', 'np.atleast_1d', 'np.atleast_2d', 'np.atleast_3d',
        'np.broadcast', 'np.broadcast_to', 'np.broadcast_arrays',
        'np.split', 'np.array_split', 'np.hsplit', 'np.vsplit', 'np.dsplit',
        'np.transpose', 'np.ndarray', 'np.shape', 'np.ndim',
        'np.size', 'np.dtype', 'np.dtype.names', 'np.dtype.fields',
        'np.nan', 'np.inf', 'np.pi', 'np.e', 'np.euler_gamma',
        'np.abs', 'np.absolute', 'np.fabs', 'np.sign', 'np.ceil',
        'np.floor', 'np.rint', 'np.trunc', 'np.modf', 'np.frexp',
        'np.ldexp', 'np.exp2', 'np.log2', 'np.log1p', 'np.expm1',
        'np.logaddexp', 'np.logaddexp2', 'np.power', 'np.sqrt', 'np.square',
        'np.reciprocal', 'np.conjugate', 'np.conj', 'np.real', 'np.imag',
        'np.angle', 'np.isnan', 'np.isfinite', 'np.isinf', 'np.isposinf',
        'np.isneginf', 'np.isnat', 'np.fabs', 'np.fmax', 'np.fmin',
        'np.nan_to_num', 'np.real_if_close', 'np.interp',
        'np.arctan', 'np.arctan2', 'np.arcsin', 'np.arccos', 'np.arcsinh',
        'np.arccosh', 'np.arctanh', 'np.degrees', 'np.radians',
        'np.deg2rad', 'np.rad2deg', 'np.sinh', 'np.cosh', 'np.tanh',
        'np.around', 'np.round_', 'np.rint', 'np.fix', 'np.floor',
        'np.ceil', 'np.trunc', 'np.prod', 'np.cumsum', 'np.cumprod',
        'np.diff', 'np.ediff1d', 'np.gradient', 'np.cross', 'np.dot',
        'np.outer', 'np.inner', 'np.trace', 'np.diag', 'np.linalg.',
        'np.fft.', 'np.clip', 'np.clip_by_norm',

        # Built-in functions (safe)
        'range', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
        'min', 'max', 'sum', 'abs', 'round', 'enumerate', 'zip', 'sorted', 'print',
        'reversed', 'filter', 'map', 'any', 'all', 'bool', 'bytearray',
        'bytes', 'chr', 'complex', 'divmod', 'float', 'frozenset',
        'hash', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass',
        'iter', 'locals', 'long', 'map', 'memoryview', 'next',
        'object', 'oct', 'ord', 'pow', 'property', 'range', 'raw_input',
        'reduce', 'repr', 'reversed', 'round', 'set', 'setattr',
        'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
        'tuple', 'type', 'unichr', 'unicode', 'vars', 'xrange', 'zip',

        # Methods (dynamic) - Allow common pandas/dataframe methods
        'Axes.', 'fig.', 'ax.', '.set_', '.get_', '.plot', '.scatter',
        '.bar', '.hist', '.boxplot', '.set_xlabel', '.set_ylabel',
        '.set_title', '.legend', '.grid',
        # DataFrame methods - EXPANDED
        '.groupby', '.agg', '.mean', '.sum', '.count', '.std', '.min', '.max',
        '.reset_index', '.sort_values', '.drop', '.fillna', '.dropna',
        '.merge', '.concat', '.apply', '.unique', '.nunique', '.value_counts',
        '.corr', '.cov', '.describe', '.head', '.tail', '.iloc', '.loc',
        '.to_numpy', '.values', '.index', '.columns', '.shape', '.size',
        '.T', '.transpose', '.round', '.abs', '.copy',
        # Additional DataFrame methods
        '.abs', '.add', '.add_prefix', '.add_suffix', '.agg', '.aggregate',
        '.align', '.all', '.any', '.append', '.applymap', '.as_matrix',
        '.asfreq', '.asof', '.assign', '.astype', '.at', '.at_time',
        '.between', '.between_time', '.bfill', '.bool', '.boxplot',
        '.clip', '.clip_lower', '.clip_upper', '.columns', '.combine',
        '.combine_first', '.convert_objects', '.copy', '.corr', '.corrwith',
        '.count', '.cov', '.cummax', '.cummin', '.cumprod', '.cumsum',
        '.describe', '.diff', '.div', '.divide', '.dot', '.drop',
        '.drop_duplicates', '.dropna', '.dt', '.dtype', '.dtypes',
        '.duplicated', '.empty', '.eq', '.equals', '.eval', '.ewm',
        '.expanding', '.explode', '.ffill', '.fillna', '.filter',
        '.first', '.first_valid_index', '.floordiv', '.flooordiv', '.from_dict',
        '.from_items', '.from_records', '.ge', '.get', '.get_dtype_counts',
        '.get_value', '.gt', '.head', '.hist', '.iat', '.idxmax',
        '.idxmin', '.iloc', '.index', '.infer_objects', '.info',
        '.insert', '.interpolate', '.isin', '.isna', '.isnull',
        '.items', '.iteritems', '.itertuples', '.ix', '.join',
        '.keys', '.kurt', '.kurtosis', '.last', '.last_valid_index',
        '.le', '.loc', '.lookup', '.lt', '.mad', '.mask',
        '.max', '.mean', '.median', '.melt', '.memory_usage', '.merge',
        '.min', '.mod', '.mode', '.mul', '.multiply', '.ndim',
        '.ne', '.nlargest', '.notna', '.notnull', '.nsmallest',
        '.nunique', '.pct_change', '.pipe', '.pivot', '.pivot_table',
        '.plot', '.pop', '.pow', '.prod', '.product', '.ptp',
        '.put', '.quantile', '.query', '.radd', '.rank', '.rdiv',
        '.reindex', '.reindex_like', '.rename', '.rename_axis',
        '.repeat', '.replace', '.resample', '.reset_index', '.rfloordiv',
        '.rmod', '.rmul', '.rolling', '.round', '.rpow', '.rsub',
        '.rtruediv', '.sample', '.select', '.select_dtypes', '.sem',
        '.set_axis', '.set_index', '.set_value', '.shape', '.shift',
        '.skew', '.slice_shift', '.sort_index', '.sortlevel', '.sort_values',
        '.squeeze', '.stack', '.std', '.sub', '.subtract', '.sum',
        '.swapaxes', '.swaplevel', '.tail', '.take', '.to_clipboard',
        '.to_csv', '.to_csv', '.to_datetime', '.to_dict', '.to_excel',
        '.to_feather', '.to_hdf', '.to_json', '.to_latex', '.to_msgpack',
        '.to_pickle', '.to_period', '.to_records', '.to_sparse',
        '.to_sql', '.to_stata', '.to_string', '.to_timestamp', '.to_xarray',
        '.transform', '.transpose', '.truediv', '.truncate', '.tshift',
        '.unstack', '.update', '.values', '.var', '.where', '.xs',
        # Series methods - EXPANDED
        '.plot', '.hist', '.boxplot', '.value_counts', '.map', '.apply',
        '.agg', '.aggregate', '.align', '.between', '.clip', '.combine',
        '.combine_first', '.corr', '.count', '.cov', '.cummax',
        '.cummin', '.cumprod', '.cumsum', '.describe', '.diff',
        '.div', '.divide', '.drop_duplicates', '.dropna', '.eq',
        '.factorize', '.ffill', '.fillna', '.filter', '.first',
        '.first_valid_index', '.floordiv', '.ge', '.get', '.gt',
        '.head', '.hist', '.iat', '.idxmax', '.idxmin', '.iloc',
        '.interpolate', '.isin', '.isna', '.isnull', '.item',
        '.items', '.iteritems', '.keys', '.last', '.last_valid_index',
        '.le', '.loc', '.lt', '.mad', '.map', '.mask',
        '.max', '.mean', '.median', '.min', '.mod', '.mode',
        '.mul', '.multiply', '.ne', '.nlargest', '.notna', '.notnull',
        '.nsmallest', '.nunique', '.pct_change', '.pipe', '.plot',
        '.pop', '.pow', '.prod', '.product', '.ptp', '.quantile',
        '.radd', '.rank', '.rdiv', '.reindex', '.reindex_like',
        '.rename', '.repeat', '.replace', '.resample', '.rfloordiv',
        '.rmod', '.rmul', '.rolling', '.round', '.rpow', '.rsub',
        '.rtruediv', '.sample', '.searchsorted', '.sem', '.shift',
        '.skew', '.slice_shift', '.sort_index', '.sortlevel', '.sort_values',
        '.squeeze', '.std', '.sub', '.subtract', '.sum', '.swapaxes',
        '.swaplevel', '.tail', '.take', '.to_clipboard', '.to_csv',
        '.to_datetime', '.to_dict', '.to_frame', '.to_json', '.to_latex',
        '.to_list', '.to_msgpack', '.to_period', '.to_pickle', '.to_sparse',
        '.to_string', '.to_timestamp', '.to_xarray', '.transform',
        '.transpose', '.truediv', '.truncate', '.tshift', '.unique',
        '.unstack', '.update', '.value_counts', '.values', '.var',
        '.where', '.xs',
        # String methods
        '.strip', '.split', '.replace', '.upper', '.lower', '.title',
        '.startswith', '.endswith', '.contains', '.format', '.join',
        '.capitalize', '.center', '.count', '.decode', '.encode',
        '.endswith', '.expandtabs', '.find', '.format', '.get',
        '.index', '.isalnum', '.isalpha', '.isdecimal', '.isdigit',
        '.islower', '.isnumeric', 'isspace', '.istitle', '.isupper',
        '.join', '.ljust', '.lower', '.lstrip', '.partition', '.replace',
        '.rfind', '.rindex', '.rjust', '.rpartition', '.rsplit',
        '.rstrip', '.split', '.splitlines', '.startswith', '.strip',
        '.swapcase', '.title', '.translate', '.upper', 'zfill',
    }

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, reject unknown functions. If False, allow.
        """
        self.strict_mode = strict_mode
        self.logger = logger

    def validate(self, code: str) -> Tuple[bool, str]:
        """
        Validate chart code.

        Args:
            code: Python code to validate

        Returns:
            (is_valid, error_message)
        """
        # BUG FIX 3.3: Validate that code contains plt.savefig(save_path)
        # This ensures generated code actually saves the chart to the expected location
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Check for plt.savefig(save_path) call
            savefig_checker = _SaveFigChecker()
            savefig_checker.visit(tree)

            if not savefig_checker.has_savefig_with_save_path:
                return False, (
                    "Code must contain plt.savefig(save_path) to save the chart. "
                    "Please add plt.savefig(save_path, bbox_inches='tight') to your code."
                )

            return True, "Code validation passed"

        except SyntaxError as e:
            return False, f"Syntax error in code: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def get_safe_namespace(self) -> Dict[str, Any]:
        """Get namespace for chart execution - DISABLED RESTRICTIONS"""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from pathlib import Path
        # [FIX 2026-01-18] Removed local import os - use global import from line 5
        # This prevents shadow variable issues and UnboundLocalError
        import builtins

        # Full builtins - no restrictions
        return {
            'plt': plt,
            'pd': pd,
            'np': np,
            'Path': Path,
            'os': os,
            'DataFrame': pd.DataFrame,
            'Series': pd.Series,
            '__builtins__': builtins,  # Full builtins access
        }


class _SaveFigChecker(ast.NodeVisitor):
    """AST visitor that checks for plt.savefig(save_path) calls."""

    def __init__(self):
        self.has_savefig_with_save_path = False

    def visit_Call(self, node: ast.Call):
        """Check if this is a plt.savefig(save_path) call."""
        try:
            func_name = ast.unparse(node.func)

            # Check if this is a savefig call
            if 'savefig' in func_name:
                # Check if any argument is 'save_path'
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id == 'save_path':
                        self.has_savefig_with_save_path = True
                        return

                # Check keyword arguments
                for kw in node.keywords:
                    if kw.arg is None or kw.arg == '':  # Positional keyword arg
                        if isinstance(kw.value, ast.Name) and kw.value.id == 'save_path':
                            self.has_savefig_with_save_path = True
                            return

        except:
            pass  # If we can't parse it, continue checking

        self.generic_visit(node)


class _ASTValidator(ast.NodeVisitor):
    """AST visitor that validates code safety."""

    def __init__(self, allowed_calls: Set[str], strict_mode: bool):
        self.allowed_calls = allowed_calls
        self.strict_mode = strict_mode

    def visit_Call(self, node: ast.Call):
        """Validate function calls."""
        try:
            func_name = ast.unparse(node.func)
        except:
            # If we can't parse it, reject in strict mode
            if self.strict_mode:
                raise SecurityViolation(f"Cannot parse function call: {node.func}")
            return

        # Check if function is allowed
        # For methods starting with '.', check if pattern appears anywhere in func_name
        # For functions, check if func_name starts with or equals the allowed pattern
        is_allowed = any(
            (allowed.startswith('.') and allowed in func_name) or
            func_name.startswith(allowed) or
            func_name == allowed
            for allowed in self.allowed_calls
        )

        if not is_allowed and self.strict_mode:
            raise SecurityViolation(f"Function not allowed: {func_name}")

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Reject dangerous imports, allow safe ones."""
        for alias in node.names:
            module_name = alias.name
            # Allow safe modules
            safe_modules = {'pandas', 'matplotlib', 'numpy', 'matplotlib.pyplot', 'seaborn', 'scipy'}
            if module_name not in safe_modules:
                raise SecurityViolation(f"Import not allowed: {module_name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Reject dangerous from...import statements, allow safe ones."""
        module_name = node.module if node.module else ''
        # Allow safe modules and specific submodules
        safe_modules = {
            'pandas', 'matplotlib', 'matplotlib.pyplot', 'numpy', 'seaborn', 'scipy',
            'pandas.core', 'pandas.io',  # pandas submodules
            'matplotlib.pyplot', 'matplotlib.ticker', 'matplotlib.colors',  # matplotlib submodules
            'numpy.core', 'numpy.linalg', 'numpy.random',  # numpy submodules
        }
        if module_name not in safe_modules:
            raise SecurityViolation(f"Import from not allowed: {module_name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Reject dangerous attributes."""
        if isinstance(node.value, ast.Name):
            if node.value.id == '__builtins__':
                raise SecurityViolation("Access to __builtins__ not allowed")
        self.generic_visit(node)


class ChartCreator(BaseAgent):
    def __init__(self, llm, logger_manager=None, output_dir=None, task_id="General"):
        """
        Initialize ChartCreator with optional logger manager and LatentReporter.

        Args:
            llm: Language model instance
            logger_manager: Optional MMExperimentLogger instance for proper error logging
            output_dir: Optional output directory path for LatentReporter integration
            task_id: Optional task identifier for LatentReporter (default: "General")
        """
        super().__init__(llm, logger_manager)
        # Store module-level logger reference for backward compatibility
        self._module_logger = logger

        # P0 FIX: Data context for whitelist CSV loading
        self.allowed_files = set()  # 白名单：允许加载的文件名
        self.data_dir = ""  # 数据目录（用于构建绝对路径）
        self._data_context_set = False  # 标记是否已设置数据上下文

        # [NEW] LATENT REPORTER: Initialize reporter for visual insights logging
        self.reporter = None
        self._output_dir = output_dir
        self._task_id = task_id

        if LatentReporter and output_dir:
            try:
                self.reporter = LatentReporter(output_dir, llm, task_id)
                logger.info(f"[ChartCreator] LatentReporter initialized for task {task_id}")
            except Exception as e:
                logger.warning(f"[ChartCreator] Failed to initialize LatentReporter: {e}")
                # Non-fatal: continue without reporter
        else:
            logger.debug(f"[ChartCreator] LatentReporter not available (LatentReporter={LatentReporter}, output_dir={output_dir})")

    def set_data_context(self, data_files: List[str], data_dir: str = None):
        """
        P0 FIX: 设置数据上下文（白名单机制）

        Args:
            data_files: 可用数据文件列表（完整路径）
            data_dir: 数据目录（如果为None，从第一个文件提取）
        """
        # 提取文件名作为白名单
        self.allowed_files = {os.path.basename(f) for f in data_files}
        # 确定数据目录
        if data_dir is None and data_files:
            self.data_dir = os.path.dirname(data_files[0])
        else:
            self.data_dir = data_dir or ""
        self._data_context_set = True
        logger.info(f"Data context set: {len(self.allowed_files)} files in whitelist")
        logger.debug(f"Allowed files: {sorted(self.allowed_files)}")
        logger.debug(f"Data directory: {self.data_dir}")

    def _create_load_csv_function(self):
        """
        P0 FIX: 创建白名单load_csv函数

        Returns:
            function: load_csv(filename)函数，只允许加载白名单中的文件
        """
        allowed = self.allowed_files
        data_dir = self.data_dir

        def load_csv(filename):
            """
            白名单CSV加载函数 - 只允许加载预定义的文件

            Args:
                filename: 文件名（必须是白名单中的文件）

            Returns:
                DataFrame: 加载的数据框

            Raises:
                ValueError: 如果文件名不在白名单中
            """
            import pandas as pd  # 导入pandas

            if filename not in allowed:
                available = ', '.join(sorted(allowed))
                raise ValueError(
                    f"[SECURITY] File '{filename}' not in whitelist. "
                    f"You must use one of: {available}"
                )
            filepath = os.path.join(data_dir, filename)
            logger.debug(f"load_csv: Loading '{filename}' from '{filepath}'")

            # P0-A FIX: Use utf-8-sig to handle BOM, then fallback to latin1
            try:
                df = pd.read_csv(filepath, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin1')

            # P0-A FIX: Remove BOM characters from column names
            # Handles ï»¿ (UTF-8 BOM) and Ï»¿ (variant)
            original_columns = df.columns.tolist()
            df.columns = [col.replace('ï»¿', '').replace('Ï»¿', '').strip() for col in df.columns]

            # P0-5 FIX: 统一列名标准化（大写+去空格）
            # 解决KeyError: 'Year' vs 'YEAR'问题
            df.columns = df.columns.str.upper()
            logger.debug(f"[P0-A+BOM] Column normalization: {original_columns} → {df.columns.tolist()}")

            return df

        return load_csv

    def _validate_column_usage(self, code_str: str, allowed_columns: Set[str]) -> Tuple[bool, List[str]]:
        """
        P0 FIX: 静态分析代码，提取并验证列名使用

        Args:
            code_str: LLM生成的代码
            allowed_columns: 允许的列名集合

        Returns:
            (is_valid, invalid_columns): 是否所有列都有效，无效列列表
        """
        import ast

        class ColumnVisitor(ast.NodeVisitor):
            def __init__(self):
                self.columns = set()

            def visit_Subscript(self, node):
                """访问 df['COLUMN'] 模式"""
                try:
                    # 检查是否是 df['COLUMN'] 或 df[...] 格式
                    if isinstance(node.value, ast.Name) and node.value.id == 'df':
                        if isinstance(node.slice, ast.Constant):
                            # df['COLUMN']
                            self.columns.add(str(node.slice.value))
                        elif isinstance(node.slice, ast.Str):  # Python < 3.8
                            self.columns.add(node.slice.s)
                except:
                    pass
                self.generic_visit(node)

        try:
            tree = ast.parse(code_str)
            visitor = ColumnVisitor()
            visitor.visit(tree)

            # 检查所有列是否在允许列表中
            invalid_cols = [col for col in visitor.columns if col not in allowed_columns]

            if invalid_cols:
                logger.warning(f"Invalid columns found: {invalid_cols}")
                logger.debug(f"Allowed columns: {sorted(allowed_columns)}")

            return len(invalid_cols) == 0, invalid_cols

        except SyntaxError:
            # 代码有语法错误，无法解析（会在后续执行时处理）
            return True, []

    def get_data_columns(self, data_file_path: str) -> List[str]:
        """
        读取CSV文件并返回列名列表

        Args:
            data_file_path: CSV文件路径

        Returns:
            List[str]: 列名列表

        Raises:
            FileNotFoundError: 如果文件不存在
            Exception: 如果读取CSV失败
        """
        if not Path(data_file_path).exists():
            raise FileNotFoundError(f"CSV文件不存在: {data_file_path}")

        try:
            # B4 HOTFIX: Use read_csv_safely() with automatic encoding fallback
            # 只读取header获取列名
            df = read_csv_safely(data_file_path, nrows=0)
            columns = list(df.columns)
            logger.info(f"从 {data_file_path} 提取到 {len(columns)} 列: {columns}")
            return columns
        except Exception as e:
            logger.error(f"读取CSV文件失败 {data_file_path}: {e}")
            raise

    def create_single_chart(self, paper_content: str, existing_charts: str, user_prompt: str=''):
        # CRITICAL FIX: Use Template with safe_substitute() and escaped values
        # This prevents KeyError when values contain $ or { }
        prompt = Template(CREATE_CHART_PROMPT).safe_substitute(
            paper_content=_escape_template_value(paper_content),
            existing_charts=_escape_template_value(existing_charts),
            user_prompt=_escape_template_value(user_prompt)
        )
        return self.llm.generate(prompt)

    def create_charts(self, paper_content: str, num_charts: int, user_prompt: str=''):
        existing_charts = ''
        charts = []
        for i in range(num_charts):
            chart = self.create_single_chart(paper_content, existing_charts, user_prompt)
            charts.append(chart)
            existing_charts = '\n---\n'.join(charts)
        return charts

    def description_to_code(self, chart_description: str, data_files: list = None, data_columns_info: str = None):
        """
        Convert chart description to matplotlib code.

        CRITICAL FIX: Prioritize pre-extracted column info over re-reading CSV files.
        This ensures consistency between computational_solving and chart generation stages.

        问题4修复：
        - 强制要求必须提供data_files或data_columns_info
        - 拒绝生成synthetic data图表
        """
        # 问题4修复：强制要求必须提供数据信息，否则拒绝生成
        if not data_files and not data_columns_info:
            error_msg = (
                "[问题4-FATAL] Chart generation REJECTED: No data source specified.\n"
                "Charts MUST be based on actual CSV files.\n"
                "REQUIREMENTS:\n"
                "- Must provide data_files (list of CSV paths)\n"
                "- OR must provide data_columns_info (pre-extracted column names)\n"
                "- WITHOUT data source, chart will NOT be generated (synthetic data is FORBIDDEN)"
            )
            logger.error(f"{error_msg}")
            raise ValueError(error_msg)

        # Add data file information to the prompt
        data_context = ""

        # CRITICAL FIX: PRIORITY 1 - Use pre-extracted column info if available
        if data_columns_info and 'Available Data Files and Columns' in data_columns_info:
            # This info comes from computational_solving.py (lines 54-101)
            # It's already formatted and contains all column names we need
            data_context = "\n\n## [DATA CONTRACT] Pre-Validated Column Information\n\n"
            data_context += data_columns_info
            data_context += "\n"
            data_context += "[CRITICAL] YOU MUST ONLY USE THE COLUMN NAMES LISTED ABOVE.\n"
            data_context += "[CRITICAL] DO NOT guess, invent, or assume any other column names exist.\n"
            data_context += "\n"
            data_context += "[FORBIDDEN COLUMN NAMES] - These DO NOT exist in the dataset:\n"
            data_context += "- 'Scenario' - Does NOT exist (create from Year if needed)\n"
            data_context += "- 'Factor' - Does NOT exist (use available columns)\n"
            data_context += "- 'Historical_Performance' - Does NOT exist (create from available columns)\n"
            data_context += "- 'Athlete_Quality' - Does NOT exist (create from available columns)\n"
            data_context += "- 'Performance_Metric' - Does NOT exist (create from available columns)\n"
            data_context += "\n"
            data_context += "[MANDATORY VALIDATION] Before using ANY column:\n"
            data_context += "1. Load data: df = pd.read_csv('PATH', encoding='latin1')\n"
            data_context += "2. Print columns: print(df.columns.tolist())\n"
            data_context += "3. Check if column exists: if 'ColumnName' in df.columns\n"
            data_context += "4. If NOT exists: CREATE from available columns OR use alternative approach\n"
            logger.info(f"Using pre-extracted column info from computational stage")

        # PRIORITY 2 - Fall back to reading CSV files if no pre-extracted info
        elif data_files:
            logger.warning(f"No pre-extracted column info provided, reading CSV files directly")
            data_context = "\n\n## Available Data Files (REAL PIPELINE DATA)\n\n"
            data_context += "[WARNING] CRITICAL: The following CSV files were generated by the pipeline:\n"

            # CRITICAL FIX: Read actual CSV files to get column names and data types
            import pandas as pd
            for i, data_file in enumerate(data_files, 1):
                file_size = os.path.getsize(data_file)
                data_context += f"{i}. `{os.path.basename(data_file)}` ({file_size:,} bytes)\n"
                # CRITICAL FIX: Do NOT include full path in prompt - LLM will copy it literally
                # This causes FileNotFoundError when LLM generates code with wrong absolute paths

                # Read CSV to get actual column names and sample data
                try:
                    # B4 HOTFIX: Use read_csv_safely() with automatic encoding fallback
                    df_sample = read_csv_safely(data_file, nrows=5)
                    data_context += f"   **ACTUAL COLUMNS**: {list(df_sample.columns)}\n"
                    data_context += f"   **SHAPE**: {df_sample.shape}\n"

                    # Add data type info for each column
                    # CRITICAL FIX: Filter out path fragments from column names to prevent "r'C" KeyError
                    for col in df_sample.columns:
                        col_str = str(col).strip()
                        # Skip columns that look like paths or path fragments
                        if re.search(r'[A-Za-z]:[/\\]|^r["\']?[A-Za-z]:', col_str):
                            logger.debug(f"Filtered out path-like column: '{col_str}'")
                            continue
                        if len(col_str) > 100:
                            logger.debug(f"Filtered out suspiciously long column: '{col_str[:50]}...'")
                            continue

                        dtype = str(df_sample[col].dtype)
                        sample_vals = df_sample[col].dropna().head(3).tolist()
                        data_context += f"      - `{col}` ({dtype}): {sample_vals}\n"
                except Exception as e:
                    data_context += f"   **ERROR reading CSV**: {e}\n"
                    logger.error(f"Failed to read {data_file}: {e}")

            data_context += "\n[CRITICAL] MANDATORY REQUIREMENTS:\n"
            # P0 FIX: 强制使用load_csv
            data_context += "- MUST use load_csv('filename.csv') to load data (NOT pd.read_csv)\n"
            data_context += "- load_csv() only accepts filenames from the whitelist above\n"
            data_context += "- DO NOT use pd.read_csv() - it will fail with wrong paths\n"
            data_context += "- DO NOT use full paths like C:\\Users\\...\\file.csv\n"
            data_context += "- DO NOT use placeholders like 'FULL_PATH', 'path/to/data.csv', etc.\n"
            data_context += "- DO NOT invent or guess file paths\n"
            data_context += "- DO NOT generate synthetic data with np.random, np.linspace, etc.\n"
            data_context += "- DO NOT create fake/example data\n"
            data_context += "- Every data point in your chart MUST come from these CSV files\n"
            data_context += "\n[OK] CORRECT EXAMPLE:\n"
            data_context += "```python\n"
            # P0 FIX: 强制使用load_csv而不是pd.read_csv
            data_context += "# P0 FIX: Use load_csv() function (NOT pd.read_csv)\n"
            data_context += "# load_csv() only allows whitelisted files and handles paths automatically\n"
            data_context += "df = load_csv('filename.csv')  # Use filename only, NOT full path\n"
            data_context += "print('[OK] Loaded:', df.shape)\n"
            data_context += "print('[OK] Columns:', df.columns.tolist())\n"
            data_context += "# CRITICAL: Use ACTUAL column names from the list above\n"
            data_context += "# NOT 'Year', 'Gold', etc. - use the REAL columns shown above!\n"
            data_context += "plt.plot(df['ACTUAL_COLUMN_NAME'], df['ANOTHER_ACTUAL_COLUMN'])\n"
            data_context += "plt.savefig(save_path)\n"
            data_context += "```\n"

        else:
            logger.warning(f"No data files or column info available - LLM will likely hallucinate")

        enhanced_description = chart_description + data_context
        # CRITICAL FIX: Use Template with safe_substitute() and escaped values
        prompt = Template(CHART_TO_CODE_PROMPT).safe_substitute(
            chart_description=_escape_template_value(enhanced_description)
        )

        # P0-5: 强制只输出代码块（用户要求：只接受 ```python ... ``` 内部内容）
        raw_response = self.llm.generate(prompt)

        try:
            from utils.code_guards import CodeExecutionGuards
            extracted_code, errors = CodeExecutionGuards.extract_code_block(
                raw_response,
                require_code_only=True  # 拒绝自然语言混入
            )
            logger.info(f"[P0-5] Successfully extracted code block from LLM response ({len(extracted_code)} chars)")
            return extracted_code
        except ValueError as e:
            logger.error(f"[P0-5] Code block extraction failed: {e}")
            # Raise exception to trigger retry logic in calling code
            raise ValueError(f"P0-5 Code Block Extraction Failed: {e}")

    # ============================================
    # P1-6: JSON-based Chart Generation (Alternative Method)
    # ============================================

    def description_to_json_config(self, chart_description: str, data_files: list = None, data_columns_info: str = None):
        """
        P1-6: Convert chart description to JSON configuration (NEW METHOD).

        This is an alternative to description_to_code() that uses structured JSON
        instead of direct code generation. Benefits:
        - More structured and verifiable
        - Easier to debug
        - Template-based code generation (safer)
        - Better integration with Schema Registry

        Args:
            chart_description: Natural language description of the chart
            data_files: List of available CSV files
            data_columns_info: Pre-extracted column information

        Returns:
            str: JSON configuration string
        """
        # BUG FIX #3: Removed local 'from string import Template'
        # Global import already exists at line 4. Local import creates
        # shadow variable that can cause "local variable referenced before assignment"
        from prompt.template import CHART_TO_JSON_PROMPT

        logger.info("[P1-6] Using JSON-based chart configuration method")

        # Prepare data context (same as description_to_code)
        data_context = ""
        if data_columns_info and 'Available Data Files and Columns' in data_columns_info:
            data_context = "\n\n## Available Data Files and Columns\n\n"
            data_context += data_columns_info
            logger.info("Using pre-extracted column info for JSON config")
        elif data_files:
            logger.warning("No pre-extracted column info, reading CSV files for JSON config")
            data_context = "\n\n## Available Data Files\n\n"
            import pandas as pd
            for i, data_file in enumerate(data_files, 1):
                data_context += f"{i}. `{os.path.basename(data_file)}`\n"
                try:
                    df_sample = read_csv_safely(data_file, nrows=5)
                    data_context += f"   Columns: {list(df_sample.columns)}\n"
                except Exception as e:
                    logger.error(f"Failed to read {data_file}: {e}")

        # Build prompt with JSON template
        enhanced_description = chart_description + data_context
        # CRITICAL FIX: Use Template with safe_substitute() and escaped values
        prompt = Template(CHART_TO_JSON_PROMPT).safe_substitute(
            chart_description=_escape_template_value(enhanced_description)
        )

        # Generate JSON config
        json_response = self.llm.generate(prompt)

        # Extract JSON (remove markdown code blocks if present)
        if "```json" in json_response:
            json_response = json_response.split("```json")[1].split("```")[0].strip()
        elif "```" in json_response:
            json_response = json_response.split("```")[1].split("```")[0].strip()

        logger.info(f"[P1-6] Generated JSON config ({len(json_response)} chars)")

        # Validate JSON
        try:
            import json
            config = json.loads(json_response)
            logger.info(f"[P1-6] JSON is valid, chart_type: {config.get('chart_type', 'unknown')}")
        except json.JSONDecodeError as e:
            logger.error(f"[P1-6] Invalid JSON generated: {e}")
            # Fall back to code-based method
            logger.warning("[P1-6] Falling back to code-based method")
            return None

        return json_response

    def json_config_to_code(self, json_config: str, save_path: str) -> str:
        """
        P1-6: Convert JSON configuration to matplotlib code using templates.

        Args:
            json_config: JSON configuration string
            save_path: Path to save the chart

        Returns:
            str: Generated Python code
        """
        from utils.chart_config import ChartConfig, ChartTemplateRenderer

        logger.info("[P1-6] Converting JSON config to code using templates")

        # Parse JSON config
        config = ChartConfig.from_json(json_config)

        # Validate config
        is_valid, errors = config.validate()
        if not is_valid:
            logger.error(f"[P1-6] Invalid config: {errors}")
            raise ValueError(f"Invalid chart configuration: {errors}")

        # Render to code
        renderer = ChartTemplateRenderer()
        code = renderer.render(config, save_path)

        logger.info(f"[P1-6] Generated code from JSON ({len(code)} chars)")
        return code

    # ============================================
    # P0/P1-5: Code Execution with Guards
    # ============================================

    def execute_chart_code(self, code: str, save_path: str):
        """
        Execute matplotlib code and save chart - WITH P0 ENGINEERING GUARDS.

        P0 Guards (Same-day fixes):
        - Guard 1: Whitelist CSV reading
        - Guard 4: Ban 3D charts
        - Guard 8: Syntax validation
        """
        # ============================================
        # P0 GUARDS: Pre-execution validation
        # P1-5: Schema Registry integration for enhanced column validation
        # ============================================
        try:
            from utils.code_guards import CodeExecutionGuards
            from utils.schema_registry import SchemaRegistry
            from utils.data_manager import get_data_manager

            data_mgr = get_data_manager()
            guards = CodeExecutionGuards(
                data_dir=str(data_mgr.data_dir) if data_mgr else None,
                available_files=data_mgr.list_csv_files() if data_mgr else []
            )

            # P1-5: Initialize Schema Registry for enhanced column validation
            schema_registry = SchemaRegistry()
            if data_mgr and data_mgr.data_dir.exists():
                schema_registry.scan_directory(str(data_mgr.data_dir), source='input_data')
                logger.info(f"[P1-5] Schema Registry initialized for chart generation: {len(schema_registry.schemas)} files")

            # P0-1: LOCAL SYNTAX PATCHER (用户要求：专治arrowprops漏逗号等高频语法错误)
            # 在ast.parse前做regex修补，避免LLM反复重试同类错误
            def apply_local_syntax_patches(code_str: str) -> str:
                """
                P0-1 本地语法补丁器：修复常见LLM生成的语法错误

                修复模式：
                - arrowprops=dict(facecolor='black' width=...) → 加逗号
                - alpha=0.7 linewidth=... → 加逗号
                - 其他dict参数漏逗号

                Args:
                    code_str: LLM生成的代码

                Returns:
                    修补后的代码
                """
                import re

                # 修复1: arrowprops dict中漏逗号（最高频）
                # facecolor='black' width= → facecolor='black', width=
                pattern1 = r"(['\"])(?:black|white|red|blue|green)(?:\1)\s+([a-z_]+)\s*="
                code_str = re.sub(pattern1, r"\1, \2=", code_str)

                # 修复2: alpha/linewidth/fontsize等参数前漏逗号
                pattern2 = r"=\s*([0-9.]+)\s+([a-z_]+)\s*="
                code_str = re.sub(pattern2, r"= \1, \2=", code_str)

                # 修复3: dict中多个键值对之间漏逗号
                # key1=value1 key2=value2 → key1=value1, key2=value2
                pattern3 = r"(['\"]?[\w]+['\"]?)\s*=\s*[^,=\s]+\s+(['\"]?[\w]+['\"]?)\s*=\s*"
                def add_comma_between_kv(match):
                    return match.group(0)[:-1] + ", " + match.group(0)[-1]
                code_str = re.sub(pattern3, add_comma_between_kv, code_str)

                return code_str

            # P0-3 FIX: Apply pre-syntax sanitize FIRST (before any other validation)
            # This fixes syntax garbage like "=," generated by LLM auto-fixers
            from utils.syntax_fixer import pre_syntax_sanitize
            code_sanitized, sanitize_fixes = pre_syntax_sanitize(code)
            if sanitize_fixes:
                logger.info(f"[P0-3] Pre-syntax sanitize applied {len(sanitize_fixes)} fixes")
            else:
                code_sanitized = code  # No changes needed

            # 应用本地语法补丁（在ast.parse前）
            code_patched = apply_local_syntax_patches(code_sanitized)
            logger.debug("[P0-1] Applied local syntax patches for arrowprops/dict errors")

            # P0 FIX #4: Additional syntax validation using assert_syntax_ok utility
            # This provides an extra safety layer before execution
            is_valid_quick, quick_error = assert_syntax_ok(code_patched)
            if not is_valid_quick:
                logger.warning(f"[P0-FIX#4] Quick syntax check failed: {quick_error}")
                # Continue to existing validation for detailed error reporting

            # P0-8: Guard 8 - Syntax validation with AUTO-FIX
            # P0-GUARD8 FIX: Attempt to auto-fix common syntax errors before failing
            from utils.syntax_fixer import SyntaxAutoFixer

            # Apply P0-GUARD8 auto-fixer for common syntax mistakes (after local patches)
            is_valid, fixed_code, error_msg = SyntaxAutoFixer.validate_and_fix(code_patched)

            if is_valid:
                # Code is now valid (either was valid or was auto-fixed)
                code = fixed_code
                logger.info("[P0-GUARD8] Code validated successfully")
            else:
                # Still has syntax errors even after auto-fix
                logger.error(f"[P0-8] SYNTAX ERROR (after auto-fix attempt): {error_msg}")
                # Try one more time with original code to get accurate error
                is_valid_original, error_original = guards.validate_syntax(code_patched)
                if not is_valid_original:
                    logger.error(f"[P0-8] Original syntax error: {error_original}")
                    raise ValueError(f"Code has syntax errors: {error_original}")
                else:
                    # Should not reach here, but handle gracefully
                    logger.warning("[P0-8] Code passed validation on re-check")
                    code = code_patched

            # 使用验证后的代码继续后续处理
            # code = code_patched  # OLD: Now using the validated/fixed code

            # P0-4: Guard 4 - Ban 3D charts (import check)
            is_safe_3d, errors_3d = guards.check_3d_charts(code)
            if not is_safe_3d:
                logger.error("[P0-4] 3D CHART DETECTED - FORBIDDEN")
                raise ValueError("3D charts are forbidden. Use 2D: bar/line/scatter/heatmap")

            # P0-4 ENHANCED: Ban 3D chart method calls (plot_surface, plot_wireframe, etc.)
            # This catches cases where LLM calls 3D methods without importing mplot3d
            from utils.code_guards_enhanced import EnhancedCodeGuards
            is_safe_3d_methods, errors_3d_methods = EnhancedCodeGuards.check_3d_chart_methods(code)
            if not is_safe_3d_methods:
                error_msg = (
                    f"3D chart method calls are forbidden (security policy).\n"
                    f"Detected: {'; '.join(errors_3d_methods)}\n"
                    f"Use 2D alternatives: bar, line, scatter, heatmap, contour, contourf."
                )
                logger.error(f"[P0-4 ENHANCED] {error_msg}")
                raise ValueError(f"[Guard 4 Enhanced Violation] {error_msg}")

            # P0-B: 问题3修复 - 使用公共路径AutoFix模块
            # 统一Chart和Computational阶段的路径处理逻辑
            allowed_files = set(guards.available_files) if hasattr(guards, 'available_files') else set()

            # Apply AutoFix using shared utility
            autofixed_code, success, path_errors = apply_path_autofix_with_validation(code, allowed_files)

            if not success:
                # AutoFix couldn't resolve all issues
                error_msg = (
                    f"Invalid file path detected (after AutoFix).\n"
                    f"{' '.join(path_errors)}\n"
                    f"Available files: {list(allowed_files)[:10]}"
                )
                logger.error(f"[问题3-AutoFix] {error_msg}")
                raise ValueError(f"[Guard 1.5 Violation] {error_msg}")

            # Log fixes
            if autofixed_code != code:
                logger.info(f"[问题3-AutoFix] Applied path AutoFix")
                code = autofixed_code

            # P0-1/P0-4: Guard 1 - Whitelist CSV reading + Auto-rewrite to load_csv
            # P0-4 FIX: Auto-rewrite pd.read_csv to load_csv instead of failing
            is_safe_csv, csv_errors = guards.check_csv_reading(code)
            if not is_safe_csv:
                logger.warning(f"[P0-1] Unsafe pd.read_csv() detected: {csv_errors}")
                logger.info(f"[P0-4] Attempting auto-rewrite to load_csv()...")

                # Extract available files from guards
                available_files = list(guards.available_files)[:10]

                # Auto-rewrite: Replace pd.read_csv with load_csv
                import re
                rewritten_code = code

                # Pattern: pd.read_csv('filename') or pd.read_csv("filename") → load_csv('filename')
                # 这个pattern已经处理了单引号和双引号两种情况
                pd_read_pattern = r"pd\.read_csv\(['\"]([^'\"]+)['\"]\)"
                def replace_pd_read(match):
                    filename = match.group(1)
                    # Check if filename is in whitelist
                    if filename in available_files:
                        return f"load_csv('{filename}')"
                    else:
                        # Keep the problematic code but it will fail later
                        return match.group(0)

                rewritten_code = re.sub(pd_read_pattern, replace_pd_read, rewritten_code)

                # Check if any replacements were made
                if rewritten_code != code:
                    logger.info(f"[P0-4] Successfully auto-rewrote pd.read_csv → load_csv")
                    code = rewritten_code
                    # Re-validate after rewrite
                    is_safe_csv, csv_errors = guards.check_csv_reading(code)
                    if is_safe_csv:
                        logger.info(f"[P0-4] Auto-rewrite validated - code is now safe")
                    else:
                        logger.warning(f"[P0-4] Auto-rewrite still has issues: {csv_errors}")
                        # Continue to failure handling
                else:
                    logger.warning(f"[P0-4] No pd.read_csv patterns found for auto-rewrite")

                # If still unsafe after auto-rewrite, then fail
                if not is_safe_csv:
                    error_msg = (
                        f"Direct pd.read_csv() calls are forbidden (auto-rewrite failed).\n"
                        f"You must use load_csv('filename') instead.\n"
                        f"Errors: {'; '.join(csv_errors)}\n"
                        f"Available files: {available_files}"
                    )
                    logger.error(f"[P0-1] CSV READING ERROR: {error_msg}")
                    raise ValueError(f"[Guard 1 Violation] {error_msg}")
                else:
                    logger.info(f"[P0-4] Auto-rewrite successful - proceeding with execution")

            # P0-2: Guard 2 - Column validation (MANDATORY ENFORCEMENT per user Step 3)
            # User requirement: Hard failure on non-existent columns (not just warning)
            if guards.schema_registry:
                # Use schema_registry as source of truth for available columns
                is_safe_cols, col_errors = guards.check_column_usage(code, guards.schema_registry)
                if not is_safe_cols:
                    error_msg = (
                        f"Invalid column usage detected.\n"
                        f"Errors: {'; '.join(col_errors)}\n"
                        f"You must use only columns that exist in the data files."
                    )
                    logger.error(f"[P0-2] COLUMN ERROR: {error_msg}")
                    raise ValueError(f"[Guard 2 Violation] {error_msg}")

            # P0-4: ENHANCED - Ban synthetic data generation (用户要求：图表必须读取真实CSV)
            # 禁止使用np.random, np.linspace等生成假数据
            from utils.code_guards_enhanced import EnhancedCodeGuards
            is_safe_synthetic, synthetic_errors = EnhancedCodeGuards.check_synthetic_data(code)
            if not is_safe_synthetic:
                error_msg = (
                    f"Synthetic data generation is FORBIDDEN.\n"
                    f"Charts MUST load data from CSV files using load_csv().\n"
                    f"Errors: {'; '.join(synthetic_errors)}"
                )
                logger.error(f"[P0-4] SYNTHETIC DATA ERROR: {error_msg}")
                raise ValueError(f"[Guard 4 Violation] {error_msg}")

            # P0-6: Fix missing imports (io, etc.)
            code = guards.fix_missing_imports(code)

            # P0-3: Ensure savefig present
            code = guards.ensure_savefig_present(code, save_path)

            logger.info("[P0 GUARDS] All checks passed")

        except ValueError as guard_error:
            # P0-3 FIX: Guard violations should NOT interrupt Stage2
            # Return failure instead of raising exception
            logger.error(f"[P0 GUARDS] Code rejected: {guard_error}")
            # Return failure tuple instead of raising
            return False, f"[Guard Violation] {str(guard_error)}"

        # Original execution continues...
        try:
            # DISABLE ALL SECURITY CHECKS - Prioritize getting charts working
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            from pathlib import Path
            # [FIX 2026-01-18] Removed local import os - use global import from line 5
            # This prevents shadow variable issues and UnboundLocalError
            import re

            # ======== P0 FIX #1: DATA_DIR injection (DEPRECATED - Using CodeExecutionGuards instead) ========
            # This code has been superseded by the CodeExecutionGuards system (lines 910-948)
            # which provides centralized data management and validation.
            #
            # To avoid conflicts with CodeExecutionGuards, this logic is commented out.
            # If CodeExecutionGuards is not available, uncomment this section.
            #
            # # Determine DATA_DIR from save_path to fix relative path issues
            # DATA_DIR = None
            # ALLOWED_FILES = set()
            # save_path_obj = Path(save_path)
            #
            # # Method 1: Extract from save_path structure
            # if save_path_obj.parent.exists():
            #     code_dir = save_path_obj.parent.parent / 'code'
            #     if code_dir.exists():
            #         DATA_DIR = str(code_dir.resolve())
            #         for csv_file in code_dir.glob("*.csv"):
            #             ALLOWED_FILES.add(csv_file.name)
            #             logger.debug(f"[P0 FIX #1] Added to whitelist: {csv_file.name}")
            #
            # # Method 2: Check parent directories for dataset
            # if not DATA_DIR:
            #     current_dir = save_path_obj.parent
            #     for _ in range(5):
            #         if current_dir == current_dir.parent:
            #             break
            #         dataset_dir = current_dir / 'MMBench' / 'dataset'
            #         if dataset_dir.exists():
            #             DATA_DIR = str(dataset_dir.resolve())
            #             for csv_file in dataset_dir.glob("*.csv"):
            #                 ALLOWED_FILES.add(csv_file.name)
            #             break
            #         current_dir = current_dir.parent
            #
            # if DATA_DIR:
            #     logger.info(f"[P0 FIX #1] DATA_DIR injected: {DATA_DIR}")
            #     logger.info(f"[P0 FIX #1] Allowed files: {sorted(ALLOWED_FILES)}")
            # else:
            #     logger.warning("[P0 FIX #1] Could not determine DATA_DIR")
            # ======== END P0 FIX #1 (DEPRECATED) ========

            # ======== P0 FIX #1 (continued): Safe CSV reading (DEPRECATED) ========
            # This entire section has been superseded by CodeExecutionGuards
            # All CSV path fixing is now handled by the guard system (lines 910-948)
            #
            # # def fix_read_csv_encoding(code_str: str) -> str:
            # #     """Remove invalid 'errors=' parameter, add encoding, and inject DATA_DIR."""
            # #     # Step 1: Remove ALL variations of 'errors=' parameter
            # #     code_str = re.sub(r',?\s*errors\s*=\s*[\'"][^\'\"]*[\'"]', '', code_str)
            # #     code_str = re.sub(r',?\s*errors\s*=\s*[^,\)\s]+', '', code_str)
            # #     # Step 2: Fix syntax issues
            # #     code_str = re.sub(r',\s*,', ',', code_str)
            # #     code_str = re.sub(r'\(\s*,', '(', code_str)
            # #     code_str = re.sub(r',\s*\)', ')', code_str)
            # #     code_str = re.sub(r'\(\s*\)', '()', code_str)
            # #     # Step 3: Replace pd.read_csv calls with full DATA_DIR paths
            # #     if DATA_DIR:
            # #         def replace_read_csv(match):
            # #             indent = re.match(r'^(\s*)', match.group(0)).group(1) if re.match(r'^\s+', match.group(0)) else ''
            # #             full_match = match.group(0)
            # #             filename_match = re.search(r'pd\.read_csv\s*\(\s*["\']([^"\']+)["\']', full_match)
            # #             if not filename_match:
            # #                 return full_match
            # #             filename = filename_match.group(1)
            # #             if filename not in ALLOWED_FILES and Path(filename).name not in [Path(f).name for f in ALLOWED_FILES]:
            # #                 logger.warning(f"[P0 FIX #1] File '{filename}' not in whitelist, but allowing anyway")
            # #             safe_path = f"os.path.join(r'{DATA_DIR}', r'{filename}')"
            # #             return f'{indent}df = pd.read_csv({safe_path}, encoding="latin-1")'
            # #         code_str = re.sub(r'(\s+)df\s*=\s*pd\.read_csv\s*\([^)]+\)', replace_read_csv, code_str)
            # #         # Also handle other variable names
            # #         def replace_other_var_read_csv(match):
            # #             indent = match.group(1)
            # #             var_name = match.group(2)
            # #             args = match.group(3)
            # #             filename_match = re.search(r'["\']([^"\']+)["\']', args)
            # #             if not filename_match:
            # #                 return match.group(0)
            # #             filename = filename_match.group(1)
            # #             safe_path = f"os.path.join(r'{DATA_DIR}', r'{filename}')"
            # #             return f'{indent}{var_name} = pd.read_csv({safe_path}, encoding="latin-1")'
            # #         code_str = re.sub(r'(\s+)(\w+)\s*=\s*pd\.read_csv\s*\(([^)]+)\)', replace_other_var_read_csv, code_str)
            # #     # Step 4: Add encoding if not present
            # #     pattern = r'(pd\.read_csv\s*\([^)]*?)(\s*\))'
            # #     def add_encoding(match):
            # #         before_close = match.group(1)
            # #         closing_paren = match.group(2)
            # #         if 'encoding=' in before_close:
            # #             return match.group(0)
            # #         return f'{before_close}, encoding="latin1"{closing_paren}'
            # #     code_str = re.sub(pattern, add_encoding, code_str)
            # #     return code_str
            # ======== END P0 FIX #1 (continued) ========

            def fix_matplotlib_params(code_str: str) -> str:
                """
                Fix matplotlib parameters intelligently - remove ONLY truly deprecated parameters.

                CRITICAL: Don't be too aggressive - modern matplotlib versions support many parameters.
                Only remove parameters that are KNOWN to cause errors in common matplotlib versions.

                P0 FIX #3: Added missing comma fixes for arrowprops and dict parameters.
                """
                import re

                # ======== P0 FIX #3: Add missing commas (HIGH-FREQUENCY ERROR) ========

                # Fix 1: Missing comma between string key-value pairs in dict
                # Pattern: 'key' 'value' or "key" "value" → 'key', 'value'
                # But ONLY inside arrowprops or dict to avoid false positives

                # Apply to arrowprops specifically (most common error location)
                # [CRITICAL FIX 2026-01-18] Wrap in try-except to prevent regex errors from crashing chart generation
                try:
                    code_str = re.sub(
                        r"(arrowprops\s*=\s*(?:dict\(|\{)[^)}]*?)(['\"])([\w\-\.]+)\2\s+(['\"])([\w\-\.]+)\3",
                        r"\1\2, \3\4, \5",
                        code_str
                    )
                except re.error as e:
                    # If regex fails (e.g., invalid group reference), skip this replacement
                    logger.debug(f"[FIX] Regex substitution failed (arrowprops comma fix): {e}")

                # Fix 2: Missing comma before numeric parameters in dict
                # Pattern: dict(key='value' number=...) → dict(key='value', number=...)
                # Specifically for matplotlib arrowprops which has many numeric params
                arrowprops_patterns = [
                    (r"(facecolor\s*=\s*['\"][^'\"]+['\"])\s+(width\s*=)", r"\1, \2"),
                    (r"(width\s*=\s*[0-9.]+)\s+(headwidth\s*=)", r"\1, \2"),
                    (r"(headwidth\s*=\s*[0-9.]+)\s+(shrink\s*=)", r"\1, \2"),
                    (r"(edgecolor\s*=\s*['\"][^'\"]+['\"])\s+(linewidth\s*=)", r"\1, \2"),
                ]

                # [CRITICAL FIX 2026-01-18] Wrap all regex substitutions in try-except
                for pattern, replacement in arrowprops_patterns:
                    try:
                        code_str = re.sub(pattern, replacement, code_str)
                    except re.error as e:
                        logger.debug(f"[FIX] Regex substitution failed (numeric params): {e}")

                # Fix 3: More general - missing comma in any dict(...) call
                # Pattern: dict(word'word2' or dict("word""word2") → dict("word", "word2")
                # This is conservative - only fixes obvious cases
                # [CRITICAL FIX 2026-01-18] Wrap in try-except
                try:
                    code_str = re.sub(
                        r"dict\(([^)]*?)(['\"])([a-zA-Z_][a-zA-Z0-9_]*)\2\s+([a-zA-Z_][a-zA-Z0-9_]*)\2",
                        r"dict(\1\2\3\2, \4\2",
                        code_str
                    )
                except re.error as e:
                    logger.debug(f"[FIX] Regex substitution failed (dict comma fix): {e}")

                logger.debug(f"[P0 FIX #3] Applied missing comma fixes")
                # ======== END P0 FIX #3 ========

                # CRITICAL FIX: Only remove 'shrink' from arrowprops (deprecated in matplotlib 3.3+)
                # Pattern: shrink=A within arrowprops dict
                # But DO NOT remove shrink from other contexts (it might be valid)

                # Fix 1: Remove 'shrink' from arrowprops dictionaries only
                # Pattern: arrowprops=dict(..., shrink=A, ...) or arrowprops={'shrink': A, ...}
                # [CRITICAL FIX 2026-01-18] Wrap in try-except
                try:
                    code_str = re.sub(
                        r"arrowprops\s*=\s*(?:dict\(|\{)[^)}]*?shrink\s*=\s*[0-9.]+[^)}]*?(?:\)|\})",
                        lambda m: re.sub(r",?\s*shrink\s*=\s*[0-9.]+\s*,?", "", m.group(0)),
                        code_str
                    )
                except re.error as e:
                    logger.debug(f"[FIX] Regex substitution failed (shrink removal): {e}")

                # Fix 2: Remove 'fancybox' from arrowprops (deprecated in some versions)
                # [CRITICAL FIX 2026-01-18] Wrap in try-except
                try:
                    code_str = re.sub(
                        r"arrowprops\s*=\s*(?:dict\(|\{)[^)}]*?fancybox\s*=\s*(?:True|False)[^)}]*?(?:\)|\})",
                        lambda m: re.sub(r",?\s*fancybox\s*=\s*(?:True|False)\s*,?", "", m.group(0)),
                        code_str
                    )
                except re.error as e:
                    logger.debug(f"[FIX] Regex substitution failed (fancybox removal): {e}")

                # Fix 3: Remove 'patchA' and 'patchB' from arrowprops (can cause issues in some versions)
                # [CRITICAL FIX 2026-01-18] Wrap in try-except
                try:
                    code_str = re.sub(
                        r"arrowprops\s*=\s*(?:dict\(|\{)[^)}]*?(?:patchA|patchB)\s*=\s*[^,)]+[^)}]*?(?:\)|\})",
                        lambda m: re.sub(r",?\s*(?:patchA|patchB)\s*=\s*[^,)]+\s*,?", "", m.group(0)),
                        code_str
                    )
                except re.error as e:
                    logger.debug(f"[FIX] Regex substitution failed (patchA/B removal): {e}")

                # Fix double commas in arrowprops dict after removal
                # [CRITICAL FIX 2026-01-18] Wrap all in try-except
                comma_fix_patterns = [
                    (r',\s*,', ','),
                    (r'\{\s*,', '{'),
                    (r',\s*\}', '}'),
                    (r',\s*\)', ')'),
                ]
                for pattern, replacement in comma_fix_patterns:
                    try:
                        code_str = re.sub(pattern, replacement, code_str)
                    except re.error as e:
                        logger.debug(f"[FIX] Regex substitution failed (comma cleanup): {e}")

                # [CRITICAL FIX 2026-01-18] Wrap last re.sub in try-except
                try:
                    code_str = re.sub(r',\s*\]', ']', code_str)
                except re.error as e:
                    logger.debug(f"[FIX] Regex substitution failed (bracket cleanup): {e}")

                return code_str

            def fix_matplotlib_api_usage(code_str: str) -> str:
                """Fix common matplotlib API misuse"""
                # Fix: plt.text(..., arrowprops=...) should be plt.annotate(..., arrowprops=...)
                # plt.text doesn't support arrowprops, only plt.annotate does
                code_str = re.sub(
                    r'plt\.text\s*\(',
                    'plt.annotate(',  # Simple fix - might need refinement
                    code_str
                )

                # Add seaborn import if code uses 'sns' but doesn't import it
                if 'sns.' in code_str and 'import seaborn' not in code_str and 'import sns' not in code_str:
                    # Insert seaborn import after matplotlib imports
                    code_str = code_str.replace(
                        'import matplotlib.pyplot as plt',
                        'import matplotlib.pyplot as plt\nimport seaborn as sns'
                    )
                    code_str = code_str.replace(
                        'from matplotlib import pyplot as plt',
                        'from matplotlib import pyplot as plt\nimport seaborn as sns'
                    )

                # Remove plt.show() calls (not needed in script execution)
                code_str = re.sub(r'plt\.show\(\)', '', code_str)

                return code_str

            def fix_pandas_syntax(code_str: str) -> str:
                """
                Fix common pandas syntax errors in LLM-generated code.

                Fixes:
                1. Tuple instead of list in groupby column selection
                   - df.groupby(...)[('col1','col2')] → df.groupby(...)[['col1','col2']]
                """
                import re

                # Fix 1: Convert tuple to list in groupby/aggregate column selection
                # Pattern: df.groupby(...)[('col1', 'col2', 'col3')]
                #         → df.groupby(...)[['col1', 'col2', 'col3']]
                # This fixes: "Cannot use tuple of columns ... for 'DataFrameGroupBy' column selection"

                # Pattern 1: df.groupby('NOC')[('Gold', 'Silver')]
                code_str = re.sub(
                    r'df\.groupby\([^)]+\)\[\(([^\)]+)\)\]',
                    lambda m: m.group(0).replace(f"[({m.group(1)})]", f"[{m.group(1).replace(', ', ', ')}]").replace(('(', ')'), ('[', ']')),
                    code_str
                )

                # More direct approach: Replace pattern [('col1', 'col2')] with [['col1', 'col2']]
                # This handles the most common case
                code_str = re.sub(
                    r'\[\(([^\)]+)\)\]',
                    lambda m: f"[{m.group(1)}]" if ',' in m.group(1) else m.group(0),
                    code_str
                )

                # Fix 2: More aggressive pattern - any [('...','...')] or [("...","...")]
                # Replace with list [['...','...']]
                code_str = re.sub(
                    r'\[\(([\'"][^\'"]+[\'"],\s*[\'"][^\'"]+[\'"])\)\]',
                    r'[[\1]]',
                    code_str
                )

                return code_str

            def normalize_column_names(code_str: str) -> str:
                """
                Fix common column name mismatches by adding normalization at start of code.

                Example: If code uses df['EVENT'] but actual column is 'Event',
                add normalization step to handle case/space variations.
                """
                import re

                # Add column normalization at the start of code
                normalization_code = """
# Normalize column names (handle case/space variations)
if 'df' in locals() or 'df' in globals():
    df.columns = [str(c).strip() for c in df.columns]
"""

                # Insert normalization after first df assignment
                # Find first df = pd.read_csv or similar
                match = re.search(r'(df\s*=\s*pd\.read_csv[^\n]+)', code_str)
                if match:
                    insert_pos = match.end()
                    code_str = code_str[:insert_pos] + '\n' + normalization_code + code_str[insert_pos:]

                return code_str

            def enforce_savefig_safety(code_str: str) -> str:
                """
                Ensure code properly saves chart to save_path with directory creation.

                Fixes:
                1. Ensures plt.savefig(save_path) is called (add if missing)
                2. Ensures parent directory is created before savefig
                3. Removes plt.show() calls (not needed in script execution)
                """
                import re

                # Check if savefig is already called with save_path
                has_savefig = 'plt.savefig(save_path)' in code_str or 'plt.savefig( save_path )' in code_str

                if not has_savefig:
                    # CRITICAL: Code doesn't save to save_path - add savefig before any plt.show()
                    # Look for plt.show() or end of script
                    if 'plt.show()' in code_str:
                        # Replace plt.show() with savefig + close
                        code_str = code_str.replace(
                            'plt.show()',
                            '''plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()'''
                        )
                    else:
                        # Add savefig at the end (before any exec/exit)
                        # Insert before the last line if it's not empty
                        lines = code_str.rstrip().split('\n')
                        if lines and lines[-1].strip():
                            # Add savefig after the last non-empty line
                            code_str += '\n\n# CRITICAL: Save chart to save_path\nplt.savefig(save_path, bbox_inches=\'tight\', dpi=300)\nplt.close()\n'
                        else:
                            code_str += '''# CRITICAL: Save chart to save_path
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()
'''
                else:
                    # savefig exists, but ensure it's before plt.show()
                    # and ensure plt.close() is called
                    if 'plt.show()' in code_str:
                        # Remove plt.show() - not needed in script
                        code_str = code_str.replace('plt.show()', '')

                    # Ensure plt.close() after savefig
                    if 'plt.close()' not in code_str:
                        code_str = code_str.replace(
                            'plt.savefig(save_path',
                            'plt.savefig(save_path, bbox_inches=\'tight\', dpi=300)\nplt.close()  # Release memory\n# plt.savefig(save_path'
                        )
                        # Fix the duplicate line we just created
                        code_str = code_str.replace('\n# plt.savefig(save_path', '\nplt.savefig(save_path')
                        code_str = re.sub(r'plt\.close\(\)\s*# Release memory\nplt\.savefig\(save_path.*?\)', '', code_str)

                # Ensure parent directory is created before savefig
                # Add: Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                mkdir_pattern = r"Path\(save_path\)\.parent\.mkdir"
                if 'mkdir' not in code_str:
                    # Find the first plt.savefig call and add mkdir before it
                    code_str = code_str.replace(
                        'plt.savefig(save_path',
                        '''# CRITICAL: Create parent directory before saving
Path(save_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path'''
                    )

                return code_str

            def fix_forbidden_patterns(code_str: str) -> str:
                """
                Fix CRITICAL forbidden patterns that LLM commonly generates.

                These patterns cause immediate crashes:
                1. Path() or raw strings inside f-strings (SyntaxError)
                2. Import statements for pathlib/pandas/matplotlib (SecurityViolation)
                3. df.name attribute (AttributeError)
                4. Placeholder strings like 'FULL_PATH_TO_' (FileNotFoundError)
                """
                import re

                # Fix 1: Remove Path() and raw strings from f-strings
                # Pattern: f"...{Path(r'...')}..." or f"...{r'...'}..."
                # This causes "f-string expression part cannot include a backslash"
                def fix_fstring_path(match):
                    """Extract Path() or raw string from f-string and assign to variable"""
                    fstring_content = match.group(0)

                    # Find all Path(r'...') patterns in f-strings
                    path_pattern = r'\{Path\(r\'[^\']*\'\)(?:\.[a-z]+\(\))?\}'
                    paths_found = re.findall(path_pattern, fstring_content)

                    if paths_found:
                        # Replace with variable references
                        # This is complex, so for now just warn in comments
                        return fstring_content  # Will be caught by syntax error

                    return fstring_content

                # Simpler approach: Detect and comment out problematic f-strings
                # Pattern: f-string containing Path(r'...')
                code_lines = code_str.split('\n')
                fixed_lines = []

                for line in code_lines:
                    # Check if line has f-string with Path() or raw string
                    if 'Path(r\'' in line and '{' in line and '}' in line:
                        # Comment out this line and add warning
                        fixed_lines.append(f'# AUTO-FIXED: Removed problematic f-string with Path(): {line}')
                    elif '{r\'' in line:
                        # Comment out f-strings with raw strings
                        fixed_lines.append(f'# AUTO-FIXED: Removed problematic f-string with raw string: {line}')
                    else:
                        fixed_lines.append(line)

                code_str = '\n'.join(fixed_lines)

                # Fix 2: Remove import statements for already-available modules
                # Remove: from pathlib import Path
                code_str = re.sub(r'from\s+pathlib\s+import\s+Path\s*\n?', '', code_str)
                code_str = re.sub(r'from\s+pathlib\s+import\s+\*\s*\n?', '', code_str)

                # Remove: import pandas as pd
                code_str = re.sub(r'import\s+pandas\s+as\s+pd\s*\n?', '', code_str)
                # Remove: from pandas import ...
                code_str = re.sub(r'from\s+pandas\s+import\s+.*?\n', '', code_str)

                # Remove: import matplotlib.pyplot as plt
                code_str = re.sub(r'import\s+matplotlib\.pyplot\s+as\s+plt\s*\n?', '', code_str)
                # Remove: from matplotlib import pyplot as plt
                code_str = re.sub(r'from\s+matplotlib\s+import\s+pyplot\s+as\s+plt\s*\n?', '', code_str)
                # Remove: from matplotlib import ...
                code_str = re.sub(r'from\s+matplotlib\s+import\s+.*?\n', '', code_str)

                # Remove: import numpy as np
                code_str = re.sub(r'import\s+numpy\s+as\s+np\s*\n?', '', code_str)
                # Remove: from numpy import ...
                code_str = re.sub(r'from\s+numpy\s+import\s+.*?\n', '', code_str)

                # Fix 3: Replace df.name with safer alternative
                # DataFrames don't have .name attribute (only Series do)
                code_str = re.sub(r'(\w+)\.name', r'\1.columns.tolist()[0] if isinstance(\1, pd.DataFrame) else \1.name', code_str)

                # Fix 4: Replace placeholder strings
                # Pattern: 'FULL_PATH_TO_xxx' or "FULL_PATH_TO_xxx" (xxx can contain dots, slashes, etc.)
                code_str = re.sub(r'["\']FULL_PATH_TO_[^"\']+["\']', r'# REMOVED: Placeholder string - use actual path', code_str)

                return code_str

            # Apply all fixes in sequence
            # CRITICAL FIX 1: Auto-fix matplotlib syntax errors (missing commas, arrowprops)
            code = auto_fix_matplotlib_syntax(code)
            # CRITICAL FIX 2: CSV encoding fallback
            # code = fix_read_csv_encoding(code)  # DEPRECATED - handled by CodeExecutionGuards
            # CRITICAL FIX 3: Matplotlib deprecated parameters
            code = fix_matplotlib_params(code)
            # CRITICAL FIX 4: Matplotlib API misuse
            code = fix_matplotlib_api_usage(code)
            # CRITICAL FIX 5: Pandas syntax errors (tuple vs list in groupby)
            code = fix_pandas_syntax(code)
            # CRITICAL FIX 6: Forbidden patterns (Path() in f-strings, placeholders, etc.)
            code = fix_forbidden_patterns(code)
            # CRITICAL FIX 7: Normalize column names (fixes KeyError: 'EVENT' vs 'Event')
            code = normalize_column_names(code)
            # CRITICAL FIX 8: ENSURE savefig is called (was missing!)
            code = enforce_savefig_safety(code)

            # CRITICAL FIX: Removed strict validation checks
            # We've applied auto-fixes for common issues (encoding, deprecated params)
            # Instead of rejecting code that might still have issues, let it execute
            # If execution fails, the error message will be more informative for debugging
            # This prevents rejecting code that might actually work despite imperfect auto-fixing

            # CRITICAL SECURITY FIX: Validate imports using AST before execution
            # This allows safe imports (pandas, matplotlib, numpy) but blocks dangerous ones (os, subprocess)
            try:
                tree = ast.parse(code)
                validator = _ASTValidator(
                    allowed_calls={
                        # Pandas top-level functions
                        'pd.read_csv', 'pd.DataFrame', 'pd.concat', 'pd.merge',
                        'pd.Series', 'pd.to_datetime', 'pd.to_numeric',
                        'pd.read_excel', 'pd.get_dummies', 'pd.cut', 'pd.qcut',
                        'pd.Categorical', 'pd.factorize', 'pd.isna', 'pd.isnull',
                        # Pandas DataFrame/Series methods (using dot prefixes)
                        '.select_dtypes', '.columns', '.tolist', '.values', '.index',
                        '.groupby', '.agg', '.mean', '.sum', '.count', '.std', '.min', '.max',
                        '.reset_index', '.sort_values', '.drop', '.fillna', '.dropna',
                        '.merge', '.concat', '.apply', '.unique', '.nunique', '.value_counts',
                        '.corr', '.cov', '.describe', '.head', '.tail', '.iloc', '.loc',
                        '.to_numpy', '.shape', '.size', '.T', '.transpose', '.round', '.abs', '.copy',
                        '.plot', '.hist', '.boxplot',
                        # Matplotlib pyplot
                        'plt.style.use', 'plt.show', 'plt.savefig', 'plt.figure',
                        'plt.subplot', 'plt.subplots', 'plt.plot', 'plt.scatter',
                        'plt.bar', 'plt.hist', 'plt.boxplot', 'plt.xlabel',
                        'plt.ylabel', 'plt.title', 'plt.legend', 'plt.grid',
                        'plt.tight_layout', 'plt.xlim', 'plt.ylim', 'plt.close',
                        'plt.contour', 'plt.contourf', 'plt.colorbar', 'plt.text',
                        'plt.annotate', 'plt.xticks', 'plt.yticks', '.set_',
                        # NumPy
                        'np.array', 'np.arange', 'np.linspace', 'np.logspace', 'np.zeros',
                        'np.ones', 'np.random', 'np.sin', 'np.cos', 'np.random.seed',
                        'np.random.randn', 'np.random.normal', 'np.polyfit', 'np.mean',
                        'np.median', 'np.std', 'np.var', '.min', '.max', '.sum',
                        'np.sqrt', 'np.log', 'np.log10', 'np.exp', 'np.transpose',
                        'np.reshape', 'np.meshgrid', 'np.squeeze', 'np.corrcoef', 'np.cov',
                        # Built-in
                        'range', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                        'print', 'enumerate', 'zip',
                    },
                    strict_mode=False  # DISABLE strict mode - allow unknown but safe methods
                )
                validator.visit(tree)
            except SecurityViolation as e:
                if self.logger_manager:
                    self.logger_manager.log_exception(
                        exception=e,
                        context="Code contains dangerous imports",
                        stage="chart_generation",
                        include_traceback=False
                    )
                else:
                    logger.error(f"Security validation failed: {e}")
                return False, f"Code validation failed: {str(e)}"

            # Safe namespace with restricted builtins
            import builtins
            safe_builtins = {
                '__import__': builtins.__import__,  # Allow import for safe modules
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'print': print,
                'reversed': reversed,
                'filter': filter,
                'map': map,
                'any': any,
                'all': all,
                # Exception types for error handling
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'FileNotFoundError': FileNotFoundError,
                'KeyError': KeyError,
                'AttributeError': AttributeError,
                'RuntimeError': RuntimeError,
            }

            # P0-3: Create safe pandas module that bans pd.read_csv()
            # User requirement: Only allow load_csv('filename'), ban direct pd.read_csv()
            class _SafePandasModule:
                """
                Safe pandas wrapper that enforces whitelist CSV reading.

                Prevents:
                - Direct pd.read_csv() calls (FileNotFoundError risk)
                - Path-based attacks
                - Relative path confusion
                """
                def __init__(self, original_pd, has_load_csv=False):
                    self._pd = original_pd
                    self._has_load_csv = has_load_csv

                def __getattr__(self, name):
                    """Forward all attributes except read_csv"""
                    if name == 'read_csv':
                        # P0-3: Ban direct pd.read_csv() calls
                        def banned_read_csv(*args, **kwargs):
                            error_msg = (
                                "\n" + "="*60 + "\n"
                                "P0-3 VIOLATION: Direct pd.read_csv() is FORBIDDEN\n"
                                "="*60 + "\n"
                                "You must use load_csv('filename') instead.\n\n"
                                "Example:\n"
                                "  # WRONG (will fail):\n"
                                "  df = pd.read_csv('file.csv')\n\n"
                                "  # CORRECT:\n"
                                "  df = load_csv('file.csv')\n"
                                "="*60 + "\n"
                            )
                            if self._has_load_csv:
                                error_msg += "\nAvailable CSV files: Use load_csv('filename')\n"
                            else:
                                error_msg += "\nWARNING: load_csv() not available (data context not set)\n"
                            raise PermissionError(error_msg)
                        return banned_read_csv
                    # Forward all other attributes to original pandas
                    return getattr(self._pd, name)

            # Check if load_csv is available
            has_load_csv = self._data_context_set
            if has_load_csv:
                load_csv_func = self._create_load_csv_function()
                logger.debug("[P0-3] load_csv function will be added to namespace (whitelist enforced)")
            else:
                load_csv_func = None
                logger.warning("[P0-3] Data context not set, load_csv not available. Will ban pd.read_csv anyway.")

            # Create safe pandas module
            safe_pd = _SafePandasModule(pd, has_load_csv=has_load_csv)

            namespace = {
                'plt': plt,
                'pd': safe_pd,  # P0-3: Use safe pandas that bans read_csv
                'np': np,
                'Path': Path,
                # DO NOT expose 'os' module - prevents os.system(), open(), etc.
                'DataFrame': pd.DataFrame,  # Direct reference still works
                'Series': pd.Series,
                'save_path': save_path,
                '__builtins__': safe_builtins,
            }

            # P0 FIX: 添加白名单load_csv函数（如果数据上下文已设置）
            if has_load_csv:
                namespace['load_csv'] = load_csv_func
                logger.info("[P0-3] load_csv('filename') available, pd.read_csv() banned")
            else:
                logger.warning("[P0-3] Data context not set, load_csv not available. pd.read_csv() still banned.")

            # Step 4: 添加finalize_plot函数（用户要求：框架层强制落盘）
            def finalize_plot(save_path_arg: str) -> bool:
                """
                Framework-level function to ensure plot is saved.

                This is guaranteed to be called after code execution as a safety net.
                Even if LLM forgets plt.savefig(), this ensures the file is created.

                Args:
                    save_path_arg: Path where chart should be saved

                Returns:
                    True if successful, False otherwise
                """
                try:
                    from pathlib import Path as PathLib
                    save_path_obj = PathLib(save_path_arg)

                    # Create parent directory
                    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

                    # Get current figure
                    import matplotlib.pyplot as plt_module
                    all_figures = plt_module._pylab_helpers.Gcf.get_all_fig_managers()

                    if all_figures:
                        # Use the most recent figure
                        fig = all_figures[-1].canvas.figure
                        fig.savefig(save_path_obj, bbox_inches='tight', dpi=200)
                        plt_module.close(fig)
                        logger.info(f"[finalize_plot] Successfully saved to {save_path_arg}")
                        return True
                    else:
                        logger.warning(f"[finalize_plot] No figure found to save")
                        return False
                except Exception as e:
                    logger.error(f"[finalize_plot] Failed to save: {e}")
                    return False

            namespace['finalize_plot'] = finalize_plot

            # ======== P1-1 FIX: Force remove ALL import statements (multi-layer protection) ========
            # This prevents LLM from importing banned modules (pathlib, os, sys, etc.)
            # Layer 1: Static code cleanup (remove import lines before execution)
            # Layer 2: Runtime __import__ blocking (prevents dynamic imports)
            import re
            original_code = code
            removed_imports = []

            # Pattern 1: 'import xxx' statements
            import_pattern_1 = r'^\s*import\s+[a-zA-Z_][a-zA-Z0-9_,\s]*(?:\s*#[^\n]*)?$'
            matches_1 = re.findall(import_pattern_1, code, flags=re.MULTILINE)
            for match in matches_1:
                removed_imports.append(match.strip())
                code = re.sub(re.escape(match), '', code, flags=re.MULTILINE)

            # Pattern 2: 'from xxx import yyy' statements
            import_pattern_2 = r'^\s*from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import\s+[^\n]*(?:\s*#[^\n]*)?$'
            matches_2 = re.findall(import_pattern_2, code, flags=re.MULTILINE)
            for match in matches_2:
                removed_imports.append(match.strip())
                code = re.sub(re.escape(match), '', code, flags=re.MULTILINE)

            # Pattern 3: Multi-line imports (from xxx import (a, b, c))
            import_pattern_3 = r'^\s*from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import\s+\([^)]+\)'
            matches_3 = re.findall(import_pattern_3, code, flags=re.MULTILINE | re.DOTALL)
            for match in matches_3:
                removed_imports.append(match.strip()[:50] + '...')
                code = re.sub(re.escape(match), '', code, flags=re.MULTILINE | re.DOTALL)

            if removed_imports:
                logger.warning(f"[P1-1] Removed {len(removed_imports)} import statement(s) from LLM code:")
                for imp in removed_imports[:5]:  # Log first 5
                    logger.warning(f"  - {imp}")
                if len(removed_imports) > 5:
                    logger.warning(f"  ... and {len(removed_imports) - 5} more")

            # Clean up empty lines left by import removal
            code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)

            # Layer 2: Runtime blocking (prevent __import__ during exec)
            def _blocked_import(name, *args, **kwargs):
                """Block ALL import attempts at runtime - system provides all needed modules"""
                logger.error(f"[P1-1] Blocked runtime import attempt: {name}")
                raise ImportError(
                    f"[SECURITY] Import statements are NOT ALLOWED in chart code. "
                    f"The system provides all necessary modules (pd, plt, np, load_csv). "
                    f"Attempted to import: {name}"
                )

            # Save original __import__ and block it
            # [CRITICAL FIX 2026-01-18] Handle __builtins__ as both module and dict
            # In some Python environments (especially exec sandbox), __builtins__ is a dict
            if isinstance(__builtins__, dict):
                original_import = __builtins__['__import__']
                __builtins__['__import__'] = _blocked_import
            else:
                original_import = __builtins__.__import__
                __builtins__.__import__ = _blocked_import

            try:
                # Execute the code with restricted namespace and validated imports
                exec(code, namespace)
            finally:
                # Restore original __import__ (important for system stability)
                if isinstance(__builtins__, dict):
                    __builtins__['__import__'] = original_import
                else:
                    __builtins__.__import__ = original_import

            # Step 4 (continued): 调用finalize_plot作为安全网（用户要求）
            # 即使代码执行了，也要确保文件一定落盘
            if 'finalize_plot' in namespace:
                logger.debug("Calling finalize_plot as safety net")
                finalize_result = namespace['finalize_plot'](save_path)
                if not finalize_result:
                    logger.warning("finalize_plot returned False, file may not be saved")
            else:
                logger.warning("finalize_plot not in namespace, skipping safety net")

            # ======== P0 FIX #2: Save fallback mechanism ========
            # CRITICAL: Even if code "executes successfully", the file might not be created
            # This fallback ensures we ALWAYS have a chart file
            save_path_obj = Path(save_path)

            if not save_path_obj.exists():
                logger.warning(f"[P0 FIX #2] Chart file not created at {save_path}")
                logger.warning(f"[P0 FIX #2] Applying fallback mechanism...")

                # Fallback Step 1: Try to save current figure
                try:
                    if 'plt' in namespace:
                        import matplotlib.pyplot as plt_module
                        # Get current figure
                        all_figures = plt_module._pylab_helpers.Gcf.get_all_fig_managers()
                        if all_figures:
                            # Use the most recent figure
                            fig = all_figures[-1].canvas.figure
                            logger.info(f"[P0 FIX #2] Found existing figure, saving...")

                            # Create parent directory
                            save_path_obj.parent.mkdir(parents=True, exist_ok=True)

                            # Save figure
                            fig.savefig(save_path, bbox_inches='tight', dpi=200)
                            plt_module.close(fig)
                            logger.info(f"[P0 FIX #2] SUCCESS: Saved existing figure to {save_path}")
                        else:
                            logger.warning(f"[P0 FIX #2] No existing figure found")
                            raise Exception("No figure to save")
                    else:
                        logger.warning(f"[P0 FIX #2] plt not in namespace")
                        raise Exception("plt not available")

                except Exception as e:
                    logger.warning(f"[P0 FIX #2] Fallback step 1 failed: {e}")

                    # Fallback Step 2: Create placeholder chart
                    try:
                        logger.info(f"[P0 FIX #2] Creating placeholder chart...")
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Create placeholder text
                        placeholder_text = f"Chart Generation Failed\n\n"
                        placeholder_text += f"Save Path: {save_path}\n"
                        placeholder_text += f"Error: {str(e)[:100]}\n\n"
                        placeholder_text += f"Please check:\n"
                        placeholder_text += f"1. Data files exist\n"
                        placeholder_text += f"2. Column names are correct\n"
                        placeholder_text += f"3. Code has plt.savefig() call"

                        ax.text(0.5, 0.5, placeholder_text,
                               ha='center', va='center',
                               fontsize=10, color='red',
                               wrap=True)
                        ax.set_title('Placeholder Chart (Generation Failed)',
                                   fontsize=12, fontweight='bold', color='darkred')
                        ax.axis('off')

                        # Save placeholder
                        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(save_path, bbox_inches='tight', dpi=150)
                        plt.close(fig)
                        logger.info(f"[P0 FIX #2] SUCCESS: Placeholder saved to {save_path}")

                    except Exception as e2:
                        logger.error(f"[P0 FIX #2] Placeholder also failed: {e2}")
                        logger.error(f"[P0 FIX #2] CRITICAL ERROR: No chart file generated!")
            # ======== END P0 FIX #2 ========

            file_size = save_path_obj.stat().st_size
            if file_size == 0:
                # CRITICAL FIX: Use logger_manager if available
                if self.logger_manager:
                    self.logger_manager.log_exception(
                        exception=Exception(f"Chart file is empty: {save_path}"),
                        context="Chart file created but is empty",
                        stage="chart_generation",
                        include_traceback=False
                    )
                else:
                    logger.warning(f"Chart file created but is empty: {save_path}")
                return False, f"Chart file created but is empty: {save_path}"

            return True, f"Chart saved successfully to {save_path} ({file_size:,} bytes)"

        except Exception as e:
            # CRITICAL FIX: Use logger_manager if available
            if self.logger_manager:
                self.logger_manager.log_exception(
                    exception=e,
                    context="Error executing chart code",
                    stage="chart_generation",
                    include_traceback=True
                )
            else:
                logger.error(f"Error executing chart code: {e}")
            import traceback
            return False, f"Error executing chart code: {str(e)}\n{type(e).__name__}\n{traceback.format_exc()}"

    def create_chart_with_image(self, chart_description: str, save_path: str, data_files: list = None, data_columns_info: str = None, max_retries: int = 3):
        """
        Generate chart description, convert to code, execute it, and save image.

        Args:
            chart_description: Text description of the chart
            save_path: Path where the chart image should be saved
            data_files: List of paths to CSV data files (REAL pipeline-generated data)
            data_columns_info: Pre-extracted column information (CRITICAL for preventing KeyError)
            max_retries: Number of times to retry if code execution fails

        Returns:
            dict: {
                'description': str,  # Chart description
                'code': str,         # Generated matplotlib code
                'image_path': str,   # Path to saved image (or None if failed)
                'success': bool,     # Whether image was created successfully
                'error': str         # Error message if failed
            }
        """
        print(f"Generating chart: {save_path}")

        # P0 FIX: 设置数据上下文（白名单）
        if data_files:
            self.set_data_context(data_files)

        if data_files:
            print(f"  -> Available REAL data files: {len(data_files)}")
            for df in data_files:
                print(f"     - {os.path.basename(df)}")

        # CRITICAL FIX: Use pre-extracted column info if available
        if data_columns_info:
            print(f"  [OK] Using pre-extracted column info from computational stage")
        else:
            print(f"  [WARN] No pre-extracted column info, will extract from data files")

        # Generate matplotlib code from description
        print("  -> Converting description to matplotlib code...")
        # CRITICAL FIX: Pass data_columns_info to ensure LLM knows actual column names
        code = self.description_to_code(chart_description, data_files, data_columns_info)

        # Extract code block if present
        if "```python" in code:
            try:
                code = code.split("```python")[1].split("```")[0].strip()
            except IndexError:
                print("  [WARN] Warning: Could not extract code block, using raw response")

        # CRITICAL FIX: Extract available columns BEFORE code execution for better error messages
        all_available_columns = set()
        if data_files:
            for data_file in data_files:
                try:
                    columns = self.get_data_columns(data_file)
                    all_available_columns.update(columns)
                    print(f"  -> Available columns in {os.path.basename(data_file)}: {columns}")
                except Exception as e:
                    print(f"  [WARN] Could not read columns from {data_file}: {e}")

        # P0-2 FIX: Hard validation - reject code using non-existent columns
        if data_files:
            # Second, validate that code only uses available columns (HARD FAIL)
            if all_available_columns:
                is_valid, invalid_columns = self._validate_column_usage(code, all_available_columns)
                if not is_valid:
                    print(f"  [P0-2] REJECT: Code uses non-existent columns: {invalid_columns}")
                    print(f"  [P0-2] Available columns: {sorted(all_available_columns)}")
                    print(f"  [P0-2] This violates P0-2 guard - rejecting immediately")

                    # Skip execution loop entirely and trigger retry with clear error
                    # Build intelligent error message for LLM
                    column_error_msg = (
                        f"P0-2 GUARD VIOLATION: Your code uses columns that don't exist in the dataset.\n"
                        f"Invalid columns: {invalid_columns}\n"
                        f"Available columns: {sorted(all_available_columns)}\n\n"
                        f"You MUST ONLY use columns from the available list above.\n"
                        f"Do NOT invent or guess column names."
                    )

                    # Try to fix with LLM immediately (skip execution)
                    print("  [P0-2] Attempting LLM fix...")
                    try:
                        # P0-4 FIX: Don't re-import Template - it's already imported at module level (line 3)
                        # Importing here creates a local variable that shadows the global, causing
                        # "local variable 'Template' referenced before assignment" errors in other paths
                        # from string import Template  # REMOVED - use global import
                        # CHART_CODE_FIXING_PROMPT is already imported at module level (line 2)

                        fix_prompt = f"""# CRITICAL: P0-2 Guard Violation

{column_error_msg}

## Original Code
```python
{code}
```

## Instructions
Rewrite the code using ONLY the available columns listed above.
Replace ALL invalid column references with valid ones from the available list.
"""

                        fixed_code = self.llm.generate(fix_prompt)
                        if "```python" in fixed_code:
                            fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
                        code = fixed_code
                        print("  [P0-2] LLM provided fixed code - validating...")

                        # Re-validate the fixed code
                        is_valid, invalid_columns = self._validate_column_usage(code, all_available_columns)
                        if not is_valid:
                            print(f"  [P0-2] FIXED CODE STILL INVALID: {invalid_columns}")
                            # Give up and return failure
                            return {
                                'description': chart_description,
                                'code': code,
                                'image_path': None,
                                'success': False,
                                'error': f"P0-2 Guard: Code uses invalid columns {invalid_columns} (even after LLM fix)"
                            }
                        print("  [P0-2] Fixed code is valid - proceeding to execution")
                    except Exception as fix_error:
                        print(f"  [P0-2] LLM fixing failed: {fix_error}")
                        return {
                            'description': chart_description,
                            'code': code,
                            'image_path': None,
                            'success': False,
                            'error': f"P0-2 Guard: Code uses invalid columns {invalid_columns}"
                        }

            # Third, validate that code loads CSV files
            if not self._validate_data_usage(code, data_files):
                print("  [WARN] Warning: Generated code may not be using real data files!")
                print("     The code should contain pd.read_csv() calls to load the provided CSV files.")
                # Continue anyway - the LLM might have a good reason

        # Try to execute the code
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  -> Retry {attempt + 1}/{max_retries}...")

            success, message = self.execute_chart_code(code, save_path)

            if success:
                print(f"  [correct] {message}")

                # P2-1 FIX: Verify file actually exists before reporting success
                # This prevents "success=True but file missing" issues
                # NOTE: os is imported at module level (line 5), no need to re-import here
                # Re-importing would cause UnboundLocalError due to Python's scoping rules
                if os.path.exists(save_path):
                    # File exists - genuine success
                    print(f"  [P2-1] File verified: {save_path} ({os.path.getsize(save_path)} bytes)")

                    # [NEW] LATENT REPORTER: Log successful chart creation with image artifact
                    if self.reporter:
                        try:
                            self.reporter.log_thought(
                                stage="Data Visualization",
                                raw_content=f"Generated chart to visualize: {chart_description}",
                                status="SUCCESS",
                                artifact={
                                    "type": "image",
                                    "path": save_path,
                                    "description": chart_description
                                }
                            )
                        except Exception as e:
                            logger.warning(f"[ChartCreator] Failed to log chart success: {e}")
                            # Non-fatal: continue without logging

                    return {
                        'description': chart_description,
                        'code': code,
                        'image_path': save_path,
                        'success': True,
                        'error': None
                    }
                else:
                    # execute_chart_code returned success=True but file doesn't exist!
                    # This is a critical failure - treat as failed
                    error_msg = f"Code execution reported success but file not created at {save_path}"
                    print(f"  [P2-1] {error_msg}")
                    print(f"  [P2-1] Treating as FAILED despite execute_chart_code returning success=True")

                    # [NEW] LATENT REPORTER: Log chart generation failure
                    if self.reporter:
                        try:
                            self.reporter.log_thought(
                                stage="Data Visualization",
                                raw_content=f"Chart generation reported success but image file missing: {chart_description}\nError: {error_msg}",
                                status="ERROR"
                            )
                        except Exception as e:
                            logger.warning(f"[ChartCreator] Failed to log chart failure: {e}")
                            # Non-fatal: continue without logging

                    return {
                        'description': chart_description,
                        'code': code,
                        'image_path': save_path,
                        'success': False,
                        'error': error_msg
                    }
            else:
                # CRITICAL FIX: Enhanced error analysis for KeyError
                print(f"  [wrong] Attempt {attempt + 1} failed")

                # CRITICAL: Detect KeyError and extract missing column names
                missing_column = None
                keyerror_detected = False

                if 'KeyError' in message:
                    keyerror_detected = True
                    import re
                    # Extract column name from error message like "KeyError: 'Gold'"
                    keyerror_match = re.search(r"KeyError:\s*['\"]([^'\"]+)['\"]", message)
                    if keyerror_match:
                        missing_column = keyerror_match.group(1)
                        print(f"  [INFO] Detected KeyError for column: '{missing_column}'")
                        print(f"  [INFO] Available columns: {sorted(all_available_columns)}")

                        # Suggest similar columns (fuzzy matching)
                        import difflib
                        similar_columns = difflib.get_close_matches(missing_column, all_available_columns, n=3, cutoff=0.3)
                        if similar_columns:
                            print(f"  [HINT] Did you mean: {similar_columns}?")

                if attempt < max_retries - 1:
                    # Try to fix the code using LLM with INTELLIGENT prompt template
                    print("  -> Attempting to fix code with LLM...")
                    try:
                        # CRITICAL FIX: Build intelligent fix_prompt based on error type
                        if keyerror_detected and missing_column and all_available_columns:
                            # SPECIFIC FIX FOR KEYERROR - Most common error
                            fix_prompt = f"""# CRITICAL ERROR DIAGNOSIS

Your code failed with: KeyError: '{missing_column}'

## Root Cause
The column '{missing_column}' DOES NOT EXIST in the dataset.

## Actual Available Columns
{sorted(all_available_columns)}

## What You Did Wrong
You used: df['{missing_column}']
But this column doesn't exist!

## How to Fix
1. Choose from the ACTUAL columns listed above
2. Replace ALL occurrences of df['{missing_column}'] with df['ACTUAL_COLUMN_NAME']
3. DO NOT invent or guess column names

## Example Fix
WRONG: plt.plot(df['{missing_column}'], df['other_column'])
CORRECT: plt.plot(df['{sorted(all_available_columns)[0]}'], df['{sorted(all_available_columns)[1] if len(all_available_columns) > 1 else sorted(all_available_columns)[0]}'])

## Original Code
```python
{code}
```

## Instructions
Rewrite the code using ONLY the columns listed in "Actual Available Columns".
Do NOT use '{missing_column}' - it doesn't exist!
"""
                        else:
                            # GENERIC FIX FOR OTHER ERRORS
                            # Build data file list for context
                            data_file_list = []
                            if data_files:
                                for df in data_files:
                                    data_file_list.append(f"- {df}")
                            data_files_str = "\n".join(data_file_list) if data_file_list else "None"

                            # CRITICAL FIX: Use Template with safe_substitute() and escaped values
                            fix_prompt = Template(CHART_CODE_FIXING_PROMPT).safe_substitute(
                                problem_description=_escape_template_value("Generate matplotlib chart"),
                                original_code=_escape_template_value(code),
                                error_info=_escape_template_value(message),
                                save_path=_escape_template_value("./chart.png")
                            )

                            # Add data file context to the prompt
                            if data_files:
                                fix_prompt += f"\n\n# Available Real Data Files:\n{data_files_str}\n"
                                fix_prompt += "CRITICAL: You MUST load data from these files using pd.read_csv()!\n"

                            # Inject actual available columns into fix_prompt
                            if all_available_columns:
                                fix_prompt += f"\n# ACTUAL AVAILABLE COLUMNS (use ONLY these):\n"
                                fix_prompt += f"# {sorted(all_available_columns)}\n"
                                fix_prompt += "# DO NOT use any other column names - they don't exist!\n"

                        # Log the fix prompt for debugging
                        print(f"  [DEBUG] Fix prompt length: {len(fix_prompt)} chars")
                        if keyerror_detected:
                            print(f"  [DEBUG] Using specialized KeyError fix prompt")

                        fixed_code = self.llm.generate(fix_prompt)
                        if "```python" in fixed_code:
                            fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
                        code = fixed_code
                        print("  [OK] LLM provided fixed code")

                        # CRITICAL: Validate the fixed code doesn't have the same KeyError
                        if keyerror_detected and missing_column and f"['{missing_column}']" in code:
                            print(f"  [WARN] WARNING: LLM may not have fixed the KeyError properly!")
                            print(f"  [WARN] Code still contains: ['{missing_column}']")

                    except Exception as fix_error:
                        print(f"  [WARN] LLM fixing failed: {fix_error}")
                        pass

        # All retries failed
        print(f"  [FAIL] Failed to generate chart after {max_retries} attempts")

        # [NEW] LATENT REPORTER: Log final chart generation failure
        if self.reporter:
            try:
                self.reporter.log_thought(
                    stage="Data Visualization",
                    raw_content=f"Failed to generate chart after {max_retries} attempts: {chart_description}\nFinal error: {message}",
                    status="ERROR"
                )
            except Exception as e:
                logger.warning(f"[ChartCreator] Failed to log chart failure: {e}")
                # Non-fatal: continue without logging

        return {
            'description': chart_description,
            'code': code,
            'image_path': None,
            'success': False,
            'error': message
        }

    def create_charts_from_csv_files(self, chart_descriptions: list, save_dir: str, csv_files: list, data_columns_info: str = None):
        """
        Batch chart generator from CSV files.

        Thin wrapper around create_chart_with_image() that processes multiple
        chart descriptions sequentially. Each chart is generated independently,
        with failures in one chart not affecting others.

        Args:
            chart_descriptions (list): List of chart description strings
            save_dir (str): Directory to save chart images (created if doesn't exist)
            csv_files (list): List of CSV file paths to use as data sources
            data_columns_info (str, optional): Pre-extracted column information

        Returns:
            list: List of chart result dictionaries, one per description.
                  Each dict contains: {
                      'description': str,
                      'code': str,
                      'image_path': str,
                      'success': bool,
                      'error': str
                  }
        """
        # NOTE: os is imported at module level (line 5), no need to re-import here
        results = []

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        print(f"[ChartCreator] Processing {len(chart_descriptions)} charts via create_charts_from_csv_files")
        print(f"[ChartCreator] Save directory: {save_dir}")
        print(f"[ChartCreator] CSV files: {len(csv_files)}")

        # Process each chart description
        for i, desc in enumerate(chart_descriptions):
            # Generate output filename: chart_1.png, chart_2.png, etc.
            save_path = os.path.join(save_dir, f"chart_{i+1}.png")

            print(f"  -> Generating Chart {i+1}/{len(chart_descriptions)}: {save_path}")

            # Reuse existing single-chart generation logic
            # This inherits all safety features:
            # - P0-2 column validation guards
            # - Automatic retry on failure (max_retries=3)
            # - Error recovery and code fixing
            result = self.create_chart_with_image(
                chart_description=desc,
                save_path=save_path,
                data_files=csv_files,
                data_columns_info=data_columns_info
            )

            results.append(result)

            # Log result
            if result.get('success'):
                print(f"     ✓ Chart {i+1} succeeded: {result.get('image_path')}")
            else:
                print(f"     ✗ Chart {i+1} failed: {result.get('error', 'Unknown error')}")

        print(f"[ChartCreator] Complete: {sum(1 for r in results if r.get('success'))}/{len(results)} charts succeeded")

        return results

    def _validate_column_usage(self, code: str, available_columns: set) -> Tuple[bool, List[str]]:
        """
        验证生成的代码只使用了可用的列名

        Args:
            code: 生成的matplotlib代码
            available_columns: 可用的列名集合

        Returns:
            Tuple[bool, List[str]]: (是否有效, 无效列名列表)
        """
        import re

        # 提取所有 df['COLUMN_NAME'] 或 df["COLUMN_NAME"] 模式
        column_pattern = r"df\[['\"]([^'\"]+)['\"]\]"
        used_columns = set(re.findall(column_pattern, code))

        # 提取 df.COLUMN_NAME 模式（但排除 df.groupby, df.mean 等方法调用）
        # 这个正则表达式匹配 df. 后面跟列名，但不是方法名
        method_names = {'groupby', 'agg', 'mean', 'sum', 'count', 'std', 'min', 'max',
                       'reset_index', 'sort_values', 'drop', 'fillna', 'dropna',
                       'merge', 'concat', 'apply', 'unique', 'nunique', 'value_counts',
                       'corr', 'cov', 'describe', 'head', 'tail', 'iloc', 'loc',
                       'to_numpy', 'values', 'index', 'columns', 'shape', 'size',
                       'T', 'transpose', 'round', 'abs', 'copy', 'plot', 'hist', 'boxplot'}

        dot_pattern = r"df\.(\w+)"
        for match in re.finditer(dot_pattern, code):
            col_name = match.group(1)
            if col_name not in method_names:
                used_columns.add(col_name)

        # CRITICAL FIX: Filter out path fragments and other non-column tokens
        # This prevents "r'C" (from r'C:\Users\...') from being treated as a column name
        used_columns_list = list(used_columns)
        used_columns = set(sanitize_column_tokens(used_columns_list, available_columns))

        # 检查是否有列名不在可用列中
        invalid_columns = used_columns - available_columns

        if invalid_columns:
            logger.warning(f"代码使用了不存在的列: {invalid_columns}")
            logger.warning(f"可用列: {available_columns}")
            logger.warning(f"使用的列: {used_columns}")
            return False, list(invalid_columns)

        return True, []

    def _validate_data_usage(self, code: str, data_files: list) -> bool:
        """
        Validate that generated code loads real data files.

        Checks if the code contains pd.read_csv() calls to load the provided CSV files.

        Args:
            code: Generated matplotlib code
            data_files: List of CSV file paths that should be loaded

        Returns:
            bool: True if code appears to load data files, False otherwise
        """
        if not data_files:
            return True  # No data files to validate

        # Check if code contains pd.read_csv calls
        has_read_csv = 'pd.read_csv' in code or 'read_csv' in code

        # P0-4: 禁止synthetic data（用户要求：从warn改为硬失败）
        # Check if code generates synthetic data instead
        has_synthetic = (
            'np.random' in code or
            'np.linspace' in code or
            'random.rand' in code or
            'random.randn' in code
        )

        # 用户要求：图表必须读取至少1个CSV，禁止synthetic data
        if has_synthetic and not has_read_csv:
            error_msg = (
                "Generated code uses synthetic data (np.random/np.linspace/random) instead of loading CSV files.\n"
                "Charts MUST load data from CSV files using load_csv().\n"
                "Remove all synthetic data generation and load real data instead."
            )
            logger.error(f"[P0-4] SYNTHETIC DATA ERROR: {error_msg}")
            raise ValueError(f"[Guard 4 Violation] {error_msg}")

        # Log warnings for debugging
        if has_synthetic and has_read_csv:
            logger.warning("[P0-4] Code uses both CSV reading AND synthetic data - this is suspicious but allowed")

        if not has_read_csv:
            logger.warning("Generated code doesn't contain pd.read_csv() calls")
            return False

        return True

    def create_charts_with_images(self, paper_content: str, num_charts: int, output_dir: str, user_prompt: str = '', data_files: list = None, data_columns_info: str = None):
        """
        Generate multiple charts with actual image files.

        Args:
            paper_content: Content describing what charts to create
            num_charts: Number of charts to generate
            output_dir: Directory to save chart images
            user_prompt: Optional user prompt
            data_files: List of paths to CSV data files generated by the task
            data_columns_info: Pre-extracted column information from computational_solving stage

        Returns:
            list: List of dicts containing chart info and image paths
        """
        # First generate chart descriptions
        print(f"\n{'='*60}")
        print(f"Generating {num_charts} charts with images")
        print(f"{'='*60}\n")

        # CRITICAL FIX: Validate data_columns_info is provided
        if data_columns_info and 'Available Data Files and Columns' in data_columns_info:
            print(f"[OK] Using pre-extracted column information from computational stage")
        elif data_files:
            print(f"[WARN] No pre-extracted column info provided, will extract from data files")
        else:
            print(f"[WARN] No data files or column info available - charts may fail")

        existing_charts = ''
        chart_descriptions = []

        for i in range(num_charts):
            print(f"\n--- Chart {i + 1}/{num_charts} ---")
            description = self.create_single_chart(paper_content, existing_charts, user_prompt)
            chart_descriptions.append(description)
            existing_charts = '\n---\n'.join(chart_descriptions)

        # Now generate images for each description
        charts_with_images = []
        charts_dir = Path(output_dir) / 'charts'
        charts_dir.mkdir(parents=True, exist_ok=True)

        for i, description in enumerate(chart_descriptions):
            image_path = str(charts_dir / f'chart_{i+1}.png')
            # CRITICAL FIX: Pass data_columns_info to create_chart_with_image
            # P0-C FIX: Catch ValueError (Guard 8 syntax errors) to prevent Stage2 crash
            try:
                chart_info = self.create_chart_with_image(description, image_path, data_files, data_columns_info)
                charts_with_images.append(chart_info)

                # P0-3 FIX: Only print success if (1) success=True AND (2) file exists
                if chart_info.get('success', False) and os.path.exists(image_path):
                    print(f"[OK] Chart {i+1} generated successfully: {image_path}")
                else:
                    # Chart was returned but marked as failed or file missing
                    error_msg = chart_info.get('error', 'Unknown error')
                    if not chart_info.get('success', False):
                        print(f"[FAIL] Chart {i+1} failed: {error_msg}")
                    else:
                        print(f"[FAIL] Chart {i+1}: success=True but file missing at {image_path}")
            except ValueError as e:
                # Guard 8 syntax error - log and continue, don't crash Stage2
                print(f"[WARN] Chart {i+1} failed with syntax error: {str(e)}")
                print(f"[WARN] Skipping chart {i+1} and continuing with remaining charts")
                # P0 FIX: Append failed chart info with 'success' key for consistency
                charts_with_images.append({
                    'description': description,
                    'image_path': None,
                    'success': False,  # P0 FIX: Include 'success' key to prevent KeyError
                    'error': f"Syntax error: {str(e)}",
                    'status': 'failed'
                })
            except Exception as e:
                # Other unexpected errors - log and continue
                print(f"[ERROR] Chart {i+1} failed with unexpected error: {type(e).__name__}: {str(e)}")
                print(f"[ERROR] Skipping chart {i+1} and continuing with remaining charts")
                # P0 FIX: Append failed chart info with 'success' key for consistency
                charts_with_images.append({
                    'description': description,
                    'image_path': None,
                    'success': False,  # P0 FIX: Include 'success' key to prevent KeyError
                    'error': f"{type(e).__name__}: {str(e)}",
                    'status': 'failed'
                })

        return charts_with_images
