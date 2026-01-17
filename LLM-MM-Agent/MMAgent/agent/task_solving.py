from .base_agent import BaseAgent
from prompt.template import (TASK_ANALYSIS_PROMPT, TASK_RESULT_PROMPT, TASK_ANSWER_PROMPT,
                             TASK_FORMULAS_PROMPT, TASK_FORMULAS_CRITIQUE_PROMPT, TASK_FORMULAS_IMPROVEMENT_PROMPT,
                             TASK_MODELING_PROMPT, TASK_MODELING_CRITIQUE_PROMPT, TASK_MODELING_IMPROVEMENT_PROMPT,
                             TASK_CODING_PROMPT, TASK_CODING_DEBUG_PROMPT, CODE_STRUCTURE_PROMPT,
                             TASK_RESULT_WITH_CODE_PROMPT)
import sys
import os
import subprocess
import selectors
import tiktoken
import json
import time
import ast
import re


class _SafeDict(dict):
    """
    Dictionary subclass that returns the placeholder itself for missing keys.

    This prevents KeyError when template contains placeholders that aren't in kwargs.
    Example: If template has "{name or func}" but kwargs doesn't have "name or func",
    this returns "{name or func}" instead of crashing.
    """
    def __missing__(self, key):
        # Return the placeholder itself (preserves original for debugging)
        return "{" + key + "}"


def safe_format(template: str, **kwargs) -> str:
    """
    Safely format a template by escaping braces in all parameter values before formatting.

    CRITICAL FIX: Uses format_map() with _SafeDict to prevent KeyError when template
    contains braces that aren't meant to be placeholders (e.g., code examples like
    "{name or func}" or f-string examples like "f'{len(df)}'").

    Args:
        template: Template string with placeholders like {key1}, {key2}, etc.
        **kwargs: Keyword arguments to substitute into the template

    Returns:
        Formatted string with all placeholders replaced

    Example:
        >>> template = "Data: {data_summary}, Code: {code}"
        >>> data_summary = "print(f'Result: {result}')"  # Contains braces!
        >>> code = "x = 1"
        >>> safe_format(template, data_summary=data_summary, code=code)
        "Data: print(f'Result: {{result}}'), Code: x = 1"
    """
    # Escape all braces in parameter values
    escaped_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            # Replace { with {{ and } with }} to escape them
            escaped_kwargs[key] = value.replace('{', '{{').replace('}', '}}')
        else:
            escaped_kwargs[key] = value

    # CRITICAL FIX: Use format_map() with _SafeDict instead of format()
    # This prevents KeyError for missing keys like "name or func"
    # Missing placeholders will be preserved as-is for debugging
    return template.format_map(_SafeDict(escaped_kwargs))


def validate_syntax(code: str) -> tuple[bool, str]:
    """
    Validate Python code syntax using AST parsing.

    Args:
        code: Python code string to validate

    Returns:
        (is_valid, error_message)
        - is_valid: True if code has valid syntax, False otherwise
        - error_message: Empty string if valid, detailed human-readable error if invalid
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        # Build detailed, human-readable error message
        error_parts = []

        # Header
        error_parts.append(f"Syntax Error at line {e.lineno}:")

        # Error description with common issues explained
        error_explanations = {
            "invalid syntax": "Invalid syntax - check for missing operators, unmatched brackets, or incomplete statements",
            "unexpected EOF": "Unexpected end of file - you may be missing a closing parenthesis, bracket, or quote",
            "unterminated string literal": "Unterminated string - missing closing quote",
            "unexpected indent": "Unexpected indentation - check your indentation levels",
            "unindent does not match any outer indentation level": "Indentation error - inconsistent use of tabs and spaces",
            "':' expected": "Missing colon (:) - likely after if, for, while, def, or class statement",
        }

        error_desc = error_explanations.get(e.msg, e.msg)
        error_parts.append(f"  Issue: {error_desc}")

        # Show the problematic line
        if e.text:
            error_parts.append(f"  Code: {e.text.strip()}")

        # Show position indicator
        if e.offset is not None:
            # Create a pointer to the exact position
            pointer = " " * (e.offset - 1) + "^" if e.offset > 0 else "^"
            error_parts.append(f"        {pointer}")

        # Suggest fixes for common errors
        if "missing colon" in error_desc.lower() or e.msg == "':' expected":
            error_parts.append("  Fix: Add colon (:) after the statement")
        elif "unexpected eof" in e.msg.lower() or "unterminated" in e.msg.lower():
            error_parts.append("  Fix: Check for missing closing ), ], }, or quote")
        elif "invalid syntax" in e.msg.lower():
            error_parts.append("  Fix: Check for missing operators, extra characters, or incomplete expressions")

        # Add context line numbers
        if e.lineno:
            error_parts.append(f"  Context: This is line {e.lineno} of the generated code")

        return False, "\n".join(error_parts)


def extract_python_code(completion: str) -> str:
    """
    P0-5 ENHANCED: Extract Python code from LLM completion with validation.

    Changes (2026-01-15):
    - Uses CodeExecutionGuards.extract_code_block() for validation
    - Detects and rejects natural language mixed in ("To operationalize", etc.)
    - Enforces ONLY code block output (raises ValueError if none found)

    Original behavior:
    - Case-insensitive (```python, ```Python, ```PYTHON)
    - Nested backticks in strings (takes longest match)
    - Multiple code blocks (returns the longest python block)
    """
    try:
        from utils.code_guards import CodeExecutionGuards

        # Use P0-5 validation with strict natural language detection
        # require_code_only=True ensures no explanatory text is allowed
        extracted_code, errors = CodeExecutionGuards.extract_code_block(
            completion,
            require_code_only=True
        )

        return extracted_code

    except ValueError as e:
        # If validation fails, return empty string to trigger retry loop
        # The error message will be logged by CodeExecutionGuards
        logger.warning(f"[P0-5] Code block validation failed: {e}")
        return ""

    except ImportError:
        # Fallback to original implementation if code_guards not available
        logger.warning("[P0-5] CodeExecutionGuards not available, using fallback extraction")

        # Original pattern for python code blocks (case-insensitive)
        pattern = r'```(?:python|py)\s*\n(.*?)```'
        matches = re.findall(pattern, completion, re.DOTALL | re.IGNORECASE)

        if not matches:
            return ""

        # Return the longest python block (handles nested backticks)
        return max(matches, key=len)


class SecureScriptExecutor:
    """
    Securely executes Python scripts with timeout protection and security validation.

    CRITICAL FIXES:
    - Uses shell=False to prevent shell injection attacks
    - Implements actual timeout enforcement (prevents infinite hangs)
    - Platform-specific: Unix uses select(), Windows uses polling
    - Path validation before execution
    - PYTHONUNBUFFERED for real-time output capture
    """

    def __init__(self, default_timeout: int = 300, logger_manager=None):
        """
        Initialize the executor with a default timeout.

        Args:
            default_timeout: Maximum execution time in seconds (default: 300s = 5 minutes)
            logger_manager: Optional MMExperimentLogger for logging timeout events
        """
        self.default_timeout = default_timeout
        self.logger_manager = logger_manager
        self.logger = logger_manager.get_logger('main') if logger_manager else None

    def build_command(self, script_path, cuda_device=0, python_executable=None):
        """
        Build a secure command list for subprocess execution.

        CRITICAL: Returns a list (not string) to prevent shell injection when shell=False.

        Args:
            script_path: Path to the Python script to execute
                Can be basename (e.g., 'main1.py') when used with cwd parameter,
                or full path when called directly
            cuda_device: GPU device number (default: 0)
            python_executable: Python executable path (default: 'python')

        Returns:
            List of command arguments [python, '-u', script_path]

        Raises:
            ValueError: If script_path is empty
        """
        if not script_path:
            raise ValueError("script_path cannot be empty")

        # CRITICAL FIX: Don't check os.path.exists() for basename-only paths
        # When subprocess.Popen uses cwd, basename is sufficient and correct
        # Existence check will happen when subprocess tries to execute
        # Checking here would fail because we're in wrong directory

        if python_executable is None:
            python_executable = "python"

        # Return as list for shell=False usage (CRITICAL: prevents shell injection)
        return [python_executable, '-u', script_path]

    def build_environment(self, cuda_device=0, additional_env=None):
        """
        Build environment variables for script execution.

        Sets PYTHONUNBUFFERED for real-time output and CUDA_VISIBLE_DEVICES for GPU control.

        Args:
            cuda_device: GPU device number (default: 0)
            additional_env: Optional additional environment variables

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        env['CUDA_VISIBLE_DEVICES'] = str(cuda_device)  # Set GPU device

        if additional_env:
            env.update(additional_env)

        return env

    def execute(self, script_path, work_dir, timeout=None, cuda_device=0):
        """
        Execute a Python script with timeout protection and security validation.

        CRITICAL: Implements actual timeout enforcement to prevent infinite hangs.
        Platform-specific: Unix uses select() with timeout, Windows uses polling.

        Args:
            script_path: Path to the Python script to execute (can be just filename or full path)
            work_dir: Working directory for execution
            timeout: Maximum execution time in seconds (None = use default)
            cuda_device: GPU device number (default: 0)

        Returns:
            str: Execution output with prefix message

        Raises:
            EnvException: If script times out, fails validation, or execution errors occur
        """
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout

        # CRITICAL FIX: Use basename ONLY when subprocess has cwd set
        # Since subprocess.Popen uses cwd=work_dir, we must pass just the filename
        # NOT an absolute path. Otherwise, Python interprets the absolute path
        # relative to cwd, causing path duplication.
        full_script_path = os.path.basename(script_path)

        # Validate and build command
        cmd = self.build_command(full_script_path, cuda_device)
        env = self.build_environment(cuda_device)

        # Track start time for timeout enforcement
        start_time = time.time()

        try:
            # CRITICAL: shell=False prevents shell injection attacks
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,  # SECURITY FIX: prevents shell injection
                cwd=work_dir,
                env=env
            )

            stdout_lines = []
            stderr_lines = []

            # Platform-specific timeout implementation
            if sys.platform == 'win32':
                # Windows: Simple polling approach with timeout
                # Check process status and timeout in small intervals
                while process.poll() is None:
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        # CRITICAL FIX: Log timeout event for diagnostics
                        if self.logger:
                            self.logger.error(f"Script timeout | Script: {script_path} | Time: {elapsed:.1f}s | Limit: {timeout}s")
                        process.terminate()
                        time.sleep(0.5)  # Give process time to terminate gracefully
                        if process.poll() is None:
                            process.kill()  # Force kill if didn't terminate
                        raise EnvException(
                            f"Script execution timed out after {elapsed:.1f}s "
                            f"(limit: {timeout}s). Script: {script_path}"
                        )
                    time.sleep(0.1)  # Check every 100ms

                # Process has completed, read all output at once
                stdout_data = process.stdout.read()
                stderr_data = process.stderr.read()

                # Split into lines for display
                for line in stdout_data.splitlines():
                    print("STDOUT:", line, end=" \n")
                    stdout_lines.append(line + "\n")

                for line in stderr_data.splitlines():
                    print("STDERR:", line, end=" \n")
                    stderr_lines.append(line + "\n")

            else:
                # Unix/Linux/macOS: Use select() with actual timeout
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                selector.register(process.stderr, selectors.EVENT_READ)

                while process.poll() is None and selector.get_map():
                    # Calculate remaining timeout
                    elapsed = time.time() - start_time
                    remaining_timeout = timeout - elapsed

                    if remaining_timeout <= 0:
                        selector.close()
                        # CRITICAL FIX: Log timeout event for diagnostics
                        if self.logger:
                            self.logger.error(f"Script timeout | Script: {script_path} | Time: {elapsed:.1f}s | Limit: {timeout}s")
                        process.terminate()
                        time.sleep(0.5)
                        if process.poll() is None:
                            process.kill()
                        raise EnvException(
                            f"Script execution timed out after {elapsed:.1f}s "
                            f"(limit: {timeout}s). Script: {script_path}"
                        )

                    # Select with remaining timeout (CRITICAL: actual timeout enforcement)
                    events = selector.select(timeout=min(1.0, remaining_timeout))

                    for key, _ in events:
                        line = key.fileobj.readline()
                        if key.fileobj == process.stdout:
                            print("STDOUT:", line, end=" ")
                            stdout_lines.append(line)
                        else:
                            print("STDERR:", line, end=" ")
                            stderr_lines.append(line)

                selector.close()

            # Drain remaining output after process completes
            for line in process.stdout:
                print("STDOUT:", line, end=" ")
                stdout_lines.append(line)
            for line in process.stderr:
                print("STDERR:", line, end=" ")
                stderr_lines.append(line)

            # Determine output based on return code
            return_code = process.returncode

            if return_code != 0:
                observation = "".join(stderr_lines)
            else:
                observation = "".join(stdout_lines)

            # If stdout is empty but return code is 0, use stderr
            if observation == "" and return_code == 0:
                observation = "".join(stderr_lines)

            return "The script has been executed. Here is the output:\n" + observation

        except FileNotFoundError as e:
            raise EnvException(f"Script validation failed: {e}")
        except EnvException:
            # Re-raise EnvException (timeout or execution error)
            raise
        except Exception as e:
            raise EnvException(
                f"Something went wrong in executing {script_path}: {e}. "
                f"Please check if it is ready to be executed."
            )


class EnvException(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message


class TaskSolver(BaseAgent):
    def __init__(self, llm, logger_manager=None):
        """
        Initialize TaskSolver with optional logger manager.

        Args:
            llm: Language model instance
            logger_manager: Optional MMExperimentLogger instance for proper error logging
        """
        super().__init__(llm, logger_manager)

        # CRITICAL FIX: Add code history tracking for context preservation
        # This prevents losing valid code (e.g., import statements) across iterations
        self.code_history = {}  # Maps task_id -> list of (iteration, code) tuples
        self.error_history = {}  # Maps task_id -> list of (iteration, error_type, error_message) tuples

    def analysis(self, prompt: str, task_description: str, user_prompt: str = ''):
        prompt = safe_format(TASK_ANALYSIS_PROMPT, prompt=prompt, task_description=task_description, user_prompt=user_prompt).strip()
        return self.llm.generate(prompt)
    
    def formulas_actor(self, prompt: str, data_summary: str, task_description: str, task_analysis: str, modeling_methods: str, user_prompt: str = ''):
        prompt = safe_format(TASK_FORMULAS_PROMPT, prompt=prompt, data_summary=data_summary, task_description=task_description, task_analysis=task_analysis, modeling_methods=modeling_methods, user_prompt=user_prompt).strip()
        return self.llm.generate(prompt)

    def formulas_critic(self, data_summary: str, task_description: str, task_analysis: str, modeling_formulas: str):
        prompt = safe_format(TASK_FORMULAS_CRITIQUE_PROMPT, data_summary=data_summary, task_description=task_description, task_analysis=task_analysis, modeling_formulas=modeling_formulas).strip()
        return self.llm.generate(prompt)
    
    def formulas_improvement(self, data_summary: str, task_description: str, task_analysis: str, modeling_formulas: str, modeling_formulas_critique: str, user_prompt: str = ''):
        prompt = safe_format(TASK_FORMULAS_IMPROVEMENT_PROMPT, data_summary=data_summary, task_description=task_description, task_analysis=task_analysis, modeling_formulas=modeling_formulas, modeling_formulas_critique=modeling_formulas_critique, user_prompt=user_prompt).strip()
        return self.llm.generate(prompt)

    def modeling(self, formulas_prompt: str, modeling_prompt: str, data_summary: str, task_description: str, task_analysis: str, modeling_methods: str, round: int = 1, user_prompt: str = ''):
        formulas = self.formulas_actor(formulas_prompt, data_summary, task_description, task_analysis, modeling_methods, user_prompt)
        for i in range(round):
            formulas_critique = self.formulas_critic(data_summary, task_description, task_analysis, formulas)
            formulas = self.formulas_improvement(data_summary, task_description, task_analysis, formulas, formulas_critique, user_prompt)
        
        modeling_method = self.modeling_actor(modeling_prompt, data_summary, task_description, task_analysis, formulas, user_prompt)
    
        return formulas, modeling_method

    def modeling_actor(self, prompt: str, data_summary: str, task_description: str, task_analysis: str, formulas: str, user_prompt: str = ''):
        prompt = safe_format(TASK_MODELING_PROMPT, prompt=prompt, data_summary=data_summary, task_description=task_description, task_analysis=task_analysis, modeling_formulas=formulas, user_prompt=user_prompt).strip()
        return self.llm.generate(prompt)

    # def modeling_critic(self, task_description: str, task_analysis: str, data_summary: str, formulas: str, modeling_process: str):
    #     prompt = TASK_MODELING_CRITIQUE_PROMPT.format(task_description=task_description, task_analysis=task_analysis, data_summary=data_summary, modeling_formulas=formulas, modeling_process=modeling_process).strip()
    #     return self.llm.generate(prompt)
    
    # def modeling_improvement(self, task_description: str, task_analysis: str, data_summary: str, formulas: str, modeling_process: str, modeling_process_critique: str):
    #     prompt = TASK_MODELING_IMPROVEMENT_PROMPT.format(task_description=task_description, task_analysis=task_analysis, data_summary=data_summary, modeling_formulas=formulas, modeling_process=modeling_process, modeling_process_critique=modeling_process_critique).strip()
    #     return self.llm.generate(prompt)

    # def modeling(self, task_description: str, task_analysis: str, data_summary: str, formulas: str, round: int = 1):
    #     process = self.modeling_actor(task_description, task_analysis, data_summary, formulas)
    #     for i in range(round):
    #         print(f'MODELING Round {i+1}')
    #         process_critique = self.modeling_critic(task_description, task_analysis, data_summary, formulas, process)
    #         process = self.modeling_improvement(task_description, task_analysis, data_summary, formulas, process, process_critique)
    #     return process
    
    def coding_actor(self, data_file, data_summary, variable_description, task_description: str, task_analysis: str, formulas: str, modeling: str, dependent_file_prompt: str, code_template: str, script_name: str, work_dir: str, user_prompt: str = ''):
        # CRITICAL FIX: Initialize new_content to prevent UnboundLocalError
        new_content = None

        # CRITICAL FIX: Use safe_format to escape braces in LLM-generated content
        # This prevents KeyError: 'country_medals' when modeling/formulas contain {var} patterns
        prompt = safe_format(TASK_CODING_PROMPT,
                            data_file=data_file,
                            data_summary=data_summary,
                            variable_description=variable_description,
                            task_description=task_description,
                            task_analysis=task_analysis,
                            modeling_formulas=formulas,
                            modeling_process=modeling,
                            dependent_file_prompt=dependent_file_prompt,
                            code_template=code_template,
                            user_prompt=user_prompt).strip()

        attempt = 0
        while attempt < 5:
            attempt += 1
            try:
                completion = self.llm.generate(prompt)

                # CRITICAL FIX: Level 1 - Extract code using improved regex
                extracted_code = extract_python_code(completion)
                if not extracted_code:
                    print(f"[WARNING] Attempt {attempt}/5: No ```python delimiter found in LLM response")
                    continue

                # CRITICAL FIX: Level 2 - Validate code is not empty
                if not extracted_code.strip():
                    print(f"[WARNING] Attempt {attempt}/5: Empty code block extracted")
                    continue

                # CRITICAL FIX: Level 3 - Validate code has meaningful content
                has_meaningful_content = any([
                    'import ' in extracted_code,
                    'def ' in extracted_code,
                    'class ' in extracted_code,
                    'if __name__' in extracted_code,
                    '#' in extracted_code,  # Has comments
                ])
                if not has_meaningful_content:
                    print(f"[WARNING] Attempt {attempt}/5: Code lacks meaningful content (no imports, functions, classes, or comments)")
                    continue

                # CRITICAL FIX: Level 4 - Validate Python syntax
                is_valid, syntax_error = validate_syntax(extracted_code)
                if not is_valid:
                    # Print error in a readable boxed format
                    print(f"[{'='*60}]")
                    print(f"[SYNTAX ERROR] Attempt {attempt}/5 - Code validation failed")
                    print(f"[{'='*60}]")
                    for line in syntax_error.split('\n'):
                        print(f"[ERROR] {line}")
                    print(f"[{'='*60}]")
                    print(f"[ACTION] Retrying with error feedback to LLM...")
                    print(f"[{'='*60}]")

                    # CRITICAL FIX: Remove old error feedback to prevent accumulation
                    user_prompt = re.sub(r'\n\n\[PREVIOUS ATTEMPT[^\n]*\n.*?(?=\n\n|$)', '', user_prompt, flags=re.DOTALL)

                    # Add new error feedback
                    user_prompt += f"\n\n[PREVIOUS ATTEMPT FAILED - SYNTAX ERROR]\n"
                    user_prompt += f"The code you generated has a Python syntax error:\n{syntax_error}\n\n"
                    user_prompt += f"Please fix this error and generate valid Python code.\n"
                    user_prompt += f"Common issues to check:\n"
                    user_prompt += f"- Add colons (:) after if, for, while, def, class statements\n"
                    user_prompt += f"- Match all parentheses, brackets, and quotes\n"
                    user_prompt += f"- Complete all expressions (no incomplete lines)\n"

                    # Regenerate prompt with error feedback
                    prompt = safe_format(TASK_CODING_PROMPT,
                                        data_file=data_file,
                                        data_summary=data_summary,
                                        variable_description=variable_description,
                                        task_description=task_description,
                                        task_analysis=task_analysis,
                                        modeling_formulas=formulas,
                                        modeling_process=modeling,
                                        dependent_file_prompt=dependent_file_prompt,
                                        code_template=code_template,
                                        user_prompt=user_prompt).strip()
                    continue

                # All validations passed
                new_content = extracted_code

                # P0-3 FIX: Import Whitelist Guard - Prevent new dependencies
                # Check if code tries to import forbidden or unknown modules
                try:
                    from utils.import_guard import ImportWhitelistGuard

                    # Validate imports against whitelist
                    import_guard = ImportWhitelistGuard()
                    imports_allowed, import_violations = import_guard.check_imports(new_content)

                    if not imports_allowed:
                        print(f"[P0-3] Import validation failed - {len(import_violations)} violation(s):")
                        for v in import_violations:
                            if v['type'] == 'forbidden_import':
                                print(f"  - FORBIDDEN: {v['import']} (line {v['line']})")
                                print(f"    Reason: {v['reason']}")
                            elif v['type'] == 'unknown_import':
                                print(f"  - UNKNOWN: {v['import']} (line {v['line']})")
                                print(f"    Reason: {v['reason']}")

                        # Reject code with forbidden imports
                        if any(v['type'] == 'forbidden_import' for v in import_violations):
                            print(f"[P0-3] Code contains forbidden imports - regenerating...")
                            user_prompt += f"\n\n[PREVIOUS ATTEMPT FAILED - FORBIDDEN IMPORT]\n"
                            user_prompt += f"Your code contains imports that are not allowed:\n"
                            for v in import_violations:
                                if v['type'] == 'forbidden_import':
                                    user_prompt += f"  - {v['import']}: {v['reason']}\n"
                            user_prompt += f"\nPlease regenerate code WITHOUT these imports.\n"
                            user_prompt += f"Use only basic data processing (pandas, numpy) and visualization (matplotlib).\n"
                            user_prompt += f"Do NOT use complex ML libraries (PCA, clustering, deep learning).\n"

                            # Regenerate prompt with error feedback
                            prompt = safe_format(TASK_CODING_PROMPT,
                                                data_file=data_file,
                                                data_summary=data_summary,
                                                variable_description=variable_description,
                                                task_description=task_description,
                                                task_analysis=task_analysis,
                                                modeling_formulas=formulas,
                                                modeling_process=modeling,
                                                dependent_file_prompt=dependent_file_prompt,
                                                code_template=code_template,
                                                user_prompt=user_prompt).strip()
                            continue
                        else:
                            # Unknown imports - warning but allow
                            print(f"[P0-3] Code contains unknown imports - proceeding with caution")
                    else:
                        print(f"[P0-3] Import validation passed - all imports allowed")

                except ImportError:
                    print(f"[WARNING] Import guard not available - skipping import validation")
                except Exception as e:
                    print(f"[WARNING] Import validation failed: {e}")

                # P0-1 FIX: AST Path Normalization - Force all CSV paths to use only filenames
                # This prevents LLM from deciding file paths (hardcoded Windows paths, relative paths, etc.)
                try:
                    from utils.ast_path_guard import ASTPathGuard
                    from utils.data_manager import get_data_manager

                    # Get allowed CSV files from data manager
                    data_mgr = get_data_manager()
                    allowed_files = data_mgr.list_csv_files() if data_mgr else []

                    # Normalize all file paths in code using AST rewriting
                    path_guard = ASTPathGuard(allowed_files=allowed_files)
                    normalized_code, violations = path_guard.normalize_code_paths(new_content)

                    if violations:
                        print(f"[P0-1] Path normalization found {len(violations)} issue(s):")
                        for v in violations:
                            if v['type'] == 'disallowed_file':
                                print(f"  - Disallowed file: {v['file']}")
                                print(f"    Allowed: {v['allowed'][:5]}...")
                            else:
                                print(f"  - {v['type']}: {v.get('message', 'Unknown')}")

                    # Use normalized code
                    if normalized_code != new_content:
                        print(f"[P0-1] Code paths normalized successfully")
                        new_content = normalized_code
                    else:
                        print(f"[P0-1] No path normalization needed (code already clean)")

                except ImportError as e:
                    print(f"[WARNING] AST path guard not available: {e}")
                    print(f"[WARNING] Proceeding without path normalization (hardcoded paths may cause issues)")
                except Exception as e:
                    print(f"[WARNING] AST path normalization failed: {e}")
                    print(f"[WARNING] Proceeding with original code")

                print(f"[OK] Attempt {attempt}/5: Valid code generated ({len(new_content)} chars)")
                break
            except Exception as e:
                # Format control.
                print(f"[WARNING] Attempt {attempt}/5: Code extraction failed - {e}")
                continue

        # CRITICAL FIX: Post-loop validation to prevent UnboundLocalError
        if new_content is None:
            error_msg = (
                f"[CRITICAL ERROR] Failed to generate valid Python code after 5 retries\n"
                f"Task: {task_description[:100] if task_description else 'Unknown'}...\n"
                f"Script: {script_name}\n"
                f"\n"
                f"Possible causes:\n"
                f"  - API rate limiting (Error 429)\n"
                f"  - Invalid LLM response format (missing ```python delimiter)\n"
                f"  - Network error\n"
                f"  - LLM returned empty or too-short code\n"
                f"  - Insufficient context in prompt\n"
            )
            print(error_msg)
            # CRITICAL FIX: Log to errors.log for tracking
            if self.logger_manager:
                self.logger_manager.log_exception(
                    exception=Exception(f"Failed to generate valid Python code for {script_name} after 5 retries"),
                    context=f"Code generation failed after 5 retries for script {script_name}",
                    stage="computational_solving",
                    task=task_description[:100] if task_description else 'Unknown',
                    include_traceback=False
                )
            raise Exception(f"Failed to generate valid Python code for {script_name} after 5 retries")

        # CRITICAL FIX: Ensure directory exists before writing
        os.makedirs(work_dir, exist_ok=True)

        # CRITICAL FIX: Only write if new_content exists
        if new_content:
            with open(os.path.join(work_dir, script_name), "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"[OK] Code written to {os.path.join(work_dir, script_name)}")

        # CRITICAL FIX: Initialize observation to prevent UnboundLocalError on timeout/exception
        observation = ""

        # Execute the script with timeout protection.
        try:
            executor = SecureScriptExecutor(logger_manager=self.logger_manager)
            observation = executor.execute(script_name, work_dir)
            ## If observation is too long, we only keep the last ~2k tokens.
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = len(enc.encode(observation))
            if tokens >= 2000:
                # FIX #1: Keep last 2000 chars instead of first 2000
                observation = observation[-2000:]
                tokens = len(enc.encode(observation))
        except Exception as e:
            error_msg = f"[ERROR] Code execution failed: {e}"
            print(error_msg)
            # CRITICAL FIX: Log to errors.log for tracking
            if self.logger_manager:
                self.logger_manager.log_exception(
                    exception=e,
                    context=f"Code execution failed for script {script_name}",
                    stage="computational_solving",
                    task=task_description[:100] if task_description else 'Unknown',
                    include_traceback=True
                )
            observation = error_msg  # CRITICAL: Ensure observation is always assigned

        return new_content, observation

    def coding_debugger(self, code_template: str, modeling: str, code: str, observation: str, script_name: str, work_dir: str, user_prompt: str = ''):
        # CRITICAL FIX: Initialize new_content to prevent UnboundLocalError
        new_content = None

        # CRITICAL FIX: Context preservation - track code history using script_name as key
        # Extract error type from observation for error pattern detection
        error_type = "Unknown"
        error_message = ""

        # Detect common error patterns
        if "KeyError:" in observation:
            error_type = "KeyError"
            error_match = re.search(r"KeyError: ('[^\']+'|\"[^\"]+\")", observation)
            if error_match:
                error_message = error_match.group(1)
        elif "NameError:" in observation:
            error_type = "NameError"
            error_match = re.search(r"NameError: name '[^']+' is not defined", observation)
            if error_match:
                error_message = error_match.group(0)
        elif "TypeError:" in observation:
            error_type = "TypeError"
            error_match = re.search(r"TypeError: (.+)", observation)
            if error_match:
                error_message = error_match.group(1).strip()
        elif "ValueError:" in observation:
            error_type = "ValueError"
            error_match = re.search(r"ValueError: (.+)", observation)
            if error_match:
                error_message = error_match.group(1).strip()
        elif "SyntaxError" in observation:
            error_type = "SyntaxError"
            error_match = re.search(r"SyntaxError: (.+)", observation)
            if error_match:
                error_message = error_match.group(1).strip()

        # Track error history
        if script_name not in self.error_history:
            self.error_history[script_name] = []
        self.error_history[script_name].append((len(self.error_history[script_name]) + 1, error_type, error_message))

        # CRITICAL FIX: Error pattern detection - check if same error occurred 3+ times
        error_count = sum(1 for _, e_type, _ in self.error_history[script_name] if e_type == error_type)
        if error_count >= 3:
            print(f"[ERROR PATTERN DETECTED] {error_type} has occurred {error_count} times")
            print(f"[FALLBACK] Activating fallback strategy...")
            # Use fallback strategy: simplify the prompt to focus on fixing this specific error
            user_prompt += f"\n\n[CRITICAL - SAME ERROR {error_count} TIMES]\n"
            user_prompt += f"The error '{error_type}: {error_message}' has occurred {error_count} times.\n"
            user_prompt += f"Please try a DIFFERENT approach to fix this error.\n"
            user_prompt += f"Do NOT repeat the same solution that failed before.\n"
            user_prompt += f"Consider alternative methods or simpler approaches.\n"

        # CRITICAL FIX: Preserve import statements from previous code
        # Extract all import statements from the current code
        import_statements = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                import_statements.append(line)

        # Build enhanced prompt with import preservation instruction
        if import_statements:
            user_prompt += f"\n\n[CRITICAL - PRESERVE IMPORTS]\n"
            user_prompt += f"The following import statements were in the previous code and MUST be preserved:\n"
            user_prompt += f"{chr(10).join(import_statements)}\n"
            user_prompt += f"Do NOT remove these import statements when fixing the code.\n"

        prompt = safe_format(TASK_CODING_DEBUG_PROMPT, code_template=code_template, modeling_process=modeling, code=code, observation=observation, user_prompt=user_prompt).strip()

        attempt = 0
        while attempt < 5:
            attempt += 1
            try:
                completion = self.llm.generate(prompt)

                # CRITICAL FIX: Level 1 - Extract code using improved regex
                extracted_code = extract_python_code(completion)
                if not extracted_code:
                    print(f"[WARNING] Debug Attempt {attempt}/5: No ```python delimiter found in LLM response")
                    continue

                # CRITICAL FIX: Level 2 - Validate code is not empty
                if not extracted_code.strip():
                    print(f"[WARNING] Debug Attempt {attempt}/5: Empty code block extracted")
                    continue

                # CRITICAL FIX: Level 3 - Validate code has meaningful content
                has_meaningful_content = any([
                    'import ' in extracted_code,
                    'def ' in extracted_code,
                    'class ' in extracted_code,
                    'if __name__' in extracted_code,
                    '#' in extracted_code,  # Has comments
                ])
                if not has_meaningful_content:
                    print(f"[WARNING] Debug Attempt {attempt}/5: Code lacks meaningful content (no imports, functions, classes, or comments)")
                    continue

                # CRITICAL FIX: Level 4 - Validate Python syntax
                is_valid, syntax_error = validate_syntax(extracted_code)
                if not is_valid:
                    # Print error in a readable boxed format
                    print(f"[{'='*60}]")
                    print(f"[SYNTAX ERROR] Debug Attempt {attempt}/5 - Code validation failed")
                    print(f"[{'='*60}]")
                    for line in syntax_error.split('\n'):
                        print(f"[ERROR] {line}")
                    print(f"[{'='*60}]")
                    print(f"[ACTION] Retrying with error feedback to LLM...")
                    print(f"[{'='*60}]")

                    # CRITICAL FIX: Remove old error feedback to prevent accumulation
                    user_prompt = re.sub(r'\n\n\[PREVIOUS DEBUG ATTEMPT[^\n]*\n.*?(?=\n\n|$)', '', user_prompt, flags=re.DOTALL)

                    # Add new error feedback
                    user_prompt += f"\n\n[PREVIOUS DEBUG ATTEMPT FAILED - SYNTAX ERROR]\n"
                    user_prompt += f"The debug code you generated has a Python syntax error:\n{syntax_error}\n\n"
                    user_prompt += f"Please fix this error and generate valid Python debug code.\n"
                    user_prompt += f"Common issues to check:\n"
                    user_prompt += f"- Add colons (:) after if, for, while, def, class statements\n"
                    user_prompt += f"- Match all parentheses, brackets, and quotes\n"
                    user_prompt += f"- Complete all expressions (no incomplete lines)\n"

                    # Regenerate prompt with error feedback
                    prompt = safe_format(TASK_CODING_DEBUG_PROMPT, 
                        code_template=code_template,
                        modeling_process=modeling,
                        code=code,
                        observation=observation,
                        user_prompt=user_prompt
                    ).strip()
                    continue

                # All validations passed
                new_content = extracted_code
                print(f"[OK] Debug Attempt {attempt}/5: Valid code generated ({len(new_content)} chars)")
                break
            except Exception as e:
                # Format control.
                print(f"[WARNING] Debug Attempt {attempt}/5: Code extraction failed - {e}")
                continue

        # CRITICAL FIX: Post-loop validation to prevent UnboundLocalError
        if new_content is None:
            error_msg = (
                f"[CRITICAL ERROR] Failed to generate valid debug code after 5 retries\n"
                f"Script: {script_name}\n"
                f"Previous error: {observation[:100] if observation else 'Unknown'}...\n"
                f"\n"
                f"Possible causes:\n"
                f"  - API rate limiting (Error 429)\n"
                f"  - Invalid LLM response format (missing ```python delimiter)\n"
                f"  - Network error\n"
                f"  - LLM returned empty or too-short code\n"
                f"  - Insufficient context in prompt\n"
            )
            print(error_msg)
            raise Exception(f"Failed to generate valid debug code for {script_name} after 5 retries")

        # CRITICAL FIX: Ensure directory exists before writing
        os.makedirs(work_dir, exist_ok=True)

        # CRITICAL FIX: Only write if new_content exists
        if new_content:
            with open(os.path.join(work_dir, script_name), "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"[OK] Debug code written to {os.path.join(work_dir, script_name)}")

        # CRITICAL FIX: Initialize observation to prevent UnboundLocalError on timeout/exception
        new_observation = ""

        # Execute the script with timeout protection.
        try:
            executor = SecureScriptExecutor(logger_manager=self.logger_manager)
            new_observation = executor.execute(script_name, work_dir)
            ## If observation is too long, we only keep the last ~2k tokens.
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = len(enc.encode(new_observation))
            if tokens >= 2000:
                # FIX #1: Keep last 2000 chars instead of first 2000
                new_observation = new_observation[-2000:]
                tokens = len(enc.encode(new_observation))
        except Exception as e:
            error_msg = f"[ERROR] Code execution failed: {e}"
            print(error_msg)
            # CRITICAL FIX: Log to errors.log for tracking
            self.logger.error(f"Debug code execution failed | Script: {script_name} | Error: {e}")
            new_observation = error_msg  # CRITICAL: Ensure observation is always assigned

        return new_content, new_observation

    def coding(self, data_file, data_summary, variable_description, task_description: str, task_analysis: str, formulas: str, modeling: str, dependent_file_prompt: str, code_template: str, script_name: str, work_dir: str, try_num: int = 5, round: int = 1, user_prompt: str = ''):
        for i in range(try_num):
            print("="*10 + f" Try: {i + 1} " + "="*10)
            iteration = 0
            max_iteration = 3
            while iteration < max_iteration:
                print("="*10 + f" Iteration: {iteration + 1} " + "="*10)
                if iteration == 0:
                    code, observation = self.coding_actor(data_file, data_summary, variable_description, task_description, task_analysis, formulas, modeling, dependent_file_prompt, code_template, script_name, work_dir, user_prompt)
                    # CRITICAL FIX: Enhanced error detection to catch all failures
                    # Check for execution errors, syntax errors, and our [ERROR] prefix
                    has_error = (
                        "[ERROR]" in observation or  # Our error prefix from SecureScriptExecutor
                        "Traceback (most recent call last):" in observation or  # Runtime errors
                        "SyntaxError" in observation or  # Syntax errors
                        "IndentationError" in observation or  # Indentation errors
                        "EnvException" in observation  # Timeout/environment errors
                    )
                    if not has_error:
                        # Extract output with bounds checking
                        output_parts = observation.split("The script has been executed. Here is the output:\n")
                        if len(output_parts) > 1:
                            return code, True, output_parts[1]
                        else:
                            # No prefix found, return observation as-is
                            return code, True, observation
                    # FIX #5: Add runtime error feedback to user_prompt for next iteration
                    else:
                        print(f"[WARNING] Runtime error detected in iteration {iteration + 1}, feeding back to debugger...")
                else:
                    code, observation = self.coding_debugger(code_template, modeling, code, observation, script_name, work_dir, user_prompt)
                    # CRITICAL FIX: Enhanced error detection to catch all failures
                    has_error = (
                        "[ERROR]" in observation or  # Our error prefix from SecureScriptExecutor
                        "Traceback (most recent call last):" in observation or  # Runtime errors
                        "SyntaxError" in observation or  # Syntax errors
                        "IndentationError" in observation or  # Indentation errors
                        "EnvException" in observation  # Timeout/environment errors
                    )
                    if not has_error:
                        # Extract output with bounds checking
                        output_parts = observation.split("The script has been executed. Here is the output:\n")
                        if len(output_parts) > 1:
                            return code, True, output_parts[1]
                        else:
                            # No prefix found, return observation as-is
                            return code, True, observation
                iteration += 1

        return code, False, None

    def result(self, task_description: str, task_analysis: str, task_formulas: str, task_modeling: str, user_prompt: str = '', execution_result: str = ''):
        if execution_result == '':
            prompt = safe_format(TASK_RESULT_PROMPT, task_description=task_description, task_analysis=task_analysis, task_formulas=task_formulas, task_modeling=task_modeling, user_prompt=user_prompt).strip()
        else:
            prompt = safe_format(TASK_RESULT_WITH_CODE_PROMPT, task_description=task_description, task_analysis=task_analysis, task_formulas=task_formulas, task_modeling=task_modeling, user_prompt=user_prompt, execution_result=execution_result).strip()
        return self.llm.generate(prompt)

    def answer(self, task_description: str, task_analysis: str, task_formulas: str, task_modeling: str, task_result: str, user_prompt: str = ''):
        prompt = safe_format(TASK_ANSWER_PROMPT, task_description=task_description, task_analysis=task_analysis, task_formulas=task_formulas, task_modeling=task_modeling, task_result=task_result, user_prompt=user_prompt).strip()
        return self.llm.generate(prompt)

    def extract_code_structure(self, task_id, code: str, save_path: str):
        """
        Extract code structure from LLM response.

        CRITICAL FIXES APPLIED (2026-01-11):
        - Fixed for-else logic (prevented count variable bug)
        - Improved JSON extraction using Coordinator's strategy
        - Replaced sys.exit() with proper exception raising
        """
        import re
        prompt = safe_format(CODE_STRUCTURE_PROMPT, code=code, save_path=save_path)
        last_error = None

        # CRITICAL FIX: Reduced retries from 5 to 3 to prevent long timeouts
        # else block only executes if all 3 attempts fail (no return)
        for attempt in range(1, 4):
            try:
                # CRITICAL FIX: Use Coordinator's robust JSON extraction strategy
                raw_response = self.llm.generate(prompt)

                # Strategy 1: Extract ```json ... ``` code blocks
                if '```json' in raw_response:
                    json_str = raw_response.split('```json', 1)[1].split('```', 1)[0].strip()
                elif '```' in raw_response:
                    json_str = raw_response.split('```', 1)[1].split('```', 1)[0].strip()
                else:
                    # Strategy 2: Find first {...} JSON object
                    json_match = re.search(r'\{[\s\S]*\}', raw_response)
                    if json_match:
                        json_str = json_match.group()
                    else:
                        json_str = raw_response.strip()

                # Strategy 3: Fix common LLM JSON issues
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", '"')
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                # Remove comments
                json_str = re.sub(r'//.*?\n', '\n', json_str)
                json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)

                # Try parsing JSON
                structure_json = json.loads(json_str)

                # Validate required key
                if 'file_outputs' not in structure_json:
                    print(f"[WARNING] Extract code structure attempt {attempt}/5: No 'file_outputs' key in JSON")
                    last_error = ValueError("Missing 'file_outputs' key")
                    continue

                # Iterate with bounds checking
                for j in range(len(structure_json['file_outputs'])):
                    if 'file_description' in structure_json['file_outputs'][j]:
                        structure_json['file_outputs'][j]['file_description'] = 'This file is generated by code for Task {}. '.format(task_id) + structure_json['file_outputs'][j]['file_description']

                # Success: return immediately
                return structure_json

            except Exception as e:
                last_error = e
                print(f"[WARNING] Extract code structure attempt {attempt}/5 failed: {e}")
                continue
        else:
            # CRITICAL FIX: This block only executes if all 5 attempts failed
            # Changed from sys.exit() to proper exception raising
            error_msg = (
                f"[CRITICAL ERROR] Failed to extract code structure after 5 attempts\n"
                f"Task ID: {task_id}\n"
                f"Last error: {last_error}\n"
            )
            print(error_msg)
            raise RuntimeError(f"Failed to extract code structure for Task {task_id} after 5 attempts. Last error: {last_error}") from last_error

    def detect_error_pattern(self, script_name: str) -> tuple:
        """
        Detect if the same error has occurred multiple times.

        Args:
            script_name: Name of the script being debugged

        Returns:
            (has_pattern, error_type, error_count) where:
            - has_pattern: True if same error occurred 3+ times
            - error_type: The error type that is repeating
            - error_count: How many times this error occurred
        """
        if script_name not in self.error_history or not self.error_history[script_name]:
            return False, None, 0

        # Count occurrences of each error type
        error_counts = {}
        for _, error_type, _ in self.error_history[script_name]:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        # Find the most frequent error
        most_frequent_error = max(error_counts.items(), key=lambda x: x[1])
        error_type, error_count = most_frequent_error

        # Check if error occurred 3+ times
        has_pattern = error_count >= 3

        if has_pattern:
            print(f"[ERROR PATTERN] {error_type} has occurred {error_count} times")

        return has_pattern, error_type, error_count

    def fallback_strategy(self, error_type: str, error_message: str, script_name: str) -> str:
        """
        Generate fallback strategy when same error repeats.

        Args:
            error_type: Type of error (e.g., "KeyError", "NameError")
            error_message: The error message
            script_name: Name of the script being debugged

        Returns:
            Enhanced prompt with fallback instructions
        """
        fallback_prompt = "\n\n[FALLBACK STRATEGY ACTIVATED]\n"

        if error_type == "KeyError":
            fallback_prompt += f"The KeyError '{error_message}' has occurred multiple times.\n"
            fallback_prompt += "Fallback approaches to try:\n"
            fallback_prompt += "1. Check if column exists before accessing: if 'col_name' in df.columns:\n"
            fallback_prompt += "2. Use iloc/positional indexing: df.iloc[:, 0]\n"
            fallback_prompt += "3. Use select_dtypes for numeric columns: df.select_dtypes(include=[np.number])\n"
            fallback_prompt += "4. Simplify the analysis to use available columns only\n"
            fallback_prompt += "5. Print df.columns.tolist() to see what columns are available\n"

        elif error_type == "NameError":
            fallback_prompt += f"A NameError has occurred multiple times.\n"
            fallback_prompt += "Fallback approaches to try:\n"
            fallback_prompt += "1. Ensure ALL required imports are at the top: import pandas as pd, import numpy as np\n"
            fallback_prompt += "2. Check variable names for typos\n"
            fallback_prompt += "3. Define variables before using them\n"
            fallback_prompt += "4. Use print() statements to debug variable scope\n"

        elif error_type == "TypeError":
            fallback_prompt += f"A TypeError has occurred multiple times.\n"
            fallback_prompt += "Fallback approaches to try:\n"
            fallback_prompt += "1. Verify data types before operations: print(type(data))\n"
            fallback_prompt += "2. Use explicit type conversion: int(), float(), str()\n"
            fallback_prompt += "3. Check function signatures and parameter types\n"
            fallback_prompt += "4. Handle mixed types in DataFrames: select_dtypes(include=[np.number])\n"

        elif error_type == "ValueError":
            fallback_prompt += f"A ValueError has occurred multiple times.\n"
            fallback_prompt += "Fallback approaches to try:\n"
            fallback_prompt += "1. Validate input data before processing\n"
            fallback_prompt += "2. Handle edge cases: empty data, missing values, NaN\n"
            fallback_prompt += "3. Use try-except blocks around risky operations\n"
            fallback_prompt += "4. Print data shape and info before operations\n"

        elif error_type == "SyntaxError":
            fallback_prompt += f"A SyntaxError has occurred multiple times.\n"
            fallback_prompt += "Fallback approaches to try:\n"
            fallback_prompt += "1. Check for missing colons (:) after if, for, while, def\n"
            fallback_prompt += "2. Match all parentheses, brackets, and quotes\n"
            fallback_prompt += "3. Complete all expressions (no incomplete lines)\n"
            fallback_prompt += "4. Use simpler code structure to avoid syntax issues\n"

        else:
            fallback_prompt += f"The error '{error_type}: {error_message}' has occurred multiple times.\n"
            fallback_prompt += "Fallback approaches to try:\n"
            fallback_prompt += "1. Use a completely different approach\n"
            fallback_prompt += "2. Simplify the code to minimal working example\n"
            fallback_prompt += "3. Add extensive debugging print statements\n"
            fallback_prompt += "4. Break the problem into smaller steps\n"

        fallback_prompt += "\nCRITICAL: Do NOT repeat the same approach that failed before.\n"
        fallback_prompt += "Try a fundamentally different method.\n"

        # Log fallback activation
        if self.logger:
            self.logger.warning(f"Fallback strategy activated | Script: {script_name} | Error: {error_type} | Message: {error_message}")

        return fallback_prompt
