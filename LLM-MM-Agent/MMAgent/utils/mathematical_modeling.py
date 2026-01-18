from agent.retrieve_method import MethodRetriever
from agent.task_solving import TaskSolver
from prompt.template import TASK_ANALYSIS_APPEND_PROMPT, TASK_FORMULAS_APPEND_PROMPT, TASK_MODELING_APPEND_PROMPT

# CRITICAL FIX: Context Pruning Strategy (from gemini 1.txt analysis)
# Implements character-based truncation to prevent context overflow
# Distinguishes between immediate predecessors (task_id - 1) and earlier dependencies
# See: test workplace/docs/41_context_pruning_analysis.md


def get_dependency_prompt(with_code, coordinator, task_id):
    """
    Generate dependency context with intelligent truncation to prevent context overflow.

    Strategy:
    - Immediate predecessor (task_id - 1): Full context with character limits
    - Earlier dependencies: Minimal context (only result interpretation)

    Args:
        with_code: Whether to include code structure information
        coordinator: Coordinator instance with memory and code_memory
        task_id: Current task ID

    Returns:
        Tuple of (task_analysis_prompt, task_formulas_prompt, task_modeling_prompt, dependent_file_prompt)
    """
    # Ensure task_id is integer for comparison
    try:
        current_id_int = int(task_id)
    except ValueError:
        current_id_int = 999999  # Fallback for non-integer task_id

    task_dependency = [int(i) for i in coordinator.DAG[str(task_id)]]
    dependent_file_prompt = ""

    if len(task_dependency) > 0:
        # Use dictionary mapping instead of list indexing to prevent IndexError
        # This prevents KeyError when task_id is not sequential (e.g., [1, 3, 5])
        dependency_analysis_text = ""
        if hasattr(coordinator, 'task_dependency_analysis') and coordinator.task_dependency_analysis:
            analysis_dict = {str(i+1): txt for i, txt in enumerate(coordinator.task_dependency_analysis)}
            task_id_str = str(task_id)
            if task_id_str in analysis_dict:
                dependency_analysis_text = analysis_dict[task_id_str]
            else:
                dependency_analysis_text = f"Tasks {task_dependency} (Analysis unavailable)"
        else:
            dependency_analysis_text = f"Tasks {task_dependency} (No dependency analysis)"

        dependency_prompt = f"""\
This task is Task {task_id}, which depends on the following tasks: {task_dependency}.
The dependencies for this task are analyzed as follows: {dependency_analysis_text}
"""

        # Context Pruning Strategy: Implement "远近亲疏" (near/far) strategy
        for dep_id in task_dependency:
            dep_id_str = str(dep_id)

            # Use .get() for safe access, return empty dict if key not found
            task_mem = coordinator.memory.get(dep_id_str, {})
            code_mem = coordinator.code_memory.get(dep_id_str, {}) if with_code else {}

            # Check if this is the immediate predecessor (task_id - 1)
            # Immediate predecessors get more detailed context
            is_immediate = (dep_id == current_id_int - 1)

            # Define truncation limits based on dependency type
            # Immediate: full context with limits
            # Earlier: minimal context (results only)
            MAX_LEN_DESC = 1000 if is_immediate else 0
            MAX_LEN_PROCESS = 1000 if is_immediate else 0
            MAX_LEN_RESULT = 2000 if is_immediate else 500
            MAX_LEN_CODE = 2000 if is_immediate else 0

            # Helper function for safe truncation
            def get_trunc(key, limit):
                """Safely get and truncate a value from task memory."""
                val = str(task_mem.get(key, ''))
                if limit == 0:
                    return ""
                if len(val) > limit:
                    return val[:limit] + f"... (truncated, total {len(val)} chars)"
                return val

            # Build dependency prompt section
            dependency_prompt += f"\n--- Dependency: Task {dep_id} ---\n"

            # 1. Task Description (immediate predecessor only)
            if is_immediate:
                desc = get_trunc('task_description', MAX_LEN_DESC)
                if desc:
                    dependency_prompt += f"# Description:\n{desc}\n"

            # 2. Modeling Process (immediate predecessor only)
            if is_immediate:
                process = get_trunc('mathematical_modeling_process', MAX_LEN_PROCESS)
                if process:
                    dependency_prompt += f"# Modeling Method:\n{process}\n"

            # 3. Result Interpretation (all dependencies, but different lengths)
            result = get_trunc('solution_interpretation', MAX_LEN_RESULT)
            if result:
                dependency_prompt += f"# Result/Conclusion:\n{result}\n"

            # 4. Code-related information (if requested)
            if with_code:
                # Code structure: only for immediate predecessor
                if is_immediate:
                    code_struct_str = str(code_mem)
                    if len(code_struct_str) > MAX_LEN_CODE:
                        code_struct_str = code_struct_str[:MAX_LEN_CODE] + "... (structure truncated)"
                    if code_struct_str and code_struct_str != "{}":
                        dependency_prompt += f"# Code Structure:\n{code_struct_str}\n"

                # CRITICAL: File outputs must ALWAYS be preserved for all dependencies
                # Even early tasks' files may be needed by current task
                file_outputs = code_mem.get('file_outputs', []) if isinstance(code_mem, dict) else []
                if file_outputs:
                    file_info = f"Task {dep_id} generated files: {file_outputs}"
                    dependent_file_prompt += f"\n# Files from Task {dep_id}:\n{file_outputs}\n"
                    # Also mention in main dependency prompt
                    dependency_prompt += f"# Generated Files:\n{file_outputs}\n"

            dependency_prompt += "---\n"

    else:
        # No dependencies case
        task_analysis_prompt = ""
        task_formulas_prompt = ""
        task_modeling_prompt = ""
        # dependent_file_prompt already initialized as ""

    # Only append prompts if there are dependencies
    if len(task_dependency) > 0:
        task_analysis_prompt = dependency_prompt + TASK_ANALYSIS_APPEND_PROMPT
        task_formulas_prompt = dependency_prompt + TASK_FORMULAS_APPEND_PROMPT
        task_modeling_prompt = dependency_prompt + TASK_MODELING_APPEND_PROMPT

    return task_analysis_prompt, task_formulas_prompt, task_modeling_prompt, dependent_file_prompt


def mathematical_modeling(task_id, problem, task_descriptions, llm, config, coordinator, with_code, logger_manager=None):
    """
    Perform mathematical modeling for a specific task.

    Args:
        task_id: Task identifier
        problem: Problem dictionary
        task_descriptions: List of task descriptions
        llm: Language model instance
        config: Configuration dictionary
        coordinator: Coordinator instance for managing task dependencies
        with_code: Whether code execution is involved
        logger_manager: Optional MMExperimentLogger for structured logging

    Returns:
        Tuple of (task_description, task_analysis, task_modeling_formulas, task_modeling_method, dependent_file_prompt)
    """
    # Log modeling start
    if logger_manager:
        logger_manager.log_progress(f"Task {task_id}: Mathematical Modeling", level='info')

    # CRITICAL FIX: Pass logger_manager to TaskSolver for proper error logging to errors.log
    ts = TaskSolver(llm, logger_manager)
    mr = MethodRetriever(llm)
    task_analysis_prompt, task_formulas_prompt, task_modeling_prompt, dependent_file_prompt = get_dependency_prompt(with_code, coordinator, task_id)

    # Task analysis - with bounds checking
    if 1 <= task_id <= len(task_descriptions):
        task_description = task_descriptions[task_id - 1]
    else:
        # Fallback: generate generic task description if task_id is out of range
        if logger_manager:
            logger_manager.log_progress(f"[WARNING] Task {task_id} ID out of range (1-{len(task_descriptions)}), using generic description", level='warning')
        task_description = f"Task {task_id}: Mathematical modeling and computational analysis"

    task_analysis = ts.analysis(task_analysis_prompt, task_description)
    
    # Hierarchical Modeling Knowledge Retrieval
    description_and_analysis = f'## Task Description\n{task_description}\n\n## Task Analysis\n{task_analysis}'
    top_modeling_methods = mr.retrieve_methods(description_and_analysis, top_k=config['top_method_num'])

    # Task Modeling
    task_modeling_formulas, task_modeling_method = ts.modeling(task_formulas_prompt, task_modeling_prompt, problem['data_description'], task_description, task_analysis, top_modeling_methods, round=config['task_formulas_round'])
    
    return task_description, task_analysis, task_modeling_formulas, task_modeling_method, dependent_file_prompt

    