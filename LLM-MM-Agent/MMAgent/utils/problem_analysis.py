import os
from utils.utils import read_json_file
from agent.data_description import DataDescription
from agent.problem_analysis import ProblemUnderstanding
from agent.coordinator import Coordinator
from agent.problem_decompose import ProblemDecompose
from prompt.template import PROBLEM_PROMPT
import shutil


def get_problem(problem_path, llm, latent_reporter=None):
    problem = read_json_file(problem_path)
    data_description = problem.get('dataset_description', {})

    # LATENT REPORTER: Log data description analysis start
    if latent_reporter and data_description:
        data_info = f"""**Data Description Analysis**

**Dataset Path**: {problem.get('dataset_path', 'Not specified')}

**Available Data Descriptions**: {len(str(data_description))} characters

**Variable Descriptions**: {len(str(problem.get('variable_description', {})))} characters

**Process**: Generating comprehensive data summary using LLM analysis of dataset structure and variables.
"""
        latent_reporter.log_thought("Data Description", data_info, "INFO")

    ds = DataDescription(llm)

    if data_description:
        data_path = problem['dataset_path']
        variable_description = problem['variable_description']
        data_summary = ds.summary(data_description=str(data_description) + '\n' + str(variable_description))
        data_summary = f'Dataset Path:\n{data_path}\n\nData Description:\n{data_summary}'

        # LATENT REPORTER: Log data summary completion
        if latent_reporter:
            summary_preview = data_summary[:500] if len(data_summary) > 500 else data_summary
            summary_info = f"""**Data Summary Generated**

**Summary Length**: {len(data_summary)} characters

**Data Summary Preview**:
{summary_preview}

{'[Full summary truncated]' if len(data_summary) > 500 else ''}

**Next Step**: Use data summary in problem understanding and task decomposition.
"""
            latent_reporter.log_success("Data Summary", summary_info)
    else:
        data_summary = ''

    problem['data_summary'] = data_summary
    problem['data_description'] = data_description

    if problem.get('addendum', ''):
        addendum = f"Addendum: \n{problem['addendum']}"
    else:
        addendum = ''

    problem_str = PROBLEM_PROMPT.format(problem_background=problem['background'], problem_requirement=problem['problem_requirement'], addendum=addendum, data_summary=data_summary).strip()
    problem['problem_str'] = problem_str
    return problem_str, problem


def problem_analysis(llm, problem_path, config, dataset_path, output_dir, logger_manager=None, latent_reporter=None):
    """
    Perform problem analysis and decomposition.

    Args:
        llm: Language model instance
        problem_path: Path to problem JSON file
        config: Configuration dictionary
        dataset_path: Path to dataset directory
        output_dir: Output directory path
        logger_manager: Optional MMExperimentLogger for structured logging
        latent_reporter: Optional LatentReporter for LLM-powered narrative logging

    Returns:
        Tuple of (problem_str, problem, order, with_code, coordinator, task_descriptions, solution)
    """
    # Log analysis start
    if logger_manager:
        logger_manager.log_progress("Starting Problem Analysis stage", level='info')

    # [FIX] Log to LatentReporter
    if latent_reporter:
        latent_reporter.log_thought("Problem Understanding", "Loading problem file and analyzing background, requirements, and available data", "INFO")

    try:
        # Get problem
        if logger_manager:
            logger_manager.log_progress("Loading problem file", level='info')
        problem_str, problem = get_problem(problem_path, llm, latent_reporter)
        problem_type = os.path.splitext(os.path.basename(problem_path))[0].split('_')[-1]

        # Initialize solution dictionary
        solution = {'tasks': []}
        solution['problem_background'] = problem['background']
        solution['problem_requirement'] = problem['problem_requirement']

        # Problem Understanding
        if logger_manager:
            logger_manager.log_progress("Starting Problem Understanding (LLM analysis)", level='info')
        if latent_reporter:
            latent_reporter.log_thought("Problem Understanding", "Analyzing problem structure using LLM to extract key components and objectives", "INFO")
        pu = ProblemUnderstanding(llm)
        problem_analysis = pu.analysis(problem_str, round=config['problem_analysis_round'])
        solution['problem_analysis'] = problem_analysis
        if logger_manager:
            logger_manager.log_progress("Problem Understanding completed", level='info')
        if latent_reporter:
            latent_reporter.log_success("Problem Understanding", "Completed deep analysis of problem structure and extracted core objectives")

        # High level probelm understanding modeling
        if logger_manager:
            logger_manager.log_progress("Starting Mathematical Modeling (LLM modeling)", level='info')
        modeling_solution = pu.modeling(problem_str, problem_analysis, round=config['problem_modeling_round'])
        if logger_manager:
            logger_manager.log_progress("Mathematical Modeling completed", level='info')
            main_logger = logger_manager.get_logger('main')
            main_logger.info('********************* Step 1: Problem Understanding finish *********************')

        # Problem Decomposition
        if logger_manager:
            logger_manager.log_progress("Starting Problem Decomposition (LLM decomposition)", level='info')
        if latent_reporter:
            latent_reporter.log_thought("Problem Decomposition", f"Breaking down complex problem into {config['num_tasks']} manageable subtasks", "INFO")
        pd = ProblemDecompose(llm)
        task_descriptions = pd.decompose_and_refine(problem_str, problem_analysis, modeling_solution, problem_type, config['num_tasks'])
        if logger_manager:
            logger_manager.log_progress(f"Problem Decomposition completed - {len(task_descriptions)} tasks identified", level='info')
            main_logger = logger_manager.get_logger('main')
            main_logger.info('********************* Step 2: Problem Decomposition finish *********************')
        if latent_reporter:
            # Enhanced logging with task details
            tasks_list = "\n".join([f"{i+1}. {task[:150]}..." for i, task in enumerate(task_descriptions)])
            decomposition_details = f"""**Problem Decomposition Completed**

**Number of Subtasks**: {len(task_descriptions)}

**Decomposed Tasks**:
{tasks_list}

**Process**: Used LLM-based decomposition to break down the complex problem into manageable, executable subtasks.

**Next Step**: Analyze task dependencies and build execution DAG.
"""
            latent_reporter.log_success("Problem Decomposition", decomposition_details)

        # Task Dependency Analysis
        with_code = len(problem['dataset_path']) > 0
        if logger_manager:
            logger_manager.log_progress("Starting Task Dependency Analysis", level='info')
        if latent_reporter:
            dag_start = f"""**Task Dependency Analysis**

**Analysis Approach**: Directed Acyclic Graph (DAG)
- **Number of Tasks**: {len(task_descriptions)}
- **With Code Execution**: {with_code}

**Process**:
1. Analyzing task dependencies and relationships
2. Building DAG to identify optimal execution order
3. Validating DAG structure (no cycles)
4. Computing topological sort for execution order

**Next Step**: Generate execution order based on dependency analysis.
"""
            latent_reporter.log_thought("Task Dependency Analysis", dag_start, "INFO")
        # [NEW] Get logger and pass to Coordinator for structured logging
        main_logger = logger_manager.get_logger('main') if logger_manager else None
        coordinator = Coordinator(llm, logger=main_logger)
        order = coordinator.analyze_dependencies(problem_str, problem_analysis, modeling_solution, task_descriptions, config['num_tasks'], with_code)
        if logger_manager:
            logger_manager.log_progress(f"Task Dependency Analysis completed - execution order: {order}", level='info')
        if latent_reporter:
            # Enhanced logging with execution order details
            order_details = "\n".join([f"Step {i+1}: Task {task_id}" for i, task_id in enumerate(order)])
            dag_completion = f"""**DAG Analysis Completed**

**Execution Order** (Topological Sort):
{order_details}

**DAG Properties**:
- **Structure**: Valid Directed Acyclic Graph
- **Cycles**: None detected
- **Execution Strategy**: Topological ordering ensures dependencies are satisfied

**Next Step**: Proceed to mathematical modeling and computational solving for each task.
"""
            latent_reporter.log_success("DAG Validated", dag_completion)

        # Parse task IDs from various formats (e.g., 'Subtask 1', 'Task 1', '1')
        parsed_order = []
        for item in order:
            # If it's already a number, use it
            if isinstance(item, int):
                parsed_order.append(item)
            # If it's a string, extract the numeric part
            else:
                # Remove any non-numeric prefix/suffix and extract the number
                import re
                match = re.search(r'\d+', str(item))
                if match:
                    parsed_order.append(int(match.group()))
                else:
                    # Fallback: try direct conversion
                    try:
                        parsed_order.append(int(item))
                    except ValueError:
                        if logger_manager:
                            logger_manager.log_progress(f"Warning: Could not parse task ID from '{item}', skipping", level='warning')
                        print(f"[WARNING] Could not parse task ID from '{item}', skipping")
        order = parsed_order

        if with_code:
            if logger_manager:
                logger_manager.log_progress(f"Copying dataset files to output directory", level='info')
            shutil.copytree(dataset_path, os.path.join(output_dir,'code'), dirs_exist_ok=True)
        print('********************* Step 3: Task Dependency Analysis finish *********************')

        if logger_manager:
            logger_manager.log_progress("Problem Analysis stage completed successfully", level='info')

        return problem, order, with_code, coordinator, task_descriptions, solution

    except Exception as e:
        # Log exception with context
        if logger_manager:
            logger_manager.log_exception(
                exception=e,
                context="Exception in Problem Analysis stage",
                stage="Problem Analysis",
                include_traceback=True
            )
        # [FIX] Log to LatentReporter
        if latent_reporter:
            latent_reporter.log_error_analysis(type(e).__name__, f"Problem Analysis failed: {str(e)}")
        raise  # Re-raise to be caught by main.py's exception handler