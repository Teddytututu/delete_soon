import json
from typing import Dict
import os
import yaml
from datetime import datetime


def normalize_task_fields(task: Dict) -> Dict:
    """
    Add backward-compatible field aliases for task data.

    CRITICAL FIX: computational_solving.py generates:
    - 'execution_result' (raw code output)
    - 'solution_interpretation' (LLM interpretation)
    - 'subtask_outcome_analysis' (final answer)

    But json_to_markdown expects:
    - 'result' (for execution output)
    - 'answer' (for final answer)

    This function adds aliases so data is not lost in reporting.

    Args:
        task: Task dictionary potentially with inconsistent field names

    Returns:
        Task dictionary with normalized fields (modified in-place for efficiency)
    """
    # Map execution_result -> result (for code execution output)
    if 'result' not in task and 'execution_result' in task:
        task['result'] = task['execution_result']

    # Map solution_interpretation -> result (LLM interpretation preferred over raw output)
    if 'solution_interpretation' in task:
        task['result'] = task['solution_interpretation']

    # Map subtask_outcome_analysis -> answer
    if 'answer' not in task and 'subtask_outcome_analysis' in task:
        task['answer'] = task['subtask_outcome_analysis']

    return task


def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_json_file(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def write_text_file(file_path: str, content: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def write_json_file(file_path: str, data:dict) -> Dict:
    with open(file_path, "w", encoding="utf-8") as json_file:
        json_file.write(json.dumps(data, indent=4, ensure_ascii=False))


def parse_llm_output_to_json(output_text: str) -> dict:
    """
    CRITICAL FIX: Robustly parse LLM output text into a Python dictionary.

    Handles multiple JSON formats:
    - Plain JSON: {...}
    - Markdown code blocks: ```json ... ``` or ``` ... ```
    - Text with embedded JSON

    Returns:
        dict: Parsed JSON data, or empty dict if parsing fails

    Raises:
        ValueError: If JSON cannot be extracted and parsed
    """
    import re

    if not output_text or not isinstance(output_text, str):
        raise ValueError(f"Invalid output_text type: {type(output_text)}")

    # Strategy 1: Try to extract JSON from markdown code blocks
    # Pattern: ```json\n{...}\n``` or ```\n{...}\n```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, output_text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 2: Try to find JSON object using regex
    # Pattern: Match outermost braces with nested content
    json_pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
    json_matches = re.findall(json_pattern, output_text, re.DOTALL)

    # Try from largest (most nested) to smallest
    for match in reversed(json_matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Strategy 3: Fallback to original method (find first { to last })
    start = output_text.find("{")
    end = output_text.rfind("}") + 1

    if start != -1 and end > start:
        json_str = output_text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nJSON string: {json_str[:200]}...")
    else:
        raise ValueError(f"No JSON object found in output. Output preview: {output_text[:200]}...")


def json_to_markdown(paper):
    """
    Converts a paper dictionary to a Markdown string with multi-level headlines.

    Args:
        paper (dict): The paper dictionary containing problem details and tasks.

    Returns:
        str: A Markdown-formatted string representing the paper.
    """
    markdown_lines = []

    # Problem Background
    markdown_lines.append("## Problem Background")
    markdown_lines.append(paper.get('problem_background', 'No background provided.') + "\n")

    # Problem Requirement
    markdown_lines.append("## Problem Requirement")
    markdown_lines.append(paper.get('problem_requirement', 'No requirements provided.') + "\n")

    # Problem Analysis
    markdown_lines.append("## Problem Analysis")
    markdown_lines.append(paper.get('problem_analysis', 'No analysis provided.') + "\n")

    # Problem Modeling
    if 'problem_modeling' in paper:
        markdown_lines.append("## Problem Modeling")
        markdown_lines.append(paper.get('problem_modeling', 'No modeling provided.') + "\n")

    # Tasks
    tasks = paper.get('tasks', [])
    if tasks:
        markdown_lines.append("## Tasks\n")
        for idx, task in enumerate(tasks, start=1):
            # CRITICAL FIX: Normalize task fields before processing
            # This ensures backward compatibility with different field naming conventions
            task = normalize_task_fields(task)

            markdown_lines.append(f"### Task {idx}")

            task_description = task.get('task_description', 'No description provided.')
            markdown_lines.append("#### Task Description")
            markdown_lines.append(task_description + "\n")

            # Task Analysis
            task_analysis = task.get('task_analysis', 'No analysis provided.')
            markdown_lines.append("#### Task Analysis")
            markdown_lines.append(task_analysis + "\n")

            # Mathematical Formulas
            task_formulas = task.get('modeling_formulas', 'No formulas provided.')  # Changed from mathematical_formulas
            markdown_lines.append("#### Mathematical Formulas")
            if isinstance(task_formulas, list):
                for formula in task_formulas:
                    markdown_lines.append(f"$${formula}$$")
            else:
                markdown_lines.append(f"$${task_formulas}$$")
            markdown_lines.append("")  # Add an empty line

            # Mathematical Modeling Process
            task_modeling = task.get('mathematical_modeling_process', 'No modeling process provided.')
            markdown_lines.append("#### Mathematical Modeling Process")
            markdown_lines.append(task_modeling + "\n")

            # Result
            task_result = task.get('result', 'No result provided.')
            markdown_lines.append("#### Result")
            markdown_lines.append(task_result + "\n")

            # Answer
            task_answer = task.get('answer', 'No answer provided.')
            markdown_lines.append("#### Answer")
            markdown_lines.append(task_answer + "\n")

            # Charts
            charts = task.get('charts', [])
            if charts:
                markdown_lines.append("#### Charts")
                for i, chart in enumerate(charts, start=1):
                    markdown_lines.append(f"##### Chart {i}")

                    # Handle new chart format (dict with image_path)
                    if isinstance(chart, dict):
                        # P0 FIX: More defensive access - use .get() with defaults
                        # Check if chart was successful and has image_path
                        success = chart.get('success', False)
                        img_path = chart.get('image_path')

                        if success and img_path:
                            # Convert absolute path to relative
                            import os
                            # Make path relative to markdown file
                            if os.path.isabs(img_path):
                                # Get the relative path from markdown dir to the image
                                markdown_file_rel_path = os.path.relpath(img_path, os.path.join(os.path.dirname(img_path), '..', 'markdown'))
                                markdown_lines.append(f"![Chart {i}]({markdown_file_rel_path})\n")
                            else:
                                markdown_lines.append(f"![Chart {i}]({img_path})\n")

                        # Add description if available
                        description = chart.get('description', '')
                        if description:
                            markdown_lines.append(description + "\n")
                    else:
                        # Old format: just append the text
                        markdown_lines.append(str(chart) + "\n")

    # Combine all lines into a single string
    markdown_str = "\n".join(markdown_lines)
    return markdown_str


def json_to_markdown_general(json_data):
    """
    Convert a JSON object to a markdown format.

    Args:
    - json_data (str or dict): The JSON data to convert. It can be a JSON string or a dictionary.

    Returns:
    - str: The markdown formatted string.
    """
    
    if isinstance(json_data, str):
        json_data = json.loads(json_data)  # If input is a JSON string, parse it.

    def recursive_markdown(data, indent=0):
        markdown_str = ""
        indent_space = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                markdown_str += f"### {key}\n"
                markdown_str += recursive_markdown(value, indent + 1)
        elif isinstance(data, list):
            for index, item in enumerate(data):
                markdown_str += f"- **Item {index + 1}**\n"
                markdown_str += recursive_markdown(item, indent + 1)
        else:
            markdown_str += f"- {data}\n"
        
        return markdown_str
    
    markdown = recursive_markdown(json_data)
    return markdown


def save_solution(solution, name, path):
    write_json_file(f'{path}/json/{name}.json', solution)
    markdown_str = json_to_markdown(solution)
    write_text_file(f'{path}/markdown/{name}.md', markdown_str)


def mkdir(path):
    """
    CRITICAL FIX: Create output directory structure using pathlib for proper recursive creation.

    Old implementation had a logic error:
    - Used `if not os.path.dirname(...)` which checks if string is empty, NOT if directory exists
    - Would fail with FileNotFoundError when parent directories don't exist

    New implementation:
    - Uses pathlib.Path.mkdir(parents=True, exist_ok=True) for robust creation
    - Automatically creates all parent directories as needed
    - Safely handles existing directories
    """
    from pathlib import Path

    p = Path(path)
    # Create base directory with all parents
    p.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    subdirs = ["json", "markdown", "latex", "code", "charts", "usage", "logs", "logs/debug"]
    for subdir in subdirs:
        (p / subdir).mkdir(parents=True, exist_ok=True)



def load_config(args, config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['model_name'] = args.model_name
    config['method_name'] = args.method_name
    return config


def get_info(args):
    """Get experiment info and initialize logging system."""
    # CRITICAL FIX: Correctly calculate project root to prevent path errors
    # __file__ is at: MMAgent/utils/utils.py
    # We need to go up 3 levels to reach: LLM-MM-Agent/
    # Structure: LLM-MM-Agent/MMAgent/utils/utils.py
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    mmagent_dir = os.path.dirname(utils_dir)  # MMAgent/
    project_root = os.path.dirname(mmagent_dir)  # LLM-MM-Agent/

    problem_path = os.path.join(project_root, 'MMBench/problem/{}.json'.format(args.task))
    config = load_config(args)

    # Use absolute paths for dataset and output directories
    dataset_dir = os.path.join(project_root, 'MMBench/dataset/', args.task)
    output_dir = os.path.join(project_root, 'MMAgent/output/{}'.format(config["method_name"]), args.task + '_{}'.format(datetime.now().strftime('%Y%m%d-%H%M%S')))

    if not os.path.exists(output_dir):
        mkdir(output_dir)

    # P0 FIX #2: Create all required subdirectories upfront to prevent FileNotFoundError
    # This prevents "logs/debug/events.jsonl not found" and similar directory errors
    # FIX 2026-01-16: Added 'markdown' and 'latex' to fix missing Stage 4 outputs
    required_subdirs = [
        'logs',           # For main log files
        'logs/debug',     # For JSONL events
        'json',           # For solution JSON
        'markdown',       # For Markdown reports (FIXED: was missing, caused no markdown output)
        'latex',          # For LaTeX paper and PDF compilation (FIXED: was missing, caused no latex/pdf output)
        'charts',         # For generated chart images
        'code',           # For generated Python scripts
        'evaluation',     # For quality evaluation reports
        'usage'           # For runtime and API usage stats
    ]

    for subdir in required_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path):
            mkdir(subdir_path)

    # Initialize logging system
    experiment_id = f"{args.task}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    from .logging_config import MMExperimentLogger
    logger_manager = MMExperimentLogger(output_dir, experiment_id, console_level='INFO')

    # Log experiment start
    main_logger = logger_manager.get_logger('main')
    # CRITICAL FIX: Replace Unicode box-drawing characters with ASCII for Windows GBK compatibility
    main_logger.info(f"{'='*80}")
    main_logger.info(f"  MM-Agent Experiment: {args.task}")
    main_logger.info(f"  Model: {config['model_name']}  |  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main_logger.info(f"{'='*80}")
    main_logger.info(f"Processing {problem_path}, config: {config}")

    return problem_path, config, dataset_dir, output_dir, logger_manager