import json
from typing import Dict
import os
import sys
import io
import yaml
from datetime import datetime
import re
from pathlib import Path

# Import json_repair for robust JSON parsing from LLM outputs
try:
    import json_repair
except ImportError:
    json_repair = None


class PlatformUtils:
    """
    Encapsulates platform-specific logic (Windows vs Linux/Mac).

    Replaces scattered 'if sys.platform == "win32"' checks throughout the codebase.
    Centralizes platform differences for easier maintenance and testing.

    Usage:
        PlatformUtils.set_console_encoding()
        if PlatformUtils.is_windows():
            # Windows-specific code
    """

    @staticmethod
    def set_console_encoding():
        """
        Force UTF-8 encoding for console output.

        Critical for Windows to prevent UnicodeEncodeError with special characters
        (Chinese characters, emojis, etc.). Also configures stderr for error logs.

        This method handles both modern Python (3.7+) with reconfigure() and
        older versions requiring TextIOWrapper replacement.
        """
        if sys.platform == 'win32':
            try:
                # Python 3.7+ preferred method
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except AttributeError:
                # Fallback for older Python versions
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    @staticmethod
    def is_windows():
        """Check if running on Windows platform."""
        return sys.platform == 'win32'

    @staticmethod
    def is_linux():
        """Check if running on Linux platform."""
        return sys.platform.startswith('linux')

    @staticmethod
    def is_macos():
        """Check if running on macOS platform."""
        return sys.platform == 'darwin'

    @staticmethod
    def get_platform_name():
        """Get human-readable platform name."""
        if PlatformUtils.is_windows():
            return 'Windows'
        elif PlatformUtils.is_macos():
            return 'macOS'
        elif PlatformUtils.is_linux():
            return 'Linux'
        else:
            return 'Unknown'


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
    """
    Write content to a text file, creating parent directories if needed.

    [CRITICAL FIX 2026-01-18] Auto-create parent directories to prevent
    FileNotFoundError when saving results to Workspace/markdown/, etc.
    """
    import os

    # [CRITICAL FIX] Automatically create parent directories if they don't exist
    # This prevents: FileNotFoundError: [Errno 2] No such file or directory: '.../Workspace/markdown/2025_C.md'
    parent_dir = os.path.dirname(file_path)
    if parent_dir:  # Only create if there's a parent directory
        os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def write_json_file(file_path: str, data:dict) -> Dict:
    """
    Write data to a JSON file, creating parent directories if needed.

    [CRITICAL FIX 2026-01-18] Auto-create parent directories to prevent
    FileNotFoundError when saving results to Workspace/json/, etc.
    """
    import os
    import json

    # [CRITICAL FIX] Automatically create parent directories if they don't exist
    # This prevents: FileNotFoundError: [Errno 2] No such file or directory: '.../Workspace/json/2025_C.json'
    parent_dir = os.path.dirname(file_path)
    if parent_dir:  # Only create if there's a parent directory
        os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


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
    """
    Get experiment info, resolve paths dynamically via config, and initialize logging.

    [NEW] Uses pathlib for modern cross-platform path handling and config-driven path resolution.
    This eliminates hardcoded paths like 'MMBench/problem/' and allows flexible data location.
    """
    # 1. Resolve Project Root safely using pathlib
    # current: LLM-MM-Agent/MMAgent/utils/utils.py
    # parents: [0]=utils, [1]=MMAgent, [2]=LLM-MM-Agent (Root)
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    # 2. Load Config FIRST (to get path settings)
    # Default to finding config.yaml in project root
    config_path = project_root / 'config.yaml'
    if not config_path.exists():
        # Fallback to local execution dir
        config_path = Path('config.yaml')

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = load_config(args, config_path=str(config_path))

    # 3. Resolve Paths based on Config
    paths_cfg = config.get('paths', {})

    # Default to 'MMBench' inside project_root if not configured
    data_root_name = paths_cfg.get('root_data', 'MMBench')
    data_root = project_root / data_root_name

    # Resolve Problem Path
    # e.g., LLM-MM-Agent/MMBench/problem/task_id.json
    problem_dir_name = paths_cfg.get('problem_dir', 'problem')
    problem_path = data_root / problem_dir_name / f"{args.task}.json"

    # Resolve Dataset Directory
    # e.g., LLM-MM-Agent/MMBench/dataset/task_id/
    dataset_dir_name = paths_cfg.get('dataset_dir', 'dataset')
    dataset_dir = data_root / dataset_dir_name / args.task

    # Resolve Output Directory
    output_root_name = paths_cfg.get('output_root', 'MMAgent/output')
    # Generate timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir_name = f"{args.task}_{timestamp}"
    output_dir = project_root / output_root_name / config["method_name"] / run_dir_name

    # 4. Create Directories with Three-Tier Architecture
    # NEW STRUCTURE: Report (presentation), Workspace (production), Memory (persistence)
    # Use pathlib mkdir with parents=True for robust recursive creation
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Define three-tier directory structure
    tiered_dirs = {
        # Tier 1: Report Layer - Core presentation and research artifacts
        "Report": None,  # For 01_Research_Journal.md, final PDF

        # Tier 2: Workspace Layer - Production output from agent operations
        "Workspace": {
            "code": None,      # Generated Python scripts (main1.py, main2.py, etc.)
            "charts": None,    # Generated visualization images
            "json": None,      # Structured JSON output
            "markdown": None,  # Human-readable markdown output
            "latex": None,     # LaTeX academic paper format
        },

        # Tier 3: Memory Layer - State persistence and raw logs
        "Memory": {
            "logs": None,           # Execution logs (debug.log, trace.jsonl, etc.)
            "checkpoints": None,    # Pipeline state checkpoints
            "usage": None,          # API usage tracking
            "evaluation": None,     # Evaluation results
        },
    }

    # Create all directories recursively
    def create_tiered_dirs(base_path, structure):
        """Recursively create directory structure."""
        for name, children in structure.items():
            dir_path = base_path / name
            dir_path.mkdir(parents=True, exist_ok=True)
            if children is not None:
                create_tiered_dirs(dir_path, children)

    create_tiered_dirs(output_dir, tiered_dirs)

    # 5. Initialize Logging with Memory directory
    # [NEW] Log files now go to Memory/logs instead of root/logs
    experiment_id = f"{args.task}_{timestamp}"
    memory_dir = output_dir / "Memory"
    from .logging_config import MMExperimentLogger
    # Convert to string for LoggerManager compatibility
    logger_manager = MMExperimentLogger(str(memory_dir), experiment_id, console_level='INFO')

    main_logger = logger_manager.get_logger('main')
    main_logger.info(f"{'='*80}")
    main_logger.info(f"  MM-Agent Experiment: {args.task}")
    main_logger.info(f"  Root: {project_root}")
    main_logger.info(f"  Data: {data_root}")
    main_logger.info(f"{'='*80}")

    # Return strings for backward compatibility with rest of the code
    return str(problem_path), config, str(dataset_dir), str(output_dir), logger_manager


def robust_json_load(json_str: str):
    """
    CRITICAL FIX: Robustly parse JSON string from LLM output using json_repair.

    This function provides a unified, robust JSON parsing mechanism for all LLM outputs
    throughout the MM-Agent system. It handles common LLM JSON formatting issues:
    - Markdown code blocks (```json ... ``` or ``` ... ```)
    - Trailing commas (common LLM error)
    - Missing quotes on keys
    - Comments (// ... or /* ... */)
    - Unescaped characters
    - Incomplete brackets/quotes

    Args:
        json_str: Raw JSON string from LLM output

    Returns:
        Parsed Python dictionary/list, or None if input is empty

    Raises:
        ValueError: If JSON cannot be parsed after all fallback attempts

    Example:
        >>> broken_json = '''```json
        ... { "name": "test", "items": [1, 2, 3,] }
        ... ```'''
        >>> result = robust_json_load(broken_json)
        >>> result
        {'name': 'test', 'items': [1, 2, 3]}

    Migration Path:
        - Replace all json.loads() on LLM output with robust_json_load()
        - Remove Tier 1/2/3 fix methods from coordinator.py
        - Remove manual regex replacement logic from task_solving.py
    """
    if not json_str:
        return None

    # Step 1: Strip Markdown code blocks (```json ... ```)
    clean_str = json_str.strip()
    if "```" in clean_str:
        # Extract content inside the first code block if present
        pattern = r"```(?:json)?\s*(.*?)```"
        match = re.search(pattern, clean_str, re.DOTALL | re.IGNORECASE)
        if match:
            clean_str = match.group(1).strip()

    # Step 2: Try standard json.loads first (fastest)
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        # Standard parsing failed, continue to more robust methods
        pass

    # Step 3: Fallback to json_repair (robust third-party library)
    if json_repair:
        try:
            # json_repair can fix most common LLM JSON errors
            decoded = json_repair.loads(clean_str)
            return decoded
        except Exception as e:
            print(f"[DEBUG] json_repair failed: {e}")
            # Continue to manual cleanup as last resort
    else:
        print("[WARNING] json_repair module not found. Please install: pip install json_repair")

    # Step 4: Fallback to manual cleanup (Legacy Tier 1-3 fixes)
    # This is a last resort if json_repair is missing or fails
    try:
        # Remove JSON comments
        clean_str = re.sub(r'//.*?\n', '\n', clean_str)
        clean_str = re.sub(r'/\*[\s\S]*?\*/', '', clean_str)

        # Fix trailing commas (very common LLM error)
        clean_str = re.sub(r',\s*}', '}', clean_str)
        clean_str = re.sub(r',\s*]', ']', clean_str)

        # Fix unquoted keys
        clean_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', clean_str)

        return json.loads(clean_str)
    except Exception as final_e:
        raise ValueError(f"Failed to parse JSON after all attempts. Error: {final_e}\nJSON preview: {clean_str[:200]}...")