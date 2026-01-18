import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# P1 FIX #5: Import schema normalization for evaluation stability
from MMAgent.utils.schema_normalization import normalize_solution


def load_json(file_path):
    """
    load json data
    
    Args:
        file_path (str): the file path
        
    Returns:
        dict: json data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File {file_path} not in valid JSON format.")
        return {}



def load_solution_json(file_path):
    """
    Load solution data with schema normalization for crash prevention (P1 FIX #5).

    Args:
        file_path (str): the file path

    Returns:
        dict: solution data in json format, normalized with safe defaults
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # P1 FIX #5: Normalize solution data to prevent AttributeError/KeyError crashes
        # This ensures all required fields exist with safe defaults, even if JSON is malformed
        data = normalize_solution(data)

        # Extract tasks for processing
        tasks = data.get('tasks', [])

        task_dict = {f"task{i+1}": task for i, task in enumerate(tasks)}

        result = {}
        # Include non-task fields from original data if available
        if isinstance(data, dict):
            for key, value in data.items():
                if key != 'tasks':
                    result[key] = value
        result.update(task_dict)

        return result
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File {file_path} not in valid JSON format.")
        return {}