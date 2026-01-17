"""
Data Models for MM-Agent

CRITICAL FIX: Define unified data classes to ensure field alignment across all stages.
This prevents "unexpected '{' in field name" errors and eliminates the need for
patchwork field renaming in utils.py.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ChartOutput:
    """Standardized chart output structure"""
    description: str
    code: str
    image_path: Optional[str] = None  # None if failed
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with consistent field names"""
        return {
            'description': self.description,
            'code': self.code,
            'image_path': self.image_path,
            'success': self.success,
            'error': self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChartOutput':
        """Create ChartOutput from dictionary"""
        return cls(
            description=data.get('description', ''),
            code=data.get('code', ''),
            image_path=data.get('image_path'),
            success=data.get('success', False),
            error=data.get('error')
        )


@dataclass
class TaskResult:
    """Standardized task execution result"""
    task_id: int
    task_description: str
    task_analysis: str
    modeling_formulas: str  # CRITICAL: Use 'modeling_formulas' not 'preliminary_formulas'
    modeling_process: str   # Mathematical modeling process
    task_code: Optional[str] = None
    is_pass: bool = False
    execution_result: Optional[str] = None
    solution_interpretation: Optional[str] = None  # Result interpretation
    subtask_outcome_analysis: Optional[str] = None  # Answer
    charts: List[ChartOutput] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with CONSISTENT field names"""
        result = {
            'task_description': self.task_description,
            'task_analysis': self.task_analysis,
            'modeling_formulas': self.modeling_formulas,  # CRITICAL: Consistent naming
            'mathematical_modeling_process': self.modeling_process,
            'charts': [chart.to_dict() for chart in self.charts]
        }

        # Add optional fields only if they exist
        if self.task_code is not None:
            result['task_code'] = self.task_code
            result['is_pass'] = self.is_pass
            result['execution_result'] = self.execution_result
            result['solution_interpretation'] = self.solution_interpretation
            result['subtask_outcome_analysis'] = self.subtask_outcome_analysis

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create TaskResult from dictionary"""
        # Convert charts
        charts = []
        if 'charts' in data:
            for chart_dict in data['charts']:
                if isinstance(chart_dict, dict):
                    charts.append(ChartOutput.from_dict(chart_dict))

        return cls(
            task_id=data.get('task_id', 0),
            task_description=data.get('task_description', ''),
            task_analysis=data.get('task_analysis', ''),
            modeling_formulas=data.get('modeling_formulas', data.get('preliminary_formulas', '')),
            modeling_process=data.get('mathematical_modeling_process', ''),
            task_code=data.get('task_code'),
            is_pass=data.get('is_pass', False),
            execution_result=data.get('execution_result'),
            solution_interpretation=data.get('solution_interpretation', data.get('result')),
            subtask_outcome_analysis=data.get('subtask_outcome_analysis', data.get('answer')),
            charts=charts
        )

    def validate(self) -> tuple:
        """
        Validate task result data.

        Returns:
            tuple: (is_valid: bool, errors: list)
        """
        errors = validate_task_result(self.to_dict())
        return (len(errors) == 0, errors)


def validate_task_result(task_dict: Dict[str, Any]) -> List[str]:
    """
    Validate task result dictionary and return list of errors.

    Returns empty list if valid, or list of error messages if invalid.
    """
    errors = []

    # Check required fields
    required_fields = ['task_description', 'task_analysis', 'modeling_formulas']
    for field in required_fields:
        if field not in task_dict:
            errors.append(f"Missing required field: {field}")

    # Check for legacy field names and suggest corrections
    if 'preliminary_formulas' in task_dict:
        errors.append("Field 'preliminary_formulas' is deprecated. Use 'modeling_formulas' instead.")

    # Validate charts if present
    if 'charts' in task_dict:
        for i, chart in enumerate(task_dict['charts']):
            if not isinstance(chart, dict):
                errors.append(f"Chart {i} is not a dictionary")
                continue

            required_chart_fields = ['description', 'code', 'success']
            for field in required_chart_fields:
                if field not in chart:
                    errors.append(f"Chart {i} missing required field: {field}")

    return errors


def normalize_task_dict(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize task dictionary to use consistent field names.

    This handles legacy field names and ensures compatibility across stages.
    """
    normalized = task_dict.copy()

    # CRITICAL: Fix field name inconsistencies
    if 'preliminary_formulas' in normalized and 'modeling_formulas' not in normalized:
        normalized['modeling_formulas'] = normalized.pop('preliminary_formulas')

    # Ensure all optional fields have default values
    normalized.setdefault('task_code', None)
    normalized.setdefault('is_pass', False)
    normalized.setdefault('execution_result', None)
    normalized.setdefault('solution_interpretation', None)
    normalized.setdefault('subtask_outcome_analysis', None)
    normalized.setdefault('charts', [])

    return normalized


# --------------------------------
# Additional Classes for Full Integration
# --------------------------------

@dataclass
class SolutionOutput:
    """Standardized solution output structure (entire problem solution)"""
    problem_background: str
    problem_requirement: str
    problem_analysis: str
    tasks: List[TaskResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with consistent field names"""
        return {
            'problem_background': self.problem_background,
            'problem_requirement': self.problem_requirement,
            'problem_analysis': self.problem_analysis,
            'tasks': [task.to_dict() for task in self.tasks]
        }


def validate_solution_output(solution_dict: Dict[str, Any]) -> tuple:
    """
    Validate solution output dictionary.

    Returns:
        tuple: (is_valid: bool, errors: list)
    """
    errors = []

    # Check required top-level fields
    required_fields = ['problem_background', 'problem_requirement', 'tasks']
    for field in required_fields:
        if field not in solution_dict:
            errors.append(f"Missing required field: {field}")

    # Validate each task
    if 'tasks' in solution_dict:
        for i, task in enumerate(solution_dict['tasks']):
            task_errors = validate_task_result(task)
            for error in task_errors:
                errors.append(f"Task {i+1}: {error}")

    return (len(errors) == 0, errors)


def convert_legacy_task_dict(task_dict: Dict[str, Any]) -> TaskResult:
    """
    Convert legacy task dictionary to TaskResult dataclass.

    This handles various legacy formats and ensures consistency.
    """
    # Normalize the dict first
    normalized = normalize_task_dict(task_dict)

    # Extract required fields
    task_description = normalized.get('task_description', '')
    task_analysis = normalized.get('task_analysis', '')
    modeling_formulas = normalized.get('modeling_formulas', '')

    # Extract optional fields
    task_id = normalized.get('task_id', 0)
    modeling_process = normalized.get('mathematical_modeling_process', '')
    task_code = normalized.get('task_code')
    is_pass = normalized.get('is_pass', False)
    execution_result = normalized.get('execution_result')
    solution_interpretation = normalized.get('solution_interpretation')
    subtask_outcome_analysis = normalized.get('subtask_outcome_analysis')

    # Convert charts
    charts = []
    if 'charts' in normalized:
        for chart_dict in normalized['charts']:
            if isinstance(chart_dict, dict):
                chart = ChartOutput(
                    description=chart_dict.get('description', ''),
                    code=chart_dict.get('code', ''),
                    image_path=chart_dict.get('image_path'),
                    success=chart_dict.get('success', False),
                    error=chart_dict.get('error')
                )
                charts.append(chart)

    # Create TaskResult
    return TaskResult(
        task_id=task_id,
        task_description=task_description,
        task_analysis=task_analysis,
        modeling_formulas=modeling_formulas,
        modeling_process=modeling_process,
        task_code=task_code,
        is_pass=is_pass,
        execution_result=execution_result,
        solution_interpretation=solution_interpretation,
        subtask_outcome_analysis=subtask_outcome_analysis,
        charts=charts
    )


# --------------------------------
# Backward Compatibility Aliases
# --------------------------------

# Alias for backward compatibility
TaskOutput = TaskResult
