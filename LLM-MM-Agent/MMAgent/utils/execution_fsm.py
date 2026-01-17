"""
Finite State Machine for Task Execution

Replaces "retry until success or max" loops with deterministic state transitions.

Core principle:
- Category A (Deterministic): Auto-fix or STOP (not retry)
- Category B (Cognitive): CoT with refinement rounds
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
from enum import Enum, auto


class TaskCategory(Enum):
    """Task categories determine execution strategy."""
    DETERMINISTIC = auto()  # No CoT, fail fast
    COGNITIVE = auto()  # CoT appropriate


class TaskState(Enum):
    """Finite states for task execution."""
    START = auto()
    VALIDATING = auto()
    AUTO_FIXING = auto()
    EXECUTING = auto()
    VERIFYING = auto()
    SUCCESS = auto()
    FAILED = auto()


class FSMError(Enum):
    """Error types with deterministic responses."""
    SYNTAX_ERROR = auto()
    PATH_ERROR = auto()
    IMPORT_ERROR = auto()
    TIMEOUT_ERROR = auto()
    SCHEMA_ERROR = auto()
    RUNTIME_ERROR = auto()
    FILE_NOT_FOUND = auto()
    TEMPLATE_ERROR = auto()


class ExecutionFSM:
    """
    Finite State Machine for task execution.

    Enforces contracts and replaces ad-hoc retry loops.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.state = TaskState.START
        self.error_history = []

    def execute_task(self,
                    task_type: str,
                    input_data: Dict[str, Any],
                    llm,
                    category: TaskCategory = TaskCategory.DETERMINISTIC) -> Dict[str, Any]:
        """
        Execute a task using FSM transitions (not retry loops).

        Args:
            task_type: Type of task (chart_generation, latex_compilation, etc.)
            input_data: Task input data
            llm: LLM instance
            category: Task category (deterministic vs cognitive)

        Returns:
            Task result with success status
        """
        self.state = TaskState.START
        self.error_history = []

        if category == TaskCategory.DETERMINISTIC:
            return self._execute_deterministic(task_type, input_data, llm)
        else:
            return self._execute_cognitive(task_type, input_data, llm)

    def _execute_deterministic(self,
                               task_type: str,
                               input_data: Dict[str, Any],
                               llm) -> Dict[str, Any]:
        """
        Execute deterministic task with FSM (no CoT, no retry).

        State transitions:
        START → VALIDATING → AUTO_FIXING → EXECUTING → VERIFYING → SUCCESS/FAILED
        """

        while self.state not in [TaskState.SUCCESS, TaskState.FAILED]:
            if self.state == TaskState.START:
                self.state = TaskState.VALIDATING

            elif self.state == TaskState.VALIDATING:
                validation_result = self._validate_input(task_type, input_data)
                if validation_result['valid']:
                    self.state = TaskState.AUTO_FIXING
                else:
                    self.error_history.append(validation_result['error'])
                    # Deterministic: Validation fails → STOP (don't retry)
                    self.state = TaskState.FAILED

            elif self.state == TaskState.AUTO_FIXING:
                # Apply automatic fixes (paths, imports, etc.)
                fixed_data = self._apply_autofix(task_type, input_data)

                if fixed_data['fixed']:
                    input_data = fixed_data['data']
                    self.state = TaskState.EXECUTING
                else:
                    # Auto-fix failed → STOP
                    self.error_history.append("Auto-fix failed")
                    self.state = TaskState.FAILED

            elif self.state == TaskState.EXECUTING:
                execution_result = self._execute_task(task_type, input_data, llm)

                if execution_result['success']:
                    self.state = TaskState.VERIFYING
                else:
                    error_type = execution_result.get('error_type')

                    # Deterministic: Check if error is auto-fixable
                    if error_type in [FSMError.PATH_ERROR, FSMError.IMPORT_ERROR]:
                        # Try auto-fix once
                        self.state = TaskState.AUTO_FIXING
                        input_data = execution_result.get('repaired_data', input_data)
                    else:
                        # Other errors → STOP (no retry)
                        self.error_history.append(execution_result['error'])
                        self.state = TaskState.FAILED

            elif self.state == TaskState.VERIFYING:
                verification_result = self._verify_output(task_type, execution_result)

                if verification_result['verified']:
                    self.state = TaskState.SUCCESS
                else:
                    # Verification failed → STOP (output doesn't meet contract)
                    self.error_history.append(verification_result['error'])
                    self.state = TaskState.FAILED

        # Build final response
        if self.state == TaskState.SUCCESS:
            return {
                'success': True,
                'data': execution_result,
                'error': None,
                'state_history': self.error_history
            }
        else:
            return {
                'success': False,
                'data': None,
                'error': self.error_history[-1] if self.error_history else 'Unknown error',
                'state_history': self.error_history
            }

    def _execute_cognitive(self,
                           task_type: str,
                           input_data: Dict[str, Any],
                           llm,
                           refinement_rounds: int = 3) -> Dict[str, Any]:
        """
        Execute cognitive task with CoT and refinement.

        This is appropriate for problem analysis, modeling, etc.
        Uses actor-critic pattern with limited refinement rounds.
        """

        # Initial generation
        result = self._generate_with_cot(task_type, input_data, llm)

        # Refinement rounds (actor-critic)
        for round_num in range(refinement_rounds):
            critique = self._critique_result(task_type, result, llm)

            if self._is_satisfactory(critique):
                break

            result = self._refine_result(task_type, result, critique, llm)

        return {
            'success': True,
            'data': result,
            'rounds': round_num + 1
        }

    def _validate_input(self, task_type: str, input_data: Dict) -> Dict[str, Any]:
        """Validate input data against contract."""
        from utils.schema_normalization import validate_task_result

        if task_type == "chart_generation":
            required_fields = ['data_files', 'chart_description', 'output_dir']
            missing = [f for f in required_fields if f not in input_data]

            if missing:
                return {
                    'valid': False,
                    'error': f"Missing required fields: {missing}"
                }

            # Check output directory exists
            if not os.path.exists(input_data['output_dir']):
                return {
                    'valid': False,
                    'error': f"Output directory does not exist: {input_data['output_dir']}"
                }

            return {'valid': True}

        elif task_type == "latex_compilation":
            # JSON structure validation
            try:
                errors = validate_task_result(input_data)
                if errors:
                    return {
                        'valid': False,
                        'error': f"JSON validation errors: {errors[:3]}"  # Show first 3
                    }
            except Exception as e:
                return {
                    'valid': False,
                    'error': f"Validation exception: {e}"
                }

            return {'valid': True}

        else:
            # Default: accept all
            return {'valid': True}

    def _apply_autofix(self, task_type: str, input_data: Dict) -> Dict[str, Any]:
        """Apply automatic fixes (deterministic transformations)."""
        from utils.path_autofix import apply_path_autofix_with_validation
        from utils.schema_normalization import normalize_chart, sanitize_json_for_latex

        if task_type == "chart_generation":
            # Fix will be applied during code generation
            return {'fixed': True, 'data': input_data}

        elif task_type == "latex_compilation":
            # Sanitize JSON data
            sanitized = sanitize_json_for_latex(input_data)
            return {'fixed': True, 'data': sanitized}

        else:
            return {'fixed': False, 'data': input_data}

    def _execute_task(self, task_type: str, input_data: Dict, llm) -> Dict[str, Any]:
        """Execute the task (delegates to appropriate executor)."""
        if task_type == "chart_generation":
            # Delegate to ChartCreator
            from agent.create_charts import ChartCreator
            cc = ChartCreator(llm)

            # Generate single chart
            charts = cc.create_charts_with_images(
                paper_content=str(input_data),
                chart_num=1,
                output_dir=input_data['output_dir'],
                data_files=input_data.get('data_files', [])
            )

            if charts and charts[0].get('success', False):
                return {'success': True, 'chart': charts[0]}
            else:
                return {
                    'success': False,
                    'error': charts[0].get('error', 'Unknown error') if charts else 'No charts generated',
                    'error_type': FSMError.RUNTIME_ERROR
                }

        elif task_type == "latex_compilation":
            # Delegate to solution_reporting
            from utils.solution_reporting import generate_paper

            try:
                generate_paper(llm, input_data['output_dir'], input_data['name'])

                # Check if PDF was created
                pdf_path = os.path.join(input_data['output_dir'], 'latex', 'solution.pdf')
                if os.path.exists(pdf_path):
                    return {'success': True, 'pdf_path': pdf_path}
                else:
                    return {
                        'success': False,
                        'error': 'PDF not created',
                        'error_type': FSMError.RUNTIME_ERROR
                    }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'error_type': FSMError.RUNTIME_ERROR
                }

        else:
            return {
                'success': False,
                'error': f'Unknown task type: {task_type}',
                'error_type': FSMError.RUNTIME_ERROR
            }

    def _verify_output(self, task_type: str, result: Dict) -> Dict[str, Any]:
        """Verify output meets contract (deterministic check)."""

        if task_type == "chart_generation":
            chart = result.get('chart', {})
            image_path = chart.get('image_path', '')

            # Contract: File MUST exist
            if not os.path.exists(image_path):
                return {
                    'verified': False,
                    'error': f'Image file not found: {image_path}'
                }

            # Contract: File MUST be > 1000 bytes
            if os.path.getsize(image_path) < 1000:
                return {
                    'verified': False,
                    'error': f'Image file too small: {image_path}'
                }

            # Contract: success MUST be True
            if not chart.get('success', False):
                return {
                    'verified': False,
                    'error': 'Chart success flag is False'
                }

            return {'verified': True}

        elif task_type == "latex_compilation":
            pdf_path = result.get('pdf_path', '')

            # Contract: PDF file MUST exist
            if not os.path.exists(pdf_path):
                return {
                    'verified': False,
                    'error': f'PDF file not found: {pdf_path}'
                }

            # Contract: PDF MUST be > 5000 bytes
            if os.path.getsize(pdf_path) < 5000:
                return {
                    'verified': False,
                    'error': f'PDF file too small: {pdf_path}'
                }

            return {'verified': True}

        else:
            # Default: accept result
            return {'verified': True}

    def _generate_with_cot(self, task_type: str, input_data: Dict, llm) -> Dict[str, Any]:
        """Generate result with chain-of-thought (cognitive tasks only)."""
        # This delegates to existing cognitive components
        # (problem_analysis, mathematical_modeling)
        # Implementation depends on task_type

        if task_type == "problem_analysis":
            from utils.problem_analysis import problem_analysis
            return problem_analysis(llm, input_data['problem'], input_data.get('config', {}))

        elif task_type == "mathematical_modeling":
            from utils.mathematical_modeling import mathematical_modeling
            return mathematical_modeling(llm, input_data['coordinator'], input_data.get('config', {}))

        else:
            raise ValueError(f"Unknown cognitive task type: {task_type}")

    def _critique_result(self, task_type: str, result: Dict, llm) -> str:
        """Generate critique of result (actor-critic pattern)."""
        # Implementation depends on task_type
        # This is where LLM evaluates its own output

        critique_prompt = f"""
        Critique the following {task_type} result:

        {result}

        Identify:
        1. What's missing?
        2. What's unclear?
        3. What could be improved?

        Be specific and actionable.
        """

        return llm.generate(critique_prompt)

    def _is_satisfactory(self, critique: str) -> bool:
        """Check if critique indicates satisfactory result."""
        # Simple heuristic: if critique is short, result is good
        return len(critique) < 200

    def _refine_result(self, task_type: str, result: Dict, critique: str, llm) -> Dict:
        """Refine result based on critique."""
        refine_prompt = f"""
        Refine the following {task_type} result based on this critique:

        Original Result:
        {result}

        Critique:
        {critique}

        Provide improved result addressing the critique.
        """

        return llm.generate(refine_prompt)


# Convenience functions for common tasks

def execute_deterministic_task(task_type: str,
                                input_data: Dict[str, Any],
                                llm,
                                logger=None) -> Dict[str, Any]:
    """
    Execute deterministic task (no CoT, fail fast).

    Use this for:
    - Chart generation
    - Code execution
    - PDF compilation
    - File operations
    """
    fsm = ExecutionFSM(logger)
    return fsm.execute_task(task_type, input_data, llm, TaskCategory.DETERMINISTIC)


def execute_cognitive_task(task_type: str,
                            input_data: Dict[str, Any],
                            llm,
                            refinement_rounds: int = 3,
                            logger=None) -> Dict[str, Any]:
    """
    Execute cognitive task (CoT with refinement).

    Use this for:
    - Problem analysis
    - Mathematical modeling
    - Method selection
    """
    fsm = ExecutionFSM(logger)
    return fsm.execute_task(task_type, input_data, llm, TaskCategory.COGNITIVE)
