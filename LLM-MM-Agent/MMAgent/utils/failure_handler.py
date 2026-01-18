"""
Failure Handler - Step 4: Minimal JSON Fallback for Failures

PROBLEM: When pipeline crashes, downstream stages fail with FileNotFoundError
because solution.json doesn't exist. This causes cascading crashes.

SOLUTION: Always write minimal solution.json, even if pipeline fails.
This prevents "file not found" errors and allows partial recovery.

Author: Engineering Fix
Date: 2026-01-15
"""

import json
import os
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FailureHandler:
    """
    Ensures minimal JSON output is always written, even on failure.

    Prevents cascading crashes by guaranteeing solution.json exists.
    """

    def __init__(self, output_dir: str, problem_id: str):
        """
        Initialize failure handler.

        Args:
            output_dir: Output directory path
            problem_id: Problem identifier (e.g., '2025_C')
        """
        self.output_dir = Path(output_dir)
        self.problem_id = problem_id
        # [FIX] Save JSON to Workspace/json instead of root json/
        self.json_dir = self.output_dir / 'Workspace' / 'json'
        self.json_dir.mkdir(parents=True, exist_ok=True)

        # Path to solution JSON
        self.solution_path = self.json_dir / f'{problem_id}.json'

        # Track what we've completed
        self.completed_stages = []

    def record_stage_completion(self, stage_name: str):
        """Record that a stage completed successfully."""
        self.completed_stages.append(stage_name)
        logger.info(f"[FailureHandler] Recorded completion: {stage_name}")

    def write_minimal_solution(
        self,
        stage: str,
        error: Optional[str] = None,
        partial_data: Optional[Dict[str, Any]] = None
    ):
        """
        Write minimal solution.json, even if pipeline failed.

        Args:
            stage: Current stage when failure occurred
            error: Error message (if any)
            partial_data: Any partial data to include
        """
        try:
            # Build minimal solution structure
            solution = {
                'problem_id': self.problem_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'FAILED' if error else 'PARTIAL',
                'failure_stage': stage,
                'completed_stages': self.completed_stages,
                'error': error,
                'error_traceback': traceback.format_exc() if error else None
            }

            # Add partial data if available
            if partial_data:
                solution['partial_data'] = partial_data

            # Write to file
            with open(self.solution_path, 'w', encoding='utf-8') as f:
                json.dump(solution, f, indent=2, ensure_ascii=False)

            logger.info(f"[FailureHandler] Wrote minimal solution: {self.solution_path}")
            logger.info(f"[FailureHandler] Completed stages: {self.completed_stages}")
            logger.info(f"[FailureHandler] Failed at: {stage}")

        except Exception as e:
            logger.error(f"[FailureHandler] CRITICAL: Failed to write minimal solution: {e}")
            # Last resort: write error to file
            try:
                error_file = self.solution_path.with_suffix('.error.txt')
                with open(error_file, 'w') as f:
                    f.write(f"Failure at stage: {stage}\n")
                    f.write(f"Error: {error}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"\n{traceback.format_exc()}")
                logger.info(f"[FailureHandler] Wrote error file: {error_file}")
            except:
                logger.critical("[FailureHandler] Could not write any error file!")

    def write_placeholder_chart(self, chart_index: int, error: str):
        """
        Write placeholder chart info when chart generation fails.

        Args:
            chart_index: Chart number (1-indexed)
            error: Error message
        """
        try:
            charts_dir = self.output_dir / 'charts'
            charts_dir.mkdir(parents=True, exist_ok=True)

            # Create placeholder text file
            placeholder_path = charts_dir / f'task1_chart{chart_index}_FAILED.txt'
            with open(placeholder_path, 'w') as f:
                f.write(f"Chart {chart_index} Generation Failed\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error: {error}\n")
                f.write(f"\nThis is a placeholder file.\n")
                f.write(f"The actual chart image could not be generated.\n")

            logger.info(f"[FailureHandler] Wrote placeholder chart: {placeholder_path}")

        except Exception as e:
            logger.error(f"[FailureHandler] Failed to write placeholder chart: {e}")

    def enhance_existing_solution(self, updates: Dict[str, Any]):
        """
        Enhance existing solution.json with additional data.

        Args:
            updates: Dictionary to merge into solution
        """
        try:
            if self.solution_path.exists():
                # Read existing solution
                with open(self.solution_path, 'r', encoding='utf-8') as f:
                    solution = json.load(f)

                # Merge updates
                solution.update(updates)

                # Write back
                with open(self.solution_path, 'w', encoding='utf-8') as f:
                    json.dump(solution, f, indent=2, ensure_ascii=False)

                logger.info(f"[FailureHandler] Enhanced solution with: {list(updates.keys())}")
            else:
                logger.warning(f"[FailureHandler] Solution file doesn't exist, cannot enhance")

        except Exception as e:
            logger.error(f"[FailureHandler] Failed to enhance solution: {e}")


def safe_pipeline_wrapper(
    output_dir: str,
    problem_id: str,
    pipeline_func,
    *args,
    **kwargs
):
    """
    Wrapper that ensures minimal output even if pipeline fails.

    Args:
        output_dir: Output directory
        problem_id: Problem ID
        pipeline_func: Function to run
        *args, **kwargs: Arguments to pass to pipeline_func

    Returns:
        Result from pipeline_func, or minimal dict on failure
    """
    handler = FailureHandler(output_dir, problem_id)

    try:
        # Run pipeline
        result = pipeline_func(*args, **kwargs)

        # Success - record completion
        handler.record_stage_completion('pipeline_complete')
        return result

    except Exception as e:
        # Failure - write minimal solution
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[safe_pipeline_wrapper] Pipeline failed: {error_msg}")

        handler.write_minimal_solution(
            stage='pipeline_execution',
            error=error_msg,
            partial_data={'exception_type': type(e).__name__}
        )

        # Return minimal dict to prevent cascading failures
        return {
            'status': 'FAILED',
            'error': error_msg,
            'problem_id': problem_id,
            'output_dir': str(output_dir)
        }


# Context manager for automatic failure handling
class FailureContext:
    """Context manager for automatic failure handling."""

    def __init__(self, output_dir: str, problem_id: str, stage: str):
        self.handler = FailureHandler(output_dir, problem_id)
        self.stage = stage
        self.partial_data = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred - write minimal solution
            error_msg = f"{exc_type.__name__}: {str(exc_val)}"
            self.handler.write_minimal_solution(
                stage=self.stage,
                error=error_msg,
                partial_data=self.partial_data
            )
        return False  # Don't suppress exception
