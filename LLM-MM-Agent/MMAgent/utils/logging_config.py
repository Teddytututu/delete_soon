"""
MM-Agent Experiment Logging System

This module provides a centralized logging system for MM-Agent experiments.
It creates separate log files for different aspects of the experiment execution.

Author: MM-Agent Team
Date: 2026-01-09
"""

import logging
import sys
import uuid
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Dict, Any


class MMExperimentLogger:
    """
    MM-Agent experiment centralized logging system.

    This class manages multiple log files for different aspects of experiment execution:
    - main.log: All log levels (DEBUG and above)
    - errors.log: Only ERROR and CRITICAL
    - llm_api.log: LLM API call details
    - code_execution.log: Code execution output
    - chart_generation.log: Chart generation details

    Console output is limited to INFO and above for clean progress display.
    """

    def __init__(self, output_dir: str, experiment_id: str, console_level="INFO"):
        """
        Initialize the logging system.

        Args:
            output_dir: Experiment output directory
            experiment_id: Unique experiment identifier
            console_level: Console log level (INFO, WARNING, ERROR)
        """
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.log_dir = self.output_dir / "logs"
        self.debug_dir = self.log_dir / "debug"

        # Create log directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Store logger instances
        self.loggers = {}

        # Setup all loggers
        self._setup_loggers(console_level)

    def _setup_loggers(self, console_level):
        """Configure all log handlers."""

        # 1. Main logger (all levels)
        main_logger = self._create_logger(
            'main',
            self.log_dir / 'main.log',
            level=logging.DEBUG,
            console_level=console_level
        )
        self.loggers['main'] = main_logger

        # 2. Error logger (ERROR and CRITICAL only)
        error_logger = self._create_logger(
            'errors',
            self.log_dir / 'errors.log',
            level=logging.ERROR,
            console_level=None  # No console output
        )
        self.loggers['errors'] = error_logger

        # CRITICAL FIX: Automatically propagate ERROR/CRITICAL from main_logger to errors.log
        # This ensures that any main_logger.error() or main_logger.critical() call
        # automatically appears in errors.log without requiring manual duplication.
        # Solves the "logging responsibilities confusion" problem.
        error_handler = logging.FileHandler(self.log_dir / 'errors.log', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        main_logger.addHandler(error_handler)

        # 3. LLM API logger
        llm_logger = self._create_logger(
            'llm_api',
            self.log_dir / 'llm_api.log',
            level=logging.DEBUG,
            console_level=None
        )
        self.loggers['llm_api'] = llm_logger

        # 4. Code execution logger
        code_logger = self._create_logger(
            'code_execution',
            self.log_dir / 'code_execution.log',
            level=logging.DEBUG,
            console_level=None
        )
        self.loggers['code_execution'] = code_logger

        # 5. Chart generation logger
        chart_logger = self._create_logger(
            'chart_generation',
            self.log_dir / 'chart_generation.log',
            level=logging.DEBUG,
            console_level=None
        )
        self.loggers['chart_generation'] = chart_logger

    def _create_logger(self, name, log_file, level, console_level=None):
        """
        Create a configured logger.

        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level (e.g., logging.DEBUG)
            console_level: Console output level (None to disable)

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f'mmagent.{name}')
        logger.setLevel(level)

        # CRITICAL FIX: Close old handlers to prevent file descriptor leaks
        # logger.handlers.clear() doesn't close the file handles!
        for h in logger.handlers[:]:
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

        logger.propagate = False

        # File handler (detailed format)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler (simple format, if enabled)
        if console_level:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, console_level))
            console_formatter = logging.Formatter(fmt='%(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def get_logger(self, name='main'):
        """
        Get a specific logger by name.

        Args:
            name: Logger name ('main', 'errors', 'llm_api', etc.)

        Returns:
            Logger instance
        """
        return self.loggers.get(name, self.loggers['main'])

    def _jsonl_path(self, filename: str) -> Path:
        """Get path to JSONL file in debug directory."""
        return self.debug_dir / filename

    def log_event(self, event: Dict[str, Any]):
        """
        Log a structured event to JSONL file for machine-readable tracking.

        P0 FIX: Ensure directory exists before writing, fallback to stderr on failure.
        This prevents "logging system crashes the pipeline" issue.

        Args:
            event: Dictionary containing event data (type, timestamp, etc.)
        """
        event = dict(event)
        event.setdefault("timestamp", datetime.now().isoformat())
        event.setdefault("experiment_id", self.experiment_id)

        path = self._jsonl_path("events.jsonl")

        # P0 FIX: Ensure directory exists (handles race conditions and external deletions)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            # P0 FIX: Fallback to stderr to prevent "exception during exception" cascades
            # DO NOT raise - let the main pipeline continue
            import sys
            print(f"[LOGGING FALLBACK] Failed to write to {path}: {e}", file=sys.stderr)
            print(f"[LOGGING FALLBACK] Event: {json.dumps(event, ensure_ascii=False)[:200]}...", file=sys.stderr)

    def log_stage_start(self, stage_name: str, stage_num: int = None):
        """
        Log the start of a stage.

        Args:
            stage_name: Name of the stage
            stage_num: Optional stage number
        """
        # CRITICAL FIX: Use ASCII-only characters for Windows GBK encoding compatibility
        # Unicode "=" fails on Windows with GBK codec
        separator = "=" * 60
        if stage_num:
            message = f"\n{separator}\n  STAGE {stage_num}: {stage_name}\n{separator}"
        else:
            message = f"\n{separator}\n  {stage_name}\n{separator}"
        self.loggers['main'].info(message)

        # Also log structured event
        self.log_event({
            "type": "stage_start",
            "stage_name": stage_name,
            "stage_num": stage_num
        })

    def log_stage_complete(self, stage_name: str, duration_seconds: int, stage_num: int = None):
        """
        Log the completion of a stage.

        Args:
            stage_name: Name of the stage
            duration_seconds: Duration in seconds
            stage_num: Optional stage number
        """
        # CRITICAL FIX: Use ASCII-only characters for Windows GBK encoding compatibility
        # Unicode checkmark causes failures on Windows with GBK codec
        if stage_num:
            message = f"[OK] STAGE {stage_num} COMPLETE ({duration_seconds}s)"
        else:
            message = f"[OK] {stage_name} COMPLETE ({duration_seconds}s)"
        self.loggers['main'].info(message)

        # Also log structured event
        self.log_event({
            "type": "stage_complete",
            "stage_name": stage_name,
            "stage_num": stage_num,
            "duration_seconds": duration_seconds,
            "status": "ok"
        })

    def log_progress(self, message: str, level='info'):
        """
        Log progress information (console + file).

        Args:
            message: Progress message
            level: Log level ('info', 'warning', 'error')
        """
        logger = self.loggers['main']
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        # CRITICAL FIX: Replace Unicode arrow with ASCII for Windows GBK compatibility
        formatted_msg = f"{timestamp} -> {message}"
        getattr(logger, level)(formatted_msg)

    def log_llm_call(self, model: str, prompt_length: int, usage: dict, stage: str):
        """
        Log LLM API call details.

        Args:
            model: Model name
            prompt_length: Length of the prompt in characters
            usage: Usage dictionary with token counts
            stage: Stage where the call was made
        """
        log_msg = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'stage': stage,
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'prompt_length': prompt_length
        }
        self.loggers['llm_api'].info(json.dumps(log_msg))

        # CRITICAL FIX: Also log structured event for machine-readable tracking
        self.log_event({
            "type": "llm_call",
            "model": model,
            "stage": stage,
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get('completion_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0),
            "prompt_length": prompt_length
        })

    def log_code_execution(self, task_id: str, script_name: str,
                          stdout: str, stderr: str, return_code: int):
        """
        Log code execution details.

        Args:
            task_id: Task identifier
            script_name: Name of the script
            stdout: Standard output
            stderr: Standard error
            return_code: Process return code
        """
        log_msg = (
            f"Task {task_id} | Script: {script_name} | "
            f"Return Code: {return_code}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}\n"
            f"{'='*80}"
        )
        self.loggers['code_execution'].debug(log_msg)

        # CRITICAL FIX: Also log structured event for machine-readable tracking
        self.log_event({
            "type": "code_execution",
            "task_id": task_id,
            "script_name": script_name,
            "return_code": return_code,
            "success": return_code == 0,
            "stdout_length": len(stdout),
            "stderr_length": len(stderr)
        })

        # If error, also log to main log and errors log
        if return_code != 0:
            self.loggers['main'].error(f"Code execution failed: {script_name}")
            self.loggers['errors'].error(log_msg)

    def log_chart_generation(self, task_id: int, num_charts: int,
                           description: str, success: bool, error: str = None):
        """
        Log chart generation details.

        Args:
            task_id: Task identifier
            num_charts: Chart number
            description: Chart description
            success: Whether generation succeeded
            error: Error message if failed
        """
        log_msg = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'num_charts': num_charts,
            'description': description[:100] + '...' if len(description) > 100 else description,
            'success': success,
            'error': error
        }
        level = 'info' if success else 'error'
        getattr(self.loggers['chart_generation'], level)(json.dumps(log_msg))

        # CRITICAL FIX: Also log structured event for machine-readable tracking
        self.log_event({
            "type": "chart_generation",
            "task_id": task_id,
            "num_charts": num_charts,
            "description": description[:100] + '...' if len(description) > 100 else description,
            "success": success,
            "error": error
        })

    def log_exception(self, exception: Exception, context: str, stage: str = None,
                      task: str = None, include_traceback: bool = True):
        """
        Log an exception with full traceback and context.

        Args:
            exception: The exception object
            context: What was being attempted when error occurred
            stage: Current stage (optional)
            task: Current task (optional)
            include_traceback: Whether to include full traceback (default: True)
        """
        import traceback

        # Build error context
        error_parts = []
        error_parts.append("=" * 80)
        error_parts.append(f"EXCEPTION OCCURRED")
        error_parts.append("=" * 80)

        if stage:
            error_parts.append(f"Stage: {stage}")
        if task:
            error_parts.append(f"Task: {task}")

        error_parts.append(f"Context: {context}")
        error_parts.append(f"Exception Type: {type(exception).__name__}")
        error_parts.append(f"Exception Message: {str(exception)}")
        error_parts.append("")

        if include_traceback:
            error_parts.append("FULL TRACEBACK:")
            error_parts.append("-" * 80)
            tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
            error_parts.extend(tb_lines)
            error_parts.append("-" * 80)

        error_parts.append("")
        error_parts.append("=" * 80)
        error_msg = "\n".join(error_parts)

        # Log to errors.log and main.log
        self.loggers['errors'].error(error_msg)
        self.loggers['main'].error(f"\n{error_msg}\n")

        # CRITICAL FIX: Also log structured event for machine-readable tracking
        self.log_event({
            "type": "exception",
            "stage": stage,
            "task": task,
            "context": context,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback_included": include_traceback
        })

        # Return formatted error for other uses
        return error_msg

    def log_latent_summary(self, log_content: str, section_name: str,
                          llm_client=None, max_summary_length: int = 500):
        """
        Generate an LLM-based latent summary of a log section.

        This method uses an LLM to create concise summaries of large log sections,
        making debugging easier by compressing verbose logs into key insights.

        Args:
            log_content: The log content to summarize
            section_name: Name of the log section (e.g., "llm_api", "code_execution")
            llm_client: Optional LLM client instance (if None, creates temporary one)
            max_summary_length: Maximum length of summary in characters

        Returns:
            Generated summary string, or None if LLM client not available
        """
        if not llm_client:
            # No LLM client available, return None
            self.loggers['main'].warning(
                f"LLM client not provided for latent summary of '{section_name}'"
            )
            return None

        try:
            # Truncate log content if too long (to save tokens)
            truncated_content = log_content[:10000]  # First 10K chars
            if len(log_content) > 10000:
                truncated_content += "\n... [CONTENT TRUNCATED FOR SUMMARIZATION]"

            # Create summarization prompt
            prompt = f"""You are analyzing logs from a mathematical modeling agent system.
Please provide a concise summary (max {max_summary_length} characters) of the following log section.

Section: {section_name}

Log Content:
{truncated_content}

Focus on:
1. What operations were performed?
2. Any errors or warnings?
3. Key outcomes or results?

Summary:"""

            # Call LLM to generate summary
            summary = llm_client.generate(prompt, system="You are a helpful log analysis assistant.")

            # Log the summary
            self.loggers['main'].info(f"\n{'='*80}\nLATENT SUMMARY: {section_name}\n{'='*80}\n{summary}\n{'='*80}\n")

            # Also log as structured event
            self.log_event({
                "type": "latent_summary",
                "section_name": section_name,
                "summary": summary[:500] + "..." if len(summary) > 500 else summary,
                "original_length": len(log_content),
                "truncated": len(log_content) > 10000
            })

            # Save summary to debug directory
            summary_path = self.debug_dir / f"{section_name}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"Section: {section_name}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Original length: {len(log_content)} characters\n")
                f.write(f"\n{summary}\n")

            return summary

        except Exception as e:
            # If summarization fails, log error but don't crash
            self.loggers['main'].error(f"Failed to generate latent summary for '{section_name}': {e}")
            return None

    def shutdown(self):
        """Close all loggers and handlers."""
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


def create_logger(output_dir: str, experiment_id: str, console_level="INFO"):
    """
    Convenience function to create an MMExperimentLogger.

    Args:
        output_dir: Experiment output directory
        experiment_id: Unique experiment identifier
        console_level: Console log level

    Returns:
        MMExperimentLogger instance
    """
    return MMExperimentLogger(output_dir, experiment_id, console_level)


# Backward compatibility alias
LoggerManager = MMExperimentLogger
