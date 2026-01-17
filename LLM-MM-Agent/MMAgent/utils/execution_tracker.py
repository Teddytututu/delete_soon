"""
Comprehensive Execution Tracker for MM-Agent

Records every step of the MM-Agent execution process in detail.
This provides complete visibility into what happened during the run.

Author: MM-Agent Team
Date: 2026-01-09
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock


class ExecutionTracker:
    """
    Comprehensive execution tracker that records every step in detail.

    This tracker captures:
    - Every LLM API call (prompts, responses, tokens, timing)
    - Every file operation (reads, writes)
    - Every decision point
    - Every stage transition
    - Errors and warnings
    - Performance metrics

    Output: logs/execution_tracker.jsonl (line-by-line JSON for easy parsing)
          logs/execution_tracker_readable.txt (human-readable format)
    """

    def __init__(self, output_dir: str, logger_manager=None):
        """
        Initialize the execution tracker.

        Args:
            output_dir: Output directory path
            logger_manager: Optional logger manager
        """
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger_manager = logger_manager

        # Tracking data
        self.events = []
        self.current_stage = None
        self.current_task = None
        self.stage_start_times = {}
        self.step_start_times = {}

        # Thread safety
        self.lock = Lock()

        # Output files
        self.jsonl_path = self.log_dir / "execution_tracker.jsonl"
        self.readable_path = self.log_dir / "execution_tracker_readable.txt"
        self.brief_path = self.log_dir / "brief_summary.txt"  # NEW: Concise summary

        # Initialize tracker
        self._initialize_tracker()

    def _initialize_tracker(self):
        """Initialize the tracker with metadata."""
        # Write header to readable log
        try:
            with open(self.readable_path, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("MM-AGENT EXECUTION TRACKING - REAL-TIME LOG\n")
                f.write("=" * 100 + "\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Output Directory: {self.output_dir}\n")
                f.write("=" * 100 + "\n")
                f.write("Events will be logged below as they happen...\n")
                f.write("=" * 100 + "\n\n")
                f.flush()
        except Exception as e:
            print(f"[ERROR] Failed to initialize readable log: {e}")

        # Write header to brief summary log
        try:
            with open(self.brief_path, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("MM-AGENT EXECUTION SUMMARY - WHAT'S HAPPENING\n")
                f.write("=" * 100 + "\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 100 + "\n")
                f.write("This file provides a brief, concise summary of key events.\n")
                f.write("For complete details, see execution_tracker_readable.txt\n")
                f.write("=" * 100 + "\n\n")
                f.flush()
        except Exception as e:
            print(f"[ERROR] Failed to initialize brief summary: {e}")

        # Track initialization event
        self.track_event("system", "tracker_initialized", {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "tracker_version": "1.0"
        })

    def _write_readable_event(self, event: dict):
        """
        Write event to human-readable log file.

        Args:
            event: Event dictionary with all details
        """
        try:
            with open(self.readable_path, 'a', encoding='utf-8') as f:
                f.write(f"\n[{event['timestamp']}] {event['event_type'].upper()} | {event['event_name']}\n")

                # Add stage/task context
                if event.get('stage'):
                    f.write(f"  Stage: {event['stage']}\n")
                if event.get('task'):
                    f.write(f"  Task: {event['task']}\n")

                # Add important data based on event type
                data = event.get('data', {})
                if data:
                    # Write key information based on event type
                    if event['event_type'] == 'llm_call':
                        f.write(f"  Model: {data.get('model', 'Unknown')}\n")
                        # CRITICAL FIX: Replace Unicode arrow with ASCII for Windows GBK compatibility
                        f.write(f"  Tokens: {data.get('prompt_tokens', 0)} -> {data.get('completion_tokens', 0)} (total: {data.get('total_tokens', 0)})\n")
                        f.write(f"  Latency: {data.get('latency_seconds', 0):.2f}s\n")

                        # Show summary of what LLM did
                        response_summary = data.get('response_summary', '')
                        if response_summary:
                            f.write(f"  Summary: {response_summary}\n")

                        # Show brief prompt preview
                        prompt_preview = data.get('prompt_preview', '')
                        if prompt_preview and len(prompt_preview) > 50:
                            f.write(f"  Prompt: {prompt_preview}\n")

                    elif event['event_type'] == 'decision':
                        # Show what decision was made and reasoning
                        f.write(f"  Selected: {data.get('selected', 'Unknown')}\n")
                        if data.get('reasoning'):
                            f.write(f"  Reasoning: {data.get('reasoning')[:300]}\n")

                    elif event['event_type'] == 'method_retrieval':
                        # Show which methods were retrieved and why
                        methods = data.get('methods_retrieved', [])
                        if methods:
                            f.write(f"  Methods Retrieved: {', '.join(methods[:3])}\n")
                        scores = data.get('scores', [])
                        if scores:
                            f.write(f"  Top Score: {max(scores):.3f}\n")

                    elif event['event_type'] == 'stage':
                        if event['event_name'] == 'stage_end':
                            f.write(f"  Duration: {data.get('duration_seconds', 0)}s\n")

                    elif event['event_type'] == 'error':
                        f.write(f"  Error: {data.get('error_message', 'Unknown')}\n")
                        if data.get('context'):
                            f.write(f"  Context: {data.get('context', 'Unknown')}\n")

                    elif event['event_type'] == 'code_generation':
                        f.write(f"  Code Length: {data.get('code_length', 0)} bytes\n")
                        f.write(f"  Success: {data.get('success', False)}\n")

                    elif event['event_type'] == 'code_execution':
                        f.write(f"  Script: {data.get('script_name', 'Unknown')}\n")
                        f.write(f"  Return Code: {data.get('return_code', -1)}\n")
                        f.write(f"  Execution Time: {data.get('execution_time_seconds', 0):.2f}s\n")

                f.write("-" * 100 + "\n")
                f.flush()  # Ensure it's written immediately
        except Exception as e:
            print(f"[ERROR] Failed to write readable event: {e}")

    def track_event(self, event_type: str, event_name: str, data: Dict[str, Any]):
        """
        Track a single event.

        Args:
            event_type: Type of event (llm_call, file_operation, stage_start, etc.)
            event_name: Name/description of the event
            data: Event data (any JSON-serializable data)
        """
        with self.lock:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "event_name": event_name,
                "stage": self.current_stage,
                "task": self.current_task,
                "data": data
            }

            self.events.append(event)

            # Write immediately to JSONL for durability
            try:
                with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"[ERROR] Failed to write event to log: {e}")

            # Write to readable log (detailed)
            self._write_readable_event(event)

            # Write to brief summary (concise)
            self._write_brief_summary(event['event_type'], event['event_name'], event.get('data', {}))

    def _write_brief_summary(self, event_type: str, event_name: str, data: dict):
        """
        Write a brief, concise summary of key events.

        This focuses on WHAT is happening, not technical details.
        """
        try:
            with open(self.brief_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%H:%M:%S')

                if event_type == 'stage' and event_name == 'stage_start':
                    stage = data.get('stage_name', 'Unknown')
                    f.write(f"[{timestamp}] >> Starting: {stage}\n")
                    f.flush()

                elif event_type == 'stage' and event_name == 'stage_end':
                    duration = data.get('duration_seconds', 0)
                    f.write(f"[{timestamp}] [OK] Completed in {duration:.1f}s\n\n")
                    f.flush()

                elif event_type == 'task' and event_name == 'task_start':
                    task = data.get('task_id', 'Unknown')
                    f.write(f"[{timestamp}]   -> Task {task}: Working...\n")
                    f.flush()

                elif event_type == 'task' and event_name == 'task_end':
                    result = data.get('result_summary', {})
                    # CRITICAL FIX: Default to 'completed' unless explicitly marked as failed
                    # This prevents false negatives when result_summary is empty or status is not set
                    status = result.get('status', 'completed')
                    if status == 'completed':
                        f.write(f"[{timestamp}]   [OK] Task completed\n")
                    else:
                        f.write(f"[{timestamp}]   [X] Task failed: {status}\n")
                    f.flush()

                elif event_type == 'llm_call':
                    # Brief summary of LLM activity
                    model = data.get('model', 'Unknown')
                    stage = data.get('stage', 'unknown')
                    summary = data.get('response_summary', '')
                    latent_summary = data.get('latent_summary', '')  # NEW: LLM-generated summary
                    tokens = data.get('total_tokens', 0)

                    # Prefer latent summary if available
                    display_summary = latent_summary if latent_summary else summary

                    # Show LLM calls with COMPLETE summaries (no truncation)
                    if display_summary:
                        # CRITICAL FIX: Show complete latent LLM summary instead of truncating to 80 chars
                        f.write(f"[{timestamp}] [LLM] {model} ({tokens}t): {display_summary}\n")
                        f.flush()

                elif event_type == 'method_retrieval':
                    # Show which methods were retrieved
                    methods = data.get('methods_retrieved', [])
                    if methods:
                        f.write(f"[{timestamp}] [LIB] Retrieved: {', '.join(methods[:2])}\n")
                        f.flush()

                elif event_type == 'code_generation':
                    task = data.get('task_id', 'Unknown')
                    length = data.get('code_length', 0)
                    success = data.get('success', False)
                    if success:
                        f.write(f"[{timestamp}] [CODE] Generated {length} bytes\n")
                        f.flush()

                elif event_type == 'code_execution':
                    task = data.get('task_id', 'Unknown')
                    return_code = data.get('return_code', -1)
                    exec_time = data.get('execution_time_seconds', 0)
                    if return_code == 0:
                        f.write(f"[{timestamp}] [RUN] Executed in {exec_time:.1f}s\n")
                        f.flush()
                    else:
                        f.write(f"[{timestamp}] [X] Execution failed\n")
                        f.flush()

                elif event_type == 'error':
                    error = data.get('error_message', 'Unknown error')
                    context = data.get('context', '')

                    # Get current context
                    stage = self.current_stage or 'Unknown'
                    task = self.current_task or 'N/A'

                    # Write detailed error message
                    f.write(f"\n[{timestamp}] [!!!] ERROR OCCURRED\n")
                    f.write(f"  Stage: {stage}\n")
                    if task != 'N/A':
                        f.write(f"  Task: {task}\n")
                    f.write(f"  Error: {error}\n")
                    if context:
                        f.write(f"  Context: {context[:200]}\n")

                    # Add helpful guidance
                    error_lower = error.lower()
                    if 'index' in error_lower or 'key' in error_lower:
                        f.write(f"  [HINT] Check data structure and file paths\n")
                    elif 'syntax' in error_lower:
                        f.write(f"  [HINT] LLM generated invalid Python code\n")
                    elif 'timeout' in error_lower:
                        f.write(f"  [HINT] Code execution took too long\n")
                    elif 'api' in error_lower or 'rate' in error_lower:
                        f.write(f"  [HINT] Check API key and quota\n")
                    elif 'memory' in error_lower:
                        f.write(f"  [HINT] System ran out of memory\n")

                    f.write(f"  [NEXT] See errors.log for full traceback\n")
                    f.write(f"\n")
                    f.flush()

        except Exception as e:
            print(f"[ERROR] Failed to write brief summary: {e}")

    def start_stage(self, stage_name: str, stage_num: Optional[int] = None):
        """
        Mark the start of a pipeline stage.

        Args:
            stage_name: Name of the stage
            stage_num: Optional stage number
        """
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.time()

        self.track_event("stage", "stage_start", {
            "stage_name": stage_name,
            "stage_num": stage_num,
            "timestamp_start": datetime.now().isoformat()
        })

        if self.logger_manager:
            self.logger_manager.log_progress(f"STAGE START: {stage_name}", level='info')

    def end_stage(self, stage_name: str, duration_seconds: float):
        """
        Mark the end of a pipeline stage.

        Args:
            stage_name: Name of the stage
            duration_seconds: Duration in seconds
        """
        self.track_event("stage", "stage_end", {
            "stage_name": stage_name,
            "duration_seconds": duration_seconds,
            "timestamp_end": datetime.now().isoformat()
        })

        if self.logger_manager:
            self.logger_manager.log_progress(f"STAGE END: {stage_name} ({duration_seconds:.2f}s)", level='info')

    def start_task(self, task_id: Any):
        """
        Mark the start of a task.

        Args:
            task_id: Task identifier
        """
        self.current_task = str(task_id)
        self.step_start_times[f"task_{task_id}"] = time.time()

        self.track_event("task", "task_start", {
            "task_id": str(task_id),
            "stage": self.current_stage
        })

    def end_task(self, task_id: Any, result_summary: Optional[Dict] = None):
        """
        Mark the end of a task.

        Args:
            task_id: Task identifier
            result_summary: Optional summary of results
        """
        duration = time.time() - self.step_start_times.get(f"task_{task_id}", time.time())

        self.track_event("task", "task_end", {
            "task_id": str(task_id),
            "stage": self.current_stage,
            "duration_seconds": duration,
            "result_summary": result_summary
        })

        if self.logger_manager:
            self.logger_manager.log_progress(f"TASK COMPLETE: {task_id} ({duration:.2f}s)", level='info')

    def _generate_response_summary(self, response: str) -> str:
        """
        Generate a ~50 word summary of the LLM response.

        Args:
            response: Full LLM response text

        Returns:
            Summary string (approximately 50 words)
        """
        # Take first few sentences, up to ~350 characters
        sentences = response.split('. ')
        summary_words = []
        char_count = 0
        max_chars = 350

        for sentence in sentences:
            if char_count + len(sentence) > max_chars:
                break
            summary_words.append(sentence)
            char_count += len(sentence) + 2  # +2 for ". "

        summary = '. '.join(summary_words)
        if len(summary_words) < len(sentences):
            summary += "..."

        # Clean up and return
        return summary.strip() if summary.strip() else response[:200] + "..."

    def track_llm_call(self, prompt: str, response: str, model: str,
                       usage: Dict[str, int], stage: str, latency: float,
                       latent_summary: Optional[str] = None):
        """
        Track an LLM API call with FULL content logging.

        Args:
            prompt: The prompt sent to LLM
            response: The response received
            model: Model name
            usage: Token usage dictionary
            stage: Stage where call was made
            latency: Request latency in seconds
            latent_summary: Optional LLM-generated summary of this call
        """
        # Generate call ID for this specific LLM call
        call_id = f"llm_{int(time.time() * 1000)}"

        # Use latent summary if available, otherwise generate response summary
        response_summary = latent_summary if latent_summary else self._generate_response_summary(response)

        # Create shorter preview for readable log (but still show meaningful content)
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt

        self.track_event("llm_call", "api_call", {
            "model": model,
            "stage": stage,
            "call_id": call_id,
            "prompt_length": len(prompt),
            "prompt_preview": prompt_preview,
            "response_length": len(response),
            "response_summary": response_summary,  # Use latent summary if available
            "latent_summary": latent_summary,  # NEW: Store LLM-generated summary
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get('completion_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0),
            "latency_seconds": latency
        })

        if self.logger_manager:
            self.logger_manager.log_llm_call(model, len(prompt), usage, stage)

    def track_file_operation(self, operation: str, file_path: str,
                            file_size: Optional[int] = None, success: bool = True,
                            error: Optional[str] = None):
        """
        Track a file operation.

        Args:
            operation: Operation type (read, write, delete, etc.)
            file_path: Path to the file
            file_size: Size of the file in bytes (for reads/writes)
            success: Whether the operation succeeded
            error: Error message if failed
        """
        self.track_event("file_operation", operation, {
            "file_path": str(file_path),
            "file_size": file_size,
            "success": success,
            "error": error
        })

    def track_decision(self, decision_name: str, options: List[str],
                      selected: str, reasoning: Optional[str] = None):
        """
        Track a decision point in the pipeline.

        Args:
            decision_name: Name of the decision
            options: List of options that were considered
            selected: The option that was selected
            reasoning: Reasoning behind the decision
        """
        self.track_event("decision", decision_name, {
            "options": options,
            "selected": selected,
            "reasoning": reasoning
        })

    def track_method_retrieval(self, query: str, methods_retrieved: List[str],
                              scores: List[float], stage: str):
        """
        Track HMML method retrieval.

        Args:
            query: The query used for retrieval
            methods_retrieved: List of retrieved methods
            scores: Similarity scores
            stage: Stage where retrieval occurred
        """
        self.track_event("method_retrieval", "hmml_query", {
            "query": query[:200],  # Truncate long queries
            "methods_retrieved": methods_retrieved,
            "scores": scores,
            "stage": stage
        })

    def track_code_generation(self, task_id: str, code_length: int,
                             success: bool, error: Optional[str] = None):
        """
        Track code generation.

        Args:
            task_id: Task identifier
            code_length: Length of generated code
            success: Whether generation succeeded
            error: Error message if failed
        """
        self.track_event("code_generation", f"task_{task_id}", {
            "task_id": str(task_id),
            "code_length": code_length,
            "success": success,
            "error": error
        })

    def track_code_execution(self, task_id: str, script_name: str,
                           return_code: int, stdout_length: int,
                           stderr_length: int, execution_time: float):
        """
        Track code execution.

        Args:
            task_id: Task identifier
            script_name: Name of the script
            return_code: Process return code
            stdout_length: Length of stdout
            stderr_length: Length of stderr
            execution_time: Execution time in seconds
        """
        self.track_event("code_execution", f"task_{task_id}", {
            "task_id": str(task_id),
            "script_name": script_name,
            "return_code": return_code,
            "stdout_length": stdout_length,
            "stderr_length": stderr_length,
            "execution_time_seconds": execution_time
        })

    def track_chart_generation(self, task_id: str, num_charts: int,
                              description: str, success: bool,
                              output_path: Optional[str] = None,
                              error: Optional[str] = None):
        """
        Track chart generation.

        Args:
            task_id: Task identifier
            num_charts: Chart number
            description: Chart description
            success: Whether generation succeeded
            output_path: Path to generated chart
            error: Error message if failed
        """
        self.track_event("chart_generation", f"task_{task_id}_chart_{num_charts}", {
            "task_id": str(task_id),
            "num_charts": num_charts,
            "description": description[:200],
            "success": success,
            "output_path": output_path,
            "error": error
        })

    def track_error(self, error_type: str, error_message: str,
                   stage: Optional[str] = None, traceback: Optional[str] = None):
        """
        Track an error.

        Args:
            error_type: Type of error
            error_message: Error message
            stage: Stage where error occurred
            traceback: Optional traceback
        """
        self.track_event("error", error_type, {
            "error_message": error_message,
            "stage": stage or self.current_stage,
            "traceback": traceback[:1000] if traceback else None  # Limit traceback length
        })

        if self.logger_manager:
            self.logger_manager.get_logger('errors').error(
                f"{error_type}: {error_message}"
            )

    def track_warning(self, warning_type: str, warning_message: str,
                     stage: Optional[str] = None):
        """
        Track a warning.

        Args:
            warning_type: Type of warning
            warning_message: Warning message
            stage: Stage where warning occurred
        """
        self.track_event("warning", warning_type, {
            "warning_message": warning_message,
            "stage": stage or self.current_stage
        })

    def generate_readable_report(self) -> str:
        """
        Generate a human-readable report of all tracked events.

        Returns:
            Path to the generated report
        """
        with open(self.readable_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("MM-Agent COMPREHENSIVE EXECUTION TRACKING REPORT\n")
            f.write("="*100 + "\n\n")

            # Group events by type
            events_by_type = {}
            for event in self.events:
                event_type = event['event_type']
                if event_type not in events_by_type:
                    events_by_type[event_type] = []
                events_by_type[event_type].append(event)

            # Write summary statistics
            f.write("-"*100 + "\n")
            f.write("EXECUTION SUMMARY\n")
            f.write("-"*100 + "\n\n")

            f.write(f"Total Events: {len(self.events)}\n")
            f.write(f"Event Types: {len(events_by_type)}\n")
            f.write(f"Time Range: {self.events[0]['timestamp']} to {self.events[-1]['timestamp']}\n\n")

            # Write breakdown by type
            f.write("Events by Type:\n")
            for event_type, events in sorted(events_by_type.items()):
                f.write(f"  {event_type}: {len(events)} events\n")
            f.write("\n")

            # Write detailed timeline
            f.write("-"*100 + "\n")
            f.write("DETAILED TIMELINE\n")
            f.write("-"*100 + "\n\n")

            for event in self.events:
                f.write(f"[{event['timestamp']}] ")
                f.write(f"{event['event_type'].upper()} | {event['event_name']}\n")

                if event.get('stage'):
                    f.write(f"  Stage: {event['stage']}\n")
                if event.get('task'):
                    f.write(f"  Task: {event['task']}\n")

                # Write key data fields
                data = event.get('data', {})
                if data:
                    for key, value in list(data.items())[:10]:  # Limit to 10 fields
                        if isinstance(value, (str, int, float, bool)):
                            value_str = str(value)[:100]
                            f.write(f"  {key}: {value_str}\n")
                        elif isinstance(value, list) and len(value) > 0:
                            f.write(f"  {key}: [{len(value)} items] {str(value)[:100]}\n")

                f.write("\n")

            # Write LLM call summary
            if 'llm_call' in events_by_type:
                f.write("-"*100 + "\n")
                f.write("LLM API CALLS SUMMARY\n")
                f.write("-"*100 + "\n\n")

                total_tokens = 0
                total_cost_estimate = 0

                for event in events_by_type['llm_call']:
                    data = event.get('data', {})
                    total_tokens += data.get('total_tokens', 0)

                f.write(f"Total LLM Calls: {len(events_by_type['llm_call'])}\n")
                f.write(f"Total Tokens Used: {total_tokens:,}\n\n")

            # Write file operations summary
            if 'file_operation' in events_by_type:
                f.write("-"*100 + "\n")
                f.write("FILE OPERATIONS SUMMARY\n")
                f.write("-"*100 + "\n\n")

                reads = len([e for e in events_by_type['file_operation']
                           if e['event_name'] == 'read'])
                writes = len([e for e in events_by_type['file_operation']
                            if e['event_name'] == 'write'])

                f.write(f"Total File Operations: {len(events_by_type['file_operation'])}\n")
                f.write(f"  Reads: {reads}\n")
                f.write(f"  Writes: {writes}\n\n")

            # Write errors and warnings
            if 'error' in events_by_type or 'warning' in events_by_type:
                f.write("-"*100 + "\n")
                f.write("ERRORS AND WARNINGS\n")
                f.write("-"*100 + "\n\n")

                if 'error' in events_by_type:
                    f.write(f"Total Errors: {len(events_by_type['error'])}\n")
                    for event in events_by_type['error']:
                        f.write(f"  [{event['timestamp']}] {event['event_name']}: ")
                        f.write(f"{event['data'].get('error_message', 'Unknown')}\n")

                if 'warning' in events_by_type:
                    f.write(f"\nTotal Warnings: {len(events_by_type['warning'])}\n")
                    for event in events_by_type['warning']:
                        f.write(f"  [{event['timestamp']}] {event['event_name']}: ")
                        f.write(f"{event['data'].get('warning_message', 'Unknown')}\n")

            f.write("\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")

        return str(self.readable_path)

    def _generate_final_brief_summary(self):
        """Generate a final summary for the brief summary file."""
        try:
            with open(self.brief_path, 'a', encoding='utf-8') as f:
                f.write("\n")
                f.write("=" * 100 + "\n")
                f.write("FINAL SUMMARY\n")
                f.write("=" * 100 + "\n")

                # Calculate duration
                if self.events:
                    start_time = datetime.fromisoformat(self.events[0]['timestamp'])
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    f.write(f"Total Duration: {duration:.1f}s ({duration/60:.1f} minutes)\n")

                # Count events by type
                event_counts = {}
                error_count = 0
                llm_call_count = 0
                stage_count = 0

                for event in self.events:
                    event_type = event['event_type']
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1

                    if event_type == 'error':
                        error_count += 1
                    elif event_type == 'llm_call':
                        llm_call_count += 1
                    elif event_type == 'stage':
                        stage_count += 1

                # Write summary
                f.write(f"Total LLM Calls: {llm_call_count}\n")
                f.write(f"Stages Completed: {stage_count // 2}\n")  # stage_start + stage_end
                f.write(f"Errors: {error_count}\n")

                # Check for errors
                if error_count > 0:
                    f.write("\n")
                    f.write("[!] PIPELINE FAILED\n")
                    f.write("\n")
                    f.write("What went wrong:\n")
                    f.write(f"  - {error_count} error(s) occurred\n")
                    f.write("\n")
                    f.write("Where to look:\n")
                    f.write(f"  - errors.log: Full tracebacks and details\n")
                    f.write(f"  - execution_tracker_readable.txt: Complete event log\n")
                    f.write(f"  - main.log: Main execution flow\n")
                    f.write("\n")
                    f.write("Common issues:\n")
                    f.write("  1. LLM generated invalid code (syntax errors)\n")
                    f.write("  2. Data structure mismatch (IndexError/KeyError)\n")
                    f.write("  3. Code execution timeout\n")
                    f.write("  4. API rate limiting or quota exceeded\n")
                    f.write("\n")
                    f.write("Recommendations:\n")
                    f.write("  - Try a better LLM model (glm-4-plus instead of glm-4-flash)\n")
                    f.write("  - Check errors.log for the exact error location\n")
                    f.write("  - Review generated code in output/*/code/ folder\n")
                else:
                    f.write("\n")
                    f.write("[OK] PIPELINE COMPLETED SUCCESSFULLY\n")
                    f.write("\n")
                    f.write("Output files:\n")
                    f.write("  - JSON: output/*/json/solution.json\n")
                    f.write("  - Markdown: output/*/markdown/solution.md\n")
                    f.write("  - Charts: output/*/charts/\n")
                    f.write("  - Code: output/*/code/\n")

                f.write("\n")
                f.write("=" * 100 + "\n")
                f.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 100 + "\n")
                f.flush()
        except Exception as e:
            print(f"[ERROR] Failed to generate final brief summary: {e}")

    def finalize(self):
        """Finalize tracking and generate summary reports."""
        # Generate final brief summary
        self._generate_final_brief_summary()

        # Add summary to readable log (events already written in real-time)
        try:
            with open(self.readable_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write("EXECUTION SUMMARY\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"Total Events: {len(self.events)}\n")
                f.write(f"Completed: {datetime.now().isoformat()}\n")
                f.write(f"Duration: {(datetime.now() - datetime.fromisoformat(self.events[0]['timestamp'])).total_seconds():.1f}s\n" if self.events else "Duration: N/A\n")

                # Count events by type
                event_counts = {}
                for event in self.events:
                    event_type = event['event_type']
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1

                f.write("\nEvents by Type:\n")
                for event_type, count in sorted(event_counts.items()):
                    f.write(f"  {event_type}: {count}\n")

                f.write("\n" + "=" * 100 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 100 + "\n")
                f.flush()
        except Exception as e:
            print(f"[ERROR] Failed to write summary: {e}")

        # Save complete event log as JSON
        json_path = self.log_dir / "execution_tracker_complete.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_events": len(self.events),
                    "time_range": {
                        "start": self.events[0]['timestamp'] if self.events else None,
                        "end": self.events[-1]['timestamp'] if self.events else None
                    }
                },
                "events": self.events
            }, f, indent=2, ensure_ascii=False)

        if self.logger_manager:
            self.logger_manager.log_progress(
                f"Execution tracking complete: {len(self.events)} events recorded",
                level='info'
            )
            self.logger_manager.log_progress(
                f"Readable report: {self.readable_path}",
                level='info'
            )
            self.logger_manager.log_progress(
                f"Complete JSON: {json_path}",
                level='info'
            )

        return str(self.readable_path)


# Global tracker instance and lock for thread-safe singleton
_tracker_instance = None
_tracker_lock = threading.Lock()


def get_tracker(output_dir: str = None, logger_manager=None) -> ExecutionTracker:
    """
    Get or create the global tracker instance (thread-safe singleton).

    This implementation uses double-checked locking to ensure thread safety
    while maintaining performance. Only one tracker instance exists globally.

    Args:
        output_dir: Output directory (required for first call)
        logger_manager: Optional logger manager

    Returns:
        ExecutionTracker instance

    Raises:
        ValueError: If output_dir is not provided on first call
    """
    global _tracker_instance

    # Fast path - check if instance exists without lock
    if _tracker_instance is not None:
        return _tracker_instance

    # Slow path - create new instance with lock
    with _tracker_lock:
        # Double-check pattern - another thread might have created it
        if _tracker_instance is None:
            if output_dir is None:
                raise ValueError("output_dir must be provided for first call to get_tracker()")
            _tracker_instance = ExecutionTracker(output_dir, logger_manager)

    return _tracker_instance


def reset_tracker() -> None:
    """
    Reset the global tracker instance (thread-safe).

    This is useful for testing or when starting a new pipeline run.
    Uses lock to prevent race conditions during reset.
    """
    global _tracker_instance

    with _tracker_lock:
        _tracker_instance = None
