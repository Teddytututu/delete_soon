"""
Execution Tracker (Lean Version)
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock

class ExecutionTracker:
    def __init__(self, output_dir: str, logger_manager=None):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger_manager = logger_manager
        self.events = []
        self.current_stage = None
        self.current_task = None
        self.stage_start_times = {}
        self.step_start_times = {}
        self.lock = Lock()

        # [MODIFIED] Only keep trace.jsonl
        self.jsonl_path = self.log_dir / "trace.jsonl"
        self._initialize_tracker()

    def _initialize_tracker(self):
        self.track_event("system", "tracker_initialized", {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "tracker_version": "2.0 (Lean)"
        })

    def track_event(self, event_type: str, event_name: str, data: Dict[str, Any]):
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

            try:
                with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"[ERROR] Failed to write trace: {e}")

    def start_stage(self, stage_name: str, stage_num: Optional[int] = None):
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.time()
        self.track_event("stage", "stage_start", {"stage_name": stage_name, "stage_num": stage_num})
        if self.logger_manager:
            self.logger_manager.log_progress(f"STAGE START: {stage_name}", level='info')

    def end_stage(self, stage_name: str, duration_seconds: float):
        self.track_event("stage", "stage_end", {"stage_name": stage_name, "duration_seconds": duration_seconds})
        if self.logger_manager:
            self.logger_manager.log_progress(f"STAGE END: {stage_name} ({duration_seconds:.2f}s)", level='info')

    def start_task(self, task_id: Any):
        self.current_task = str(task_id)
        self.step_start_times[f"task_{task_id}"] = time.time()
        self.track_event("task", "task_start", {"task_id": str(task_id)})

    def end_task(self, task_id: Any, result_summary: Optional[Dict] = None):
        duration = time.time() - self.step_start_times.get(f"task_{task_id}", time.time())
        self.track_event("task", "task_end", {"task_id": str(task_id), "duration_seconds": duration, "result_summary": result_summary})
        if self.logger_manager:
            self.logger_manager.log_progress(f"TASK COMPLETE: {task_id} ({duration:.2f}s)", level='info')

    def track_llm_call(self, prompt, response, model, usage, stage, latency, latent_summary=None):
        call_id = f"llm_{int(time.time() * 1000)}"
        self.track_event("llm_call", "api_call", {
            "model": model, "stage": stage, "call_id": call_id,
            "prompt_length": len(prompt), "response_length": len(response),
            "total_tokens": usage.get('total_tokens', 0), "latency_seconds": latency
        })
        if self.logger_manager:
            self.logger_manager.log_llm_call(model, len(prompt), usage, stage)

    def track_file_operation(self, operation, file_path, file_size=None, success=True, error=None):
        self.track_event("file_operation", operation, {"file_path": str(file_path), "success": success})

    def track_decision(self, decision_name, options, selected, reasoning=None):
        self.track_event("decision", decision_name, {"selected": selected, "reasoning": reasoning})

    def track_method_retrieval(self, query, methods_retrieved, scores, stage):
        self.track_event("method_retrieval", "hmml_query", {"query": query, "methods": methods_retrieved})

    def track_code_generation(self, task_id, code_length, success, error=None):
        self.track_event("code_generation", f"task_{task_id}", {"success": success, "error": error})

    def track_code_execution(self, task_id, script_name, return_code, stdout_length, stderr_length, execution_time):
        self.track_event("code_execution", f"task_{task_id}", {"return_code": return_code, "execution_time": execution_time})

    def track_chart_generation(self, task_id, num_charts, description, success, output_path=None, error=None):
        self.track_event("chart_generation", f"task_{task_id}", {"success": success, "error": error})

    def track_error(self, error_type, error_message, stage=None, traceback=None):
        self.track_event("error", error_type, {"message": error_message, "stage": stage or self.current_stage})
        if self.logger_manager:
            self.logger_manager.get_logger('errors').error(f"{error_type}: {error_message}")

    # ========================================================================
    # [TRUTH MODE] CRITICAL FIX: log_error() for complete traceback capture
    # ========================================================================

    def log_error(self, stage: str, error_obj: Exception, context: str = None):
        """
        【Truth Mode】记录完整的Python traceback到trace.jsonl

        这是实现"日志不说谎"的核心方法。当Pipeline崩溃时，
        此方法会将完整的Python堆栈信息记录为CRITICAL_FAILURE事件，
        供LatentReporter进行法医式尸检分析。

        Args:
            stage: 发生错误的阶段 (如 "main_pipeline", "task_solving", "code_execution")
            error_obj: 异常对象 (必须是Exception的实例)
            context: 导致错误的代码片段或上下文信息 (可选)

        Example:
            >>> try:
            ...     some_code()
            ... except Exception as e:
            ...     tracker.log_error("main_pipeline", e, context="While processing task 1")

        Output in trace.jsonl:
            {"timestamp": "...", "type": "CRITICAL_FAILURE", "data": {
                "stage": "main_pipeline",
                "error_type": "ValueError",
                "error_message": "invalid literal for int()...",
                "traceback": "Traceback (most recent call last):\n  File ...",
                "context_snippet": "While processing task 1"
            }}
        """
        import traceback as tb_module

        try:
            # 提取完整的Traceback
            # 这才是Latent LLM需要看到的"真相"
            tb_str = "".join(tb_module.format_exception(
                type(error_obj),
                error_obj,
                error_obj.__traceback__
            ))

            # 构建CRITICAL_FAILURE事件payload
            payload = {
                "stage": stage,
                "error_type": type(error_obj).__name__,
                "error_message": str(error_obj),
                "traceback": tb_str,  # <--- 真相在这里！
                "context_snippet": context
            }

            # 写入trace.jsonl，使用特殊的"CRITICAL_FAILURE"类型
            # 这样LatentReporter的_read_crash_site()方法可以快速定位
            self._write("CRITICAL_FAILURE", payload)

            # 同时记录到logger_manager的errors.log (如果存在)
            if self.logger_manager:
                error_logger = self.logger_manager.get_logger('errors')
                error_logger.critical(
                    f"[CRITICAL_FAILURE] Stage: {stage} | "
                    f"Error: {type(error_obj).__name__}: {str(error_obj)} | "
                    f"Context: {context or 'No context'}"
                )

        except Exception as logging_error:
            # 最后的防线：如果连记录错误都失败了，打印到stderr
            # 绝不能让错误记录失败导致原始错误被掩盖
            print(f"[CRITICAL] Failed to log error to trace.jsonl: {logging_error}")
            print(f"[ORIGINAL ERROR] {type(error_obj).__name__}: {str(error_obj)}")

    def _write(self, type_str: str, content: Dict[str, Any]):
        """
        内部方法：直接写入事件到trace.jsonl

        Args:
            type_str: 事件类型 (如 "CRITICAL_FAILURE", "stage", "task")
            content: 事件内容 (会被放入"data"字段)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": type_str,
            "data": content
        }

        try:
            with self.lock:
                with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            # 如果写文件失败，至少打印到stderr
            print(f"[ERROR] Failed to write to trace.jsonl: {e}")
            print(f"[EVENT CONTENT] {json.dumps(event, ensure_ascii=False)[:200]}...")

    def track_warning(self, warning_type, warning_message, stage=None):
        self.track_event("warning", warning_type, {"message": warning_message})

    def finalize(self):
        json_path = self.log_dir / "trace_full.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"events": self.events}, f, indent=2, ensure_ascii=False)
        except Exception: pass

        if self.logger_manager:
            self.logger_manager.log_progress(f"Trace saved: {self.jsonl_path}", level='info')
        return str(self.jsonl_path)

_tracker_instance = None
_tracker_lock = threading.Lock()

def get_tracker(output_dir: str = None, logger_manager=None) -> ExecutionTracker:
    global _tracker_instance
    if _tracker_instance is not None: return _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            if output_dir is None: raise ValueError("output_dir required")
            _tracker_instance = ExecutionTracker(output_dir, logger_manager)
    return _tracker_instance

def reset_tracker():
    global _tracker_instance
    with _tracker_lock: _tracker_instance = None
