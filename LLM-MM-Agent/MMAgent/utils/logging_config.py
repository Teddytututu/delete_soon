"""
MM-Agent Experiment Logging System (Lean Version)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

class MMExperimentLogger:
    def __init__(self, output_dir: str, experiment_id: str, console_level="INFO"):
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id

        # [MODIFIED] Simplified directory structure
        self.log_dir = self.output_dir / "logs"

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.loggers = {}
        self._setup_loggers(console_level)

    def _setup_loggers(self, console_level):
        """Configure SINGLE debug logger for all events."""
        # [MODIFIED] Only create ONE debug.log
        debug_logger = self._create_logger(
            'main',
            self.log_dir / 'debug.log',
            level=logging.DEBUG,
            console_level=console_level
        )
        self.loggers['main'] = debug_logger

        # [MODIFIED] Map all other loggers to the same debug logger
        self.loggers['errors'] = debug_logger
        self.loggers['llm_api'] = debug_logger
        self.loggers['code_execution'] = debug_logger
        self.loggers['chart_generation'] = debug_logger

    def _create_logger(self, name, log_file, level, console_level=None):
        logger = logging.getLogger(f'mmagent.{name}')
        logger.setLevel(level)

        for h in logger.handlers[:]:
            try: h.close()
            except: pass
            logger.removeHandler(h)

        logger.propagate = False

        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        if console_level:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, console_level))
            console_formatter = logging.Formatter(fmt='%(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def get_logger(self, name='main'):
        return self.loggers.get(name, self.loggers['main'])

    def _jsonl_path(self, filename: str) -> Path:
        # [MODIFIED] Standardize trace filename
        if filename == "events.jsonl":
            return self.log_dir / "trace.jsonl"
        return self.log_dir / filename

    def log_event(self, event: Dict[str, Any]):
        event = dict(event)
        event.setdefault("timestamp", datetime.now().isoformat())
        event.setdefault("experiment_id", self.experiment_id)

        path = self._jsonl_path("events.jsonl")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            import sys
            print(f"[LOGGING FALLBACK] Failed to write to {path}: {e}", file=sys.stderr)

    def log_stage_start(self, stage_name: str, stage_num: int = None):
        separator = "=" * 60
        msg = f"\n{separator}\n  STAGE {stage_num}: {stage_name}\n{separator}" if stage_num else f"\n{separator}\n  {stage_name}\n{separator}"
        self.loggers['main'].info(msg)
        self.log_event({"type": "stage_start", "stage_name": stage_name, "stage_num": stage_num})

    def log_stage_complete(self, stage_name: str, duration_seconds: int, stage_num: int = None):
        msg = f"[OK] STAGE {stage_num} COMPLETE ({duration_seconds}s)" if stage_num else f"[OK] {stage_name} COMPLETE ({duration_seconds}s)"
        self.loggers['main'].info(msg)
        self.log_event({"type": "stage_complete", "stage_name": stage_name, "duration_seconds": duration_seconds})

    def log_progress(self, message: str, level='info'):
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        getattr(self.loggers['main'], level)(f"{timestamp} -> {message}")

    def log_llm_call(self, model, prompt_length, usage, stage):
        # Only log event, no separate file
        self.log_event({
            "type": "llm_call", "model": model, "stage": stage,
            "total_tokens": usage.get('total_tokens', 0)
        })

    def log_code_execution(self, task_id, script_name, stdout, stderr, return_code):
        log_msg = f"Task {task_id} | Script: {script_name} | Return: {return_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        self.loggers['main'].debug(log_msg)
        self.log_event({"type": "code_execution", "task_id": task_id, "return_code": return_code})

    def log_chart_generation(self, task_id, num_charts, description, success, error=None):
        self.log_event({"type": "chart_generation", "task_id": task_id, "success": success, "error": error})

    def log_exception(self, exception, context, stage=None, task=None, include_traceback=True):
        import traceback
        tb = traceback.format_exc() if include_traceback else str(exception)
        msg = f"EXCEPTION: {context}\n{tb}"
        self.loggers['main'].error(msg)
        self.log_event({"type": "exception", "context": context, "message": str(exception)})
        return msg

    def log_latent_summary(self, log_content, section_name, llm_client=None, max_summary_length=500):
        # Stub for compatibility
        return None

    def shutdown(self):
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

def create_logger(output_dir, experiment_id, console_level="INFO"):
    return MMExperimentLogger(output_dir, experiment_id, console_level)

# Backward compatibility alias
LoggerManager = MMExperimentLogger
