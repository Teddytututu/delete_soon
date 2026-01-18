"""
Latent Reporter: Post-Processing Research Journal Generator

This module implements an observer-pattern reporter that reads execution traces
and generates MCM/ICM-style research journal entries using LLM.

Key Features:
- Reads trace.jsonl for comprehensive event analysis
- Stage-level retrospection (problem_analysis, mathematical_modeling)
- Error diagnosis and solution documentation
- Result validation and sensitivity discussion
- Automatic chart embedding with generated captions

Author: MM-Agent Team
Date: 2026-01-18 (Refactored for chat with claude3.txt proposal)
"""

import json
import os
import sys
import io
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

# [FIX 2026-01-18] Windows Console Encoding Fix
# Set UTF-8 encoding for stdout/stderr on Windows to display emoji characters
# This prevents UnicodeEncodeError when printing emoji (🛑, 🟢, 🔴) in error messages
if sys.platform == 'win32':
    try:
        # Reconfigure stdout and stderr with UTF-8 encoding
        # Use 'replace' error handler to prevent crashes on unencodable characters
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=sys.stdout.line_buffering
            )
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=sys.stderr.line_buffering
            )
    except Exception as e:
        # If reconfiguration fails, continue silently (non-critical)
        pass

# Import journal prompts
from MMAgent.prompt import journal_prompts

logger = logging.getLogger("LatentReporter")


class LatentReporter:
    """
    潜伏报告器：观察者模式的科研日记生成器

    核心职责：
    1. 观察 Agent 执行过程（读取 trace.jsonl）
    2. 调用 LLM 将技术日志转化为科研风格的叙述
    3. 自动嵌入图表等产物
    4. 生成 Markdown 格式的实验日记

    Usage:
        reporter = LatentReporter(output_dir, llm_client, tracker_file="path/to/trace.jsonl")
        reporter.reflect_on_stage("problem_analysis")
        reporter.reflect_on_stage("mathematical_modeling")
        reporter.diagnose_failure("KeyError: 'YEAR'")
        reporter.summarize_results(solution_dict)
    """

    def __init__(self, output_dir, llm_client, tracker_file=None):
        """
        初始化潜伏报告器

        Args:
            output_dir: 主输出目录 (例如 output/MM-Agent/Task_Timestamp/)
            llm_client: LLM 实例 (用于生成叙述)
            tracker_file: trace.jsonl 的路径 (可选，默认自动定位)
        """
        # --- 路径修正逻辑 (反递归 Bug 修复) ---
        base_path = Path(output_dir)
        if base_path.name in ["Workspace", "Memory", "Report", "logs", "code", "json"]:
            self.root_dir = base_path.parent
        else:
            self.root_dir = base_path

        self.report_dir = self.root_dir / "Report"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 定义核心文件路径
        self.journal_path = self.report_dir / "01_Research_Journal.md"

        # 自动定位 trace.jsonl (如果未传入)
        if tracker_file:
            self.trace_file = Path(tracker_file)
        else:
            self.trace_file = self.root_dir / "Memory" / "logs" / "trace.jsonl"

        self.llm = llm_client
        self.logger = logging.getLogger("main")

        # 初始化日记文件头
        if not self.journal_path.exists():
            self._init_journal()

    def _init_journal(self):
        """写入日记的头部信息"""
        header = f"""# 2025_C 数学建模科研日记
**Task ID**: Auto-Generated
**Status**: In Progress

> 本文档由 Latent LLM 实时生成，记录了 Agent 的思维实验过程。

---

## 1. Problem Background
[待生成]

"""
        try:
            with open(self.journal_path, 'w', encoding='utf-8') as f:
                f.write(header)
            self.logger.info(f"Research journal initialized: {self.journal_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize journal: {e}")

    def _read_recent_events(self, k=50) -> List[Dict[str, Any]]:
        """
        读取 trace.jsonl 的最后 k 行

        Args:
            k: 读取的行数

        Returns:
            List[Dict]: 事件列表
        """
        events = []
        if not self.trace_file.exists():
            self.logger.warning(f"Trace file not found: {self.trace_file}")
            return []

        try:
            with open(self.trace_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 简单策略：取最近的上下文
                # 实际生产中可根据 stage 过滤
                for line in lines[-k:]:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.warning(f"Failed to read trace events: {e}")

        return events

    # ========================================================================
    # [TRUTH MODE] CRITICAL FIX: _read_crash_site() for CRITICAL_FAILURE events
    # ========================================================================

    def _read_crash_site(self) -> List[Dict[str, Any]]:
        """
        【Truth Mode】专门读取CRITICAL_FAILURE事件，定位崩溃点

        与_read_recent_events()不同，此方法只读取"type": "CRITICAL_FAILURE"的事件，
        这些事件包含完整的Python traceback，是诊断Pipeline崩溃的关键证据。

        Returns:
            List[Dict]: 所有CRITICAL_FAILURE事件的data字段列表

        Example:
            >>> crashes = reporter._read_crash_site()
            >>> if crashes:
            ...     last_crash = crashes[-1]
            ...     print(last_crash['error_type'])  # 'KeyError'
            ...     print(last_crash['traceback'])   # 完整的堆栈信息
        """
        failures = []
        if not self.trace_file.exists():
            self.logger.error("Trace file not found for crash analysis")
            return []

        try:
            with open(self.trace_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        # 只关心CRITICAL_FAILURE类型的事件
                        if event.get('type') == 'CRITICAL_FAILURE':
                            # 提取data字段，包含traceback等信息
                            failures.append(event.get('data', {}))
                    except json.JSONDecodeError:
                        # 跳过格式错误的行
                        continue
        except Exception as e:
            self.logger.error(f"Failed to read crash site from trace file: {e}")

        return failures

    def _append_markdown(self, content: str):
        """
        追加内容到日记文件

        Args:
            content: Markdown 格式的内容
        """
        try:
            with open(self.journal_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{content}\n")
            self.logger.info(f"LatentReporter updated journal at {self.journal_path}")
        except Exception as e:
            self.logger.error(f"Failed to append to journal: {e}")

    # ========================================================================
    # 核心接口：阶段性复盘
    # ========================================================================

    def reflect_on_stage(self, stage_name: str):
        """
        阶段性复盘：读取 trace.jsonl 并生成阶段分析

        Args:
            stage_name: 阶段名称
                - "problem_analysis": 问题分析阶段
                - "mathematical_modeling": 数学建模阶段
        """
        self.logger.info(f"LatentReporter reflecting on stage: {stage_name}")

        # 1. 读取最近的事件
        events = self._read_recent_events(k=100)  # 读取足够的上下文
        events_str = json.dumps(events, indent=2, ensure_ascii=False)

        # 2. 根据阶段选择 prompt
        if stage_name == "problem_analysis":
            prompt = journal_prompts.STAGE_REFLECTION_ANALYSIS.format(events=events_str)
            title = "## 2. Problem Restatement & Hypotheses"
        elif stage_name == "mathematical_modeling":
            prompt = journal_prompts.STAGE_REFLECTION_MODELING.format(events=events_str)
            title = "## 3. Modeling Process"
        else:
            self.logger.warning(f"Unknown stage: {stage_name}, skipping reflection")
            return

        # 3. 调用 LLM 生成内容
        try:
            response = self.llm.chat(
                prompt,
                system_prompt=journal_prompts.SYSTEM_PROMPT
            )
            self._append_markdown(f"{title}\n\n{response}")
            self.logger.info(f"Stage reflection completed: {stage_name}")
        except Exception as e:
            self.logger.error(f"Failed to generate stage reflection: {e}")
            # Fallback: 写入原始事件
            self._append_markdown(f"{title}\n\n[LLM Generation Failed: {e}]\n\n**Raw Events**:\n```json\n{events_str[:500]}\n```")

    # ========================================================================
    # 核心接口：错误诊断
    # ========================================================================

    def diagnose_failure(self, fallback_error_msg: str):
        """
        【Truth Mode】当Pipeline崩溃时调用，进行法医式尸检分析

        这是实现"日志不说谎"的核心方法。它会：
        1. 读取trace.jsonl中的CRITICAL_FAILURE事件（包含完整traceback）
        2. 将traceback喂给LLM进行分析
        3. 在Research Journal中生成详细的错误报告
        4. 附加原始堆栈信息供人工验证

        Args:
            fallback_error_msg: 如果找不到trace.jsonl时的备用错误信息
                                (通常是main.py except块捕获的str(e))

        Example:
            >>> try:
            ...     run_pipeline()
            ... except Exception as e:
            ...     reporter.diagnose_failure(str(e))
        """
        self.logger.warning(">>> LatentReporter: Starting Forensic Analysis (法医式尸检)...")

        # 1. 读取最详细的堆栈
        failures = self._read_crash_site()

        # 2. 构建上下文
        context = ""
        if failures:
            # ✅ 找到了记录在案的底层错误
            last_fail = failures[-1]
            context = f"""
!!! DETECTED CRITICAL FAILURE IN LOGS !!!

**Error Type**: {last_fail.get('error_type')}
**Error Message**: {last_fail.get('error_message')}
**Stage**: {last_fail.get('stage')}

--- RAW TRACEBACK (The Truth) ---
{last_fail.get('traceback')}
---------------------------------

**Context Snippet**:
{last_fail.get('context_snippet', 'No context available')}

**Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            self.logger.info(f"Found {len(failures)} critical failure(s) in trace, analyzing the most recent one")
        else:
            # ❌ 日志里没记下来？程序崩溃得太快，或者Tracker被绕过了
            context = f"""
!!! SYSTEM CRASH (UNLOGGED) !!!

The system crashed without writing a structured error log to trace.jsonl.

**Python Exception detected by Main Loop**:
{fallback_error_msg}

**Possible Causes**:
1. tracker.log_error() was not called before exception bubbled up
2. Exception occurred before tracker initialization
3. File system error prevented writing to trace.jsonl
4. Exception occurred in ExecutionTracker.log_error() itself

**Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Note**: Analysis is based on fallback error message only, which may be incomplete.
"""
            self.logger.warning("No CRITICAL_FAILURE events found in trace, using fallback error message")

        # 3. 调用LLM进行分析
        prompt = journal_prompts.ERROR_DIAGNOSIS.format(events=context)

        try:
            response = self.llm.chat(
                prompt,
                system_prompt=journal_prompts.SYSTEM_PROMPT
            )

            # 4. 狠狠地写入Markdown，用红色警告
            with open(self.journal_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n## 🛑 FATAL ERROR ANALYSIS\n")
                f.write(f"> **Status**: MISSION FAILED\n")
                f.write(f"> **Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"> **Crash Events Found**: {len(failures) if failures else 0}\n\n")
                f.write(response)

                # 把原始堆栈也附在最后，防止LLM还是解释不清
                # 使用HTML <details>标签实现折叠效果
                f.write(f"\n\n<details><summary>🔍 Raw Traceback & Context (Click to expand)</summary>\n\n")
                f.write(f"```text\n{context}\n```\n")
                f.write(f"</details>\n")

            self.logger.info("Fatal error analysis completed and written to journal")

        except Exception as e:
            self.logger.error(f"Reporter itself crashed during diagnosis: {e}")
            # 最后的防线：直接写入原始上下文，不依赖LLM
            self._append_markdown(f"## 🛑 FATAL ERROR ANALYSIS\n\n**Reporter Failed**: {e}\n\n**Raw Context**:\n```text\n{context}\n```")

    # ========================================================================
    # 核心接口：结果分析
    # ========================================================================

    def summarize_results(self, solution_content: Any):
        """
        结果分析：总结计算结果并进行灵敏度讨论

        Args:
            solution_content: 结果数据 (dict 或 JSON 字符串)
        """
        self.logger.info("LatentReporter summarizing results")

        # 转换为 JSON 字符串
        if isinstance(solution_content, dict):
            solution_str = json.dumps(solution_content, indent=2, ensure_ascii=False)
        elif isinstance(solution_content, str):
            solution_str = solution_content
        else:
            solution_str = str(solution_content)

        # 1. 尝试寻找图表
        charts_dir = self.root_dir / "Workspace" / "charts"
        chart_md = ""
        if charts_dir.exists():
            chart_md = "\n### Visualizations\n"
            chart_files = list(charts_dir.glob("*.png")) + list(charts_dir.glob("*.jpg"))
            for chart in chart_files:
                # 使用相对路径引用，确保 Markdown 可移植
                rel_path = f"../Workspace/charts/{chart.name}"
                chart_md += f"\n![{chart.stem}]({rel_path})\n*{chart.stem}*\n"

        # 2. 调用 LLM 生成结果分析
        prompt = journal_prompts.RESULT_VALIDATION.format(solution=solution_str)

        try:
            response = self.llm.chat(
                prompt,
                system_prompt=journal_prompts.SYSTEM_PROMPT
            )
            self._append_markdown(f"## 4. Computational Experiments & Validation\n{chart_md}\n\n{response}")
            self.logger.info("Result summarization completed")
        except Exception as e:
            self.logger.error(f"Failed to generate result summary: {e}")
            self._append_markdown(f"## 4. Computational Experiments & Validation\n{chart_md}\n\n**Result Data**:\n```json\n{solution_str[:1000]}\n```\n\n[LLM Analysis Failed: {e}]")

    # ========================================================================
    # 核心接口：日记终结
    # ========================================================================

    def finalize_journal(self):
        """
        在实验结束时调用，添加总结信息

        This method adds a concluding section to the journal with summary statistics.
        """
        try:
            footer = f"""

## 🏁 实验结束

**结束时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

*本日志由 MM-Agent Latent Reporter 自动生成*
*For detailed execution traces, see: `../Memory/logs/trace.jsonl`*
"""
            with open(self.journal_path, "a", encoding='utf-8') as f:
                f.write(footer)

            self.logger.info("Research journal finalized")

        except Exception as e:
            self.logger.error(f"Failed to finalize journal: {e}")

    # ========================================================================
    # 向后兼容方法（保持与旧版 main.py 的兼容性）
    # ========================================================================

    def log_thought(self, stage: str, raw_content: str, status: str = "INFO", artifact: dict = None):
        """
        向后兼容方法：实时记录思考事件（不调用 LLM，快速写入）

        这是旧版 LatentReporter 的接口，保留用于向后兼容。
        新架构推荐使用 reflect_on_stage() 进行深度分析。
        """
        # 直接写入，不调用 LLM（快速路径）
        timestamp = datetime.now().strftime('%H:%M:%S')
        status_icons = {
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "INFO": "ℹ️",
            "THINKING": "🧠"
        }
        icon = status_icons.get(status, "📝")

        entry = f"### {icon} [{timestamp}] {stage}\n\n{raw_content}\n"

        # 嵌入产物
        if artifact:
            if artifact.get('type') == 'image':
                img_path = artifact.get('path')
                desc = artifact.get('description', 'Figure')
                if img_path and os.path.exists(img_path):
                    try:
                        rel_path = os.path.relpath(img_path, self.report_dir)
                        entry += f"\n![{desc}]({rel_path})\n*{desc}*\n"
                    except (ValueError, Exception):
                        entry += f"\n> [图片已生成: {os.path.basename(img_path)}]\n"

            elif artifact.get('type') == 'code':
                code_path = artifact.get('path')
                if code_path:
                    entry += f"\n> 📄 **Generated Script**: `{os.path.basename(code_path)}`\n"

            elif artifact.get('type') == 'data':
                data_path = artifact.get('path')
                desc = artifact.get('description', 'Data file')
                if data_path:
                    entry += f"\n> 📊 **Data Output**: [{desc}]({os.path.basename(data_path)})\n"

        entry += "\n---\n"

        try:
            with open(self.journal_path, "a", encoding='utf-8') as f:
                f.write(entry)
            self.logger.debug(f"Journal entry added: {stage} [{status}]")
        except Exception as e:
            self.logger.error(f"Failed to write journal entry: {e}")

    def log_success(self, stage: str, achievement: str, artifact: dict = None):
        """向后兼容方法：记录成功事件"""
        self.log_thought(stage, achievement, "SUCCESS", artifact)

    def log_error_analysis(self, error_type: str, error_message: str, attempted_fix: str = None):
        """向后兼容方法：记录错误分析"""
        content = f"**错误类型**: {error_type}\n**错误详情**: {error_message}"
        if attempted_fix:
            content += f"\n**尝试修复**: {attempted_fix}"
        self.log_thought("Error Analysis", content, "ERROR")


# ============================================================================
# 便捷函数：用于快速创建和初始化 LatentReporter
# ============================================================================

def create_latent_reporter(output_dir: str, llm_client, task_id: str = "Unknown") -> LatentReporter:
    """
    创建并初始化一个 LatentReporter 实例

    Args:
        output_dir: 输出目录路径
        llm_client: LLM 实例
        task_id: 任务 ID (保留用于兼容性，当前未使用)

    Returns:
        LatentReporter: 初始化完成的报告器实例
    """
    return LatentReporter(output_dir, llm_client)
