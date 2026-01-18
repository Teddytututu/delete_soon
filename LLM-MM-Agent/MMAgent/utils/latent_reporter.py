"""
Latent Reporter: LLM-powered narrative logging system

This module provides intelligent, human-readable logging by using LLM to transform
technical execution logs into engaging research journal entries.

Key Features:
- Real-time narrative generation during pipeline execution
- Automatic error analysis and solution documentation
- Chart and artifact embedding with auto-generated captions
- Research Journal in Markdown format for human consumption

Author: MM-Agent Team
Date: 2025-01-18
"""

import os
import time
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LatentReporter:
    """
    潜伏报告器：使用 LLM 将技术日志转化为人类可读的科研日记

    核心职责：
    1. 观察 Agent 执行过程（实时）
    2. 调用 LLM 将技术细节翻译成科研风格的叙述
    3. 自动嵌入代码、图表等产物
    4. 生成 Markdown 格式的实验日记

    Usage:
        reporter = LatentReporter(output_dir, llm_client, task_id="2025_C")
        reporter.log_thought(
            stage="Data Cleaning",
            raw_content="Encountered KeyError: 'Year', fixed by mapping to 'YEAR'",
            status="WARNING"
        )
    """

    def __init__(self, output_dir, llm_client, task_id="Unknown"):
        """
        初始化潜伏报告器

        Args:
            output_dir: 主输出目录 (例如 output/MM-Agent/Task_Timestamp/)
            llm_client: LLM 实例 (用于生成叙述)
            task_id: 任务名称 (例如 2025_C)
        """
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / "Report"
        self.journal_path = self.report_dir / "01_Research_Journal.md"
        self.llm = llm_client
        self.task_id = task_id
        self.logger = logging.getLogger("LatentReporter")

        # 1. 确保报告目录存在
        os.makedirs(self.report_dir, exist_ok=True)

        # 2. 如果日记文件不存在，初始化它
        if not self.journal_path.exists():
            self._init_journal()

    def _init_journal(self):
        """写入日记的头部信息"""
        header = f"""# 🧪 智能实验记录: {self.task_id}

**启动时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**实验目标**: 自动化数学建模与求解 (Automated Modeling & Solving)

---
> *本日志由 Latent LLM 实时观察并生成，记录 Agent 的思考过程、错误与修复。*

"""
        try:
            with open(self.journal_path, "w", encoding='utf-8') as f:
                f.write(header)
            self.logger.info(f"Research journal initialized: {self.journal_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize journal: {e}")

    def log_thought(self, stage: str, raw_content: str, status: str = "INFO", artifact: dict = None):
        """
        核心方法：记录一个思考或事件

        Args:
            stage: 当前阶段 (如 "Data Cleaning", "Modeling", "Coding")
            raw_content: 原始的技术日志或代码片段
            status: 状态 (SUCCESS, WARNING, ERROR, INFO, THINKING)
            artifact: 关联产物 (例如 {"type": "image", "path": "...", "description": "..."})
        """
        try:
            # 1. 调用 LLM 将技术日志转化为"人话" (叙事)
            narrative = self._generate_narrative(stage, raw_content, status)

            # 2. 格式化为 Markdown
            timestamp = datetime.now().strftime('%H:%M:%S')
            status_icons = {
                "SUCCESS": "✅",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "INFO": "ℹ️",
                "THINKING": "🧠"
            }
            icon = status_icons.get(status, "📝")

            entry = f"### {icon} [{timestamp}] {stage}\n\n"
            entry += f"{narrative}\n"

            # 3. 嵌入产物 (图片/代码链接)
            if artifact:
                if artifact.get('type') == 'image':
                    img_path = artifact.get('path')
                    desc = artifact.get('description', 'Figure')
                    if img_path and os.path.exists(img_path):
                        # 计算相对路径以便 Markdown 渲染
                        try:
                            rel_path = os.path.relpath(img_path, self.report_dir)
                            entry += f"\n![{desc}]({rel_path})\n*{desc}*\n"
                        except ValueError:
                            # 跨驱动器路径无法计算相对路径
                            entry += f"\n> [图片已生成: {os.path.basename(img_path)}]\n"
                        except Exception as e:
                            self.logger.warning(f"Failed to compute relative path for image: {e}")
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

            # 4. 追加写入文件 (实时落盘)
            with open(self.journal_path, "a", encoding='utf-8') as f:
                f.write(entry)

            self.logger.debug(f"Journal entry added: {stage} [{status}]")

        except Exception as e:
            # 即使报告器挂了，也不能影响主程序
            self.logger.error(f"Failed to write latent report: {e}", exc_info=True)

    def _generate_narrative(self, stage, content, status):
        """
        使用 LLM 生成简短的科研风格叙述

        Args:
            stage: 当前阶段名称
            content: 原始技术内容
            status: 状态标志

        Returns:
            str: LLM 生成的叙述文本
        """
        # 优化：如果是简单的短内容且是 INFO 状态，不需要每次都调 LLM (省钱且快)
        if len(content) < 100 and status == "INFO":
            return content

        # 构建 LLM prompt
        prompt = f"""
You are a senior data scientist writing a lab notebook. Your task is to summarize
the following execution event into a brief, professional, first-person observation.

Context:
- Stage: {stage}
- Status: {status}

Raw Content:
{content[:2000]}

Requirements:
- Be concise (2-4 sentences in Chinese).
- If SUCCESS: Explain what was achieved and why it matters.
- If ERROR: Explain the likely cause and the attempted solution.
- If WARNING: Highlight the risk and the mitigation strategy.
- If INFO: Simply state what happened.
- Tone: Professional, objective, insightful, like a researcher's diary.
- Use first-person perspective ("I", "we", "our model").

Journal Entry (Chinese):
"""

        try:
            # 调用 LLM 生成叙述（使用低温度以获得更稳定的输出）
            narrative = self.llm.generate(
                prompt,
                system="You are a helpful research assistant writing lab notes.",
                usage=False,  # 不计入使用统计，避免污染主流程的 tracking
                temperature=0.3  # 低温度，输出更稳定
            ).strip()

            # 清理可能的 markdown 代码块标记
            if narrative.startswith("```"):
                lines = narrative.split('\n')
                # 找到第一个非 ``` 行
                for i, line in enumerate(lines):
                    if not line.strip().startswith("```"):
                        narrative = '\n'.join(lines[i:])
                        break
                # 移除末尾的 ```
                if narrative.endswith("```"):
                    narrative = narrative[:-3].strip()

            return narrative

        except Exception as e:
            self.logger.warning(f"LLM narrative generation failed: {e}, falling back to raw content")
            # Fallback: 返回原始内容的摘要
            return f"**[系统记录]**: {content[:300]}..."

    def log_error_analysis(self, error_type: str, error_message: str, attempted_fix: str = None):
        """
        专门记录错误分析的便捷方法

        Args:
            error_type: 错误类型 (如 "KeyError", "ValueError", "TimeoutError")
            error_message: 错误消息
            attempted_fix: 尝试的修复方法
        """
        content = f"**错误类型**: {error_type}\n**错误详情**: {error_message}"
        if attempted_fix:
            content += f"\n**尝试修复**: {attempted_fix}"

        self.log_thought(
            stage="Error Analysis",
            raw_content=content,
            status="ERROR"
        )

    def log_success(self, stage: str, achievement: str, artifact: dict = None):
        """
        记录成功的便捷方法

        Args:
            stage: 阶段名称
            achievement: 成就描述
            artifact: 关联产物
        """
        self.log_thought(
            stage=stage,
            raw_content=achievement,
            status="SUCCESS",
            artifact=artifact
        )

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
*For detailed execution traces, see: `../Memory/trace_stream.jsonl`*
"""
            with open(self.journal_path, "a", encoding='utf-8') as f:
                f.write(footer)

            self.logger.info("Research journal finalized")

        except Exception as e:
            self.logger.error(f"Failed to finalize journal: {e}")


# ============================================================================
# 便捷函数：用于快速创建和初始化 LatentReporter
# ============================================================================

def create_latent_reporter(output_dir: str, llm_client, task_id: str = "Unknown") -> LatentReporter:
    """
    创建并初始化一个 LatentReporter 实例

    Args:
        output_dir: 输出目录路径
        llm_client: LLM 实例
        task_id: 任务 ID

    Returns:
        LatentReporter: 初始化完成的报告器实例
    """
    return LatentReporter(output_dir, llm_client, task_id)
