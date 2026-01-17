"""
Minimal Chart Renderer - "减法修复"版本

根据chat with claude2.txt和chat with claude3.txt的分析：
- 移除过度工程化（Guards, AST, autofix）
- 统一成功判定：只认文件是否存在
- 简化JSON结构：{title, image_path, success}

核心原则：
1. LLM 不写 savefig
2. LLM 不写路径
3. LLM 不决定 success
4. 只要 PNG 文件存在 → Chart 成功

Date: 2026-01-17
Priority: P0 (Critical - fixes broken chart pipeline)
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def render_chart(
    df: pd.DataFrame,
    chart_spec: Dict[str, Any],
    save_path: str
) -> bool:
    """
    最小、稳定的 chart renderer

    成功判定：只认 save_path 文件是否存在且大小 > 0

    Args:
        df: pandas DataFrame with data
        chart_spec: Dictionary with chart configuration
            {
                "type": "line|bar|scatter|histogram",
                "x": "column_name",
                "y": "column_name" (optional for histogram),
                "title": "chart title",
                "xlabel": "x-axis label" (optional),
                "ylabel": "y-axis label" (optional)
            }
        save_path: Path where PNG should be saved

    Returns:
        bool: True if file exists and has content, False otherwise
    """
    save_path = Path(save_path)

    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract chart parameters
        chart_type = chart_spec.get("type", "line")
        x = chart_spec.get("x")
        y = chart_spec.get("y")
        title = chart_spec.get("title", "")
        xlabel = chart_spec.get("xlabel", x or "")
        ylabel = chart_spec.get("ylabel", y or "")

        # Render chart based on type
        if chart_type == "line":
            ax.plot(df[x], df[y], marker='o', linewidth=2)
        elif chart_type == "bar":
            ax.bar(df[x], df[y])
        elif chart_type == "scatter":
            ax.scatter(df[x], df[y])
        elif chart_type == "histogram":
            ax.hist(df[x].dropna(), bins=30, edgecolor='black')
            ylabel = y or "Frequency"
        else:
            # Default to line for unsupported types
            logger.warning(f"Unsupported chart type: {chart_type}, using line")
            ax.plot(df[x], df[y], marker='o', linewidth=2)

        # Set labels and title
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Tight layout
        fig.tight_layout()

        # === 唯一保存点 ===
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # === 唯一成功判定 ===
        if save_path.exists() and save_path.stat().st_size > 0:
            logger.info(f"[MINIMAL] Chart saved successfully: {save_path} ({save_path.stat().st_size} bytes)")
            return True
        else:
            logger.error(f"[MINIMAL] Chart file NOT created or empty: {save_path}")
            return False

    except Exception as e:
        logger.error(f"[MINIMAL] Chart rendering failed: {e}")
        return False


def create_chart_json(
    title: str,
    save_path: str,
    success: bool
) -> Dict[str, Any]:
    """
    创建最小、稳定的 Chart JSON 结构

    LaTeX 100% 能用的结构：
    {
        "title": "...",
        "image_path": "...",
        "success": true/false
    }

    Args:
        title: Chart title
        save_path: Path to saved PNG
        success: Whether rendering succeeded

    Returns:
        Dict with minimal chart metadata
    """
    return {
        "title": title,
        "image_path": save_path if success else None,
        "success": bool(success)
    }


def render_charts_from_specs(
    dataframes: Dict[str, pd.DataFrame],
    chart_specs: List[Dict[str, Any]],
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    批量渲染图表（用于替换 create_charts_with_images）

    Args:
        dataframes: Dictionary of {filename: DataFrame}
        chart_specs: List of chart specification dicts
        output_dir: Directory to save PNG files

    Returns:
        List of chart JSON dicts with {title, image_path, success}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    charts = []

    for i, spec in enumerate(chart_specs, start=1):
        # Determine which DataFrame to use
        csv_file = spec.get("csv_file")
        if csv_file and csv_file in dataframes:
            df = dataframes[csv_file]
        elif len(dataframes) == 1:
            # Use the only available DataFrame
            df = list(dataframes.values())[0]
        else:
            logger.error(f"[MINIMAL] Cannot determine DataFrame for chart {i}: {spec}")
            charts.append(create_chart_json(
                title=spec.get("title", f"Chart {i}"),
                save_path="",
                success=False
            ))
            continue

        # Generate save path
        filename = f"chart_{i}.png"
        save_path = output_path / filename

        # Render chart
        success = render_chart(df, spec, str(save_path))

        # Create JSON entry
        chart_json = create_chart_json(
            title=spec.get("title", f"Chart {i}"),
            save_path=str(save_path),
            success=success
        )

        charts.append(chart_json)

        # Log result
        if success:
            logger.info(f"[OK] Chart {i} generated: {save_path}")
        else:
            logger.error(f"[FAIL] Chart {i} failed to generate")

    # Log summary
    success_count = sum(1 for c in charts if c["success"])
    logger.info(f"\n{'='*60}")
    logger.info(f"Charts generated: {success_count}/{len(charts)} successful")
    logger.info(f"{'='*60}\n")

    return charts


# ============================================================================
# Integration Helper: LLM → Chart Spec
# ============================================================================

def parse_llm_chart_description(
    description: str,
    available_columns: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    将 LLM 的自然语言描述转换为 chart_spec

    这是一个轻量级解析器，不需要复杂的 LLM 调用。

    Args:
        description: LLM chart description (natural language)
        available_columns: {csv_file: [col1, col2, ...]}

    Returns:
        chart_spec dict
    """
    # Simple keyword-based extraction
    # In production, you might use LLM for this, but keep it minimal for now

    spec = {
        "type": "line",  # Default
        "title": "Chart",
        "x": None,
        "y": None
    }

    # Extract chart type
    description_lower = description.lower()
    if "bar" in description_lower:
        spec["type"] = "bar"
    elif "scatter" in description_lower:
        spec["type"] = "scatter"
    elif "histogram" in description_lower or "distribution" in description_lower:
        spec["type"] = "histogram"

    # Extract title (first sentence or phrase)
    if "." in description:
        spec["title"] = description.split(".")[0].strip()
    else:
        spec["title"] = description[:50].strip()

    # TODO: Extract x/y columns from available_columns
    # This would require more sophisticated parsing or LLM call

    return spec


if __name__ == "__main__":
    # Self-test
    import numpy as np

    print("="*60)
    print("MINIMAL CHART RENDERER - SELF-TEST")
    print("="*60)

    # Create sample data
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 15, 25, 30]
    })

    # Test 1: Line chart
    spec = {
        "type": "line",
        "x": "x",
        "y": "y",
        "title": "Test Line Chart"
    }

    success = render_chart(df, spec, "test_chart.png")
    print(f"\n[Test 1] Line chart: {'SUCCESS' if success else 'FAILED'}")

    if success:
        import os
        size = os.path.getsize("test_chart.png")
        print(f"  File size: {size} bytes")

    print("\n" + "="*60)
    print("Self-test complete!")
    print("="*60)
