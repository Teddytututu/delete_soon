"""
模板驱动图表生成 Prompt - 问题2修复

核心思想：
- LLM 不再生成 matplotlib 代码
- 只能选择图表类型和填充变量
- 系统自动渲染模板生成代码

这彻底解决了问题2中提到的语法错误问题。
"""

TEMPLATE_DRIVEN_CHART_PROMPT = r"""You are a chart specification expert. Your task is to analyze a chart description and extract the necessary parameters to generate it using pre-defined templates.

## Chart Description

$chart_description

## Available Data Files
$data_files_info

## Your Task - Extract Chart Parameters (DO NOT write Python code!)

You MUST respond with a JSON object containing these parameters:

```json
{
  "chart_type": "one of: line, bar, scatter, histogram, multi_line",
  "csv_file": "filename.csv from the available files above",
  "x_column": "column name for x-axis",
  "y_columns": ["column", "names", "for", "y-axis"],
  "title": "chart title"
}
```

## Chart Types and Their Uses

**1. line** - Line Chart
   - Best for: Time series, trends over time
   - X-axis: Usually time or year column
   - Y-axis: One numeric column

**2. bar** - Bar Chart
   - Best for: Comparing categories
   - X-axis: Category column
   - Y-axis: One numeric column

**3. scatter** - Scatter Plot
   - Best for: Showing relationships between two variables
   - X-axis: One numeric column
   - Y-axis: One numeric column

**4. histogram** - Histogram
   - Best for: Distribution analysis
   - X-axis: One numeric column
   - Y-columns: Not needed (auto-calculated)

**5. multi_line** - Multi-Line Chart
   - Best for: Comparing multiple trends over time
   - X-axis: Time or year column
   - Y-columns: Multiple numeric columns (2-5 max)

## CRITICAL RULES (问题2修复 - Template-Based Generation)

### ❌ FORBIDDEN - DO NOT DO THIS:
- ❌ Write matplotlib code yourself
- ❌ Use plt.plot(), ax.bar(), etc.
- ❌ Create custom visualizations
- ❌ Use arbitrary Python structures

### ✅ MANDATORY - YOU MUST:
1. **ONLY** choose from the 5 chart types listed above
2. **ONLY** use columns that exist in the available CSV files
3. **EXTRACT** parameters from the chart description
4. **RESPOND** with valid JSON only

## Column Selection Guidelines

1. **Check available columns first** - Use only columns that exist
2. **Match chart description intent** - What relationship is being shown?
3. **Choose appropriate chart type** - Use the guidelines above
4. **Handle missing data gracefully** - If a column doesn't exist, use alternatives

## Examples

### Example 1: Time Series Trend
Chart description: "Show how athlete participation changed over years"

Response:
```json
{
  "chart_type": "line",
  "csv_file": "clean_athletes.csv",
  "x_column": "YEAR",
  "y_columns": ["ID"],
  "title": "Athlete Participation Over Time"
}
```

### Example 2: Category Comparison
Chart description: "Compare medal counts across countries"

Response:
```json
{
  "chart_type": "bar",
  "csv_file": "clean_medal_counts.csv",
  "x_column": "NOC",
  "y_columns": ["TOTAL"],
  "title": "Medal Counts by Country"
}
```

### Example 3: Distribution
Chart description: "Show distribution of athlete ages"

Response:
```json
{
  "chart_type": "histogram",
  "csv_file": "clean_athletes.csv",
  "x_column": "AGE",
  "y_columns": [],
  "title": "Athlete Age Distribution"
}
```

## Response Format

**IMPORTANT**:
1. Respond ONLY with the JSON object
2. Do NOT include ```json``` markdown code blocks
3. Do NOT add any explanatory text
4. Ensure valid JSON syntax

Your response:
"""


# ============================================================================
# Template Selection Guide (Internal Use)
# ============================================================================

"""
Sources:
- [Matplotlib Official Examples Gallery](https://matplotlib.org/stable/gallery/index.html)
- [GeeksforGeeks - savefig() in Python](https://www.geeksforgeeks.org/python/matplotlib-pyplot-savefig-in-python/)
- [Free Templates - 365datascience](https://365datascience.com/resources-center/templates/bar-and-line-chart-with-matplotlib-in-python/)
"""
