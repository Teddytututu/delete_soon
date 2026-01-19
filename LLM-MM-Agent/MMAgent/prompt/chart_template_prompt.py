"""
Template-Driven Chart Generation Prompt - 45 Chart Types

Core Concept:
- LLM no longer generates matplotlib code (eliminates syntax errors)
- LLM only selects chart type and fills parameters
- System automatically renders templates to generate code

This彻底solves the syntax error problem mentioned in the architectural plan.

Version: 3.0 (45 Templates)
Date: 2026-01-19
"""

TEMPLATE_SELECTION_PROMPT = r"""You are a Lead Data Visualization Specialist. Your goal is to configure a visualization based on the user's request and the available data.

### 1. Available Data Context
${data_context}

### 2. User Request
"${user_description}"

### 3. Available Chart Templates (45 Types)

The system supports the following chart types organized by category. Choose the BEST fit for the user's request.

## Category A: Basic Charts (15 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **line** | Trends over time/continuous variables | x_column, y_column |
| **bar** | Comparing categories | x_column, y_column |
| **scatter** | Correlation between two variables | x_column, y_column |
| **histogram** | Distribution of a single variable | x_column, bins (optional) |
| **multi_line** | Multiple trends comparison | x_column, y_columns (list) |
| **stacked_bar** | Composition comparison | x_column, y_columns (list) |
| **stacked_area** | Composition over time | x_column, y_columns (list) |
| **boxplot** | Statistical distribution | x_column (category), y_column (value) |
| **violin** | Distribution density | x_column, y_column |
| **heatmap** | Correlation matrix or density | x_column, y_column, value_column |
| **pie** | Proportions (use sparingly) | label_column, value_column |
| **horizontal_bar** | Long label categories | y_column, x_column |

## Category B: Bubble & Density (3 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **bubble** | 3D relationship (x, y, size) | x_column, y_column, size_column |
| **kde** | Kernel density estimation | x_column, y_column (optional) |
| **density** | 2D density plot | x_column, y_column |

## Category C: Area & Step (3 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **area** | Filled area under line | x_column, y_column |
| **step** | Step-wise changes | x_column, y_column |
| **stem** | Discrete values with stems | x_column, y_column |

## Category D: Strip & Swarm (2 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **strip** | 1D scatter with categories | x_column (category), y_column |
| **swarm** | Categorical distribution | x_column (category), y_column |

## Category E: Error & Range (2 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **errorbar** | Means with error bars | x_column, y_column, yerr_column |
| **filled_between** | Confidence interval bands | x_column, y_min_column, y_max_column |

## Category F: Specialized (10 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **donut** | Ring chart (modern pie) | label_column, value_column |
| **lollipop** | Stem-ended bar chart | x_column, y_column |
| **radar** | Multi-variable comparison | categories_column, values_column |
| **polar** | Polar coordinate plot | theta_column, r_column |
| **hexbin** | 2D histogram bins | x_column, y_column |
| **contour** | Contour plot (3D surface) | x_column, y_column, z_column |
| **quiver** | Vector field | x_column, y_column, u_column, v_column |
| **sankey** | Flow diagram | (requires special preparation) |
| **treemap** | Hierarchical proportions | (requires special preparation) |
| **dendrogram** | Hierarchical clustering | (requires special preparation) |

## Category G: Statistical (5 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **pairplot** | All pairwise relationships | (multiple numeric columns) |
| **jointplot** | Scatter + marginal distributions | x_column, y_column |
| **clustermap** | Hierarchical clustered heatmap | (correlation matrix) |
| **regplot** | Linear regression with scatter | x_column, y_column |
| **residplot** | Residual plot | x_column, y_column |

## Category H: Time Series (2 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **lag_plot** | Time series lag correlation | column, lag (optional) |
| **autocorrelation** | Autocorrelation function | column |

## Category I: Business (3 types)

| Template ID | Description | Required Params |
|:---|:---|:---|
| **pareto** | 80/20 principle analysis | category_column, value_column |
| **timeline** | Event timeline | date_column, event_column |
| **calendar_heatmap** | Daily activity patterns | date_column, value_column |

### 4. Constraints
1. **Source of Truth**: You MUST use the exact column names from "Available Data Context". Do not invent columns like 'Year' if the data says 'YEAR'.
2. **Strict JSON**: Output ONLY a valid JSON object. No markdown, no comments.
3. **Data Type Match**: Ensure the columns you pick match the chart requirements (e.g., don't put strings on a Histogram X-axis).
4. **Choose Wisely**: Select the most appropriate chart type for the user's request and data characteristics.

### 5. Output Format (JSON)

```json
{
    "template_id": "one_of_the_45_ids_above",
    "data_file": "filename.csv",
    "parameters": {
        "x_column": "EXACT_COLUMN_NAME",
        "y_column": "EXACT_COLUMN_NAME",
        "title": "A descriptive title for the chart"
    },
    "reasoning": "Brief explanation of why you chose this chart and columns"
}
```

**Note**: For templates requiring multiple columns (e.g., multi_line), use `y_columns` (array) instead of `y_column`. For specialized parameters, add them to the parameters object.

Your response:
"""


# ============================================================================
# Helper Functions for Template Integration
# ============================================================================

def get_template_summary() -> str:
    """
    Get a quick summary of all 45 available templates.

    Returns:
        str: Formatted summary of template categories and counts
    """
    summary = """
## Template Summary (45 Total)

A. Basic Charts (15): line, bar, scatter, histogram, multi_line, stacked_bar, stacked_area, boxplot, violin, heatmap, pie, horizontal_bar, subplot_2x2, subplot_scatter_hist, time_series

B. Bubble & Density (3): bubble, kde, density

C. Area & Step (3): area, step, stem

D. Strip & Swarm (2): strip, swarm

E. Error & Range (2): errorbar, filled_between

F. Specialized (10): donut, lollipop, radar, polar, hexbin, contour, quiver, sankey, treemap, dendrogram

G. Statistical (5): pairplot, jointplot, clustermap, regplot, residplot

H. Time Series (2): lag_plot, autocorrelation

I. Business (3): pareto, timeline, calendar_heatmap
"""
    return summary


def get_template_requirements(template_id: str) -> dict:
    """
    Get required parameters for a specific template.

    Args:
        template_id: Template identifier

    Returns:
        dict: Required and optional parameters for the template
    """
    requirements = {
        # Basic Charts
        "line": {"required": ["x_column", "y_column"], "optional": ["title", "color"]},
        "bar": {"required": ["x_column", "y_column"], "optional": ["title", "color"]},
        "scatter": {"required": ["x_column", "y_column"], "optional": ["title", "color", "size_column"]},
        "histogram": {"required": ["x_column"], "optional": ["title", "bins", "color"]},
        "multi_line": {"required": ["x_column", "y_columns"], "optional": ["title"]},
        "stacked_bar": {"required": ["x_column", "y_columns"], "optional": ["title"]},
        "stacked_area": {"required": ["x_column", "y_columns"], "optional": ["title"]},
        "boxplot": {"required": ["x_column", "y_column"], "optional": ["title"]},
        "violin": {"required": ["x_column", "y_column"], "optional": ["title"]},
        "heatmap": {"required": ["x_column", "y_column", "value_column"], "optional": ["title", "cmap"]},
        "pie": {"required": ["label_column", "value_column"], "optional": ["title"]},
        "horizontal_bar": {"required": ["y_column", "x_column"], "optional": ["title"]},

        # Bubble & Density
        "bubble": {"required": ["x_column", "y_column", "size_column"], "optional": ["title", "color_column"]},
        "kde": {"required": ["x_column"], "optional": ["y_column", "title"]},
        "density": {"required": ["x_column", "y_column"], "optional": ["title", "cmap"]},

        # Area & Step
        "area": {"required": ["x_column", "y_column"], "optional": ["title", "color"]},
        "step": {"required": ["x_column", "y_column"], "optional": ["title", "color"]},
        "stem": {"required": ["x_column", "y_column"], "optional": ["title", "linefmt"]},

        # Strip & Swarm
        "strip": {"required": ["x_column", "y_column"], "optional": ["title"]},
        "swarm": {"required": ["x_column", "y_column"], "optional": ["title"]},

        # Error & Range
        "errorbar": {"required": ["x_column", "y_column", "yerr_column"], "optional": ["title"]},
        "filled_between": {"required": ["x_column", "y_min_column", "y_max_column"], "optional": ["title", "color"]},

        # Specialized
        "donut": {"required": ["label_column", "value_column"], "optional": ["title"]},
        "lollipop": {"required": ["x_column", "y_column"], "optional": ["title", "color"]},
        "radar": {"required": ["categories_column", "values_column"], "optional": ["title"]},
        "polar": {"required": ["theta_column", "r_column"], "optional": ["title"]},
        "hexbin": {"required": ["x_column", "y_column"], "optional": ["title", "gridsize"]},
        "contour": {"required": ["x_column", "y_column", "z_column"], "optional": ["title", "cmap"]},
        "quiver": {"required": ["x_column", "y_column", "u_column", "v_column"], "optional": ["title"]},
        "sankey": {"required": [], "optional": ["title"]},  # Special preparation needed
        "treemap": {"required": [], "optional": ["title"]},  # Special preparation needed
        "dendrogram": {"required": [], "optional": ["title"]},  # Special preparation needed

        # Statistical
        "pairplot": {"required": [], "optional": ["title"]},  # Uses all numeric columns
        "jointplot": {"required": ["x_column", "y_column"], "optional": ["title", "kind"]},
        "clustermap": {"required": [], "optional": ["title"]},  # Uses correlation matrix
        "regplot": {"required": ["x_column", "y_column"], "optional": ["title"]},
        "residplot": {"required": ["x_column", "y_column"], "optional": ["title"]},

        # Time Series
        "lag_plot": {"required": ["column"], "optional": ["title", "lag"]},
        "autocorrelation": {"required": ["column"], "optional": ["title"]},

        # Business
        "pareto": {"required": ["category_column", "value_column"], "optional": ["title"]},
        "timeline": {"required": ["date_column", "event_column"], "optional": ["title"]},
        "calendar_heatmap": {"required": ["date_column", "value_column"], "optional": ["title"]},
    }

    return requirements.get(template_id, {"required": [], "optional": []})


# ============================================================================
# Sources and References
# ============================================================================

"""
Research Sources for 45 Template Expansion:

1. Top 50 matplotlib Visualizations
   https://machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/

2. The Ultimate Guide to Data Visualization - Matplotlib & Seaborn
   https://medium.com/@roshmitadey/the-ultimate-guide-to-data-visualization-matplotlib-seaborn-3d230cd6f773

3. Matplotlib Official Gallery
   https://matplotlib.org/stable/gallery/index.html

4. SciencePlots - Matplotlib Styles Library
   https://github.com/garrettj403/SciencePlots

5. DataCamp - Types of Data Plots and How to Create Them in Python
   https://www.datacamp.com/tutorial/types-of-data-plots-and-how-to-create-them-in-python

6. Publication-Quality Plots in Python with Matplotlib
   https://www.fschuch.com/en/blog/2025/07/05/publication-quality-plots-in-python-with-matplotlib/
"""

# Alias for backward compatibility
TEMPLATE_DRIVEN_CHART_PROMPT = TEMPLATE_SELECTION_PROMPT

