"""
Chart Configuration System - P1-6 Enhancement

This module implements a JSON-based chart configuration system that:
1. Defines structured schemas for chart configurations
2. Validates chart configurations against schemas
3. Provides templates for converting JSON to matplotlib code
4. Enables safer, more controllable chart generation

Benefits:
- Structured: JSON schema ensures valid configurations
- Safe: Template-based code generation prevents arbitrary code
- Debuggable: JSON configs are easier to inspect and debug
- Maintainable: Templates can be updated without changing LLM prompts

Author: Engineering Enhancement
Date: 2026-01-15
Priority: P1-6 (High complexity, high impact on chart quality)
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    PIE = "pie"
    AREA = "area"


class AggregationType(Enum):
    """Data aggregation methods."""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STD = "std"
    NONE = "none"


@dataclass
class DataSource:
    """Data source configuration."""
    file: str  # Filename to load
    sheet: Optional[str] = None  # For Excel files (not used currently)
    filters: Optional[List[Dict[str, Any]]] = None  # Data filters


@dataclass
class AxisConfig:
    """Axis configuration."""
    label: str
    column: Optional[str] = None  # Column name for data
    scale: str = "linear"  # linear, log, symlog, logit
    limits: Optional[List[float]] = None  # [min, max]
    ticks: Optional[List[float]] = None  # Custom tick values
    format: Optional[str] = None  # Tick format string


@dataclass
class SeriesConfig:
    """Data series configuration."""
    name: str
    x_column: str
    y_column: str
    color: Optional[str] = None  # Color name or hex code
    marker: Optional[str] = None  # Marker style (o, s, ^, etc.)
    linestyle: Optional[str] = None  # -, --, -., :
    alpha: float = 1.0  # Transparency (0-1)
    aggregation: str = "none"  # Aggregation method


@dataclass
class LegendConfig:
    """Legend configuration."""
    show: bool = True
    position: str = "best"  # best, upper right, upper left, etc.
    title: Optional[str] = None
    fontsize: int = 10


@dataclass
class ChartConfig:
    """
    Complete chart configuration.

    This defines the structure of chart JSON configurations.
    LLM generates JSON matching this structure, then templates convert it to code.
    """

    # Required fields
    chart_type: str  # Must match ChartType enum
    title: str
    width: int = 10
    height: int = 6
    dpi: int = 100

    # Data sources
    data_source: Optional[DataSource] = None

    # Axes
    x_axis: Optional[AxisConfig] = None
    y_axis: Optional[AxisConfig] = None
    color_axis: Optional[AxisConfig] = None  # For color-coded plots

    # Series (for plots with multiple series)
    series: List[SeriesConfig] = field(default_factory=list)

    # Styling
    color_palette: Optional[str] = None  # viridis, plasma, inferno, etc.
    grid: bool = True
    legend: Optional[LegendConfig] = None

    # Annotations
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    # Additional matplotlib parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChartConfig':
        """Create from dictionary."""
        # Handle nested objects
        if 'data_source' in data and data['data_source']:
            data['data_source'] = DataSource(**data['data_source'])

        if 'x_axis' in data and data['x_axis']:
            data['x_axis'] = AxisConfig(**data['x_axis'])

        if 'y_axis' in data and data['y_axis']:
            data['y_axis'] = AxisConfig(**data['y_axis'])

        if 'color_axis' in data and data['color_axis']:
            data['color_axis'] = AxisConfig(**data['color_axis'])

        if 'series' in data and data['series']:
            data['series'] = [SeriesConfig(**s) for s in data['series']]

        if 'legend' in data and data['legend']:
            data['legend'] = LegendConfig(**data['legend'])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ChartConfig':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate configuration.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check chart type
        try:
            ChartType(self.chart_type)
        except ValueError:
            errors.append(f"Invalid chart_type: {self.chart_type}. Must be one of {[t.value for t in ChartType]}")

        # Check dimensions
        if self.width <= 0 or self.height <= 0:
            errors.append(f"Invalid dimensions: {self.width}x{self.height}")

        # Check data source
        if not self.data_source and not self.series:
            errors.append("Either data_source or series must be specified")

        # Check series configuration
        if self.series:
            for i, s in enumerate(self.series):
                if not s.x_column or not s.y_column:
                    errors.append(f"Series {i}: x_column and y_column are required")

        return len(errors) == 0, errors


class ChartTemplateRenderer:
    """
    Renders chart configurations into matplotlib code.

    This uses templates to convert JSON configs into executable Python code.
    Templates are safer than direct LLM code generation.
    """

    def __init__(self):
        """Initialize template renderer."""
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load chart type templates."""
        return {
            "bar": self._bar_template,
            "line": self._line_template,
            "scatter": self._scatter_template,
            "histogram": self._histogram_template,
            "box": self._box_template,
        }

    def render(self, config: ChartConfig, save_path: str) -> str:
        """
        Render chart configuration to Python code.

        Args:
            config: Chart configuration
            save_path: Path to save the chart

        Returns:
            Python code string
        """
        # Validate config
        is_valid, errors = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid chart configuration: {errors}")

        # Get template for chart type
        template_func = self.templates.get(config.chart_type)
        if not template_func:
            raise ValueError(f"Unsupported chart type: {config.chart_type}")

        # Render template
        code = template_func(config, save_path)
        return code

    def _bar_template(self, config: ChartConfig, save_path: str) -> str:
        """Template for bar charts."""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "import numpy as np",
            "",
            f"# Chart: {config.title}",
            f"# Type: Bar Chart",
            "",
        ]

        # Data loading
        if config.data_source:
            code_lines.append(f"df = load_csv('{config.data_source.file}')")
            code_lines.append("")

        # Figure setup
        code_lines.extend([
            f"fig, ax = plt.subplots(figsize=({config.width}, {config.height}))",
            "",
        ])

        # Plot based on series configuration
        if config.series:
            for series in config.series:
                if series.aggregation != "none":
                    code_lines.append(
                        f"# Aggregate data: {series.y_column} by {series.x_column}"
                    )
                    code_lines.append(
                        f"agg_data = df.groupby('{series.x_column}')['{series.y_column}'].{series.aggregation}()"
                    )
                    code_lines.append(
                        f"ax.bar(agg_data.index, agg_data.values, label='{series.name}')"
                    )
                else:
                    code_lines.append(
                        f"ax.bar(df['{series.x_column}'], df['{series.y_column}'], label='{series.name}')"
                    )
        else:
            # Simple bar chart
            x_col = config.x_axis.column if config.x_axis else "x"
            y_col = config.y_axis.column if config.y_axis else "y"
            code_lines.append(f"ax.bar(df['{x_col}'], df['{y_col}'])")

        # Styling
        code_lines.extend(self._styling_template(config))

        # Save
        code_lines.extend([
            "",
            f"plt.savefig('{save_path}', bbox_inches='tight', dpi={config.dpi})",
            "plt.close()",
        ])

        return "\n".join(code_lines)

    def _line_template(self, config: ChartConfig, save_path: str) -> str:
        """Template for line charts."""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "import numpy as np",
            "",
            f"# Chart: {config.title}",
            f"# Type: Line Chart",
            "",
        ]

        # Data loading
        if config.data_source:
            code_lines.append(f"df = load_csv('{config.data_source.file}')")
            code_lines.append("")

        # Figure setup
        code_lines.extend([
            f"fig, ax = plt.subplots(figsize=({config.width}, {config.height}))",
            "",
        ])

        # Plot series
        if config.series:
            for series in config.series:
                line_style = series.linestyle or "-"
                marker = series.marker or ""
                color = f", color='{series.color}'" if series.color else ""

                code_lines.append(
                    f"ax.plot(df['{series.x_column}'], df['{series.y_column}'], "
                    f"label='{series.name}', linestyle='{line_style}', marker='{marker}'{color})"
                )

        # Styling
        code_lines.extend(self._styling_template(config))

        # Save
        code_lines.extend([
            "",
            f"plt.savefig('{save_path}', bbox_inches='tight', dpi={config.dpi})",
            "plt.close()",
        ])

        return "\n".join(code_lines)

    def _scatter_template(self, config: ChartConfig, save_path: str) -> str:
        """Template for scatter plots."""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "import numpy as np",
            "",
            f"# Chart: {config.title}",
            f"# Type: Scatter Plot",
            "",
        ]

        # Data loading
        if config.data_source:
            code_lines.append(f"df = load_csv('{config.data_source.file}')")
            code_lines.append("")

        # Figure setup
        code_lines.extend([
            f"fig, ax = plt.subplots(figsize=({config.width}, {config.height}))",
            "",
        ])

        # Plot series
        if config.series:
            for series in config.series:
                color = f", c='{series.color}'" if series.color else ""
                marker = series.marker or "o"
                alpha = series.alpha

                code_lines.append(
                    f"ax.scatter(df['{series.x_column}'], df['{series.y_column}'], "
                    f"label='{series.name}', marker='{marker}', alpha={alpha}{color})"
                )

        # Styling
        code_lines.extend(self._styling_template(config))

        # Save
        code_lines.extend([
            "",
            f"plt.savefig('{save_path}', bbox_inches='tight', dpi={config.dpi})",
            "plt.close()",
        ])

        return "\n".join(code_lines)

    def _histogram_template(self, config: ChartConfig, save_path: str) -> str:
        """Template for histograms."""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "import numpy as np",
            "",
            f"# Chart: {config.title}",
            f"# Type: Histogram",
            "",
        ]

        # Data loading
        if config.data_source:
            code_lines.append(f"df = load_csv('{config.data_source.file}')")
            code_lines.append("")

        # Figure setup
        code_lines.extend([
            f"fig, ax = plt.subplots(figsize=({config.width}, {config.height}))",
            "",
        ])

        # Plot histogram
        if config.series:
            for series in config.series:
                bins = config.extra_params.get('bins', 30)
                code_lines.append(
                    f"ax.hist(df['{series.y_column}'].dropna(), bins={bins}, "
                    f"label='{series.name}', alpha={series.alpha})"
                )
        elif config.y_axis and config.y_axis.column:
            bins = config.extra_params.get('bins', 30)
            code_lines.append(
                f"ax.hist(df['{config.y_axis.column}'].dropna(), bins={bins})"
            )

        # Styling
        code_lines.extend(self._styling_template(config))

        # Save
        code_lines.extend([
            "",
            f"plt.savefig('{save_path}', bbox_inches='tight', dpi={config.dpi})",
            "plt.close()",
        ])

        return "\n".join(code_lines)

    def _box_template(self, config: ChartConfig, save_path: str) -> str:
        """Template for box plots."""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "import numpy as np",
            "",
            f"# Chart: {config.title}",
            f"# Type: Box Plot",
            "",
        ]

        # Data loading
        if config.data_source:
            code_lines.append(f"df = load_csv('{config.data_source.file}')")
            code_lines.append("")

        # Figure setup
        code_lines.extend([
            f"fig, ax = plt.subplots(figsize=({config.width}, {config.height}))",
            "",
        ])

        # Plot box plot
        if config.series:
            data_to_plot = []
            labels = []
            for series in config.series:
                data_to_plot.append(f"df['{series.y_column}'].dropna()")
                labels.append(series.name)

            code_lines.append(f"data_to_plot = [{', '.join(data_to_plot)}]")
            code_lines.append(f"labels = {labels}")
            code_lines.append("ax.boxplot(data_to_plot, labels=labels)")
        elif config.y_axis and config.y_axis.column:
            code_lines.append(f"ax.boxplot(df['{config.y_axis.column}'].dropna())")

        # Styling
        code_lines.extend(self._styling_template(config))

        # Save
        code_lines.extend([
            "",
            f"plt.savefig('{save_path}', bbox_inches='tight', dpi={config.dpi})",
            "plt.close()",
        ])

        return "\n".join(code_lines)

    def _styling_template(self, config: ChartConfig) -> List[str]:
        """Generate styling code snippets."""
        lines = []

        # Title
        if config.title:
            lines.append(f"ax.set_title('{config.title}')")

        # Axis labels
        if config.x_axis and config.x_axis.label:
            lines.append(f"ax.set_xlabel('{config.x_axis.label}')")
        if config.y_axis and config.y_axis.label:
            lines.append(f"ax.set_ylabel('{config.y_axis.label}')")

        # Grid
        if config.grid:
            lines.append("ax.grid(True)")

        # Legend
        if config.legend and config.legend.show:
            lines.append(f"ax.legend(loc='{config.legend.position}')")

        # Limits
        if config.x_axis and config.x_axis.limits:
            lines.append(f"ax.set_xlim({config.x_axis.limits[0]}, {config.x_axis.limits[1]})")
        if config.y_axis and config.y_axis.limits:
            lines.append(f"ax.set_ylim({config.y_axis.limits[0]}, {config.y_axis.limits[1]})")

        # Scale
        if config.x_axis and config.x_axis.scale != "linear":
            lines.append(f"ax.set_xscale('{config.x_axis.scale}')")
        if config.y_axis and config.y_axis.scale != "linear":
            lines.append(f"ax.set_yscale('{config.y_axis.scale}')")

        return lines


def validate_chart_json(json_str: str) -> tuple[bool, List[str], Optional[ChartConfig]]:
    """
    Validate chart JSON configuration.

    Args:
        json_str: JSON string to validate

    Returns:
        (is_valid, errors, config_object)
    """
    try:
        config = ChartConfig.from_json(json_str)
        is_valid, errors = config.validate()
        return is_valid, errors, config if is_valid else None
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"], None
    except Exception as e:
        return False, [f"Parse error: {e}"], None
