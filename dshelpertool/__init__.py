# coding: utf-8
"""
DSHelperTool: A lightweight helper package for data science and analysis tasks.
"""

__version__ = "0.1.0"

from . import cleaning, dtypes, easter_egg, eda, overview, report, stats, viz
from .cleaning import drop_cols, update_col
from .dtypes import to_datetime_cols, to_numeric_cols
from .easter_egg import fortune_cookie
from .eda import check_skew, fill_missing, plot_skew

# Import commonly used functions for easier access
from .overview import get_duplicates, quick_look, value_counts_all
from .report import export_to_excel, generate_data_dictionary, generate_summary_report
from .stats import (
    correlation_matrix,
    describe_all,
    group_stats,
    hypothesis_test,
    outlier_detection,
)
from .viz import (
    plot_boxplots,
    plot_categorical,
    plot_correlation,
    plot_distributions,
    plot_group_comparison,
    plot_missing,
    plot_scatter_matrix,
    plot_time_series,
)
