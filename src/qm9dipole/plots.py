"""Shared figure style, so every notebook and the final figures read as one set.

Colors follow the entity, never its rank: each model keeps its color in every figure, and
the quantum model (M4) is reserved slot 1. The categorical hues are the dataviz reference
palette in its validated order (colorblind-safe for adjacent series in line charts; for
scatter plots, where every pair must differ, use at most the first three). Magnitudes use
one blue ramp, light to dark. Figures are light mode, on the chart surface below.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots, in validated order: blue, orange, aqua, yellow, magenta, green,
#: violet, red.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

#: One fixed color per model. The mean baseline is a reference, so it is muted gray.
MODEL_COLORS: dict[str, str] = {
    "qkrr": SERIES[0], "rbf_krr": SERIES[1], "xgb": SERIES[2], "rf": SERIES[3],
    "ridge": SERIES[4], "linear": SERIES[5], "mean": MUTED,
}
MODEL_LABELS: dict[str, str] = {
    "qkrr": "quantum kernel ridge", "rbf_krr": "RBF kernel ridge", "xgb": "XGBoost",
    "rf": "random forest", "ridge": "ridge", "linear": "linear (OLS)", "mean": "mean",
}

#: Sequential blue ramp (steps 100 → 700) for continuous magnitudes such as μ.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "qm9_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

#: Diverging map for signed quantities such as correlations: red (−) ↔ neutral gray ↔ blue (+).
DIVERGING = LinearSegmentedColormap.from_list("qm9_diverging", ["#e34948", "#f0efec", "#2a78d6"])

#: Field roles in the EDA (CLAUDE.md rule 1): usable by headline models, or exploration only.
ROLE_COLORS: dict[str, str] = {"legal": SERIES[0], "dft": SERIES[1]}
ROLE_LABELS: dict[str, str] = {"legal": "function of Z and R (headline-legal)",
                               "dft": "DFT output (exploration only)"}

LINE_WIDTH = 1.75  # points (~2 px at 96 dpi)
MARKER_SIZE = 6.0  # points (~8 px)


def use_style() -> None:
    """Apply the shared matplotlib style: recessive solid hairline grid, thin marks."""
    for cmap in (SEQUENTIAL, DIVERGING):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        # The generic family walks this list silently, so a missing font never warns.
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
        "lines.linewidth": LINE_WIDTH, "lines.markersize": MARKER_SIZE,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_SECONDARY,
        "image.cmap": "qm9_blue",
    })
