"""
visualization.py
----------------
Plotting utilities for trajectory analysis and optimisation results.

Functions
---------
porkchop_plot          : ΔV contour map over (departure date, TOF)
pareto_2d              : 2-objective slice of a Pareto front
pareto_3d              : 3-objective scatter
compare_propellants_2d : Overlay Pareto fronts for all propellants (2D)
compare_propellants_3d : Same in 3D

All functions return a matplotlib Figure that can be saved or displayed.
"""

from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from astropy.time import Time
import astropy.units as u

from mars_transfer.trajectory.lambert import solve_lambert
from mars_transfer.ephemeris.ephemeris import epoch_range

# Human-readable objective labels
OBJ_LABELS = [
    "Propellant consumed [kg]",
    "Time of flight [days]",
    "Mission cost [M USD]",
]

# One colour per propellant key
PROP_COLORS = {
    "kerolox":  "#e05c2a",
    "hydrolox": "#2a7fe0",
    "storable": "#9b2ae0",
    "methalox": "#2ae07f",
}


# ---------------------------------------------------------------------------
# Porkchop plot
# ---------------------------------------------------------------------------

def porkchop_plot(
    depart_window: tuple[Time, Time],
    tof_bounds: tuple[float, float],
    n_depart: int = 80,
    n_tof: int = 80,
    dv_min: float = 3.0,
    dv_max: float = 12.0,
    n_levels: int = 25,
) -> plt.Figure:
    """
    Contour plot of total ΔV (km/s) over departure date × TOF.

    Parameters
    ----------
    depart_window : (start_epoch, end_epoch)
    tof_bounds    : (min_days, max_days)
    n_depart      : grid resolution along departure-date axis
    n_tof         : grid resolution along TOF axis
    dv_min/max    : colour-scale bounds [km/s]
    n_levels      : number of contour levels

    Returns
    -------
    matplotlib Figure
    """
    depart_epochs = epoch_range(depart_window[0], depart_window[1], n_depart)
    tof_grid      = np.linspace(tof_bounds[0], tof_bounds[1], n_tof)

    dv_grid = np.full((n_tof, n_depart), np.nan)

    for i, dep in enumerate(depart_epochs):
        for j, tof in enumerate(tof_grid):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = solve_lambert(dep, tof)
                dv_grid[j, i] = r["dv_total"]
            except Exception:
                pass

    # X axis: days since start of window
    jd0          = depart_window[0].jd
    x_days       = np.array([e.jd - jd0 for e in depart_epochs])
    start_label  = depart_window[0].datetime.strftime("%Y-%m-%d")

    # Clamp for cleaner colour scale
    dv_plot = np.clip(dv_grid, dv_min, dv_max)

    fig, ax = plt.subplots(figsize=(12, 7))
    levels  = np.linspace(dv_min, dv_max, n_levels)

    cf = ax.contourf(x_days, tof_grid, dv_plot, levels=levels, cmap="plasma_r")
    cs = ax.contour(x_days, tof_grid, dv_plot,
                    levels=levels[::3], colors="white", linewidths=0.6, alpha=0.5)
    ax.clabel(cs, fmt="%.1f km/s", fontsize=7, inline=True)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Total ΔV (TMI + MOI) [km/s]", fontsize=10)

    ax.set_xlabel(f"Days from {start_label}", fontsize=11)
    ax.set_ylabel("Time of Flight [days]", fontsize=11)
    ax.set_title("Earth → Mars Porkchop Plot", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pareto front plots
# ---------------------------------------------------------------------------

def pareto_2d(
    pareto_F: np.ndarray,
    x_obj: int = 1,
    y_obj: int = 0,
    color: str = "steelblue",
    label: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    2-D scatter of two objectives from a Pareto front.

    Parameters
    ----------
    pareto_F : (n, 3) objective array — columns [propellant, tof, cost]
    x_obj    : column index for x-axis (0, 1, or 2)
    y_obj    : column index for y-axis
    color    : marker colour
    label    : legend label (optional)
    ax       : existing Axes to draw into (creates figure if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.scatter(
        pareto_F[:, x_obj], pareto_F[:, y_obj],
        c=color, alpha=0.75, edgecolors="k", linewidths=0.4, s=45, label=label,
    )
    ax.set_xlabel(OBJ_LABELS[x_obj], fontsize=10)
    ax.set_ylabel(OBJ_LABELS[y_obj], fontsize=10)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    fig.tight_layout()
    return fig


def pareto_3d(
    pareto_F: np.ndarray,
    title: str = "3-D Pareto Front",
    color_by: int = 2,
) -> plt.Figure:
    """
    3-D scatter of all three objectives.

    Parameters
    ----------
    pareto_F : (n, 3) objective array
    title    : plot title
    color_by : which objective to encode as colour (0, 1, or 2)
    """
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        pareto_F[:, 0], pareto_F[:, 1], pareto_F[:, 2],
        c=pareto_F[:, color_by], cmap="viridis", alpha=0.8, s=35,
    )
    ax.set_xlabel(OBJ_LABELS[0], fontsize=9, labelpad=8)
    ax.set_ylabel(OBJ_LABELS[1], fontsize=9, labelpad=8)
    ax.set_zlabel(OBJ_LABELS[2], fontsize=9, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(sc, ax=ax, label=OBJ_LABELS[color_by], shrink=0.5, pad=0.1)
    fig.tight_layout()
    return fig


def compare_propellants_2d(
    results: dict[str, dict],
    x_obj: int = 1,
    y_obj: int = 0,
) -> plt.Figure:
    """
    Overlay Pareto fronts for all propellants on a single 2-D plot.

    Parameters
    ----------
    results : output of run_all_propellants() — dict keyed by propellant key
    x_obj, y_obj : objective indices for axes
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, res in results.items():
        pF = res["pareto_F"]
        if len(pF) == 0:
            continue
        name  = res["propellant"].name
        color = PROP_COLORS.get(key, "grey")
        ax.scatter(
            pF[:, x_obj], pF[:, y_obj],
            c=color, alpha=0.75, edgecolors="k", linewidths=0.3,
            s=40, label=name,
        )

    ax.set_xlabel(OBJ_LABELS[x_obj], fontsize=11)
    ax.set_ylabel(OBJ_LABELS[y_obj], fontsize=11)
    ax.set_title("Pareto Front Comparison by Propellant", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def compare_propellants_3d(results: dict[str, dict]) -> plt.Figure:
    """3-D version of compare_propellants_2d."""
    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection="3d")

    for key, res in results.items():
        pF = res["pareto_F"]
        if len(pF) == 0:
            continue
        name  = res["propellant"].name
        color = PROP_COLORS.get(key, "grey")
        ax.scatter(
            pF[:, 0], pF[:, 1], pF[:, 2],
            c=color, alpha=0.75, s=30, label=name,
        )

    ax.set_xlabel(OBJ_LABELS[0], fontsize=9, labelpad=8)
    ax.set_ylabel(OBJ_LABELS[1], fontsize=9, labelpad=8)
    ax.set_zlabel(OBJ_LABELS[2], fontsize=9, labelpad=8)
    ax.set_title("3-D Pareto Fronts by Propellant", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_design_point(result: dict, title: str = "") -> plt.Figure:
    """
    Bar chart summary of a single design-point result dict from solve_lambert().
    Useful for inspecting a specific Pareto solution.

    Parameters
    ----------
    result : dict returned by solve_lambert(), augmented with vehicle/cost info
             Expected keys: dv_tmi, dv_moi, v_inf_depart, v_inf_arrive, tof_days
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: ΔV breakdown
    ax = axes[0]
    labels = ["TMI burn\n(LEO→escape)", "MOI burn\n(capture)"]
    values = [result["dv_tmi"], result["dv_moi"]]
    bars = ax.bar(labels, values, color=["#2a7fe0", "#e05c2a"], edgecolor="k", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f} km/s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ΔV [km/s]")
    ax.set_title("ΔV Budget")
    ax.set_ylim(0, max(values) * 1.25)

    # Right: key mission numbers as text
    ax = axes[1]
    ax.axis("off")
    info = [
        ("Time of flight",    f"{result['tof_days']:.1f} days"),
        ("v∞ at Earth",       f"{result['v_inf_depart']:.3f} km/s"),
        ("v∞ at Mars",        f"{result['v_inf_arrive']:.3f} km/s"),
        ("C3",                f"{result['C3']:.2f} km²/s²"),
        ("Total ΔV",          f"{result['dv_total']:.3f} km/s"),
    ]
    y = 0.85
    for label, val in info:
        ax.text(0.05, y, f"{label}:", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0.55, y, val, fontsize=10, transform=ax.transAxes)
        y -= 0.15

    fig.suptitle(title or "Mission Design Point Summary", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
