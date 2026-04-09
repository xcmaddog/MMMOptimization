"""
visualization.py
----------------
Pareto front and mission result plots for the Mars transfer optimiser.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

OBJ_LABELS = ["TOF [days]", "Fuel consumed [kg]", "Mission cost [M USD]"]

PROP_COLORS = {
    "kerolox":  "#e05c2a",
    "hydrolox": "#2a7fe0",
    "storable": "#9b2ae0",
    "methalox": "#2ae07f",
}


def compare_propellants_2d(
    results: dict[str, dict],
    x_obj: int = 0,
    y_obj: int = 1,
) -> plt.Figure:
    """Overlay 2-D Pareto fronts for all propellants."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, res in results.items():
        pF = res["pareto_F"]
        if len(pF) == 0:
            continue
        color = PROP_COLORS.get(key, "grey")
        name  = res["propellant"].name
        ax.scatter(pF[:, x_obj], pF[:, y_obj],
                   c=color, alpha=0.75, edgecolors="k",
                   linewidths=0.3, s=45, label=name)
    ax.set_xlabel(OBJ_LABELS[x_obj], fontsize=11)
    ax.set_ylabel(OBJ_LABELS[y_obj], fontsize=11)
    ax.set_title("Pareto Front Comparison by Propellant", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def pareto_3d(
    pareto_F: np.ndarray,
    labels: list[str] | None = None,
    title: str = "3-D Pareto Front",
) -> plt.Figure:
    """3-D scatter of all three objectives, coloured by propellant if labels given."""
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    if labels is not None:
        unique = list(dict.fromkeys(labels))   # preserve order
        for key in unique:
            idx = [i for i, l in enumerate(labels) if l == key]
            color = PROP_COLORS.get(key, "grey")
            ax.scatter(pareto_F[idx, 0], pareto_F[idx, 1], pareto_F[idx, 2],
                       c=color, alpha=0.8, s=30, label=key)
        ax.legend(fontsize=8)
    else:
        sc = ax.scatter(pareto_F[:, 0], pareto_F[:, 1], pareto_F[:, 2],
                        c=pareto_F[:, 2], cmap="viridis", alpha=0.8, s=30)
        fig.colorbar(sc, ax=ax, label=OBJ_LABELS[2], shrink=0.5)

    ax.set_xlabel(OBJ_LABELS[0], fontsize=9, labelpad=8)
    ax.set_ylabel(OBJ_LABELS[1], fontsize=9, labelpad=8)
    ax.set_zlabel(OBJ_LABELS[2], fontsize=9, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def mission_summary_bar(result: dict, title: str = "") -> plt.Figure:
    """Bar chart + text summary for a single evaluated design point."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    vals   = [result["fuel_kg"], 0]   # phase3 fuel is 0 in current setup
    labels = ["Phase 2 propellant", "Phase 3 propellant"]
    bars = ax.bar(labels, vals, color=["#2a7fe0", "#e05c2a"],
                  edgecolor="k", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,.0f} kg", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Propellant mass [kg]")
    ax.set_title("Fuel Budget")
    ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)

    ax = axes[1]
    ax.axis("off")
    info = [
        ("Time of flight",  f"{result['tof_days']:.1f} days"),
        ("Fuel consumed",   f"{result['fuel_kg']:,.0f} kg"),
        ("Mission cost",    f"${result['cost_usd']/1e6:.2f} M"),
        ("Stable orbit",    str(result["feasible"])),
        ("Status",          result["status"]),
    ]
    y = 0.90
    for lbl, val in info:
        ax.text(0.02, y, f"{lbl}:", fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.52, y, val, fontsize=10, transform=ax.transAxes)
        y -= 0.17

    fig.suptitle(title or "Mission Design Point Summary",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig