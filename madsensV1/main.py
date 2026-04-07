"""
main.py
-------
Entry point for the 3M Aerospace Earth-to-Mars trajectory optimisation tool.

Runs four cases defined in the project spec, saves figures to ./output/.

Usage
-----
    python main.py                  # full run, all propellants
    python main.py --porkchop-only  # just generate the porkchop plot
    python main.py --propellant hydrolox  # single propellant
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works everywhere
import matplotlib.pyplot as plt

from mars_transfer.ephemeris.ephemeris import epoch_from_date
from mars_transfer.vehicle.vehicle import VehicleConfig, PROPELLANTS
from mars_transfer.optimization.runner import (
    run_single,
    run_all_propellants,
    merge_pareto_fronts,
)
from mars_transfer.visualization.visualization import (
    porkchop_plot,
    compare_propellants_2d,
    compare_propellants_3d,
    pareto_3d,
    plot_design_point,
)
from mars_transfer.trajectory.lambert import solve_lambert

# ---------------------------------------------------------------------------
# Mission parameters — edit here to change the scenario
# ---------------------------------------------------------------------------

# 2026-2027 Earth-Mars launch window (next favourable synodic alignment)
DEPART_START = epoch_from_date(2026, 9,  1)
DEPART_END   = epoch_from_date(2027, 3, 31)
TOF_BOUNDS   = (100.0, 400.0)   # days

# Base vehicle (propellant field is overridden per run in run_all_propellants)
BASE_VEHICLE = VehicleConfig(
    payload_mass_kg    = 5_000.0,    # kg
    structural_mass_kg = 8_000.0,    # kg
    propellant         = PROPELLANTS["hydrolox"],
    max_propellant_kg  = 150_000.0,  # kg
    max_g_loading      = 4.0,
)

# NSGA-II settings
POP_SIZE = 200
N_GEN    = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, out_dir: str = "output") -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main cases
# ---------------------------------------------------------------------------

def case_porkchop(out_dir: str = "output") -> None:
    """Generate a porkchop plot for the 2026-2027 window."""
    print("\n[Case 0] Porkchop plot...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = porkchop_plot(
            depart_window=(DEPART_START, DEPART_END),
            tof_bounds=TOF_BOUNDS,
            n_depart=80,
            n_tof=80,
        )
    _save(fig, "porkchop.png", out_dir)


def case_single_propellant(prop_key: str, out_dir: str = "output") -> dict:
    """Run optimisation for one propellant and save plots."""
    from dataclasses import replace
    prop    = PROPELLANTS[prop_key]
    vehicle = replace(BASE_VEHICLE, propellant=prop)

    print(f"\n[Case] Propellant: {prop.name}")
    result = run_single(
        vehicle, (DEPART_START, DEPART_END), TOF_BOUNDS,
        pop_size=POP_SIZE, n_gen=N_GEN, verbose=True,
    )

    pF = result["pareto_F"]
    if len(pF) == 0:
        print("  No feasible solutions found.")
        return result

    # Fuel vs Time
    fig = plt.figure(figsize=(8, 5))
    ax  = fig.add_subplot(111)
    ax.scatter(pF[:, 1], pF[:, 0], c="#2a7fe0", edgecolors="k", linewidths=0.4, s=40)
    ax.set_xlabel("Time of Flight [days]"); ax.set_ylabel("Propellant consumed [kg]")
    ax.set_title(f"Pareto Front — {prop.name}"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"pareto_{prop_key}_fuel_vs_time.png", out_dir)

    # 3D
    fig = pareto_3d(pF, title=f"3-D Pareto — {prop.name}")
    _save(fig, f"pareto_{prop_key}_3d.png", out_dir)

    return result


def case_all_propellants(out_dir: str = "output") -> dict[str, dict]:
    """Run optimisation for all propellants and produce comparison plots."""
    print("\n[Case] All propellants — full Pareto sweep")
    results = run_all_propellants(
        BASE_VEHICLE, (DEPART_START, DEPART_END), TOF_BOUNDS,
        pop_size=POP_SIZE, n_gen=N_GEN, verbose=True,
    )

    # Fuel vs Time comparison
    fig = compare_propellants_2d(results, x_obj=1, y_obj=0)
    _save(fig, "compare_fuel_vs_time.png", out_dir)

    # Cost vs Time comparison
    fig = compare_propellants_2d(results, x_obj=1, y_obj=2)
    _save(fig, "compare_cost_vs_time.png", out_dir)

    # Cost vs Fuel comparison
    fig = compare_propellants_2d(results, x_obj=0, y_obj=2)
    _save(fig, "compare_cost_vs_fuel.png", out_dir)

    # 3D comparison
    fig = compare_propellants_3d(results)
    _save(fig, "compare_3d.png", out_dir)

    # Global Pareto front across all propellants
    gF, gX, labels = merge_pareto_fronts(results)
    if len(gF) > 0:
        fig = pareto_3d(gF, title="Global Pareto Front (all propellants)")
        _save(fig, "global_pareto_3d.png", out_dir)

    # Print summary table
    print("\n--- Summary: Extremes of the global Pareto front ---")
    if len(gF) > 0:
        _print_extreme(gF, gX, labels, obj=0, label="Min propellant")
        _print_extreme(gF, gX, labels, obj=1, label="Min time (sprint)")
        _print_extreme(gF, gX, labels, obj=2, label="Min cost")

    return results


def _print_extreme(F, X, labels, obj, label):
    idx = np.argmin(F[:, obj])
    t_jd, tof, m_prop = X[idx]
    prop_kg, tof_days, cost_M = F[idx]
    dep = Time(t_jd, format="jd", scale="tdb")
    print(f"\n  {label} [{labels[idx]}]")
    print(f"    Departure  : {dep.iso[:10]}")
    print(f"    TOF        : {tof_days:.1f} days")
    print(f"    Propellant : {prop_kg:,.0f} kg")
    print(f"    Cost       : ${cost_M:.2f} M")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import numpy as np
    from astropy.time import Time

    # make _print_extreme work (needs np and Time in scope)
    globals()["np"] = np
    globals()["Time"] = Time

    parser = argparse.ArgumentParser(description="Mars transfer trajectory optimiser")
    parser.add_argument("--porkchop-only", action="store_true",
                        help="Only generate the porkchop plot")
    parser.add_argument("--propellant", choices=list(PROPELLANTS.keys()), default=None,
                        help="Run a single propellant instead of all")
    parser.add_argument("--out", default="output", help="Output directory")
    args = parser.parse_args()

    case_porkchop(args.out)

    if args.porkchop_only:
        return

    if args.propellant:
        case_single_propellant(args.propellant, args.out)
    else:
        case_all_propellants(args.out)

    print("\nDone. Figures saved to:", args.out)


if __name__ == "__main__":
    main()
