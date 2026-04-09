"""
main.py
-------
Entry point for the 3M Aerospace Mars transfer multi-objective optimisation.

Usage
-----
    python main.py                        # full sweep, all propellants
    python main.py --propellant hydrolox  # single propellant
    python main.py --workers 4            # parallel (4 processes)
    python main.py --pop 20 --gen 5       # quick smoke test

Output figures are saved to ./output/.
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimizer.objective import build_objective_functions, ObjectiveFunction
from optimizer.runner import (
    run_single,
    run_all_propellants,
    merge_pareto_fronts,
    print_pareto_summary,
)
from optimizer.propellants import PROPELLANTS
from visualization.visualization import (
    compare_propellants_2d,
    pareto_3d,
    mission_summary_bar,
)


def _save(fig: plt.Figure, name: str, out_dir: str = "output") -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mars transfer trajectory optimiser")
    parser.add_argument("--propellant", choices=list(PROPELLANTS.keys()),
                        default=None, help="Run one propellant only")
    parser.add_argument("--pop",     type=int,   default=50,   help="Population size")
    parser.add_argument("--gen",     type=int,   default=30,   help="Generations")
    parser.add_argument("--workers", type=int,   default=1,    help="Parallel workers")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--out",     default="output")
    args = parser.parse_args()

    # --- Build objective functions (one per propellant) ---
    print("Building objective functions...")
    obj_fns = build_objective_functions(
        payload_mass_kg    = 5_000.0,
        structural_mass_kg = 8_000.0,
        stage_sep_speed_m_s= 50.0,
        phase3_lead_hours  = 10.0,
        p1_dt_s            = 180.0,
        p2_dt_s            = 180.0,
        p3_dt_s            = 180.0,
        p2_total_days      = 300.0,
        p3_total_days      = 10.0,
    )

    # --- Optimise ---
    if args.propellant:
        results = {
            args.propellant: run_single(
                obj_fns[args.propellant],
                pop_size=args.pop,
                n_gen=args.gen,
                n_workers=args.workers,
                seed=args.seed,
            )
        }
    else:
        results = run_all_propellants(
            objective_fns=obj_fns,
            pop_size=args.pop,
            n_gen=args.gen,
            n_workers=args.workers,
            seed=args.seed,
        )

    # --- Merge global front ---
    gF, gX, g_labels = merge_pareto_fronts(results)
    print_pareto_summary(gF, gX, g_labels)

    # --- Figures ---
    if len(results) > 1:
        for x_obj, y_obj, fname in [
            (0, 1, "pareto_tof_vs_fuel.png"),
            (0, 2, "pareto_tof_vs_cost.png"),
            (1, 2, "pareto_fuel_vs_cost.png"),
        ]:
            fig = compare_propellants_2d(results, x_obj=x_obj, y_obj=y_obj)
            _save(fig, fname, args.out)

    if len(gF) > 0:
        fig = pareto_3d(gF, labels=g_labels, title="Global Pareto Front")
        _save(fig, "global_pareto_3d.png", args.out)

    print(f"\nDone. Outputs saved to: {args.out}/")


if __name__ == "__main__":
    main()