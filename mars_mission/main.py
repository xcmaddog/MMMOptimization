"""
main.py
-------
Entry point for the 3M Aerospace Earth-to-Mars trajectory optimisation tool.

Design space summary
--------------------
The optimizer searches over 7 variables:

    launch_epoch_jd    JD 2461270–2461410   Sep 2026 – Jan 2027 window
                       (the 2D N-body sim is accurate only when Mars is
                        near the ecliptic; this window satisfies that)
    tof_days           130 – 300 days
    thrust_newtons     50 000 – 500 000 N
    m_struct_kg        2 000 – 15 000 kg   (structure + payload dry mass)
    moi_fuel_fraction  0.10 – 0.55         (fraction of fuel held for MOI)
    total_fuel_kg      5 000 – 150 000 kg
    leo_coast_days     0 – 1 days

Initial population seeding
--------------------------
NSGA-II's initial population is seeded with Lambert warm-start solutions
centred on five confirmed low-ΔV (epoch, TOF) anchor points, avoiding the
cold-start problem of finding feasible trajectories from scratch.

Recommended settings
--------------------
  Quick smoke test:  --pop 10 --gen 5   --propellant hydrolox
  Moderate run:      --pop 50 --gen 50  --workers 4
  Full run:          --pop 100 --gen 150 --workers 8

Feasibility note
----------------
A run is feasible when Phase 3 reports stable_orbit_detected=True, meaning
the rocket completed two periapsis passages around Mars within the 90-day
observation window.  This requires both a close Mars approach from Phase 2
AND a successful MOI retro-burn in Phase 3.  Larger populations and more
generations give NSGA-II more budget to discover feasible solutions.

Usage
-----
    python main.py                             # all propellants, default settings
    python main.py --propellant hydrolox       # single propellant
    python main.py --pop 50 --gen 50 --workers 4   # parallel
    python main.py --pop 10 --gen 5 --propellant hydrolox  # quick smoke test
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimizer import (
    build_objective_functions, ObjectiveFunction,
    run_single, run_all_propellants,
    merge_pareto_fronts, print_pareto_summary,
    PROPELLANTS, KNOWN_GOOD_ANCHORS,
    LOWER_BOUNDS, UPPER_BOUNDS, IDX_EPOCH, IDX_TOF,
)
from visualization.visualization import (
    compare_propellants_2d, pareto_3d, mission_summary_bar,
)
from astropy.time import Time


def _save(fig: plt.Figure, name: str, out_dir: str = "output") -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _print_design_space() -> None:
    """Print a summary of the epoch window and known-good anchors."""
    lo = Time(LOWER_BOUNDS[IDX_EPOCH], format="jd", scale="tdb").iso[:10]
    hi = Time(UPPER_BOUNDS[IDX_EPOCH], format="jd", scale="tdb").iso[:10]
    print(f"\nEpoch window : {lo} → {hi}")
    print(f"TOF range    : {LOWER_BOUNDS[IDX_TOF]:.0f} – {UPPER_BOUNDS[IDX_TOF]:.0f} days")
    print("\nKnown-good Lambert anchor points (seed starts):")
    for jd, tof in KNOWN_GOOD_ANCHORS:
        dep = Time(jd, format="jd", scale="tdb").iso[:10]
        print(f"  {dep}  TOF={tof:.0f}d")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mars transfer trajectory optimiser — 3M Aerospace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--propellant", choices=list(PROPELLANTS.keys()), default=None,
        help="Single propellant to optimise (default: all four)",
    )
    parser.add_argument("--pop",     type=int,   default=50,  help="Population size")
    parser.add_argument("--gen",     type=int,   default=50,  help="Generations")
    parser.add_argument("--workers", type=int,   default=1,   help="Parallel worker processes")
    parser.add_argument("--n-seeds", type=int,   default=20,  help="Lambert warm-start seeds")
    parser.add_argument("--seed-frac", type=float, default=0.5,
                        help="Fraction of initial population filled with seeds")
    parser.add_argument("--seed",    type=int,   default=42,  help="RNG seed")
    parser.add_argument("--out",     default="output",        help="Output directory")
    parser.add_argument("--info",    action="store_true",
                        help="Print design-space info and exit")
    args = parser.parse_args()

    if args.info:
        _print_design_space()
        return

    # ── build objective functions ──────────────────────────────────────────
    print("Building objective functions...")
    obj_fns = build_objective_functions(
        stage_sep_speed_m_s  = 50.0,
        phase3_lead_hours    = 10.0,
        moi_thrust_fraction  = 0.05,
        p1_dt_s              = 300.0,
        p2_dt_s              = 300.0,
        p3_dt_s              = 300.0,
        p2_total_days        = 380.0,
        p3_total_days        = 90.0,
    )

    # ── optimise ──────────────────────────────────────────────────────────
    run_kwargs = dict(
        pop_size   = args.pop,
        n_gen      = args.gen,
        n_workers  = args.workers,
        n_seeds    = args.n_seeds,
        seed_frac  = args.seed_frac,
        rng_seed   = args.seed,
        verbose    = True,
    )

    if args.propellant:
        results = {
            args.propellant: run_single(obj_fns[args.propellant], **run_kwargs)
        }
    else:
        results = run_all_propellants(objective_fns=obj_fns, **run_kwargs)

    # ── merge global Pareto front ──────────────────────────────────────────
    gF, gX, g_labels = merge_pareto_fronts(results)
    print_pareto_summary(gF, gX, g_labels)

    # ── figures ───────────────────────────────────────────────────────────
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

        # Print departure-date distribution of feasible solutions
        print(f"\n{len(gF)} global Pareto solutions:")
        print(f"  TOF range  : {gF[:,0].min():.1f} – {gF[:,0].max():.1f} days")
        print(f"  Fuel range : {gF[:,1].min():,.0f} – {gF[:,1].max():,.0f} kg")
        print(f"  Cost range : ${gF[:,2].min():.1f} M – ${gF[:,2].max():.1f} M")
    else:
        print(
            "\nNo feasible solutions found.\n"
            "Tips:\n"
            "  • Increase population (--pop 100) and generations (--gen 150)\n"
            "  • Add parallel workers (--workers 4) to evaluate more solutions\n"
            "  • The Sep–Oct 2026 window is the most compatible with the 2D sim\n"
            "  • Check output/ for any partial results"
        )

    print(f"\nDone. Outputs saved to: {args.out}/")


if __name__ == "__main__":
    main()