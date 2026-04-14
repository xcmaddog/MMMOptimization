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
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimizer import (
    build_objective_functions, ObjectiveFunction,
    DESIGN_VARIABLE_SPEC,
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



# ── Pareto reporting helpers ──────────────────────────────────────────────────

_VAR_NAMES = [s["name"] for s in DESIGN_VARIABLE_SPEC]


def _print_pareto_design_vars(F: "np.ndarray", X: "np.ndarray", labels: list[str]) -> None:
    """Print a table of objectives + design variables for every Pareto solution."""
    from astropy.time import Time
    print()
    print("Global Pareto Front — full design vectors")
    print("=" * 100)
    header = (
        f"{'#':>3}  {'Propellant':12}  {'TOF[d]':>7}  {'Fuel[kg]':>10}  "
        f"{'Cost[M$]':>9}  {'Epoch':>12}  {'tof_dv':>7}  {'Thrust[N]':>10}  "
        f"{'Struct[kg]':>10}  {'MOI_frac':>8}  {'TFuel[kg]':>10}  {'Coast[d]':>8}"
    )
    print(header)
    print("-" * 100)
    for i, (f_row, x_row, lbl) in enumerate(zip(F, X, labels)):
        tof_days, fuel_kg, cost_M = f_row
        try:
            dep_date = Time(float(x_row[0]), format="jd", scale="tdb").iso[:10]
        except Exception:
            dep_date = f"{x_row[0]:.1f}"
        print(
            f"{i+1:>3}  {lbl:12}  {tof_days:>7.1f}  {fuel_kg:>10,.0f}  "
            f"{cost_M:>9.2f}  {dep_date:>12}  {x_row[1]:>7.1f}  {x_row[2]:>10.0f}  "
            f"{x_row[3]:>10.0f}  {x_row[4]:>8.3f}  {x_row[5]:>10.0f}  {x_row[6]:>8.3f}"
        )
    print("=" * 100)


def _save_pareto_csv(F: "np.ndarray", X: "np.ndarray", labels: list[str],
                     out_dir: str) -> None:
    """Save the Pareto front (objectives + design variables) to a CSV file."""
    import csv as _csv
    from astropy.time import Time
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pareto_solutions.csv")
    obj_names = ["tof_days", "fuel_kg", "cost_MUSD"]
    with open(path, "w", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow(["solution", "propellant"] + obj_names + _VAR_NAMES + ["departure_date"])
        for i, (f_row, x_row, lbl) in enumerate(zip(F, X, labels)):
            try:
                dep = Time(float(x_row[0]), format="jd", scale="tdb").iso[:10]
            except Exception:
                dep = str(x_row[0])
            writer.writerow(
                [i + 1, lbl]
                + [f"{v:.6g}" for v in f_row]
                + [f"{v:.6g}" for v in x_row]
                + [dep]
            )
    print(f"  Saved {path}")


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
    parser.add_argument(
        "--no-feasible-seeds", action="store_true",
        help="Disable simulator-verified feasible seeding (faster, less reliable)",
    )
    parser.add_argument(
        "--seed-eval-budget", type=int, default=120,
        help="Max seed evaluations for feasible seeding (per propellant)",
    )
    parser.add_argument(
        "--seed-max-seconds", type=float, default=60.0,
        help="Max wall-clock seconds to spend generating feasible seeds per propellant",
    )
    parser.add_argument(
        "--seed-quiet", action="store_true",
        help="Suppress feasible-seed progress logging",
    )
    parser.add_argument(
        "--seed-cache-only", action="store_true",
        help="Only use cached feasible seeds (no new simulator evaluations)",
    )
    parser.add_argument(
        "--seed-min-eval-seconds", type=float, default=5.0,
        help="Minimum remaining seconds required to start a new seed evaluation",
    )
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
        seed_feasible = not args.no_feasible_seeds,
        seed_eval_budget = args.seed_eval_budget,
        seed_max_seconds = args.seed_max_seconds,
        seed_verbose = not args.seed_quiet,
        seed_cache_only = args.seed_cache_only,
        seed_min_eval_seconds = args.seed_min_eval_seconds,
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

        print(f"\n{len(gF)} global Pareto solutions:")
        print(f"  TOF range  : {gF[:,0].min():.1f} – {gF[:,0].max():.1f} days")
        print(f"  Fuel range : {gF[:,1].min():,.0f} – {gF[:,1].max():,.0f} kg")
        print(f"  Cost range : ${gF[:,2].min():.1f} M – ${gF[:,2].max():.1f} M")
        _print_pareto_design_vars(gF, gX, g_labels)
        _save_pareto_csv(gF, gX, g_labels, args.out)
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
