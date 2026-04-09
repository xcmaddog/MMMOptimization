"""
runner.py
---------
NSGA-II optimisation runner.

Parallelism is handled inside MarsTransferProblem._evaluate() via
concurrent.futures.ProcessPoolExecutor, so this file has no direct
dependency on any pymoo parallel infrastructure.

Typical usage
-------------
    from optimizer.runner import run_all_propellants
    from optimizer.objective import build_objective_functions

    obj_fns = build_objective_functions(
        payload_mass_kg=5_000,
        p2_total_days=300,
        p3_total_days=10,
    )
    results = run_all_propellants(
        obj_fns,
        pop_size=50,
        n_gen=30,
        n_workers=4,
    )

Note on n_workers
-----------------
Each worker is a separate OS process, so there is no GIL contention.
A safe default is os.cpu_count() - 1.
Set n_workers=1 for serial execution (recommended when debugging).
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from optimizer.objective import ObjectiveFunction, build_objective_functions
from optimizer.problem import MarsTransferProblem
from optimizer.propellants import PROPELLANTS, Propellant


# ---------------------------------------------------------------------------
# Single-propellant run
# ---------------------------------------------------------------------------

def run_single(
    objective_fn: ObjectiveFunction,
    pop_size: int = 50,
    n_gen: int = 30,
    n_workers: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run NSGA-II for one propellant / ObjectiveFunction.

    Parameters
    ----------
    objective_fn : ObjectiveFunction
    pop_size     : NSGA-II population size
    n_gen        : number of generations
    n_workers    : parallel worker processes (1 = serial)
    seed         : RNG seed
    verbose      : print progress summary

    Returns
    -------
    dict with keys:
        pareto_F   : np.ndarray (n, 3)  — [tof_days, fuel_kg, cost_MUSD]
        pareto_X   : np.ndarray (n, N_VAR) — design vectors on the Pareto front
        result     : raw pymoo Result object
        propellant : Propellant used
        runtime_s  : wall-clock time [s]
    """
    prop = objective_fn.propellant
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Propellant : {prop.name}")
        print(f"  pop={pop_size}  gen={n_gen}  workers={n_workers}  seed={seed}")
        print(f"{'='*60}")

    problem = MarsTransferProblem(objective_fn, n_workers=n_workers)
    algo    = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    term    = get_termination("n_gen", n_gen)

    t0 = time.perf_counter()
    res = minimize(problem, algo, term, seed=seed, verbose=False)
    runtime_s = time.perf_counter() - t0

    pareto_F = res.F if res.F is not None else np.empty((0, 3))
    pareto_X = res.X if res.X is not None else np.empty((0, 9))

    if verbose:
        print(f"  Done in {runtime_s:.1f} s  |  Pareto solutions: {len(pareto_F)}")

    return {
        "pareto_F":   pareto_F,
        "pareto_X":   pareto_X,
        "result":     res,
        "propellant": prop,
        "runtime_s":  runtime_s,
    }


# ---------------------------------------------------------------------------
# Full propellant sweep
# ---------------------------------------------------------------------------

def run_all_propellants(
    objective_fns: dict[str, ObjectiveFunction] | None = None,
    pop_size: int = 50,
    n_gen: int = 30,
    n_workers: int = 1,
    seed: int = 42,
    verbose: bool = True,
    **obj_fn_kwargs,
) -> dict[str, dict]:
    """
    Run NSGA-II for every propellant in the catalogue.

    If objective_fns is None, they are built via
    build_objective_functions(**obj_fn_kwargs).

    Returns
    -------
    dict keyed by propellant key — each value is a run_single() result dict.
    """
    if objective_fns is None:
        objective_fns = build_objective_functions(**obj_fn_kwargs)

    results = {}
    for key, obj_fn in objective_fns.items():
        results[key] = run_single(
            obj_fn,
            pop_size=pop_size,
            n_gen=n_gen,
            n_workers=n_workers,
            seed=seed,
            verbose=verbose,
        )
    return results


# ---------------------------------------------------------------------------
# Merge Pareto fronts across propellants
# ---------------------------------------------------------------------------

def merge_pareto_fronts(
    results: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Combine per-propellant Pareto fronts and return the global
    non-dominated set.

    Returns
    -------
    F      : np.ndarray (n, 3)
    X      : np.ndarray (n, N_VAR)
    labels : list[str] — propellant key for each solution
    """
    all_F, all_X, all_labels = [], [], []
    for key, res in results.items():
        pF, pX = res["pareto_F"], res["pareto_X"]
        if len(pF) == 0:
            continue
        all_F.append(pF)
        all_X.append(pX)
        all_labels.extend([key] * len(pF))

    if not all_F:
        return np.empty((0, 3)), np.empty((0, 9)), []

    F_cat  = np.vstack(all_F)
    X_cat  = np.vstack(all_X)
    nd_idx = _nondominated_indices(F_cat)
    return F_cat[nd_idx], X_cat[nd_idx], [all_labels[i] for i in nd_idx]


def _nondominated_indices(F: np.ndarray) -> list[int]:
    """Return row indices of non-dominated solutions. O(n²), fine for <1000 pts."""
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_pareto_summary(
    F: np.ndarray, X: np.ndarray, labels: list[str]
) -> None:
    """Print a table of the global Pareto front extremes."""
    if len(F) == 0:
        print("No feasible solutions found.")
        return

    obj_names = ["Min TOF [days]", "Min Fuel [kg]", "Min Cost [M USD]"]
    print("\n--- Global Pareto Front Extremes ---")
    for i, label in enumerate(obj_names):
        idx = int(np.argmin(F[:, i]))
        print(f"\n  {label}  [{labels[idx]}]")
        print(f"    TOF     : {F[idx, 0]:.1f} days")
        print(f"    Fuel    : {F[idx, 1]:,.0f} kg")
        print(f"    Cost    : ${F[idx, 2]:.2f} M")