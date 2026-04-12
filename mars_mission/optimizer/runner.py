"""
runner.py
---------
NSGA-II runner with seeded initial population and optional parallelism.
"""

from __future__ import annotations

import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from optimizer.objective import (
    ObjectiveFunction, build_objective_functions, generate_seeds, N_VAR,
)
from optimizer.problem import MarsTransferProblem
from optimizer.propellants import PROPELLANTS, Propellant


def run_single(
    objective_fn: ObjectiveFunction,
    pop_size:   int   = 100,
    n_gen:      int   = 100,
    n_workers:  int   = 1,
    n_seeds:    int   = 20,
    seed_frac:  float = 0.5,
    rng_seed:   int   = 42,
    verbose:    bool  = True,
) -> dict:
    """
    Run NSGA-II for one propellant.

    The initial population is seed_frac × pop_size Lambert warm starts
    plus (1 − seed_frac) × pop_size uniform-random individuals.

    Returns
    -------
    dict: pareto_F, pareto_X, result, propellant, runtime_s
    """
    prop = objective_fn.propellant
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Propellant : {prop.name}")
        print(f"  pop={pop_size}  gen={n_gen}  workers={n_workers}  seed={rng_seed}")
        print(f"{'='*60}")

    problem = MarsTransferProblem(
        objective_fn,
        n_workers=n_workers,
        n_seeds=n_seeds,
        seed_frac=seed_frac,
        rng_seed=rng_seed,
    )
    algo = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    term = get_termination("n_gen", n_gen)

    t0 = time.perf_counter()
    res = minimize(problem, algo, term, seed=rng_seed, verbose=False)
    runtime_s = time.perf_counter() - t0

    pareto_F = res.F if res.F is not None else np.empty((0, 3))
    pareto_X = res.X if res.X is not None else np.empty((0, N_VAR))

    if verbose:
        print(f"  Done in {runtime_s:.1f} s  |  Pareto solutions: {len(pareto_F)}")

    return {
        "pareto_F": pareto_F, "pareto_X": pareto_X,
        "result": res, "propellant": prop, "runtime_s": runtime_s,
    }


def run_all_propellants(
    objective_fns: dict[str, ObjectiveFunction] | None = None,
    pop_size: int = 100, n_gen: int = 100,
    n_workers: int = 1, n_seeds: int = 20,
    seed_frac: float = 0.5, rng_seed: int = 42,
    verbose: bool = True, **obj_fn_kwargs,
) -> dict[str, dict]:
    """Run NSGA-II for every propellant in the catalogue."""
    if objective_fns is None:
        objective_fns = build_objective_functions(**obj_fn_kwargs)
    return {
        key: run_single(obj_fn, pop_size=pop_size, n_gen=n_gen,
                        n_workers=n_workers, n_seeds=n_seeds,
                        seed_frac=seed_frac, rng_seed=rng_seed, verbose=verbose)
        for key, obj_fn in objective_fns.items()
    }


def merge_pareto_fronts(results: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Global non-dominated set from all per-propellant Pareto fronts."""
    all_F, all_X, all_labels = [], [], []
    for key, res in results.items():
        pF, pX = res["pareto_F"], res["pareto_X"]
        if len(pF):
            all_F.append(pF); all_X.append(pX)
            all_labels.extend([key] * len(pF))
    if not all_F:
        return np.empty((0, 3)), np.empty((0, N_VAR)), []
    F = np.vstack(all_F); X = np.vstack(all_X)
    idx = _nondominated(F)
    return F[idx], X[idx], [all_labels[i] for i in idx]


def _nondominated(F: np.ndarray) -> list[int]:
    n = len(F); dom = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dom[i] = True; break
    return [i for i in range(n) if not dom[i]]


def print_pareto_summary(F, X, labels) -> None:
    if len(F) == 0:
        print("No feasible solutions found."); return
    print("\n--- Global Pareto Front Extremes ---")
    for i, lbl in enumerate(["Min TOF [days]", "Min Fuel [kg]", "Min Cost [M USD]"]):
        idx = int(np.argmin(F[:, i]))
        print(f"\n  {lbl}  [{labels[idx]}]")
        print(f"    TOF  : {F[idx,0]:.1f} days")
        print(f"    Fuel : {F[idx,1]:,.0f} kg")
        print(f"    Cost : ${F[idx,2]:.2f} M")