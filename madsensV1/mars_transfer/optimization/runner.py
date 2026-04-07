"""
runner.py
---------
High-level interface for running multi-objective optimisation and
merging Pareto fronts across propellant options.

Typical usage
-------------
    from mars_transfer.optimization.runner import run_single, run_all_propellants
    from mars_transfer.ephemeris.ephemeris import epoch_from_date
    from mars_transfer.vehicle.vehicle import VehicleConfig, PROPELLANTS

    vehicle = VehicleConfig(
        payload_mass_kg=5000,
        structural_mass_kg=8000,
        propellant=PROPELLANTS["hydrolox"],
        max_propellant_kg=150_000,
    )
    depart_window = (epoch_from_date(2026, 9, 1), epoch_from_date(2027, 3, 31))

    result = run_single(vehicle, depart_window)
    # result["pareto_F"]  — objective values on the Pareto front
    # result["pareto_X"]  — corresponding design variables

    all_results = run_all_propellants(vehicle, depart_window)
"""

from __future__ import annotations

# from copy import replace as _replace
from dataclasses import replace
import numpy as np
from astropy.time import Time

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from mars_transfer.vehicle.vehicle import VehicleConfig, PROPELLANTS
from mars_transfer.optimization.problem import MarsTransferProblem


def run_single(
    vehicle: VehicleConfig,
    depart_window: tuple[Time, Time],
    tof_bounds: tuple[float, float] = (100.0, 400.0),
    pop_size: int = 200,
    n_gen: int = 150,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run NSGA-II on a single vehicle / propellant configuration.

    Parameters
    ----------
    vehicle        : VehicleConfig
    depart_window  : (start_epoch, end_epoch)
    tof_bounds     : (min_days, max_days)
    pop_size       : NSGA-II population size
    n_gen          : number of generations
    seed           : RNG seed for reproducibility
    verbose        : print generation progress

    Returns
    -------
    dict with keys:
        "pareto_F"   : np.ndarray (n, 3) — Pareto-front objective values
                       columns: [propellant_kg, tof_days, cost_MUSD]
        "pareto_X"   : np.ndarray (n, 3) — corresponding design variables
                       columns: [t_depart_jd, tof_days, m_prop_loaded_kg]
        "result"     : raw pymoo Result object
        "problem"    : MarsTransferProblem instance
        "propellant" : Propellant used
    """
    prob = MarsTransferProblem(vehicle, depart_window, tof_bounds)

    algo = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    term = get_termination("n_gen", n_gen)

    if verbose:
        print(f"  Running NSGA-II | propellant: {vehicle.propellant.name} "
              f"| pop={pop_size} | gen={n_gen}")

    res = minimize(
        prob,
        algo,
        term,
        seed=seed,
        verbose=False,   # pymoo's own verbose is noisy; we handle progress above
    )

    # res.F and res.X are the non-dominated (Pareto) solutions pymoo found.
    # pymoo already filters to feasible non-dominated solutions when constraints
    # are defined via n_ieq_constr.
    pareto_F = res.F if res.F is not None else np.empty((0, 3))
    pareto_X = res.X if res.X is not None else np.empty((0, 3))

    if verbose:
        print(f"    → {len(pareto_F)} Pareto-optimal solutions found.")

    return {
        "pareto_F":   pareto_F,
        "pareto_X":   pareto_X,
        "result":     res,
        "problem":    prob,
        "propellant": vehicle.propellant,
    }


def run_all_propellants(
    base_vehicle: VehicleConfig,
    depart_window: tuple[Time, Time],
    tof_bounds: tuple[float, float] = (100.0, 400.0),
    pop_size: int = 200,
    n_gen: int = 150,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run NSGA-II for every propellant in the catalogue.

    base_vehicle's propellant field is overwritten for each run —
    all other vehicle parameters are shared.

    Returns
    -------
    dict keyed by propellant key (e.g. "hydrolox"), each value is a
    run_single() result dict.
    """
    results = {}
    for key, prop in PROPELLANTS.items():
        if verbose:
            print(f"\n=== Propellant: {prop.name} ===")
        v = replace(base_vehicle, propellant=prop)
        results[key] = run_single(
            v, depart_window, tof_bounds,
            pop_size=pop_size, n_gen=n_gen,
            seed=seed, verbose=verbose,
        )
    return results


def merge_pareto_fronts(results: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Combine Pareto fronts from multiple propellant runs and extract the
    global non-dominated set.

    Returns
    -------
    F      : np.ndarray (n, 3) — global Pareto-front objective values
    X      : np.ndarray (n, 3) — corresponding design variables
    labels : list[str] of length n — propellant key for each solution
    """
    all_F, all_X, all_labels = [], [], []
    for key, res in results.items():
        pF = res["pareto_F"]
        pX = res["pareto_X"]
        if len(pF) == 0:
            continue
        all_F.append(pF)
        all_X.append(pX)
        all_labels.extend([key] * len(pF))

    if not all_F:
        return np.empty((0, 3)), np.empty((0, 3)), []

    F_cat = np.vstack(all_F)
    X_cat = np.vstack(all_X)

    # Non-dominated sort on the merged set
    nd_idx = _nondominated_indices(F_cat)

    return F_cat[nd_idx], X_cat[nd_idx], [all_labels[i] for i in nd_idx]


def _nondominated_indices(F: np.ndarray) -> list[int]:
    """
    Return indices of non-dominated rows in objective matrix F (minimisation).
    Simple O(n^2) sort — fine for Pareto front sizes typical here (<1000).
    """
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
