"""
problem.py
----------
pymoo Problem subclass for the Mars transfer optimisation.

Parallelism strategy (pymoo 0.6.x)
------------------------------------
pymoo 0.6.x's ElementwiseProblem + StarmapParallelRunner approach requires
the runner to be picklable, which pool objects are not. The correct pattern
for this version is to subclass Problem (not ElementwiseProblem) and
manually parallelize inside _evaluate() using concurrent.futures
ProcessPoolExecutor, whose worker function is a picklable top-level function.

Each worker process receives a single design vector x and an ObjectiveFunction
instance (which is picklable because it holds only plain Python data). The
worker calls obj_fn.objectives(x) independently, with its own copy of the
simulator modules and its own local cache.

Three objectives (all minimised):
    F[0]  tof_days      — total mission duration [days]
    F[1]  fuel_kg       — total propellant consumed [kg]
    F[2]  cost_MUSD     — total mission cost [million USD]

One inequality constraint (pymoo: satisfied when G <= 0):
    G[0]  feasibility   — -1.0 if stable orbit achieved, +1.0 otherwise

One instance = one propellant. Use runner.py to sweep all propellants.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pymoo.core.problem import Problem

from optimizer.objective import (
    ObjectiveFunction,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    N_VAR,
    BIG,
)
from optimizer.propellants import Propellant


# ---------------------------------------------------------------------------
# Top-level worker function — must be at module level to be picklable
# ---------------------------------------------------------------------------

def _evaluate_one(args: tuple) -> tuple[list[float], list[float]]:
    """
    Evaluate a single design vector in a worker process.

    Parameters
    ----------
    args : (obj_fn, x)
        obj_fn : ObjectiveFunction
        x      : np.ndarray shape (N_VAR,)

    Returns
    -------
    (F_list, G_list) where F has 3 objectives and G has 1 constraint value.
    """
    obj_fn, x = args
    tof, fuel, cost_M = obj_fn.objectives(x)
    g = obj_fn.feasibility_violation(x)   # cached — no extra simulator call
    return [tof, fuel, cost_M], [g]


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class MarsTransferProblem(Problem):
    """
    pymoo Problem for one propellant / vehicle configuration.

    Parameters
    ----------
    objective_fn : ObjectiveFunction
        Pre-built objective function for a specific propellant.
    n_workers    : int
        Number of parallel worker processes.
        1 = serial (recommended for debugging).
        >1 = ProcessPoolExecutor with that many workers.
    """

    def __init__(self, objective_fn: ObjectiveFunction, n_workers: int = 1):
        self.obj_fn    = objective_fn
        self.n_workers = n_workers

        super().__init__(
            n_var=N_VAR,
            n_obj=3,
            n_ieq_constr=1,
            xl=LOWER_BOUNDS.copy(),
            xu=UPPER_BOUNDS.copy(),
        )

    # ------------------------------------------------------------------
    # pymoo interface — called once per generation with the whole population
    # ------------------------------------------------------------------

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        """
        Evaluate the entire population X of shape (pop_size, N_VAR).

        When n_workers > 1, each row of X is dispatched to a separate
        worker process via ProcessPoolExecutor. The worker calls
        _evaluate_one() which is picklable (top-level function).
        """
        args_list = [(self.obj_fn, X[i]) for i in range(len(X))]

        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(_evaluate_one, args_list))
        else:
            results = [_evaluate_one(a) for a in args_list]

        F = np.array([r[0] for r in results])
        G = np.array([r[1] for r in results])

        out["F"] = F
        out["G"] = G

    # ------------------------------------------------------------------
    # Convenience accessor
    # ------------------------------------------------------------------

    @property
    def propellant(self) -> Propellant:
        return self.obj_fn.propellant