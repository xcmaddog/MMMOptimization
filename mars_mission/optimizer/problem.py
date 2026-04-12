"""
problem.py
----------
pymoo Problem with seeded initial population and parallel evaluation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

from optimizer.objective import (
    ObjectiveFunction, generate_seeds,
    LOWER_BOUNDS, UPPER_BOUNDS, N_VAR,
)
from optimizer.propellants import Propellant


# ---------------------------------------------------------------------------
# Worker — top-level so multiprocessing can pickle it
# ---------------------------------------------------------------------------

def _evaluate_one(args: tuple) -> tuple[list[float], list[float]]:
    """Evaluate one design vector; returns ([f0,f1,f2], [g0])."""
    obj_fn, x = args
    tof, fuel, cost_M = obj_fn.objectives(x)
    g = obj_fn.feasibility_violation(x)
    return [tof, fuel, cost_M], [g]


# ---------------------------------------------------------------------------
# Seeded sampling
# ---------------------------------------------------------------------------

class SeededSampling(Sampling):
    """
    pymoo Sampling that blends Lambert warm-start seeds with random individuals.

    Parameters
    ----------
    seeds      : pre-computed seed array (n_seeds, N_VAR)
    seed_frac  : fraction of population filled from seeds; rest is random
    rng_seed   : for reproducible random fill
    """

    def __init__(self, seeds: np.ndarray, seed_frac: float = 0.5, rng_seed: int = 0):
        super().__init__()
        self._seeds    = seeds
        self._seed_frac = seed_frac
        self._rng      = np.random.default_rng(rng_seed)

    def _do(self, problem, n_samples, **kwargs):
        n_seed = min(len(self._seeds), max(1, int(n_samples * self._seed_frac)))
        n_rand = n_samples - n_seed
        chosen = self._seeds[np.arange(n_seed) % len(self._seeds)]
        rand   = self._rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(n_rand, N_VAR))
        return np.clip(np.vstack([chosen, rand]), LOWER_BOUNDS, UPPER_BOUNDS)


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class MarsTransferProblem(Problem):
    """
    pymoo Problem for one propellant.

    Parameters
    ----------
    objective_fn : ObjectiveFunction
    n_workers    : parallel worker processes (1 = serial)
    seeds        : pre-computed seeds; generated automatically if None
    n_seeds      : how many seeds to build (ignored if seeds is supplied)
    seed_frac    : fraction of initial population seeded vs. random
    rng_seed     : for reproducibility
    """

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        n_workers: int            = 1,
        seeds: np.ndarray | None  = None,
        n_seeds: int              = 20,
        seed_frac: float          = 0.5,
        rng_seed: int             = 42,
    ):
        self.obj_fn    = objective_fn
        self.n_workers = n_workers

        if seeds is None:
            seeds = generate_seeds(objective_fn.propellant,
                                   n_seeds=n_seeds, rng_seed=rng_seed)
        sampling = SeededSampling(seeds, seed_frac=seed_frac, rng_seed=rng_seed)

        super().__init__(
            n_var        = N_VAR,
            n_obj        = 3,
            n_ieq_constr = 1,
            xl           = LOWER_BOUNDS.copy(),
            xu           = UPPER_BOUNDS.copy(),
            sampling     = sampling,
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        args_list = [(self.obj_fn, X[i]) for i in range(len(X))]
        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                results = list(ex.map(_evaluate_one, args_list))
        else:
            results = [_evaluate_one(a) for a in args_list]
        out["F"] = np.array([r[0] for r in results])
        out["G"] = np.array([r[1] for r in results])

    @property
    def propellant(self) -> Propellant:
        return self.obj_fn.propellant