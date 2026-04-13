"""
runner.py
---------
NSGA-II runner with seeded initial population, tqdm progress bar, and
optional parallelism via ProcessPoolExecutor.
"""

from __future__ import annotations

import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from optimizer.objective import (
    ObjectiveFunction, build_objective_functions, generate_seeds, N_VAR,
)
from optimizer.problem import MarsTransferProblem
from optimizer.propellants import PROPELLANTS, Propellant
from optimizer.cache import EvalCache, get_default_cache


# ---------------------------------------------------------------------------
# Progress bar callback
# ---------------------------------------------------------------------------

class ProgressCallback(Callback):
    """
    pymoo Callback that drives a tqdm progress bar and prints per-generation
    summaries to stdout.

    Shows:
        • Generation number / total
        • Cumulative evaluations
        • Current Pareto-front size (feasible solutions only)
        • Elapsed time and estimated time remaining
        • Cache hit rate since last generation
    """

    def __init__(self, n_gen: int, pop_size: int, propellant_name: str,
                 show_bar: bool = True):
        super().__init__()
        self.n_gen          = n_gen
        self.pop_size       = pop_size
        self.propellant_name = propellant_name
        self.show_bar       = show_bar and _TQDM_AVAILABLE
        self._bar           = None
        self._t0            = time.perf_counter()
        self._last_evals    = 0

    def initialize(self, algorithm):
        if self.show_bar:
            total_evals = (self.n_gen + 1) * self.pop_size
            self._bar = tqdm(
                total=self.n_gen,
                desc=f"  {self.propellant_name[:12]:12s}",
                unit="gen",
                ncols=90,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} gen "
                    "[{elapsed}<{remaining}, {rate_fmt}]  {postfix}"
                ),
            )

    def _update(self, algorithm):
        gen      = algorithm.n_gen
        n_evals  = algorithm.evaluator.n_eval
        opt_pop  = algorithm.opt
        n_pareto = len(opt_pop) if opt_pop is not None else 0
        elapsed  = time.perf_counter() - self._t0
        new_evals = n_evals - self._last_evals
        self._last_evals = n_evals

        if self.show_bar and self._bar is not None:
            self._bar.set_postfix(
                pareto=n_pareto,
                evals=n_evals,
                refresh=False,
            )
            self._bar.update(1)
        else:
            # Fallback: plain text
            rate = new_evals / max(elapsed / gen, 1e-9)
            eta_s = (self.n_gen - gen) * (elapsed / max(gen, 1))
            eta  = _fmt_time(eta_s)
            print(
                f"  gen {gen:4d}/{self.n_gen}  |  "
                f"evals={n_evals:6d}  |  "
                f"pareto={n_pareto:3d}  |  "
                f"elapsed={_fmt_time(elapsed)}  |  "
                f"ETA={eta}",
                flush=True,
            )

    def finalize(self):
        if self._bar is not None:
            self._bar.close()


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Single-propellant run
# ---------------------------------------------------------------------------

def run_single(
    objective_fn: ObjectiveFunction,
    pop_size:    int   = 100,
    n_gen:       int   = 100,
    n_workers:   int   = 1,
    n_seeds:     int   = 20,
    seed_frac:   float = 0.5,
    rng_seed:    int   = 42,
    verbose:     bool  = True,
    show_progress: bool = True,
) -> dict:
    """
    Run NSGA-II for one propellant.

    Initial population = seed_frac × pop_size Lambert warm starts
                       + (1 − seed_frac) × pop_size uniform-random individuals.

    Parameters
    ----------
    objective_fn   : ObjectiveFunction
    pop_size       : population size
    n_gen          : number of generations
    n_workers      : parallel worker processes (1 = serial)
    n_seeds        : Lambert warm-start seeds to generate
    seed_frac      : fraction of initial population seeded vs. random
    rng_seed       : for reproducibility
    verbose        : print propellant header and final summary
    show_progress  : show tqdm progress bar per generation

    Returns
    -------
    dict: pareto_F, pareto_X, result, propellant, runtime_s
    """
    prop = objective_fn.propellant

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Propellant : {prop.name}")
        print(f"  pop={pop_size}  gen={n_gen}  workers={n_workers}  seed={rng_seed}")
        # Print cache stats if available
        if objective_fn._disk_cache is not None:
            s = objective_fn._disk_cache.stats()
            print(f"  Disk cache : {s['total']} entries "
                  f"({s['feasible']} feasible)  [{s['db_path']}]")
        print(f"{'='*62}")

    problem = MarsTransferProblem(
        objective_fn,
        n_workers=n_workers,
        n_seeds=n_seeds,
        seed_frac=seed_frac,
        rng_seed=rng_seed,
    )
    algo     = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    term     = get_termination("n_gen", n_gen)
    callback = ProgressCallback(
        n_gen=n_gen,
        pop_size=pop_size,
        propellant_name=prop.name,
        show_bar=show_progress,
    )

    t0 = time.perf_counter()
    res = minimize(
        problem, algo, term,
        seed=rng_seed,
        verbose=False,
        callback=callback,
    )
    callback.finalize()
    runtime_s = time.perf_counter() - t0

    pareto_F = res.F if res.F is not None else np.empty((0, 3))
    pareto_X = res.X if res.X is not None else np.empty((0, N_VAR))

    if verbose:
        print(
            f"  Finished in {_fmt_time(runtime_s)}  |  "
            f"Pareto solutions: {len(pareto_F)}"
        )

    return {
        "pareto_F": pareto_F, "pareto_X": pareto_X,
        "result": res, "propellant": prop, "runtime_s": runtime_s,
    }


# ---------------------------------------------------------------------------
# Full propellant sweep
# ---------------------------------------------------------------------------

def run_all_propellants(
    objective_fns: dict[str, ObjectiveFunction] | None = None,
    pop_size:    int   = 100,
    n_gen:       int   = 100,
    n_workers:   int   = 1,
    n_seeds:     int   = 20,
    seed_frac:   float = 0.5,
    rng_seed:    int   = 42,
    verbose:     bool  = True,
    show_progress: bool = True,
    **obj_fn_kwargs,
) -> dict[str, dict]:
    """Run NSGA-II for every propellant in the catalogue."""
    if objective_fns is None:
        objective_fns = build_objective_functions(**obj_fn_kwargs)
    results = {}
    for key, obj_fn in objective_fns.items():
        results[key] = run_single(
            obj_fn,
            pop_size=pop_size, n_gen=n_gen,
            n_workers=n_workers, n_seeds=n_seeds,
            seed_frac=seed_frac, rng_seed=rng_seed,
            verbose=verbose, show_progress=show_progress,
        )
    return results


# ---------------------------------------------------------------------------
# Merge and utilities
# ---------------------------------------------------------------------------

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