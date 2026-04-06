"""
problem.py
----------
pymoo-compatible multi-objective optimization problem for Earth-to-Mars transfer.

Design variables (one row per population member):
    x[0] : t_depart_jd    — departure epoch as Julian Date
    x[1] : tof_days       — time of flight [days]
    x[2] : m_prop_kg      — propellant mass loaded [kg]

Objectives (all minimised by pymoo):
    F[0] : propellant mass consumed [kg]
    F[1] : time of flight [days]
    F[2] : total mission cost [USD / 1e6]   (scaled to similar magnitude as others)

Inequality constraints (pymoo convention: satisfied when G <= 0):
    G[0] : dv_required - dv_available       (fuel feasibility)
    G[1] : m_prop - max_propellant_kg       (tank capacity)
    G[2] : v_inf_arrive - MAX_V_INF_ARRIVE  (arrival speed cap)

pymoo evaluates the entire population as a batch: x is (pop_size, 3).
We loop over rows; each call to solve_lambert is independent and fast.
Failed Lambert solves (degenerate geometry) are penalised rather than
allowed to crash the run.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time
from pymoo.core.problem import Problem

from mars_transfer.trajectory.lambert import solve_lambert
from mars_transfer.vehicle.vehicle import (
    VehicleConfig,
    propellant_mass_required,
    max_delta_v,
)
from mars_transfer.cost.cost import estimate_cost

# Hard upper bound on Mars arrival v_inf.
# Above this the MOI burn becomes impractical for a reasonable vehicle.
MAX_V_INF_ARRIVE = 5.0   # km/s

# Large penalty value for infeasible / failed evaluations
_BIG = 1e9


class MarsTransferProblem(Problem):
    """
    pymoo Problem subclass for multi-objective Mars transfer optimisation.

    One instance = one propellant choice.  To compare propellants, instantiate
    one problem per entry in PROPELLANTS, run NSGA-II on each, then merge the
    Pareto fronts (see runner.py).

    Parameters
    ----------
    vehicle        : VehicleConfig  (propellant, masses, tank size)
    depart_window  : (start_epoch, end_epoch) as astropy Time objects
    tof_bounds     : (min_days, max_days)
    """

    def __init__(
        self,
        vehicle: VehicleConfig,
        depart_window: tuple[Time, Time],
        tof_bounds: tuple[float, float] = (100.0, 400.0),
    ):
        self.vehicle = vehicle
        self._m_dry  = vehicle.payload_mass_kg + vehicle.structural_mass_kg

        jd_lo = depart_window[0].jd
        jd_hi = depart_window[1].jd
        tof_lo, tof_hi = tof_bounds

        xl = np.array([jd_lo, tof_lo, 0.0])
        xu = np.array([jd_hi, tof_hi, vehicle.max_propellant_kg])

        super().__init__(
            n_var=3,
            n_obj=3,
            n_ieq_constr=3,
            xl=xl,
            xu=xu,
        )

    # ------------------------------------------------------------------
    # pymoo interface
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """
        Batch evaluation.  x is shape (pop_size, 3).
        Fills out["F"] (objectives) and out["G"] (constraints).
        """
        n = x.shape[0]
        F = np.full((n, 3), _BIG)
        G = np.full((n, 3), _BIG)

        for i in range(n):
            t_jd, tof, m_prop = x[i]
            F[i], G[i] = self._eval_one(t_jd, tof, m_prop)

        out["F"] = F
        out["G"] = G

    # ------------------------------------------------------------------
    # Single-point evaluation
    # ------------------------------------------------------------------

    def _eval_one(
        self, t_jd: float, tof_days: float, m_prop_kg: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate objectives and constraints for one design point."""
        try:
            dep = Time(t_jd, format="jd", scale="tdb")
            result = solve_lambert(dep, tof_days)
        except Exception:
            return np.full(3, _BIG), np.full(3, _BIG)

        dv_req = result["dv_total"]   # km/s

        # How much dv the loaded propellant can deliver
        dv_avail = max_delta_v(self.vehicle.propellant.isp_vac, m_prop_kg, self._m_dry)

        # Propellant actually needed (may be less than loaded)
        try:
            m_prop_needed = propellant_mass_required(
                dv_req, self.vehicle.propellant.isp_vac, self._m_dry
            )
        except Exception:
            return np.full(3, _BIG), np.full(3, _BIG)

        cost = estimate_cost(self.vehicle, m_prop_needed)

        # --- Objectives ---
        f0 = m_prop_needed                  # kg  — minimise propellant
        f1 = tof_days                       # days — minimise time
        f2 = cost.total / 1e6              # M USD — minimise cost

        # --- Inequality constraints (satisfied when <= 0) ---
        g0 = dv_req - dv_avail                              # fuel feasibility
        g1 = m_prop_kg - self.vehicle.max_propellant_kg     # tank capacity
        g2 = result["v_inf_arrive"] - MAX_V_INF_ARRIVE      # arrival speed

        return np.array([f0, f1, f2]), np.array([g0, g1, g2])
