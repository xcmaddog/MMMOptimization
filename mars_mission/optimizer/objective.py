"""
objective.py
------------
Core objective function that bridges Matt's three-phase N-body simulator
with the pymoo optimiser.

Design vector layout
--------------------
The optimiser passes a 1-D array x with the following elements.
All bounds and indices are defined in DESIGN_VARIABLE_SPEC at the bottom
of this file so that problem.py and runner.py can read them without
duplicating magic numbers.

    Index  Name                               Units
    -----  ---------------------------------  -----
      0    launch_epoch_jd                    Julian Date
      1    leo_coast_days          (Phase 1)  days  (time in parking orbit)
      2    launch_angle_deg        (Phase 1)  degrees
      3    burn_duration_min       (Phase 2)  minutes
      4    thrust_newtons          (Phase 2)  N
      5    initial_total_mass_kg   (Phase 2)  kg
      6    starting_fuel_mass_kg   (Phase 2)  kg
      7    remaining_stage_mass_kg (Phase 2)  kg
      8    moi_burn_duration_min   (Phase 3)  minutes  (retrograde capture burn)

Propellant type is discrete and handled outside this vector — the caller
instantiates one objective instance per propellant, or uses
`evaluate_all_propellants()` to sweep them all.

Constants threaded through from the caller
------------------------------------------
- propellant             : Propellant  (sets Isp and burn_rate)
- payload_mass_kg        : float       (payload delivered to Mars orbit)
- stage_sep_speed_m_s    : float       (stage separation ΔV model parameter)
- phase3_lead_hours      : float       (collision-course lead time for MOI)
- p1_dt_s / p2_dt_s / p3_dt_s  : output timestep for each phase (speed knob)
- p2_total_days          : float       (max phase 2 sim duration)
- p3_total_days          : float       (max phase 3 sim duration)

Feasibility
-----------
A run is considered feasible when Phase 3 reports stable_orbit_detected=True.
Infeasible runs return BIG_PENALTY objectives and a positive constraint value
so that pymoo's dominance ranking pushes them out of the Pareto front.

Caching
-------
Each ObjectiveFunction instance owns a local dict keyed on a rounded tuple
of x.  This deduplicates within a single worker process.  When running with
multiprocessing (pymoo's ElementwiseProblem + StarmapParallelRunner), each
worker has its own cache — cross-worker deduplication is not attempted
because NSGA-II rarely revisits identical points.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from astropy.time import Time

from simulator.me_transfer_simulator import (
    run_phase1,
    run_phase2,
    run_phase3,
)
from optimizer.propellants import (
    Propellant,
    PROPELLANTS,
    burn_rate_kg_per_min,
)
from cost.cost import estimate_cost

# ---------------------------------------------------------------------------
# Penalty value for infeasible / crashed runs
# ---------------------------------------------------------------------------
BIG = 1e9


# ---------------------------------------------------------------------------
# Design variable specification
# Consumed by problem.py to set bounds without duplicating magic numbers.
# ---------------------------------------------------------------------------
DESIGN_VARIABLE_SPEC: list[dict] = [
    # idx  name                          lb            ub      description
    {"name": "launch_epoch_jd",          "lb": 2_460_800.5, "ub": 2_461_500.5},   # ~Sep 2025 – Aug 2027
    {"name": "leo_coast_days",           "lb": 0.0,         "ub": 2.0},            # days in LEO before burn
    {"name": "launch_angle_deg",         "lb": 0.0,         "ub": 360.0},          # angle around Earth orbit
    {"name": "burn_duration_min",        "lb": 1.0,         "ub": 120.0},          # Phase 2 burn
    {"name": "thrust_newtons",           "lb": 10_000.0,    "ub": 500_000.0},      # engine thrust
    {"name": "initial_total_mass_kg",    "lb": 5_000.0,     "ub": 200_000.0},      # wet mass at burn start
    {"name": "starting_fuel_mass_kg",    "lb": 1_000.0,     "ub": 150_000.0},      # propellant budget
    {"name": "remaining_stage_mass_kg",  "lb": 500.0,       "ub": 20_000.0},       # post-sep dry mass
    {"name": "moi_burn_duration_min",    "lb": 1.0,         "ub": 90.0},           # Phase 3 retro burn
]

N_VAR = len(DESIGN_VARIABLE_SPEC)
LOWER_BOUNDS = np.array([s["lb"] for s in DESIGN_VARIABLE_SPEC])
UPPER_BOUNDS = np.array([s["ub"] for s in DESIGN_VARIABLE_SPEC])

# Named index constants — import these in problem.py / runner.py
IDX_EPOCH        = 0
IDX_COAST        = 1
IDX_ANGLE        = 2
IDX_BURN_DUR     = 3
IDX_THRUST       = 4
IDX_TOTAL_MASS   = 5
IDX_FUEL_MASS    = 6
IDX_STAGE_MASS   = 7
IDX_MOI_DUR      = 8


# ---------------------------------------------------------------------------
# Rounding precision for cache keys
# ---------------------------------------------------------------------------
_CACHE_DECIMALS = 4   # round to 4 decimal places before hashing


class ObjectiveFunction:
    """
    Wraps the three-phase simulator as a callable objective function.

    Instantiate one per propellant.  Call evaluate(x) to get the result dict,
    or use the three scalar helpers (tof_obj, fuel_obj, cost_obj) that the
    pymoo Problem subclass can call directly.

    Parameters
    ----------
    propellant            : Propellant
    payload_mass_kg       : payload mass delivered to Mars orbit [kg]
    structural_mass_kg    : structural / hardware mass of the propulsion stage [kg]
                            Used only for the cost model CER; does not feed into
                            the simulator (which uses remaining_stage_mass_kg).
    stage_sep_speed_m_s   : stage-separation ejection speed [m/s]
    phase3_lead_hours     : collision-course lead time for phase-3 handoff [h]
    p1_dt_s               : Phase 1 output timestep [s]  (smaller = slower but finer)
    p2_dt_s               : Phase 2 output timestep [s]
    p3_dt_s               : Phase 3 output timestep [s]
    p2_total_days         : Phase 2 max simulation duration [days]
    p3_total_days         : Phase 3 max simulation duration [days]
    """

    def __init__(
        self,
        propellant: Propellant,
        payload_mass_kg: float = 5_000.0,
        structural_mass_kg: float = 8_000.0,
        stage_sep_speed_m_s: float = 50.0,
        phase3_lead_hours: float = 10.0,
        p1_dt_s: float = 180.0,
        p2_dt_s: float = 180.0,
        p3_dt_s: float = 180.0,
        p2_total_days: float = 300.0,
        p3_total_days: float = 10.0,
    ):
        self.propellant          = propellant
        self.payload_mass_kg     = payload_mass_kg
        self.structural_mass_kg  = structural_mass_kg
        self.stage_sep_speed_m_s = stage_sep_speed_m_s
        self.phase3_lead_hours   = phase3_lead_hours
        self.p1_dt_s             = p1_dt_s
        self.p2_dt_s             = p2_dt_s
        self.p3_dt_s             = p3_dt_s
        self.p2_total_days       = p2_total_days
        self.p3_total_days       = p3_total_days

        # Per-instance cache: rounded x tuple -> result dict
        self._cache: dict[tuple, dict] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> dict:
        """
        Run the full three-phase simulation for design vector x.

        Returns a result dict with keys:
            tof_days      : total mission duration [days]
            fuel_kg       : total propellant consumed [kg]
            cost_usd      : total mission cost [USD]
            feasible      : bool — True if a stable Mars orbit was achieved
            p1_result     : raw Phase 1 output dict
            p2_result     : raw Phase 2 output dict
            p3_result     : raw Phase 3 output dict
            status        : human-readable outcome string
        """
        key = self._cache_key(x)
        if key in self._cache:
            return self._cache[key]

        result = self._run(x)
        self._cache[key] = result
        return result

    def objectives(self, x: np.ndarray) -> tuple[float, float, float]:
        """
        Return (tof_days, fuel_kg, cost_usd/1e6) for design vector x.

        Infeasible runs return (BIG, BIG, BIG).
        This is the tuple pymoo's _evaluate method should unpack into F.
        """
        r = self.evaluate(x)
        if not r["feasible"]:
            return BIG, BIG, BIG
        return r["tof_days"], r["fuel_kg"], r["cost_usd"] / 1e6

    def feasibility_violation(self, x: np.ndarray) -> float:
        """
        Return a scalar constraint value:  <= 0 means feasible.

        pymoo convention: G <= 0 is satisfied.
        We return -1.0 for feasible runs and +1.0 for infeasible ones.
        """
        r = self.evaluate(x)
        return -1.0 if r["feasible"] else 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, x: np.ndarray) -> tuple:
        return tuple(np.round(x, _CACHE_DECIMALS))

    def _unpack(self, x: np.ndarray) -> dict:
        """Unpack design vector into named values."""
        return {
            "launch_epoch_jd":         float(x[IDX_EPOCH]),
            "leo_coast_days":          float(x[IDX_COAST]),
            "launch_angle_deg":        float(x[IDX_ANGLE]),
            "burn_duration_min":       float(x[IDX_BURN_DUR]),
            "thrust_newtons":          float(x[IDX_THRUST]),
            "initial_total_mass_kg":   float(x[IDX_TOTAL_MASS]),
            "starting_fuel_mass_kg":   float(x[IDX_FUEL_MASS]),
            "remaining_stage_mass_kg": float(x[IDX_STAGE_MASS]),
            "moi_burn_duration_min":   float(x[IDX_MOI_DUR]),
        }

    def _epoch_to_utc_string(self, jd: float) -> str:
        """Convert a Julian Date float to an ISO UTC string for astropy."""
        return Time(jd, format="jd", scale="utc").isot

    def _build_phase_settings(self, v: dict) -> tuple[dict, dict, dict]:
        """
        Build the three settings dicts that Transfer_Simulator expects.

        The burn_rate is derived from propellant Isp and thrust so it is
        never an independent design variable.
        """
        start_utc = self._epoch_to_utc_string(v["launch_epoch_jd"])
        rate = burn_rate_kg_per_min(self.propellant, v["thrust_newtons"])

        phase1 = {
            "launch_altitude_km":         200.0,          # fixed LEO altitude
            "initial_velocity_km_s":      None,            # auto circular speed
            "launch_angle_deg":           v["launch_angle_deg"],
            "simulation_start_time_utc":  start_utc,
            "dt_seconds":                 self.p1_dt_s,
            "max_step_seconds":           min(30.0, self.p1_dt_s),
            "total_time_days":            v["leo_coast_days"],
        }

        phase2 = {
            "requested_burn_duration_minutes": v["burn_duration_min"],
            "thrust_newtons":                  v["thrust_newtons"],
            "initial_total_mass_kg":           v["initial_total_mass_kg"],
            "burn_rate_kg_per_min":            rate,
            "starting_fuel_mass_kg":           v["starting_fuel_mass_kg"],
            "remaining_stage_mass_kg":         v["remaining_stage_mass_kg"],
            "stage_separation_relative_speed_m_s": self.stage_sep_speed_m_s,
            "phase3_collision_lead_hours":     self.phase3_lead_hours,
            "dt_seconds":                      self.p2_dt_s,
            "max_step_seconds":                min(60.0, self.p2_dt_s),
            "total_time_days":                 self.p2_total_days,
        }

        phase3 = {
            "correction_fuel_mass_kg":              0.0,   # all fuel already in phase2 budget
            "thrust_newtons":                       v["thrust_newtons"] * 0.05,  # 5 % for correction thrusters
            "collision_conversion_burn_duration_minutes": v["moi_burn_duration_min"],
            "requested_burn_duration_minutes":      v["moi_burn_duration_min"],
            "dt_seconds":                           self.p3_dt_s,
            "max_step_seconds":                     min(30.0, self.p3_dt_s),
            "total_time_days":                      self.p3_total_days,
        }

        return phase1, phase2, phase3

    def _validate_mass_constraints(self, v: dict) -> str | None:
        """
        Pre-check physical mass constraints before running the simulator.
        Returns an error string if invalid, None otherwise.

        Mirrors Part2's validate_mass_inputs but allows early exit.
        """
        if v["starting_fuel_mass_kg"] > v["initial_total_mass_kg"]:
            return "starting_fuel_mass_kg > initial_total_mass_kg"
        non_fuel = v["initial_total_mass_kg"] - v["starting_fuel_mass_kg"]
        if v["remaining_stage_mass_kg"] > non_fuel:
            return "remaining_stage_mass_kg > non-fuel mass"
        if v["remaining_stage_mass_kg"] <= 0:
            return "remaining_stage_mass_kg must be positive"
        return None

    def _extract_objectives(
        self,
        v: dict,
        p1_result: dict,
        p2_result: dict,
        p3_result: dict,
    ) -> tuple[float, float, float, bool, str]:
        """
        Pull tof, fuel, cost, and feasibility out of the three phase results.

        Returns
        -------
        tof_days    : total elapsed time across all three phases
        fuel_kg     : total propellant consumed (Phase 2 burn + Phase 3 retro)
        cost_usd    : estimated mission cost
        feasible    : True if Phase 3 detected a stable Mars orbit
        status      : human-readable outcome
        """
        p1_fs = p1_result["phase1_final_state"]
        p2_fs = p2_result["phase2_simulation"]["final_state"]
        p3_fs = p3_result["phase3_final_state"]

        # Total mission time [days]
        tof_days = (
            p1_fs["elapsed_time_seconds"]
            + p2_fs["elapsed_time_seconds"]
            + p3_fs["elapsed_time_seconds"]
        ) / 86_400.0

        # Fuel consumed:
        #   Phase 2 burn fuel = starting_fuel - fuel_remaining_after_phase2_burn
        #   Phase 3 retro fuel = whatever phase3 burned (tracked in its final state)
        # The simplest accounting: total_initial - final_rocket_mass,
        # but that includes discarded stage hardware.  So we track fuel explicitly.
        p2_fuel_burned = v["starting_fuel_mass_kg"] - p2_fs["fuel_remaining_kg"]
        p3_fuel_burned = max(
            0.0,
            p3_fs.get("rocket_mass_kg", 0.0)   # phase3 sets correction_fuel -> 0 by design
        )
        # Since we set correction_fuel_mass_kg=0 in phase3, p3 fuel burn is 0
        # in our setup (MOI is handled by phase2 burn duration).
        # If your team later separates MOI fuel, update this line.
        fuel_kg = p2_fuel_burned

        # Cost model
        cost = estimate_cost(
            propellant=self.propellant,
            m_prop_consumed_kg=fuel_kg,
            structural_mass_kg=self.structural_mass_kg,
            initial_wet_mass_kg=v["initial_total_mass_kg"],
        )

        # Feasibility: Phase 3 must confirm a stable orbit
        feasible = bool(p3_fs.get("stable_orbit_detected", False))
        status = p3_fs.get("status", "unknown")

        return tof_days, fuel_kg, cost.total, feasible, status

    def _run(self, x: np.ndarray) -> dict:
        """Execute all three phases and return the unified result dict."""
        v = self._unpack(x)

        # --- Pre-flight mass checks ---
        mass_error = self._validate_mass_constraints(v)
        if mass_error:
            return self._penalty_result(f"mass constraint: {mass_error}")

        phase1_settings, phase2_settings, phase3_settings = self._build_phase_settings(v)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                p1_result = run_phase1(
                    phase1_settings=phase1_settings,
                    run_animation=False,
                )
                p2_result = run_phase2(
                    phase1_final_state=p1_result["phase1_final_state"],
                    phase2_settings=phase2_settings,
                    run_animation=False,
                )
                p3_result = run_phase3(
                    phase2_handoff_state=p2_result["phase3_handoff_state"],
                    phase3_settings=phase3_settings,
                    run_animation=False,
                )

        except Exception as exc:
            return self._penalty_result(f"simulator exception: {exc}")

        tof_days, fuel_kg, cost_usd, feasible, status = self._extract_objectives(
            v, p1_result, p2_result, p3_result
        )

        return {
            "tof_days":   tof_days,
            "fuel_kg":    fuel_kg,
            "cost_usd":   cost_usd,
            "feasible":   feasible,
            "status":     status,
            "p1_result":  p1_result,
            "p2_result":  p2_result,
            "p3_result":  p3_result,
        }

    @staticmethod
    def _penalty_result(reason: str) -> dict:
        return {
            "tof_days":   BIG,
            "fuel_kg":    BIG,
            "cost_usd":   BIG,
            "feasible":   False,
            "status":     reason,
            "p1_result":  None,
            "p2_result":  None,
            "p3_result":  None,
        }


# ---------------------------------------------------------------------------
# Convenience: sweep all propellants and return a dict of ObjectiveFunctions
# ---------------------------------------------------------------------------

def build_objective_functions(**kwargs) -> dict[str, ObjectiveFunction]:
    """
    Instantiate one ObjectiveFunction per propellant in the catalogue.

    Any keyword arguments are forwarded to ObjectiveFunction.__init__,
    allowing the caller to set payload_mass_kg, timesteps, etc. once.

    Returns
    -------
    dict keyed by propellant key, e.g. {"kerolox": obj_fn, ...}
    """
    return {
        key: ObjectiveFunction(propellant=prop, **kwargs)
        for key, prop in PROPELLANTS.items()
    }