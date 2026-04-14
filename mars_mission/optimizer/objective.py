"""
objective.py
------------
Core objective function bridging Matt's three-phase N-body simulator with pymoo.

Architecture notes
------------------
1.  Mass split: TMI fuel vs. MOI fuel
    Phase 2 burns `starting_fuel_mass_kg`, drops the spent stage, and keeps
    `remaining_stage_mass_kg`. Phase 3 burns `correction_fuel_mass_kg` from
    that remaining mass.  The hard constraint from Part2 is:

        remaining_stage_mass_kg  <  initial_total_mass_kg − starting_fuel_mass_kg

    We satisfy this by adding MASS_BUFFER_KG to initial_total_mass_kg, so
    remaining_stage is always strictly less than the non-TMI-fuel portion.

2.  Launch angle: finite-burn arc correction
    A TMI burn of duration T_burn spans an arc of the LEO parking orbit:
        arc_deg = (T_burn / T_orbit) × 360

    Lambert assumes an instantaneous burn at a single point.  We shift the
    burn start so its midpoint aligns with the Lambert departure v_inf:
        launch_angle = arctan2(−v_inf_x, v_inf_y) − arc_deg/2   (mod 360)

3.  Epoch bounds: Sep–Dec 2026 Earth-Mars synodic window
    The simulator is 2D (ecliptic plane).  Diagnostic scanning shows that
    Mars is within 5–10% of the ecliptic during the September–October 2026
    opposition window, making 2D trajectories physically accurate.
    Feasible Lambert dV values (< 12 km/s total) exist only in the range
    JD 2461270–2461410 (2026-08-12 to 2027-01-01).

    KNOWN GOOD (epoch_jd, tof_days) SEED POINTS from Lambert scan:
        (2461294.5, 250)  →  dV = 8.77 km/s   (2026-09-11 dep, 250d TOF)
        (2461314.5, 200)  →  dV = 11.82 km/s  (2026-10-01 dep, 200d TOF)
        (2461314.5, 250)  →  dV = 8.22 km/s   (2026-10-01 dep, 250d TOF)
        (2461335.5, 200)  →  dV = 9.47 km/s   (2026-10-22 dep, 200d TOF)
        (2461370.5, 210)  →  dV = 7.18 km/s   (2026-11-26 dep, 210d TOF)

Design vector layout
--------------------
   Idx  Name               Units   Bounds
    0   launch_epoch_jd    JD      2461270 – 2461410
    1   tof_days           days    130 – 300
    2   thrust_newtons     N       50 000 – 500 000
    3   m_struct_kg        kg      2 000 – 15 000
    4   moi_fuel_fraction  –       0.10 – 0.55
    5   total_fuel_kg      kg      5 000 – 150 000
    6   leo_coast_days     days    0 – 1
"""

from __future__ import annotations

import os, sys, warnings
import numpy as np
import astropy.units as u
from astropy.time import Time

from simulator.me_transfer_simulator import run_phase1, run_phase2, run_phase3

_MT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mars_transfer'))
if _MT not in sys.path:
    sys.path.insert(0, _MT)
from mars_transfer.ephemeris.ephemeris import get_heliocentric_state
from mars_transfer.trajectory.lambert import _lambert_universal, MU_SUN, solve_lambert

from optimizer.propellants import Propellant, PROPELLANTS, burn_rate_kg_per_min
from cost.cost import estimate_cost
from optimizer.cache import EvalCache, get_default_cache, DEFAULT_CACHE_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G0_KM_S2     = 9.80665e-3
EARTH_MU_KM  = 398_600.4418
LEO_ALT_KM   = 200.0
LEO_R_KM     = 6_378.0 + LEO_ALT_KM
LEO_PERIOD_S = 2.0 * np.pi * np.sqrt(LEO_R_KM**3 / EARTH_MU_KM)
MASS_BUFFER_KG = 200.0
BIG = 1e9

# Earth-escape delta-v from 200 km LEO (Oberth-effect lower bound).
# Any design that cannot meet this analytically is rejected before the simulator runs.
_V_CIRC_LEO_KM_S = np.sqrt(EARTH_MU_KM / LEO_R_KM)
_DV_LEO_ESCAPE_KM_S = (np.sqrt(2.0) - 1.0) * _V_CIRC_LEO_KM_S  # ≈ 3.22 km/s

# ---------------------------------------------------------------------------
# Design variable specification
# ---------------------------------------------------------------------------
# Epoch bounds: Sep 2026 – Jan 2027 Earth-Mars window (2D sim compatible)
DESIGN_VARIABLE_SPEC: list[dict] = [
    {"name": "launch_epoch_jd",   "lb": 2_461_270.5, "ub": 2_461_410.5},
    {"name": "tof_days",          "lb": 130.0,        "ub": 300.0},
    {"name": "thrust_newtons",    "lb": 50_000.0,     "ub": 500_000.0},
    {"name": "m_struct_kg",       "lb": 2_000.0,      "ub": 15_000.0},
    {"name": "moi_fuel_fraction", "lb": 0.10,         "ub": 0.55},
    {"name": "total_fuel_kg",     "lb": 20_000.0,     "ub": 200_000.0},
    {"name": "leo_coast_days",    "lb": 0.0,          "ub": 1.0},
]

N_VAR        = len(DESIGN_VARIABLE_SPEC)
LOWER_BOUNDS = np.array([s["lb"] for s in DESIGN_VARIABLE_SPEC])
UPPER_BOUNDS = np.array([s["ub"] for s in DESIGN_VARIABLE_SPEC])

IDX_EPOCH    = 0
IDX_TOF      = 1
IDX_THRUST   = 2
IDX_STRUCT   = 3
IDX_MOI_FRAC = 4
IDX_FUEL     = 5
IDX_COAST    = 6

# Known-good (epoch_jd, tof_days) anchor points confirmed from Lambert scan
KNOWN_GOOD_ANCHORS: list[tuple[float, float]] = [
    (2_461_294.5, 250.0),   # 2026-09-11, 250d  dV=8.77 km/s
    (2_461_314.5, 250.0),   # 2026-10-01, 250d  dV=8.22 km/s
    (2_461_314.5, 200.0),   # 2026-10-01, 200d  dV=11.82 km/s
    (2_461_335.5, 200.0),   # 2026-10-22, 200d  dV=9.47 km/s
    (2_461_370.5, 210.0),   # 2026-11-26, 210d  dV=7.18 km/s
]


# ---------------------------------------------------------------------------
# Vehicle mass derivation
# ---------------------------------------------------------------------------

def derive_masses(x: np.ndarray, propellant: Propellant) -> dict:
    """
    Derive Phase 2/3 mass inputs from design vector.

    Layout
    ------
    total_fuel   = tmi_fuel + moi_fuel
    moi_fuel     = total_fuel × moi_fuel_fraction
    tmi_fuel     = total_fuel × (1 − moi_fuel_fraction)
    m_remaining  = m_struct + moi_fuel          (kept after stage sep)
    m_wet        = m_remaining + tmi_fuel + BUFFER
                                                (BUFFER ensures strict < constraint)
    """
    thrust     = float(x[IDX_THRUST])
    m_struct   = float(x[IDX_STRUCT])
    moi_frac   = float(x[IDX_MOI_FRAC])
    total_fuel = float(x[IDX_FUEL])

    moi_fuel   = total_fuel * moi_frac
    tmi_fuel   = total_fuel * (1.0 - moi_frac)
    m_remaining = m_struct + moi_fuel
    m_wet       = m_remaining + tmi_fuel + MASS_BUFFER_KG

    rate       = burn_rate_kg_per_min(propellant, thrust)
    burn_dur_s = (tmi_fuel / (rate / 60.0)) if rate > 0 else 0.0

    return {
        "tmi_fuel":         tmi_fuel,
        "moi_fuel":         moi_fuel,
        "m_wet":            m_wet,
        "m_remaining":      m_remaining,
        "burn_rate_kg_min": rate,
        "burn_dur_s":       burn_dur_s,
        "burn_dur_min":     burn_dur_s / 60.0,
    }


def check_masses(m: dict, isp_vac_s: float | None = None) -> str | None:
    """
    Return an error string if Part2 mass constraints are violated, else None.

    If isp_vac_s is supplied, also checks that the TMI burn delivers enough
    delta-v to escape Earth from 200 km LEO.  This catches configurations
    that would waste a full simulator run before re-impacting Earth.
    """
    if m["tmi_fuel"] <= 0 or m["moi_fuel"] <= 0:
        return "fuel components must be positive"
    if m["burn_rate_kg_min"] <= 0:
        return "burn_rate is zero (thrust or Isp is zero)"
    non_tmi = m["m_wet"] - m["tmi_fuel"]
    if m["m_remaining"] >= non_tmi:
        return f"remaining ({m['m_remaining']:.1f}) >= non_tmi ({non_tmi:.1f})"
    if isp_vac_s is not None:
        dv_tmi = isp_vac_s * G0_KM_S2 * np.log(m["m_wet"] / m["m_remaining"])
        if dv_tmi < _DV_LEO_ESCAPE_KM_S:
            return (
                f"TMI dV={dv_tmi:.3f} km/s < escape dV={_DV_LEO_ESCAPE_KM_S:.3f} km/s"
            )
    return None


# ---------------------------------------------------------------------------
# Launch angle
# ---------------------------------------------------------------------------

def compute_launch_angle(epoch_jd: float, tof_days: float, burn_dur_s: float) -> float:
    """
    Parking-orbit launch angle [deg] so the burn midpoint aligns with the
    Lambert departure v_inf direction.  Returns 0.0 on failure.

    Formula
    -------
    lambert_angle = arctan2(−v_inf_x, v_inf_y)   mod 360
    burn_arc_deg  = (burn_dur_s / LEO_PERIOD_S) × 360
    launch_angle  = lambert_angle − burn_arc_deg / 2   mod 360
    """
    try:
        dep = Time(epoch_jd, format="jd", scale="tdb")
        arr = dep + tof_days * u.day
        r_earth, v_earth = get_heliocentric_state("earth", dep)
        r_mars,  _       = get_heliocentric_state("mars",  arr)
        v1, _ = _lambert_universal(MU_SUN, r_earth, r_mars, tof_days * 86_400.0)
        v_inf = (v1 - v_earth)[:2]
        n = np.linalg.norm(v_inf)
        if n < 1e-9:
            return 0.0
        d = v_inf / n
        lambert_angle = np.degrees(np.arctan2(-d[0], d[1])) % 360.0
        burn_arc_deg  = (burn_dur_s / LEO_PERIOD_S) * 360.0
        return (lambert_angle - burn_arc_deg / 2.0) % 360.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Phase settings builders
# ---------------------------------------------------------------------------

def build_phase_settings(
    x: np.ndarray,
    masses: dict,
    propellant: Propellant,
    launch_angle: float,
    start_utc: str,
    p1_dt_s: float,
    p2_dt_s: float,
    p3_dt_s: float,
    p2_total_days: float,
    p3_total_days: float,
    stage_sep_speed_m_s: float,
    phase3_lead_hours: float,
    moi_thrust_fraction: float,
) -> tuple[dict, dict, dict]:

    p1 = {
        "launch_altitude_km":        LEO_ALT_KM,
        "initial_velocity_km_s":     None,
        "launch_angle_deg":          launch_angle,
        "simulation_start_time_utc": start_utc,
        "dt_seconds":                p1_dt_s,
        "max_step_seconds":          min(30.0, p1_dt_s),
        "total_time_days":           float(x[IDX_COAST]),
    }
    p2 = {
        "requested_burn_duration_minutes": masses["burn_dur_min"],
        "thrust_newtons":                  float(x[IDX_THRUST]),
        "initial_total_mass_kg":           masses["m_wet"],
        "burn_rate_kg_per_min":            masses["burn_rate_kg_min"],
        "starting_fuel_mass_kg":           masses["tmi_fuel"],
        "remaining_stage_mass_kg":         masses["m_remaining"],
        "stage_separation_relative_speed_m_s": stage_sep_speed_m_s,
        "phase3_collision_lead_hours":     phase3_lead_hours,
        "dt_seconds":                      p2_dt_s,
        "max_step_seconds":                p2_dt_s,   # adaptive steps handle near-planet accuracy
        "total_time_days":                 p2_total_days,
        "adaptive_steps":                  True,
        "close_approach_radius_km":        500_000.0,
    }
    p3 = {
        "correction_fuel_mass_kg":             masses["moi_fuel"],
        "thrust_newtons":                      float(x[IDX_THRUST]) * moi_thrust_fraction,
        "collision_conversion_burn_duration_minutes": 30.0,
        "requested_burn_duration_minutes":     60.0,
        "dt_seconds":                          p3_dt_s,
        "max_step_seconds":                    min(30.0, p3_dt_s),
        "total_time_days":                     p3_total_days,
    }
    return p1, p2, p3


# ---------------------------------------------------------------------------
# ObjectiveFunction
# ---------------------------------------------------------------------------

class ObjectiveFunction:
    """
    Wraps the three-phase N-body simulator as a callable objective.

    One instance per propellant.  Instances are picklable for multiprocessing.

    Parameters
    ----------
    propellant           : Propellant
    stage_sep_speed_m_s  : stage-separation ejection speed [m/s]
    phase3_lead_hours    : collision-course lead time for Phase-3 handoff [h]
    moi_thrust_fraction  : MOI thruster force as fraction of TMI engine thrust
    p1/2/3_dt_s          : output timestep per phase [s]
    p2_total_days        : Phase 2 max duration [days]  (TOF + 60-day buffer)
    p3_total_days        : Phase 3 observation window [days]
    """

    def __init__(
        self,
        propellant: Propellant,
        stage_sep_speed_m_s: float  = 50.0,
        phase3_lead_hours: float    = 10.0,
        moi_thrust_fraction: float  = 0.05,
        p1_dt_s: float              = 300.0,
        p2_dt_s: float              = 300.0,
        p3_dt_s: float              = 300.0,
        p2_total_days: float        = 380.0,
        p3_total_days: float        = 90.0,
        cache_path: str | None      = DEFAULT_CACHE_PATH,
    ):
        self.propellant          = propellant
        self.stage_sep_speed_m_s = stage_sep_speed_m_s
        self.phase3_lead_hours   = phase3_lead_hours
        self.moi_thrust_fraction = moi_thrust_fraction
        self.p1_dt_s             = p1_dt_s
        self.p2_dt_s             = p2_dt_s
        self.p3_dt_s             = p3_dt_s
        self.p2_total_days       = p2_total_days
        self.p3_total_days       = p3_total_days
        self._cache: dict[tuple, dict] = {}
        self._disk_cache: EvalCache | None = (
            EvalCache(cache_path) if cache_path else None
        )

    # ── public ───────────────────────────────────────────────────────────────

    def evaluate(self, x: np.ndarray) -> dict:
        """
        Run all three phases and return a result dict:
            tof_days, fuel_kg, cost_usd, feasible, status,
            p1_result, p2_result, p3_result

        Checks in-memory cache first, then disk cache, then runs simulator.
        """
        key = tuple(np.round(x, 4))
        # 1. In-memory cache (fastest — no disk I/O)
        if key in self._cache:
            return self._cache[key]
        # 2. Disk cache (shared across workers and runs)
        if self._disk_cache is not None:
            cached = self._disk_cache.get(x, self.propellant.key)
            if cached is not None:
                self._cache[key] = cached
                return cached
        # 3. Run the simulator
        r = self._run(x)
        self._cache[key] = r
        if self._disk_cache is not None:
            self._disk_cache.put(x, self.propellant.key, r)
        return r

    def objectives(self, x: np.ndarray) -> tuple[float, float, float]:
        """(tof_days, fuel_kg, cost_usd/1e6).  Returns (BIG, BIG, BIG) if infeasible."""
        r = self.evaluate(x)
        if not r["feasible"]:
            return BIG, BIG, BIG
        return r["tof_days"], r["fuel_kg"], r["cost_usd"] / 1e6

    def feasibility_violation(self, x: np.ndarray) -> float:
        """pymoo G convention: -1.0 = feasible (G ≤ 0), +1.0 = infeasible (G > 0)."""
        return -1.0 if self.evaluate(x)["feasible"] else 1.0

    # ── private ──────────────────────────────────────────────────────────────

    def _run(self, x: np.ndarray) -> dict:
        masses = derive_masses(x, self.propellant)
        err = check_masses(masses, isp_vac_s=self.propellant.isp_vac_s)
        if err:
            return self._penalty(f"mass: {err}")

        epoch_jd = float(x[IDX_EPOCH])
        tof_days = float(x[IDX_TOF])
        angle    = compute_launch_angle(epoch_jd, tof_days, masses["burn_dur_s"])
        utc      = Time(epoch_jd, format="jd", scale="tdb").utc.isot
        p2_days  = min(tof_days + 60.0, self.p2_total_days)

        p1s, p2s, p3s = build_phase_settings(
            x, masses, self.propellant, angle, utc,
            self.p1_dt_s, self.p2_dt_s, self.p3_dt_s,
            p2_days, self.p3_total_days,
            self.stage_sep_speed_m_s, self.phase3_lead_hours,
            self.moi_thrust_fraction,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p1r = run_phase1(phase1_settings=p1s, run_animation=False)
                p2r = run_phase2(
                    phase1_final_state=p1r["phase1_final_state"],
                    phase2_settings=p2s, run_animation=False,
                )
                p3r = run_phase3(
                    phase2_handoff_state=p2r["phase3_handoff_state"],
                    phase3_settings=p3s, run_animation=False,
                )
        except Exception as exc:
            return self._penalty(f"simulator: {exc}")

        return self._extract(x, masses, p1r, p2r, p3r)

    def _extract(self, x, masses, p1r, p2r, p3r) -> dict:
        p1fs = p1r["phase1_final_state"]
        p2fs = p2r["phase2_simulation"]["final_state"]
        p3fs = p3r["phase3_final_state"]

        tof_days = (
            p1fs["elapsed_time_seconds"]
            + p2fs["elapsed_time_seconds"]
            + p3fs["elapsed_time_seconds"]
        ) / 86_400.0

        tmi_burned = masses["tmi_fuel"] - p2fs["fuel_remaining_kg"]
        moi_burned = masses["moi_fuel"] - p3fs["fuel_remaining_kg"]
        fuel_kg    = max(0.0, tmi_burned) + max(0.0, moi_burned)

        cost = estimate_cost(
            propellant          = self.propellant,
            m_prop_consumed_kg  = fuel_kg,
            structural_mass_kg  = float(x[IDX_STRUCT]),
            initial_wet_mass_kg = masses["m_wet"],
        )

        return {
            "tof_days":  tof_days,
            "fuel_kg":   fuel_kg,
            "cost_usd":  cost.total,
            "feasible":  bool(p3fs.get("stable_orbit_detected", False)),
            "status":    p3fs.get("status", "unknown"),
            "p1_result": p1r,
            "p2_result": p2r,
            "p3_result": p3r,
        }

    @staticmethod
    def _penalty(reason: str) -> dict:
        return {
            "tof_days": BIG, "fuel_kg": BIG, "cost_usd": BIG,
            "feasible": False, "status": reason,
            "p1_result": None, "p2_result": None, "p3_result": None,
        }


# ---------------------------------------------------------------------------
# Seeder
# ---------------------------------------------------------------------------

def generate_seeds(
    propellant: Propellant,
    n_seeds: int = 20,
    rng_seed: int = 0,
    objective_fn: "ObjectiveFunction | None" = None,
    require_feasible: bool = False,
    max_evals: int | None = None,
    max_seconds: float | None = 60.0,
    verbose: bool = True,
    progress_every: int = 10,
    fallback_to_lambert: bool = True,
    cache_seed_limit: int = 50,
    cache_only: bool = False,
    min_eval_seconds: float = 5.0,
) -> np.ndarray:
    """
    Generate warm-start design vectors from the confirmed Lambert anchor points.

    For each of the five known-good (epoch, TOF) pairs:
    - Size the vehicle to exactly deliver the Lambert dV budget.
    - Add random variation in thrust, m_struct, and a fuel margin (10–40 %)
      so seeds span a range of the objective space.
    - Jitter epoch and TOF slightly around each anchor.

    If require_feasible and an ObjectiveFunction is supplied, runs the full
    simulator to keep only seeds that are actually mission-feasible. This is
    slower but avoids cold-starting the optimizer in infeasible space.

    Pads with uniform-random samples if fewer than n_seeds survive mass checks
    (unless require_feasible=True, in which case only feasible seeds are returned).

    Returns
    -------
    np.ndarray of shape (n_seeds, N_VAR), within bounds.
    """
    import time
    rng = np.random.default_rng(rng_seed)
    Isp = propellant.isp_vac_s

    seeds: list[np.ndarray] = []
    feasible_only = bool(require_feasible and objective_fn is not None)
    eval_budget = max_evals
    if feasible_only and eval_budget is None:
        eval_budget = max(40, n_seeds * 6)
    eval_count = 0
    t0 = time.perf_counter()

    if feasible_only and verbose:
        print(
            f"Seeding with feasible Lambert starts for {propellant.name} "
            f"(target={n_seeds}, eval_budget={eval_budget}, "
            f"max_seconds={max_seconds if max_seconds is not None else 'none'})"
        )

    # First, pull any feasible seeds from the disk cache (fast).
    if feasible_only and objective_fn is not None and objective_fn._disk_cache is not None:
        cached = objective_fn._disk_cache.fetch_feasible(
            propellant.key, limit=max(cache_seed_limit, n_seeds)
        )
        for x in cached:
            if len(seeds) >= n_seeds:
                break
            x = np.clip(x, LOWER_BOUNDS, UPPER_BOUNDS)
            m = derive_masses(x, propellant)
            if check_masses(m, isp_vac_s=propellant.isp_vac_s) is None:
                seeds.append(x)
        if verbose and len(cached):
            print(f"  cache seeds={len(cached)}  accepted={len(seeds)}")

    def maybe_add(x: np.ndarray) -> None:
        nonlocal eval_count
        if feasible_only and cache_only:
            return
        if feasible_only and max_seconds is not None:
            elapsed = time.perf_counter() - t0
            if elapsed >= max_seconds:
                return
            remaining = max_seconds - elapsed
            if remaining < max(0.0, min_eval_seconds):
                return
            # Avoid starting a long evaluation if we're likely to overrun.
            if eval_count > 0:
                avg_eval = elapsed / eval_count
                if (elapsed + avg_eval) >= max_seconds:
                    return
        if not feasible_only:
            seeds.append(x)
            return
        if eval_budget is not None and eval_count >= eval_budget:
            return
        eval_count += 1
        try:
            r = objective_fn.evaluate(x)
        except Exception:
            return
        if r.get("feasible", False):
            seeds.append(x)
        if verbose and feasible_only and (eval_count % max(1, progress_every) == 0):
            elapsed = time.perf_counter() - t0
            print(
                f"  seed evals={eval_count}  feasible={len(seeds)}  "
                f"elapsed={elapsed:.1f}s"
            )

    # How many variants to generate per anchor
    variants_per_anchor = max(1, n_seeds // len(KNOWN_GOOD_ANCHORS) + 2)

    for jd_anchor, tof_anchor in KNOWN_GOOD_ANCHORS:
        if feasible_only and max_seconds is not None:
            if (time.perf_counter() - t0) >= max_seconds:
                break
        if feasible_only and cache_only:
            break
        dep = Time(jd_anchor, format="jd", scale="tdb")

        try:
            r = solve_lambert(dep, tof_anchor)
        except Exception:
            continue

        dv_tmi, dv_moi = r["dv_tmi"], r["dv_moi"]
        if not (0 < dv_tmi < 20 and 0 < dv_moi < 20):
            continue

        for _ in range(variants_per_anchor):
            # Random vehicle parameters
            m_struct = rng.uniform(LOWER_BOUNDS[IDX_STRUCT],
                                   min(12_000.0, UPPER_BOUNDS[IDX_STRUCT]))
            thrust   = rng.uniform(LOWER_BOUNDS[IDX_THRUST],
                                   min(400_000.0, UPPER_BOUNDS[IDX_THRUST]))

            # Minimum fuel to deliver the exact Lambert dV budget
            moi_min     = m_struct * (np.exp(dv_moi / (Isp * G0_KM_S2)) - 1.0)
            m_after_tmi = m_struct + moi_min
            tmi_min     = m_after_tmi * (np.exp(dv_tmi / (Isp * G0_KM_S2)) - 1.0)
            total_min   = tmi_min + moi_min

            if total_min <= 0 or not np.isfinite(total_min):
                continue

            # Random margin gives objective-space diversity
            margin     = rng.uniform(1.10, 1.40)
            total_fuel = np.clip(total_min * margin,
                                 LOWER_BOUNDS[IDX_FUEL], UPPER_BOUNDS[IDX_FUEL])

            # Natural MOI fraction with small jitter
            moi_frac = np.clip(
                moi_min / total_min + rng.uniform(-0.03, 0.03),
                LOWER_BOUNDS[IDX_MOI_FRAC], UPPER_BOUNDS[IDX_MOI_FRAC],
            )

            # Small epoch/TOF jitter around the anchor
            jd  = np.clip(jd_anchor  + rng.uniform(-5.0, 5.0),
                          LOWER_BOUNDS[IDX_EPOCH], UPPER_BOUNDS[IDX_EPOCH])
            tof = np.clip(tof_anchor + rng.uniform(-10.0, 10.0),
                          LOWER_BOUNDS[IDX_TOF], UPPER_BOUNDS[IDX_TOF])

            x = np.array([
                jd, tof, thrust, m_struct, moi_frac, total_fuel,
                rng.uniform(0.0, 0.5),
            ])
            x = np.clip(x, LOWER_BOUNDS, UPPER_BOUNDS)

            # Quick mass sanity check before adding
            m = derive_masses(x, propellant)
            if check_masses(m, isp_vac_s=propellant.isp_vac_s) is None:
                maybe_add(x)

    if feasible_only and not cache_only:
        # Try additional random samples to reach target count (within budget)
        while len(seeds) < n_seeds:
            if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
                break
            if eval_budget is not None and eval_count >= eval_budget:
                break
            x = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
            x = np.clip(x, LOWER_BOUNDS, UPPER_BOUNDS)
            m = derive_masses(x, propellant)
            if check_masses(m, isp_vac_s=propellant.isp_vac_s) is None:
                maybe_add(x)
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"Feasible seeding complete: {len(seeds)} seeds in "
                f"{elapsed:.1f}s (evals={eval_count})."
            )
        if len(seeds) >= n_seeds or not fallback_to_lambert:
            return np.array(seeds)
        if verbose:
            print("  Falling back to Lambert-only seeds to fill remaining slots.")

    # If we need more, fill with non-feasible Lambert/random seeds (fast)
    while len(seeds) < n_seeds:
        x = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
        x = np.clip(x, LOWER_BOUNDS, UPPER_BOUNDS)
        m = derive_masses(x, propellant)
        if check_masses(m, isp_vac_s=propellant.isp_vac_s) is None:
            seeds.append(x)

    return np.array(seeds[:n_seeds])


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_objective_functions(**kwargs) -> dict[str, ObjectiveFunction]:
    """One ObjectiveFunction per propellant. kwargs → ObjectiveFunction.__init__."""
    return {k: ObjectiveFunction(propellant=p, **kwargs) for k, p in PROPELLANTS.items()}
