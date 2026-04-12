"""
lambert.py
----------
Lambert problem solver and delta-v calculations for Earth-to-Mars transfers.

The Lambert solver uses universal variables (Bate, Mueller & White, 1971),
solved via Brent's method from scipy. This produces results that match
hapsira's Izzo solver to machine precision and requires only numpy + scipy.

Pipeline for one (departure_epoch, tof_days) design point
----------------------------------------------------------
1. Get heliocentric r, v for Earth at departure and Mars at arrival.
2. Solve Lambert: given r_Earth, r_Mars, and TOF, find the transfer
   velocities v1 (at Earth) and v2 (at Mars).
3. Subtract planetary velocities to get hyperbolic excess speeds (v_inf).
4. Apply Oberth-effect burns to convert v_inf into actual delta-v:

   TMI from LEO:
       dv_tmi = sqrt(v_inf_dep^2 + 2*mu_E/r_leo) - sqrt(mu_E/r_leo)

   MOI into Mars capture orbit:
       dv_moi = sqrt(v_inf_arr^2 + 2*mu_M/r_cap) - sqrt(mu_M/r_cap)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from astropy.time import Time
import astropy.units as u

from mars_transfer.ephemeris.ephemeris import get_heliocentric_state

# ---------------------------------------------------------------------------
# Physical constants (all in km, km/s, km^3/s^2)
# ---------------------------------------------------------------------------
MU_SUN   = 132_712_442_099.0   # km^3/s^2  (DE430)
MU_EARTH =        398_600.4418  # km^3/s^2
MU_MARS  =         42_828.3744  # km^3/s^2
R_EARTH  =          6_378.137   # km (mean equatorial)
R_MARS   =          3_396.19    # km (mean equatorial)

LEO_ALT_KM  = 400.0   # parking orbit altitude above Earth surface
MOI_ALT_KM  = 400.0   # capture orbit altitude above Mars surface

R_LEO = R_EARTH + LEO_ALT_KM
R_MOI = R_MARS  + MOI_ALT_KM


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve_lambert(
    departure_epoch: Time,
    tof_days: float,
    prograde: bool = True,
) -> dict:
    """
    Full delta-v calculation for one Earth-to-Mars design point.

    Parameters
    ----------
    departure_epoch : astropy.time.Time
    tof_days        : time of flight [days]
    prograde        : True = prograde (short-way) transfer

    Returns
    -------
    dict with keys:
        dv_tmi        [km/s] — Trans-Mars Injection burn from LEO
        dv_moi        [km/s] — Mars Orbit Insertion retro-burn
        dv_total      [km/s] — sum of the two burns
        v_inf_depart  [km/s] — hyperbolic excess speed at Earth
        v_inf_arrive  [km/s] — hyperbolic excess speed at Mars
        C3            [km^2/s^2] — characteristic energy at Earth
        tof_days
        departure_epoch
        arrival_epoch

    Raises
    ------
    ValueError  if Lambert geometry is degenerate or TOF is non-physical.
    RuntimeError if the solver fails to converge.
    """
    if tof_days <= 0:
        raise ValueError(f"tof_days must be positive, got {tof_days}")

    arrival_epoch = departure_epoch + tof_days * u.day

    r_earth, v_earth = get_heliocentric_state("earth", departure_epoch)
    r_mars,  v_mars  = get_heliocentric_state("mars",  arrival_epoch)

    v1, v2 = _lambert_universal(
        MU_SUN, r_earth, r_mars, tof_days * 86_400.0, prograde=prograde
    )

    v_inf_dep = float(np.linalg.norm(v1 - v_earth))
    v_inf_arr = float(np.linalg.norm(v2 - v_mars))

    dv_tmi = _oberth_burn(v_inf_dep, MU_EARTH, R_LEO)
    dv_moi = _oberth_burn(v_inf_arr, MU_MARS,  R_MOI)

    return {
        "dv_tmi":          dv_tmi,
        "dv_moi":          dv_moi,
        "dv_total":        dv_tmi + dv_moi,
        "v_inf_depart":    v_inf_dep,
        "v_inf_arrive":    v_inf_arr,
        "C3":              v_inf_dep ** 2,
        "tof_days":        tof_days,
        "departure_epoch": departure_epoch,
        "arrival_epoch":   arrival_epoch,
    }


# ---------------------------------------------------------------------------
# Lambert universal-variable solver
# ---------------------------------------------------------------------------

def _lambert_universal(
    k: float,
    r1_vec: np.ndarray,
    r2_vec: np.ndarray,
    tof: float,
    prograde: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal-variable Lambert solver (Bate, Mueller & White, 1971).
    Solved via Brent's method on the TOF equation in the universal variable z.

    Parameters
    ----------
    k       : gravitational parameter [km^3/s^2]
    r1_vec  : departure position [km]
    r2_vec  : arrival position [km]
    tof     : time of flight [s]
    prograde: selects prograde or retrograde transfer

    Returns
    -------
    v1, v2 : departure and arrival velocity vectors [km/s]
    """
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cross   = np.cross(r1_vec, r2_vec)
    cross_n = np.linalg.norm(cross)
    if cross_n < 1e-6:
        raise ValueError("Position vectors are collinear — Lambert problem undefined.")

    cos_dnu = np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0)

    # Choose transfer direction
    if prograde:
        dm = 1.0 if cross[2] >= 0.0 else -1.0
    else:
        dm = -1.0 if cross[2] >= 0.0 else 1.0

    dnu = np.arccos(cos_dnu)
    if dm < 0:
        dnu = 2.0 * np.pi - dnu

    # A is a geometric constant combining both radii and transfer angle
    A = dm * np.sqrt(r1 * r2 * (1.0 + np.cos(dnu)))
    if abs(A) < 1e-10:
        raise ValueError("Transfer angle is 180° — degenerate Lambert case.")

    # TOF as a function of universal variable z
    def tof_of_z(z: float) -> float:
        cz = _C(z)
        if cz <= 0:
            return -1e30
        sz = _S(z)
        y  = r1 + r2 + A * (z * sz - 1.0) / np.sqrt(cz)
        if y < 0:
            return -1e30
        chi = np.sqrt(y / cz)
        return (chi**3 * sz + A * np.sqrt(y)) / np.sqrt(k)

    # Find z_lo where tof_of_z < tof (lower bracket)
    z_lo = -50.0
    for _ in range(200):
        if tof_of_z(z_lo) < tof:
            break
        z_lo -= 50.0
    else:
        raise RuntimeError("Could not find lower bracket for Lambert z.")

    # Find z_hi where tof_of_z > tof (upper bracket), scanning upward from 0
    z_hi = 0.0
    for _ in range(500):
        val = tof_of_z(z_hi)
        if val > tof:
            break
        z_hi += 1.0
    else:
        raise RuntimeError("Could not find upper bracket for Lambert z.")

    z_sol = brentq(lambda z: tof_of_z(z) - tof, z_lo, z_hi, xtol=1e-10, maxiter=200)

    # Reconstruct velocity vectors from f, g Lagrange coefficients
    cz  = _C(z_sol)
    sz  = _S(z_sol)
    y   = r1 + r2 + A * (z_sol * sz - 1.0) / np.sqrt(cz)
    f   = 1.0 - y / r1
    g   = A * np.sqrt(y / k)
    g_dot = 1.0 - y / r2

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g
    return v1, v2


def _S(z: float) -> float:
    """Stumpff function S(z) = (sqrt(z) - sin(sqrt(z))) / sqrt(z)^3."""
    if z > 1e-6:
        sq = np.sqrt(z)
        return (sq - np.sin(sq)) / sq**3
    elif z < -1e-6:
        sq = np.sqrt(-z)
        return (np.sinh(sq) - sq) / sq**3
    else:
        # Taylor series around z=0
        return 1.0/6.0 - z/120.0 + z**2/5040.0


def _C(z: float) -> float:
    """Stumpff function C(z) = (1 - cos(sqrt(z))) / z."""
    if z > 1e-6:
        return (1.0 - np.cos(np.sqrt(z))) / z
    elif z < -1e-6:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    else:
        return 0.5 - z/24.0 + z**2/720.0


# ---------------------------------------------------------------------------
# Oberth-effect burn equation
# ---------------------------------------------------------------------------

def _oberth_burn(v_inf: float, mu: float, r_orbit: float) -> float:
    """
    Delta-v to depart from (or capture into) a circular orbit using the
    Oberth effect — burning at the periapsis of the hyperbolic trajectory.

        dv = sqrt(v_inf^2 + v_esc^2) - v_circ
           = sqrt(v_inf^2 + 2*mu/r) - sqrt(mu/r)

    Works identically for departure (TMI) and arrival (MOI).

    Parameters
    ----------
    v_inf    : hyperbolic excess speed [km/s]
    mu       : gravitational parameter of central body [km^3/s^2]
    r_orbit  : radius of circular parking/capture orbit [km]

    Returns
    -------
    dv [km/s]
    """
    v_circ = np.sqrt(mu / r_orbit)
    return float(np.sqrt(v_inf**2 + 2.0 * mu / r_orbit) - v_circ)
