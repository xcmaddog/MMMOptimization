"""
ephemeris.py
------------
Heliocentric state vectors for Earth and Mars using astropy's built-in
ephemeris. No external kernel downloads required.

All outputs are in km and km/s in a Sun-centred frame aligned with ICRS,
which is the frame expected by the Lambert solver in trajectory/lambert.py.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u

SUPPORTED_BODIES = ("earth", "mars", "sun")


def get_heliocentric_state(body: str, epoch: Time) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the heliocentric position and velocity of a solar system body.

    Parameters
    ----------
    body  : "earth" or "mars" (case-insensitive)
    epoch : astropy.time.Time

    Returns
    -------
    r_km  : np.ndarray (3,) — position in km
    v_kms : np.ndarray (3,) — velocity in km/s
    """
    body = body.lower()
    if body not in SUPPORTED_BODIES:
        raise ValueError(f"Body '{body}' not supported. Choose from {SUPPORTED_BODIES}.")

    pos,     vel     = get_body_barycentric_posvel(body,  epoch)
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)

    r_km  = (pos - sun_pos).xyz.to(u.km).value
    v_kms = (vel - sun_vel).xyz.to(u.km / u.s).value
    return r_km, v_kms


def epoch_from_date(
    year: int, month: int, day: int,
    hour: int = 0, minute: int = 0, second: float = 0.0,
) -> Time:
    """Construct an astropy Time from calendar components."""
    iso = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:06.3f}"
    return Time(iso, format="isot", scale="tdb")


def epoch_range(start: Time, end: Time, n_points: int) -> list[Time]:
    """Return n_points equally-spaced epochs between start and end."""
    jds = np.linspace(start.jd, end.jd, n_points)
    return [Time(jd, format="jd", scale="tdb") for jd in jds]
