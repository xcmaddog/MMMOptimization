"""
vehicle.py
----------
Propellant catalogue, vehicle configuration, and rocket equation utilities.

All units are kg and km/s throughout.

Tsiolkovsky rocket equation:
    dv = Isp * g0 * ln(m_wet / m_dry)
    m_prop = m_dry * (exp(dv / (Isp * g0)) - 1)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

G0 = 9.80665e-3   # km/s^2  (standard gravity, converted to match km/s dv units)


@dataclass(frozen=True)
class Propellant:
    """Immutable propellant definition."""
    name:         str
    isp_vac:      float   # vacuum Isp [s]
    cost_per_kg:  float   # USD/kg of propellant


# Discrete propellant options.  Isp values are representative vacuum figures.
PROPELLANTS: dict[str, Propellant] = {
    "kerolox": Propellant(
        name="Kerosene/LOX (RP-1)",
        isp_vac=311.0,
        cost_per_kg=0.50,
    ),
    "hydrolox": Propellant(
        name="LH2/LOX",
        isp_vac=450.0,
        cost_per_kg=2.50,
    ),
    "storable": Propellant(
        name="NTO/MMH (hypergolic)",
        isp_vac=340.0,
        cost_per_kg=10.0,
    ),
    "methalox": Propellant(
        name="Methane/LOX",
        isp_vac=380.0,
        cost_per_kg=1.00,
    ),
}


@dataclass
class VehicleConfig:
    """
    Fixed parameters of the launch vehicle / spacecraft.

    Attributes
    ----------
    payload_mass_kg      : Mass of delivered payload [kg]
    structural_mass_kg   : Dry mass of propulsion stage (tanks, engine, etc.) [kg]
    propellant           : Propellant choice (sets Isp and cost/kg)
    max_propellant_kg    : Maximum propellant capacity (tank constraint) [kg]
    max_g_loading        : Maximum allowable acceleration during burns [g]
    """
    payload_mass_kg:    float
    structural_mass_kg: float
    propellant:         Propellant
    max_propellant_kg:  float
    max_g_loading:      float = 4.0


def propellant_mass_required(dv: float, isp: float, m_dry: float) -> float:
    """
    Propellant mass needed to achieve dv starting from m_dry.

    Parameters
    ----------
    dv    : required delta-v [km/s]
    isp   : specific impulse [s]
    m_dry : dry mass [kg]

    Returns
    -------
    m_prop [kg]
    """
    return m_dry * (np.exp(dv / (isp * G0)) - 1.0)


def max_delta_v(isp: float, m_prop: float, m_dry: float) -> float:
    """
    Maximum delta-v achievable given a propellant budget.

    Parameters
    ----------
    isp    : specific impulse [s]
    m_prop : available propellant [kg]
    m_dry  : dry mass [kg]

    Returns
    -------
    dv_max [km/s]
    """
    return isp * G0 * np.log((m_dry + m_prop) / m_dry)
