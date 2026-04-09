"""
propellants.py
--------------
Propellant catalogue and helpers for converting between Isp-based
rocket-equation variables and the burn_rate_kg_per_min that Matt's
Phase 2 simulator expects as a direct input.

Relationship
------------
    Isp [s]  =  thrust [N]  /  (mdot [kg/s]  *  g0 [m/s²])

    =>  mdot [kg/s]  =  thrust [N]  /  (Isp [s]  *  g0 [m/s²])
    =>  burn_rate [kg/min]  =  mdot [kg/s]  *  60

So for a given (propellant, thrust), burn_rate is fully determined —
it is not an independent design variable.

Usage
-----
    from optimizer.propellants import PROPELLANTS, burn_rate_kg_per_min

    prop = PROPELLANTS["hydrolox"]
    rate = burn_rate_kg_per_min(prop, thrust_newtons=100_000.0)
"""

from __future__ import annotations
from dataclasses import dataclass

G0_M_S2 = 9.80665          # standard gravity [m/s²]
G0_KM_S2 = G0_M_S2 / 1000  # [km/s²]


@dataclass(frozen=True)
class Propellant:
    """Immutable propellant definition."""
    key:          str     # short identifier used as dict key
    name:         str     # human-readable label
    isp_vac_s:    float   # vacuum specific impulse [s]
    cost_per_kg:  float   # estimated propellant cost [USD/kg]


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
# Isp values are representative vacuum figures for the engine cycle.
# cost_per_kg values are rough market estimates — adjust as needed.
# ---------------------------------------------------------------------------
PROPELLANTS: dict[str, Propellant] = {
    "kerolox": Propellant(
        key="kerolox",
        name="Kerosene / LOX  (RP-1)",
        isp_vac_s=311.0,
        cost_per_kg=0.50,
    ),
    "hydrolox": Propellant(
        key="hydrolox",
        name="LH2 / LOX",
        isp_vac_s=450.0,
        cost_per_kg=2.50,
    ),
    "storable": Propellant(
        key="storable",
        name="NTO / MMH  (hypergolic)",
        isp_vac_s=340.0,
        cost_per_kg=10.0,
    ),
    "methalox": Propellant(
        key="methalox",
        name="Methane / LOX",
        isp_vac_s=380.0,
        cost_per_kg=1.00,
    ),
}


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def burn_rate_kg_per_min(propellant: Propellant, thrust_newtons: float) -> float:
    """
    Derive the mass-flow rate in kg/min from propellant Isp and engine thrust.

    Parameters
    ----------
    propellant    : Propellant  (carries isp_vac_s)
    thrust_newtons: float  [N]

    Returns
    -------
    burn_rate : float [kg/min]
    """
    if thrust_newtons <= 0.0 or propellant.isp_vac_s <= 0.0:
        return 0.0
    mdot_kg_s = thrust_newtons / (propellant.isp_vac_s * G0_M_S2)
    return mdot_kg_s * 60.0


def effective_isp(thrust_newtons: float, burn_rate_kg_per_min_val: float) -> float:
    """
    Back-calculate Isp from thrust and burn rate (useful for validation).

    Parameters
    ----------
    thrust_newtons         : float [N]
    burn_rate_kg_per_min_val : float [kg/min]

    Returns
    -------
    isp : float [s]
    """
    mdot_kg_s = burn_rate_kg_per_min_val / 60.0
    if mdot_kg_s <= 0.0:
        return 0.0
    return thrust_newtons / (mdot_kg_s * G0_M_S2)


def delta_v_km_s(
    propellant: Propellant,
    m_wet_kg: float,
    m_prop_kg: float,
) -> float:
    """
    Tsiolkovsky rocket equation.

    Parameters
    ----------
    propellant  : Propellant
    m_wet_kg    : total wet mass at burn start [kg]
    m_prop_kg   : propellant mass consumed [kg]

    Returns
    -------
    dv : float [km/s]
    """
    import numpy as np
    m_dry = max(m_wet_kg - m_prop_kg, 1.0)
    return propellant.isp_vac_s * G0_KM_S2 * np.log(m_wet_kg / m_dry)