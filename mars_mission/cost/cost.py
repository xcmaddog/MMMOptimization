"""
cost.py
-------
Parametric mission cost model.

Three components:
    1. Propellant cost   — mass consumed × cost_per_kg
    2. Vehicle cost      — power-law CER on structural mass
    3. Launch cost       — total wet mass lifted to LEO × cost_per_kg_to_leo

All figures in USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from optimizer.propellants import Propellant

COST_PER_KG_TO_LEO: float = 2_700.0   # USD/kg  (Falcon 9 baseline)


@dataclass
class MissionCost:
    """Itemised mission cost breakdown [USD]."""
    propellant_cost: float
    vehicle_cost:    float
    launch_cost:     float

    @property
    def total(self) -> float:
        return self.propellant_cost + self.vehicle_cost + self.launch_cost

    def __repr__(self) -> str:
        return (
            f"MissionCost:\n"
            f"  propellant : ${self.propellant_cost:>15,.0f}\n"
            f"  vehicle    : ${self.vehicle_cost:>15,.0f}\n"
            f"  launch     : ${self.launch_cost:>15,.0f}\n"
            f"  TOTAL      : ${self.total:>15,.0f}"
        )


def estimate_cost(
    propellant: Propellant,
    m_prop_consumed_kg: float,
    structural_mass_kg: float,
    initial_wet_mass_kg: float,
    cost_per_kg_to_leo: float = COST_PER_KG_TO_LEO,
) -> MissionCost:
    """
    Estimate total mission cost.

    Parameters
    ----------
    propellant           : Propellant  (carries cost_per_kg)
    m_prop_consumed_kg   : propellant actually burned across all phases [kg]
    structural_mass_kg   : dry stage mass (tanks, engine, structure) [kg]
    initial_wet_mass_kg  : total mass lifted to LEO at mission start [kg]
    cost_per_kg_to_leo   : launch cost [USD/kg to LEO]
    """
    propellant_cost = m_prop_consumed_kg * propellant.cost_per_kg
    vehicle_cost    = _vehicle_cer(structural_mass_kg)
    launch_cost     = initial_wet_mass_kg * cost_per_kg_to_leo

    return MissionCost(
        propellant_cost=propellant_cost,
        vehicle_cost=vehicle_cost,
        launch_cost=launch_cost,
    )


def _vehicle_cer(structural_mass_kg: float) -> float:
    """
    Parametric vehicle production cost (power-law CER).
    C = A × m^B,  A = 5000 USD/kg^B,  B = 0.6.
    """
    A, B = 5_000.0, 0.6
    return A * max(structural_mass_kg, 1.0) ** B