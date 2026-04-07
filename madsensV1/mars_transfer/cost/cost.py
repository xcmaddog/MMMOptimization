"""
cost.py
-------
Low-fidelity parametric mission cost model.

Three components:
    1. Propellant cost  — mass * cost_per_kg
    2. Vehicle cost     — parametric CER on structural mass (power-law)
    3. Launch cost      — wet mass * cost_per_kg_to_LEO

All values in USD.

These are trade-study estimates, not high-fidelity cost models.
Adjust COST_PER_KG_TO_LEO and the CER coefficients as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from mars_transfer.vehicle.vehicle import VehicleConfig

# Current market: Falcon 9 ~$2,700/kg, Falcon Heavy ~$1,700/kg to LEO.
# Use Falcon 9 as the conservative baseline.
COST_PER_KG_TO_LEO: float = 2_700.0   # USD/kg


@dataclass
class MissionCost:
    """Itemised cost breakdown [USD]."""
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
    vehicle: VehicleConfig,
    m_prop_consumed: float,
    cost_per_kg_to_leo: float = COST_PER_KG_TO_LEO,
) -> MissionCost:
    """
    Estimate total mission cost given a vehicle and propellant consumed.

    Parameters
    ----------
    vehicle             : VehicleConfig
    m_prop_consumed     : propellant mass actually burned [kg]
    cost_per_kg_to_leo  : launch cost [USD/kg]

    Returns
    -------
    MissionCost
    """
    propellant_cost = m_prop_consumed * vehicle.propellant.cost_per_kg
    vehicle_cost    = _vehicle_cer(vehicle.structural_mass_kg)
    m_wet           = m_prop_consumed + vehicle.structural_mass_kg + vehicle.payload_mass_kg
    launch_cost     = m_wet * cost_per_kg_to_leo

    return MissionCost(
        propellant_cost=propellant_cost,
        vehicle_cost=vehicle_cost,
        launch_cost=launch_cost,
    )


def _vehicle_cer(structural_mass_kg: float) -> float:
    """
    Parametric vehicle production cost (power-law CER).
    C = A * m^B

    Coefficients calibrated roughly against liquid upper-stage analogues
    (Centaur, ICPS).  A = 5000 USD/kg^B, B = 0.6 (economy of scale).
    """
    A, B = 5_000.0, 0.6
    return A * structural_mass_kg ** B
