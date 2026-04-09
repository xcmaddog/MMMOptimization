"""
tests/test_objective.py
-----------------------
Unit tests for propellant helpers, cost model, and objective function
plumbing — without running the full simulator (which is slow).

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from optimizer.propellants import (
    PROPELLANTS, Propellant,
    burn_rate_kg_per_min, effective_isp, delta_v_km_s, G0_M_S2,
)
from cost.cost import estimate_cost, MissionCost
from optimizer.objective import (
    ObjectiveFunction, DESIGN_VARIABLE_SPEC,
    LOWER_BOUNDS, UPPER_BOUNDS, N_VAR, BIG,
    IDX_EPOCH, IDX_COAST, IDX_ANGLE, IDX_BURN_DUR,
    IDX_THRUST, IDX_TOTAL_MASS, IDX_FUEL_MASS, IDX_STAGE_MASS, IDX_MOI_DUR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hydrolox():
    return PROPELLANTS["hydrolox"]

@pytest.fixture
def kerolox():
    return PROPELLANTS["kerolox"]

@pytest.fixture
def nominal_x():
    """A physically consistent design vector near the centre of the bounds."""
    x = np.zeros(N_VAR)
    x[IDX_EPOCH]      = 2_461_000.5   # mid-window JD
    x[IDX_COAST]      = 0.5           # half-day coast in LEO
    x[IDX_ANGLE]      = 133.0         # launch angle
    x[IDX_BURN_DUR]   = 60.0          # 60-min burn
    x[IDX_THRUST]     = 100_000.0     # 100 kN
    x[IDX_TOTAL_MASS] = 12_000.0      # 12 t wet
    x[IDX_FUEL_MASS]  = 8_500.0       # 8.5 t propellant
    x[IDX_STAGE_MASS] = 2_500.0       # 2.5 t remaining stage
    x[IDX_MOI_DUR]    = 32.0          # 32-min MOI burn
    return x


# ---------------------------------------------------------------------------
# Propellant catalogue
# ---------------------------------------------------------------------------

class TestPropellants:
    def test_all_keys_present(self):
        for key in ("kerolox", "hydrolox", "storable", "methalox"):
            assert key in PROPELLANTS

    def test_isp_positive(self):
        for p in PROPELLANTS.values():
            assert p.isp_vac_s > 0

    def test_hydrolox_higher_isp_than_kerolox(self):
        assert PROPELLANTS["hydrolox"].isp_vac_s > PROPELLANTS["kerolox"].isp_vac_s

    def test_burn_rate_positive(self, hydrolox):
        rate = burn_rate_kg_per_min(hydrolox, thrust_newtons=100_000.0)
        assert rate > 0.0

    def test_burn_rate_zero_thrust(self, hydrolox):
        rate = burn_rate_kg_per_min(hydrolox, thrust_newtons=0.0)
        assert rate == 0.0

    def test_effective_isp_roundtrip(self, hydrolox):
        thrust = 100_000.0
        rate   = burn_rate_kg_per_min(hydrolox, thrust)
        isp_back = effective_isp(thrust, rate)
        assert abs(isp_back - hydrolox.isp_vac_s) < 1e-6

    def test_higher_isp_lower_burn_rate(self):
        """Same thrust → hydrolox uses less fuel per minute than kerolox."""
        thrust = 100_000.0
        rate_h = burn_rate_kg_per_min(PROPELLANTS["hydrolox"], thrust)
        rate_k = burn_rate_kg_per_min(PROPELLANTS["kerolox"],  thrust)
        assert rate_h < rate_k

    def test_delta_v_positive(self, hydrolox):
        dv = delta_v_km_s(hydrolox, m_wet_kg=10_000.0, m_prop_kg=5_000.0)
        assert dv > 0.0

    def test_delta_v_increases_with_propellant(self, hydrolox):
        dv_lo = delta_v_km_s(hydrolox, m_wet_kg=10_000.0, m_prop_kg=2_000.0)
        dv_hi = delta_v_km_s(hydrolox, m_wet_kg=10_000.0, m_prop_kg=7_000.0)
        assert dv_hi > dv_lo

    def test_delta_v_higher_isp_gives_more_dv(self):
        m_wet, m_prop = 10_000.0, 5_000.0
        dv_h = delta_v_km_s(PROPELLANTS["hydrolox"], m_wet, m_prop)
        dv_k = delta_v_km_s(PROPELLANTS["kerolox"],  m_wet, m_prop)
        assert dv_h > dv_k


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_returns_mission_cost(self, hydrolox):
        c = estimate_cost(hydrolox, 5_000.0, 8_000.0, 12_000.0)
        assert isinstance(c, MissionCost)

    def test_total_equals_sum(self, hydrolox):
        c = estimate_cost(hydrolox, 5_000.0, 8_000.0, 12_000.0)
        assert abs(c.total - (c.propellant_cost + c.vehicle_cost + c.launch_cost)) < 1e-6

    def test_all_components_positive(self, hydrolox):
        c = estimate_cost(hydrolox, 5_000.0, 8_000.0, 12_000.0)
        assert c.propellant_cost > 0
        assert c.vehicle_cost    > 0
        assert c.launch_cost     > 0

    def test_more_fuel_costs_more(self, hydrolox):
        c_lo = estimate_cost(hydrolox, 2_000.0, 8_000.0, 12_000.0)
        c_hi = estimate_cost(hydrolox, 8_000.0, 8_000.0, 12_000.0)
        assert c_hi.total > c_lo.total

    def test_storable_propellant_costs_more_per_kg_than_kerolox(self):
        m = 5_000.0
        c_kero  = estimate_cost(PROPELLANTS["kerolox"],  m, 8_000, 12_000)
        c_store = estimate_cost(PROPELLANTS["storable"], m, 8_000, 12_000)
        assert c_store.propellant_cost > c_kero.propellant_cost


# ---------------------------------------------------------------------------
# Design variable spec
# ---------------------------------------------------------------------------

class TestDesignVariableSpec:
    def test_n_var_matches_spec(self):
        assert N_VAR == len(DESIGN_VARIABLE_SPEC)

    def test_bounds_arrays_correct_length(self):
        assert len(LOWER_BOUNDS) == N_VAR
        assert len(UPPER_BOUNDS) == N_VAR

    def test_lower_below_upper(self):
        assert np.all(LOWER_BOUNDS < UPPER_BOUNDS)


# ---------------------------------------------------------------------------
# ObjectiveFunction — unit tests without running the real simulator
# ---------------------------------------------------------------------------

class TestObjectiveFunction:
    def test_unpack_roundtrip(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        v   = obj._unpack(nominal_x)
        assert abs(v["launch_angle_deg"]       - nominal_x[IDX_ANGLE])      < 1e-9
        assert abs(v["burn_duration_min"]       - nominal_x[IDX_BURN_DUR])   < 1e-9
        assert abs(v["starting_fuel_mass_kg"]   - nominal_x[IDX_FUEL_MASS])  < 1e-9

    def test_epoch_to_utc_string(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        utc = obj._epoch_to_utc_string(nominal_x[IDX_EPOCH])
        # Just check it's a parseable ISO string
        from astropy.time import Time
        t = Time(utc, format="isot", scale="utc")
        assert t.jd > 0

    def test_burn_rate_derived_from_isp(self, hydrolox, nominal_x):
        obj    = ObjectiveFunction(propellant=hydrolox)
        v      = obj._unpack(nominal_x)
        _, p2, _ = obj._build_phase_settings(v)
        rate_expected = burn_rate_kg_per_min(hydrolox, nominal_x[IDX_THRUST])
        assert abs(p2["burn_rate_kg_per_min"] - rate_expected) < 1e-9

    def test_mass_constraint_fuel_exceeds_total(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        bad_x = nominal_x.copy()
        bad_x[IDX_FUEL_MASS]  = 15_000.0   # > initial_total_mass_kg = 12 000
        bad_x[IDX_TOTAL_MASS] = 12_000.0
        v = obj._unpack(bad_x)
        err = obj._validate_mass_constraints(v)
        assert err is not None

    def test_mass_constraint_stage_exceeds_non_fuel(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        bad_x = nominal_x.copy()
        # non-fuel = 12000 - 8500 = 3500 kg,  but stage = 4000 kg
        bad_x[IDX_STAGE_MASS] = 4_000.0
        v = obj._unpack(bad_x)
        err = obj._validate_mass_constraints(v)
        assert err is not None

    def test_valid_mass_constraint_passes(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        v = obj._unpack(nominal_x)
        err = obj._validate_mass_constraints(v)
        assert err is None

    def test_cache_key_consistent(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        k1 = obj._cache_key(nominal_x)
        k2 = obj._cache_key(nominal_x.copy())
        assert k1 == k2

    def test_cache_key_differs_on_change(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        x2  = nominal_x.copy()
        x2[IDX_ANGLE] += 1.0
        assert obj._cache_key(nominal_x) != obj._cache_key(x2)

    def test_penalty_result_structure(self, hydrolox):
        r = ObjectiveFunction._penalty_result("test")
        assert r["feasible"] is False
        assert r["tof_days"] == BIG
        assert r["fuel_kg"]  == BIG
        assert r["cost_usd"] == BIG

    def test_phase_settings_include_isp_derived_burn_rate(self, hydrolox, nominal_x):
        """burn_rate in phase2 settings should match Isp-derived value."""
        obj = ObjectiveFunction(propellant=hydrolox)
        v   = obj._unpack(nominal_x)
        _, p2, _ = obj._build_phase_settings(v)
        expected = burn_rate_kg_per_min(hydrolox, v["thrust_newtons"])
        assert abs(p2["burn_rate_kg_per_min"] - expected) < 1e-9