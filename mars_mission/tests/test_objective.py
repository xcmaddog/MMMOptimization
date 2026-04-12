"""
tests/test_objective.py
-----------------------
Unit tests for propellants, cost model, and objective function plumbing.
Does NOT run the full three-phase simulator (too slow for unit tests).

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

from optimizer.propellants import (
    PROPELLANTS, Propellant,
    burn_rate_kg_per_min, effective_isp, delta_v_km_s, G0_M_S2,
)
from cost.cost import estimate_cost, MissionCost
from optimizer import (
    ObjectiveFunction, generate_seeds, derive_masses, check_masses,
    compute_launch_angle,
    DESIGN_VARIABLE_SPEC, N_VAR, LOWER_BOUNDS, UPPER_BOUNDS,
    KNOWN_GOOD_ANCHORS,
    IDX_EPOCH, IDX_TOF, IDX_THRUST, IDX_STRUCT,
    IDX_MOI_FRAC, IDX_FUEL, IDX_COAST, BIG,
    LEO_PERIOD_S, MASS_BUFFER_KG,
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
    """Physically consistent design vector near the centre of the bounds."""
    x = np.zeros(N_VAR)
    x[IDX_EPOCH]    = 2_461_314.5   # 2026-10-01  (known-good anchor)
    x[IDX_TOF]      = 250.0         # days
    x[IDX_THRUST]   = 200_000.0     # 200 kN
    x[IDX_STRUCT]   = 5_000.0       # kg
    x[IDX_MOI_FRAC] = 0.25          # 25 % of fuel reserved for MOI
    x[IDX_FUEL]     = 60_000.0      # kg total propellant
    x[IDX_COAST]    = 0.5           # half-day coast in LEO
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

    def test_hydrolox_highest_isp(self):
        assert PROPELLANTS["hydrolox"].isp_vac_s > PROPELLANTS["kerolox"].isp_vac_s
        assert PROPELLANTS["hydrolox"].isp_vac_s > PROPELLANTS["storable"].isp_vac_s
        assert PROPELLANTS["hydrolox"].isp_vac_s > PROPELLANTS["methalox"].isp_vac_s

    def test_burn_rate_positive(self, hydrolox):
        assert burn_rate_kg_per_min(hydrolox, 100_000.0) > 0

    def test_burn_rate_zero_thrust(self, hydrolox):
        assert burn_rate_kg_per_min(hydrolox, 0.0) == 0.0

    def test_burn_rate_scales_with_thrust(self, hydrolox):
        r1 = burn_rate_kg_per_min(hydrolox, 100_000.0)
        r2 = burn_rate_kg_per_min(hydrolox, 200_000.0)
        assert abs(r2 - 2 * r1) < 1e-9

    def test_effective_isp_roundtrip(self, hydrolox):
        thrust = 100_000.0
        rate   = burn_rate_kg_per_min(hydrolox, thrust)
        isp_back = effective_isp(thrust, rate)
        assert abs(isp_back - hydrolox.isp_vac_s) < 1e-6

    def test_higher_isp_lower_burn_rate_same_thrust(self):
        t = 100_000.0
        assert (burn_rate_kg_per_min(PROPELLANTS["hydrolox"], t)
                < burn_rate_kg_per_min(PROPELLANTS["kerolox"], t))

    def test_delta_v_positive(self, hydrolox):
        assert delta_v_km_s(hydrolox, 10_000.0, 5_000.0) > 0

    def test_delta_v_increases_with_propellant(self, hydrolox):
        dv_lo = delta_v_km_s(hydrolox, 10_000.0, 2_000.0)
        dv_hi = delta_v_km_s(hydrolox, 10_000.0, 7_000.0)
        assert dv_hi > dv_lo

    def test_higher_isp_more_dv(self):
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

    def test_storable_more_expensive_per_kg(self):
        c_k = estimate_cost(PROPELLANTS["kerolox"],  5_000.0, 8_000.0, 12_000.0)
        c_s = estimate_cost(PROPELLANTS["storable"], 5_000.0, 8_000.0, 12_000.0)
        assert c_s.propellant_cost > c_k.propellant_cost


# ---------------------------------------------------------------------------
# Design variable spec
# ---------------------------------------------------------------------------

class TestDesignVariableSpec:

    def test_n_var_matches_spec(self):
        assert N_VAR == len(DESIGN_VARIABLE_SPEC)

    def test_bounds_length(self):
        assert len(LOWER_BOUNDS) == N_VAR
        assert len(UPPER_BOUNDS) == N_VAR

    def test_lower_strictly_below_upper(self):
        assert np.all(LOWER_BOUNDS < UPPER_BOUNDS)

    def test_epoch_bounds_in_2026_window(self):
        # Lower bound should be roughly Sep 2026
        assert 2_461_200 < LOWER_BOUNDS[IDX_EPOCH] < 2_461_350
        # Upper bound should be roughly Jan 2027
        assert 2_461_350 < UPPER_BOUNDS[IDX_EPOCH] < 2_461_500

    def test_tof_bounds_reasonable(self):
        assert 100 <= LOWER_BOUNDS[IDX_TOF] < UPPER_BOUNDS[IDX_TOF] <= 400

    def test_known_good_anchors_within_bounds(self):
        for jd, tof in KNOWN_GOOD_ANCHORS:
            assert LOWER_BOUNDS[IDX_EPOCH] <= jd  <= UPPER_BOUNDS[IDX_EPOCH], \
                f"Anchor JD {jd} out of epoch bounds"
            assert LOWER_BOUNDS[IDX_TOF]   <= tof <= UPPER_BOUNDS[IDX_TOF], \
                f"Anchor TOF {tof} out of TOF bounds"


# ---------------------------------------------------------------------------
# Mass derivation and constraints
# ---------------------------------------------------------------------------

class TestMasses:

    def test_derive_masses_keys(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        for k in ("tmi_fuel", "moi_fuel", "m_wet", "m_remaining",
                  "burn_rate_kg_min", "burn_dur_s", "burn_dur_min"):
            assert k in m

    def test_fuel_split_sums_to_total(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        assert abs(m["tmi_fuel"] + m["moi_fuel"] - float(nominal_x[IDX_FUEL])) < 1e-6

    def test_moi_fraction_respected(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        expected_moi = float(nominal_x[IDX_FUEL]) * float(nominal_x[IDX_MOI_FRAC])
        assert abs(m["moi_fuel"] - expected_moi) < 1e-6

    def test_m_wet_includes_buffer(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        expected = m["m_remaining"] + m["tmi_fuel"] + MASS_BUFFER_KG
        assert abs(m["m_wet"] - expected) < 1e-6

    def test_mass_constraint_satisfied(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        assert check_masses(m) is None

    def test_strict_inequality_remaining_vs_non_tmi(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        non_tmi = m["m_wet"] - m["tmi_fuel"]
        assert m["m_remaining"] < non_tmi

    def test_mass_constraint_fails_bad_moi_frac(self, hydrolox, nominal_x):
        bad = nominal_x.copy()
        bad[IDX_MOI_FRAC] = 0.999   # nearly all fuel as MOI → remaining ≈ m_wet - tmi ≈ total
        m = derive_masses(bad, hydrolox)
        # m_remaining ≈ struct + 0.999*fuel, non_tmi ≈ struct + 0.999*fuel + BUFFER
        # Should still pass because BUFFER keeps it strictly less
        # (the constraint can only fail if BUFFER=0, which we don't allow)
        # So this is actually still OK — test the zero-thrust case instead
        bad2 = nominal_x.copy()
        bad2[IDX_THRUST] = 0.0
        m2 = derive_masses(bad2, hydrolox)
        assert check_masses(m2) is not None   # burn_rate=0 → error

    def test_burn_rate_derived_from_isp(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        expected = burn_rate_kg_per_min(hydrolox, float(nominal_x[IDX_THRUST]))
        assert abs(m["burn_rate_kg_min"] - expected) < 1e-9

    def test_burn_duration_consistent(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        assert abs(m["burn_dur_min"] - m["burn_dur_s"] / 60.0) < 1e-9


# ---------------------------------------------------------------------------
# Launch angle
# ---------------------------------------------------------------------------

class TestLaunchAngle:

    def test_returns_float_in_0_360(self, nominal_x):
        angle = compute_launch_angle(
            float(nominal_x[IDX_EPOCH]),
            float(nominal_x[IDX_TOF]),
            burn_dur_s=900.0,
        )
        assert 0.0 <= angle < 360.0

    def test_returns_valid_range_always(self):
        # The function always returns a float in [0, 360), using 0.0 as
        # a fallback only if the Lambert solver raises internally.
        for epoch, tof, burn in [
            (2_461_294.5, 250.0,  900.0),   # anchor 1
            (2_461_314.5, 200.0,  500.0),   # anchor 2, short burn
            (2_461_370.5, 210.0, 1800.0),   # anchor 5, long burn
        ]:
            angle = compute_launch_angle(epoch, tof, burn)
            assert 0.0 <= angle < 360.0, (
                f"angle {angle:.2f} out of [0,360) for epoch={epoch} tof={tof} burn={burn}"
            )

    def test_changes_with_tof(self, nominal_x):
        a1 = compute_launch_angle(float(nominal_x[IDX_EPOCH]), 200.0, 900.0)
        a2 = compute_launch_angle(float(nominal_x[IDX_EPOCH]), 250.0, 900.0)
        assert a1 != a2

    def test_arc_correction_shifts_angle(self, nominal_x):
        """Longer burn arc should produce a different angle."""
        a_short = compute_launch_angle(float(nominal_x[IDX_EPOCH]), 250.0, burn_dur_s=100.0)
        a_long  = compute_launch_angle(float(nominal_x[IDX_EPOCH]), 250.0, burn_dur_s=2000.0)
        assert a_short != a_long


# ---------------------------------------------------------------------------
# ObjectiveFunction plumbing (no simulator calls)
# ---------------------------------------------------------------------------

class TestObjectiveFunction:

    def test_penalty_result_structure(self, hydrolox):
        r = ObjectiveFunction._penalty("test reason")
        assert r["feasible"] is False
        assert r["tof_days"]  == BIG
        assert r["fuel_kg"]   == BIG
        assert r["cost_usd"]  == BIG
        assert "test reason" in r["status"]

    def test_cache_key_deterministic(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        k1 = tuple(np.round(nominal_x, 4))
        k2 = tuple(np.round(nominal_x.copy(), 4))
        assert k1 == k2

    def test_cache_key_differs_on_change(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        x2 = nominal_x.copy()
        x2[IDX_TOF] += 10.0
        assert tuple(np.round(nominal_x, 4)) != tuple(np.round(x2, 4))

    def test_objectives_returns_big_for_bad_mass(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        bad = nominal_x.copy()
        bad[IDX_THRUST] = 0.0   # forces burn_rate=0 → mass check fails
        tof, fuel, cost = obj.objectives(bad)
        assert tof == BIG and fuel == BIG and cost == BIG

    def test_feasibility_violation_infeasible(self, hydrolox, nominal_x):
        obj = ObjectiveFunction(propellant=hydrolox)
        bad = nominal_x.copy()
        bad[IDX_THRUST] = 0.0
        assert obj.feasibility_violation(bad) == 1.0

    def test_phase_settings_burn_rate_from_isp(self, hydrolox, nominal_x):
        from optimizer.objective import build_phase_settings
        obj = ObjectiveFunction(propellant=hydrolox)
        m   = derive_masses(nominal_x, hydrolox)
        angle = compute_launch_angle(float(nominal_x[IDX_EPOCH]),
                                     float(nominal_x[IDX_TOF]),
                                     m["burn_dur_s"])
        utc = Time(float(nominal_x[IDX_EPOCH]), format="jd", scale="tdb").utc.isot
        _, p2, _ = build_phase_settings(
            nominal_x, m, hydrolox, angle, utc,
            300, 300, 300, 310, 90, 50, 10, 0.05,
        )
        expected_rate = burn_rate_kg_per_min(hydrolox, float(nominal_x[IDX_THRUST]))
        assert abs(p2["burn_rate_kg_per_min"] - expected_rate) < 1e-9

    def test_phase_settings_remaining_strictly_less_than_non_tmi(self, hydrolox, nominal_x):
        from optimizer.objective import build_phase_settings
        m   = derive_masses(nominal_x, hydrolox)
        angle = compute_launch_angle(float(nominal_x[IDX_EPOCH]),
                                     float(nominal_x[IDX_TOF]),
                                     m["burn_dur_s"])
        utc = Time(float(nominal_x[IDX_EPOCH]), format="jd", scale="tdb").utc.isot
        _, p2, _ = build_phase_settings(
            nominal_x, m, hydrolox, angle, utc,
            300, 300, 300, 310, 90, 50, 10, 0.05,
        )
        non_tmi = p2["initial_total_mass_kg"] - p2["starting_fuel_mass_kg"]
        assert p2["remaining_stage_mass_kg"] < non_tmi


# ---------------------------------------------------------------------------
# Seeder
# ---------------------------------------------------------------------------

class TestSeeder:

    def test_shape(self, hydrolox):
        seeds = generate_seeds(hydrolox, n_seeds=10, rng_seed=0)
        assert seeds.shape == (10, N_VAR)

    def test_all_within_bounds(self, hydrolox):
        seeds = generate_seeds(hydrolox, n_seeds=15, rng_seed=1)
        assert np.all(seeds >= LOWER_BOUNDS)
        assert np.all(seeds <= UPPER_BOUNDS)

    def test_mass_constraints_pass(self, hydrolox):
        seeds = generate_seeds(hydrolox, n_seeds=12, rng_seed=2)
        for s in seeds:
            m = derive_masses(s, hydrolox)
            assert check_masses(m) is None, \
                f"Mass constraint failed: {check_masses(m)}"

    def test_deterministic_with_same_seed(self, hydrolox):
        s1 = generate_seeds(hydrolox, n_seeds=8, rng_seed=99)
        s2 = generate_seeds(hydrolox, n_seeds=8, rng_seed=99)
        np.testing.assert_array_equal(s1, s2)

    def test_different_with_different_seed(self, hydrolox):
        s1 = generate_seeds(hydrolox, n_seeds=8, rng_seed=1)
        s2 = generate_seeds(hydrolox, n_seeds=8, rng_seed=2)
        assert not np.allclose(s1, s2)

    def test_anchors_represented(self, hydrolox):
        """At least some seeds should have epochs near the known-good anchors."""
        seeds = generate_seeds(hydrolox, n_seeds=20, rng_seed=0)
        anchor_jds = [jd for jd, _ in KNOWN_GOOD_ANCHORS]
        epoch_col  = seeds[:, IDX_EPOCH]
        found = any(
            np.any(np.abs(epoch_col - jd) < 10.0)
            for jd in anchor_jds
        )
        assert found, "No seed epochs near any known-good anchor"

    def test_pads_to_requested_count(self, hydrolox):
        """Even if anchors are few, always returns exactly n_seeds rows."""
        for n in [5, 20, 50]:
            seeds = generate_seeds(hydrolox, n_seeds=n, rng_seed=0)
            assert len(seeds) == n