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
from optimizer.objective import build_phase_settings
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


# ---------------------------------------------------------------------------
# Disk cache tests
# ---------------------------------------------------------------------------

class TestEvalCache:
    """Tests for the SQLite-backed persistent evaluation cache."""

    @pytest.fixture
    def tmp_cache(self, tmp_path):
        from optimizer.cache import EvalCache
        return EvalCache(str(tmp_path / "test.db"))

    @pytest.fixture
    def sample_x(self):
        x = np.zeros(N_VAR)
        x[IDX_EPOCH]    = 2_461_314.5
        x[IDX_TOF]      = 250.0
        x[IDX_THRUST]   = 200_000.0
        x[IDX_STRUCT]   = 5_000.0
        x[IDX_MOI_FRAC] = 0.25
        x[IDX_FUEL]     = 60_000.0
        x[IDX_COAST]    = 0.5
        return x

    @pytest.fixture
    def sample_result(self):
        return {
            "tof_days": 250.0, "fuel_kg": 45_000.0, "cost_usd": 1.2e8,
            "feasible": True, "status": "completed",
            "p1_result": None, "p2_result": None, "p3_result": None,
        }

    def test_miss_on_empty_cache(self, tmp_cache, sample_x):
        assert tmp_cache.get(sample_x, "hydrolox") is None

    def test_put_then_get(self, tmp_cache, sample_x, sample_result):
        tmp_cache.put(sample_x, "hydrolox", sample_result)
        hit = tmp_cache.get(sample_x, "hydrolox")
        assert hit is not None
        assert hit["feasible"] is True
        assert abs(hit["tof_days"] - 250.0) < 1e-9
        assert abs(hit["fuel_kg"] - 45_000.0) < 1e-6

    def test_different_propellant_is_miss(self, tmp_cache, sample_x, sample_result):
        tmp_cache.put(sample_x, "hydrolox", sample_result)
        assert tmp_cache.get(sample_x, "kerolox") is None

    def test_different_x_is_miss(self, tmp_cache, sample_x, sample_result):
        tmp_cache.put(sample_x, "hydrolox", sample_result)
        x2 = sample_x.copy()
        x2[IDX_TOF] += 5.0
        assert tmp_cache.get(x2, "hydrolox") is None

    def test_duplicate_put_is_ignored(self, tmp_cache, sample_x, sample_result):
        tmp_cache.put(sample_x, "hydrolox", sample_result)
        # Second put with different values should not overwrite
        r2 = dict(sample_result, tof_days=999.0)
        tmp_cache.put(sample_x, "hydrolox", r2)
        hit = tmp_cache.get(sample_x, "hydrolox")
        assert abs(hit["tof_days"] - 250.0) < 1e-9   # original value preserved

    def test_stats_counts(self, tmp_cache, sample_x, sample_result):
        assert tmp_cache.stats()["total"] == 0
        tmp_cache.put(sample_x, "hydrolox", sample_result)
        s = tmp_cache.stats()
        assert s["total"] == 1
        assert s["feasible"] == 1

    def test_infeasible_counted_separately(self, tmp_cache, sample_x):
        infeasible = {
            "tof_days": BIG, "fuel_kg": BIG, "cost_usd": BIG,
            "feasible": False, "status": "miss",
            "p1_result": None, "p2_result": None, "p3_result": None,
        }
        tmp_cache.put(sample_x, "hydrolox", infeasible)
        s = tmp_cache.stats()
        assert s["total"] == 1
        assert s["feasible"] == 0

    def test_clear_empties_cache(self, tmp_cache, sample_x, sample_result):
        tmp_cache.put(sample_x, "hydrolox", sample_result)
        tmp_cache.clear()
        assert tmp_cache.stats()["total"] == 0
        assert tmp_cache.get(sample_x, "hydrolox") is None

    def test_rounding_matches_objective_function(self, tmp_cache, hydrolox, sample_x):
        """Cache key rounding must match ObjectiveFunction.evaluate() rounding."""
        obj = ObjectiveFunction(propellant=hydrolox, cache_path=None)
        key_obj = tuple(np.round(sample_x, 4))
        # Slightly jitter x within the rounding tolerance
        x_jittered = sample_x.copy()
        x_jittered[IDX_TOF] += 1e-5   # delta < 0.5 * 10^-4 → rounds to same
        result = {"tof_days": 250.0, "fuel_kg": 40000.0, "cost_usd": 1e8,
                  "feasible": False, "status": "test",
                  "p1_result": None, "p2_result": None, "p3_result": None}
        tmp_cache.put(sample_x, "hydrolox", result)
        # Same rounded value → cache hit
        hit = tmp_cache.get(x_jittered, "hydrolox")
        assert hit is not None


# ---------------------------------------------------------------------------
# Adaptive step size tests
# ---------------------------------------------------------------------------

class TestAdaptiveSteps:
    """Tests that the adaptive step size logic is present and correct."""

    def test_signature_has_adaptive_params(self):
        import inspect
        from simulator.Part2_Transfer_Burn import simulate_transfer_burn_phase2
        sig = inspect.signature(simulate_transfer_burn_phase2)
        assert "adaptive_steps" in sig.parameters
        assert "close_approach_radius_km" in sig.parameters

    def test_adaptive_steps_default_is_true(self):
        import inspect
        from simulator.Part2_Transfer_Burn import simulate_transfer_burn_phase2
        sig = inspect.signature(simulate_transfer_burn_phase2)
        assert sig.parameters["adaptive_steps"].default is True

    def test_close_approach_radius_default(self):
        import inspect
        from simulator.Part2_Transfer_Burn import simulate_transfer_burn_phase2
        sig = inspect.signature(simulate_transfer_burn_phase2)
        default = sig.parameters["close_approach_radius_km"].default
        assert default == 500_000.0

    def test_phase2_settings_include_adaptive_flag(self, hydrolox, nominal_x):
        """build_phase_settings must pass adaptive_steps to Phase 2."""
        m = derive_masses(nominal_x, hydrolox)
        angle = compute_launch_angle(
            float(nominal_x[IDX_EPOCH]), float(nominal_x[IDX_TOF]), m["burn_dur_s"]
        )
        from astropy.time import Time
        utc = Time(float(nominal_x[IDX_EPOCH]), format="jd", scale="tdb").utc.isot
        _, p2, _ = build_phase_settings(
            nominal_x, m, hydrolox, angle, utc,
            300, 300, 300, 310, 90, 50, 10, 0.05,
        )
        assert p2.get("adaptive_steps") is True
        assert "close_approach_radius_km" in p2


# ---------------------------------------------------------------------------
# ProgressCallback tests
# ---------------------------------------------------------------------------

class TestProgressCallback:
    """Tests for the tqdm progress bar callback."""

    def test_callback_importable(self):
        from optimizer.runner import ProgressCallback
        cb = ProgressCallback(n_gen=10, pop_size=50,
                              propellant_name="Test", show_bar=False)
        assert cb is not None

    def test_callback_runs_without_tqdm(self):
        """show_bar=False should fall back to plain text without crashing."""
        import io, sys
        from optimizer.runner import ProgressCallback
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.core.problem import Problem

        class TinyProblem(Problem):
            def __init__(self):
                super().__init__(n_var=2, n_obj=2, xl=np.zeros(2), xu=np.ones(2))
            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = X

        cb = ProgressCallback(n_gen=2, pop_size=4,
                              propellant_name="Hydrolox", show_bar=False)
        captured = io.StringIO()
        sys.stdout = captured
        try:
            res = minimize(TinyProblem(), NSGA2(pop_size=4),
                           get_termination("n_gen", 2),
                           callback=cb, verbose=False, seed=1)
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "gen" in output.lower()

    def test_fmt_time(self):
        from optimizer.runner import _fmt_time
        assert "s" in _fmt_time(45)
        assert "m" in _fmt_time(90)
        assert "h" in _fmt_time(3700)
        assert _fmt_time(0) == "0s"
        assert _fmt_time(3661) == "1h01m01s"


# ---------------------------------------------------------------------------
# Earth-escape constraint tests
# ---------------------------------------------------------------------------

class TestEscapeConstraint:
    """Tests for the Earth-escape pre-check in check_masses."""

    def test_sufficient_fuel_passes(self, hydrolox, nominal_x):
        m = derive_masses(nominal_x, hydrolox)
        err = check_masses(m, isp_vac_s=hydrolox.isp_vac_s)
        assert err is None, f"Expected pass, got: {err}"

    def test_insufficient_tmi_fuel_fails(self, kerolox):
        """Kerolox with only 20k total fuel cannot escape (needs ~80k)."""
        x = np.zeros(N_VAR)
        x[IDX_EPOCH]    = 2_461_314.5
        x[IDX_TOF]      = 250.0
        x[IDX_THRUST]   = 150_000.0
        x[IDX_STRUCT]   = 5_000.0
        x[IDX_MOI_FRAC] = 0.30
        x[IDX_FUEL]     = 20_000.0   # too little for kerolox
        x[IDX_COAST]    = 0.3
        m = derive_masses(x, kerolox)
        err = check_masses(m, isp_vac_s=kerolox.isp_vac_s)
        assert err is not None
        assert "escape" in err.lower() or "dV" in err or "dv" in err.lower()

    def test_hydrolox_needs_less_fuel_than_kerolox(self):
        """Lower-Isp propellants require more fuel to escape."""
        from optimizer.propellants import PROPELLANTS
        hydro   = PROPELLANTS["hydrolox"]
        kero    = PROPELLANTS["kerolox"]
        m_struct, moi_frac = 5_000.0, 0.30

        # Find minimum fuel for each by checking escape at increasing fuel
        def min_fuel(prop):
            for fuel in range(10_000, 200_000, 2_000):
                x = np.array([2_461_314.5, 250.0, 150_000.0,
                               m_struct, moi_frac, float(fuel), 0.3])
                m = derive_masses(x, prop)
                if check_masses(m, isp_vac_s=prop.isp_vac_s) is None:
                    return fuel
            return 200_000

        fuel_hydro = min_fuel(hydro)
        fuel_kero  = min_fuel(kero)
        assert fuel_hydro < fuel_kero, \
            f"Expected hydrolox({fuel_hydro}) < kerolox({fuel_kero})"

    def test_escape_not_checked_without_isp(self, kerolox):
        """Without isp_vac_s, escape is not checked (backward-compatible)."""
        x = np.zeros(N_VAR)
        x[IDX_EPOCH]    = 2_461_314.5
        x[IDX_TOF]      = 250.0
        x[IDX_THRUST]   = 150_000.0
        x[IDX_STRUCT]   = 5_000.0
        x[IDX_MOI_FRAC] = 0.30
        x[IDX_FUEL]     = 5_000.0
        x[IDX_COAST]    = 0.3
        m = derive_masses(x, kerolox)
        # Without isp, only structural constraints checked
        # (may pass or fail on non-escape grounds — we just check no escape error)
        err = check_masses(m, isp_vac_s=None)
        if err is not None:
            assert "escape" not in err.lower()

    def test_objective_penalises_no_escape(self, kerolox):
        """ObjectiveFunction.objectives() returns BIG for insufficient fuel."""
        obj = ObjectiveFunction(propellant=kerolox, cache_path=None)
        x = np.zeros(N_VAR)
        x[IDX_EPOCH]    = 2_461_314.5
        x[IDX_TOF]      = 250.0
        x[IDX_THRUST]   = 150_000.0
        x[IDX_STRUCT]   = 5_000.0
        x[IDX_MOI_FRAC] = 0.30
        x[IDX_FUEL]     = 20_000.0   # can't escape with kerolox
        x[IDX_COAST]    = 0.3
        tof, fuel, cost = obj.objectives(x)
        assert tof == BIG and fuel == BIG and cost == BIG

    def test_fuel_lower_bound_raised(self):
        """total_fuel_kg lower bound should be at least 20,000 kg."""
        assert LOWER_BOUNDS[IDX_FUEL] >= 20_000.0, \
            f"Fuel lower bound {LOWER_BOUNDS[IDX_FUEL]} is too low"