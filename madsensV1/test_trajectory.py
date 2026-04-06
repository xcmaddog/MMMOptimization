"""
Tests for the mars_transfer package.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from astropy.time import Time

from mars_transfer.ephemeris.ephemeris import (
    get_heliocentric_state, epoch_from_date, epoch_range,
)
from mars_transfer.trajectory.lambert import (
    solve_lambert, _lambert_universal, _oberth_burn, _S, _C,
    MU_SUN, MU_EARTH, MU_MARS, R_EARTH, R_MARS, R_LEO, R_MOI,
)
from mars_transfer.vehicle.vehicle import (
    PROPELLANTS, VehicleConfig,
    propellant_mass_required, max_delta_v, G0,
)
from mars_transfer.cost.cost import estimate_cost, MissionCost


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nominal_epoch():
    return epoch_from_date(2026, 11, 26)

@pytest.fixture
def base_vehicle():
    return VehicleConfig(
        payload_mass_kg    = 5_000.0,
        structural_mass_kg = 8_000.0,
        propellant         = PROPELLANTS["hydrolox"],
        max_propellant_kg  = 150_000.0,
    )


# ---------------------------------------------------------------------------
# Ephemeris tests
# ---------------------------------------------------------------------------

class TestEphemeris:
    def test_earth_state_shape(self, nominal_epoch):
        r, v = get_heliocentric_state("earth", nominal_epoch)
        assert r.shape == (3,) and v.shape == (3,)

    def test_earth_distance_plausible(self, nominal_epoch):
        """Earth should be ~1 AU from the Sun."""
        AU = 1.496e8   # km
        r, _ = get_heliocentric_state("earth", nominal_epoch)
        assert 0.97 * AU < np.linalg.norm(r) < 1.03 * AU

    def test_mars_distance_plausible(self, nominal_epoch):
        """Mars should be between 1.38 and 1.67 AU."""
        AU = 1.496e8
        r, _ = get_heliocentric_state("mars", nominal_epoch)
        dist = np.linalg.norm(r)
        assert 1.38 * AU < dist < 1.67 * AU

    def test_earth_speed_plausible(self, nominal_epoch):
        """Earth's heliocentric speed ≈ 30 km/s."""
        _, v = get_heliocentric_state("earth", nominal_epoch)
        assert 28.0 < np.linalg.norm(v) < 32.0

    def test_invalid_body_raises(self, nominal_epoch):
        with pytest.raises(ValueError):
            get_heliocentric_state("jupiter", nominal_epoch)

    def test_epoch_range_length(self, nominal_epoch):
        end = epoch_from_date(2027, 3, 31)
        epochs = epoch_range(nominal_epoch, end, 50)
        assert len(epochs) == 50

    def test_epoch_range_monotonic(self, nominal_epoch):
        end = epoch_from_date(2027, 3, 31)
        epochs = epoch_range(nominal_epoch, end, 20)
        jds = [e.jd for e in epochs]
        assert all(jds[i] < jds[i+1] for i in range(len(jds)-1))


# ---------------------------------------------------------------------------
# Stumpff functions
# ---------------------------------------------------------------------------

class TestStumpff:
    def test_S_at_zero(self):
        assert abs(_S(0.0) - 1/6) < 1e-10

    def test_C_at_zero(self):
        assert abs(_C(0.0) - 0.5) < 1e-10

    def test_S_positive(self):
        """For z=pi^2 (semicircle), S = (pi - 2) / pi^2 ≈ 0.0908."""
        z = np.pi**2
        expected = (np.pi - np.sin(np.pi)) / np.pi**3  # standard formula
        assert abs(_S(z) - (np.sqrt(z) - np.sin(np.sqrt(z))) / z**1.5) < 1e-10

    def test_C_positive(self):
        z = np.pi**2
        assert abs(_C(z) - (1 - np.cos(np.pi)) / z) < 1e-10

    def test_S_negative(self):
        z = -1.0
        sq = np.sqrt(1.0)
        expected = (np.sinh(sq) - sq) / sq**3
        assert abs(_S(z) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Lambert solver
# ---------------------------------------------------------------------------

class TestLambert:
    """Validate against hapsira reference values (computed offline)."""

    # Reference from hapsira on these exact inputs:
    # r1 = [-1.47e8+1e4, 1e5, 0]  r2 = [1.52e8, 1.5e8, 0]  tof=210 days
    R1_REF = np.array([-1.46990e8, 1.0e5, 0.0])
    R2_REF = np.array([ 1.52000e8, 1.5e8, 0.0])
    TOF_S  = 210 * 86400.0
    V1_REF = np.array([ 9.30731888, -30.97187625, 0.0])
    V2_REF = np.array([-11.15307939, 18.93857533,  0.0])

    def test_lambert_matches_reference_v1(self):
        v1, _ = _lambert_universal(MU_SUN, self.R1_REF, self.R2_REF, self.TOF_S)
        assert np.linalg.norm(v1 - self.V1_REF) < 1e-4   # km/s tolerance

    def test_lambert_matches_reference_v2(self):
        _, v2 = _lambert_universal(MU_SUN, self.R1_REF, self.R2_REF, self.TOF_S)
        assert np.linalg.norm(v2 - self.V2_REF) < 1e-4

    def test_lambert_collinear_raises(self):
        r1 = np.array([1e8, 0.0, 0.0])
        r2 = np.array([2e8, 0.0, 0.0])
        with pytest.raises(ValueError, match="[Cc]ollinear"):
            _lambert_universal(MU_SUN, r1, r2, 200 * 86400.0)

    def test_solve_lambert_returns_keys(self, nominal_epoch):
        result = solve_lambert(nominal_epoch, 210.0)
        for key in ("dv_tmi", "dv_moi", "dv_total", "v_inf_depart",
                    "v_inf_arrive", "C3", "tof_days"):
            assert key in result

    def test_solve_lambert_dv_total_plausible(self, nominal_epoch):
        """Total ΔV for Earth→Mars should be 4–12 km/s for typical geometries."""
        result = solve_lambert(nominal_epoch, 210.0)
        assert 4.0 < result["dv_total"] < 12.0

    def test_solve_lambert_c3_nonnegative(self, nominal_epoch):
        result = solve_lambert(nominal_epoch, 210.0)
        assert result["C3"] >= 0.0

    def test_solve_lambert_negative_tof_raises(self, nominal_epoch):
        with pytest.raises(ValueError):
            solve_lambert(nominal_epoch, -10.0)

    def test_oberth_burn_positive(self):
        dv = _oberth_burn(3.0, MU_EARTH, R_LEO)
        assert dv > 0.0

    def test_oberth_burn_increases_with_vinf(self):
        dv_lo = _oberth_burn(2.0, MU_EARTH, R_LEO)
        dv_hi = _oberth_burn(5.0, MU_EARTH, R_LEO)
        assert dv_hi > dv_lo

    def test_oberth_zero_vinf_is_zero(self):
        """v_inf=0 means we're on an escape trajectory: dv = v_esc - v_circ."""
        dv = _oberth_burn(0.0, MU_EARTH, R_LEO)
        v_circ = np.sqrt(MU_EARTH / R_LEO)
        v_esc  = np.sqrt(2 * MU_EARTH / R_LEO)
        assert abs(dv - (v_esc - v_circ)) < 1e-10


# ---------------------------------------------------------------------------
# Rocket equation
# ---------------------------------------------------------------------------

class TestRocketEquation:
    def test_roundtrip(self):
        """propellant_mass_required and max_delta_v are inverses."""
        isp, m_dry, dv_in = 450.0, 13_000.0, 7.2
        m_prop = propellant_mass_required(dv_in, isp, m_dry)
        dv_out = max_delta_v(isp, m_prop, m_dry)
        assert abs(dv_out - dv_in) < 1e-8

    def test_more_isp_less_propellant(self):
        m_dry, dv = 13_000.0, 7.0
        m_kero  = propellant_mass_required(dv, PROPELLANTS["kerolox"].isp_vac,  m_dry)
        m_hydro = propellant_mass_required(dv, PROPELLANTS["hydrolox"].isp_vac, m_dry)
        assert m_hydro < m_kero   # higher Isp → less propellant

    def test_propellant_positive(self):
        assert propellant_mass_required(5.0, 311.0, 5000.0) > 0

    def test_dv_increases_with_mass_ratio(self):
        isp, m_dry = 380.0, 10_000.0
        dv_small = max_delta_v(isp, 10_000.0, m_dry)
        dv_large = max_delta_v(isp, 50_000.0, m_dry)
        assert dv_large > dv_small


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_cost_is_positive(self, base_vehicle):
        cost = estimate_cost(base_vehicle, m_prop_consumed=50_000.0)
        assert isinstance(cost, MissionCost)
        assert cost.total > 0

    def test_all_components_positive(self, base_vehicle):
        cost = estimate_cost(base_vehicle, m_prop_consumed=50_000.0)
        assert cost.propellant_cost > 0
        assert cost.vehicle_cost    > 0
        assert cost.launch_cost     > 0

    def test_total_equals_sum(self, base_vehicle):
        cost = estimate_cost(base_vehicle, m_prop_consumed=40_000.0)
        assert abs(cost.total - (cost.propellant_cost +
                                 cost.vehicle_cost +
                                 cost.launch_cost)) < 1e-6

    def test_more_propellant_costs_more(self, base_vehicle):
        c_lo = estimate_cost(base_vehicle, 20_000.0)
        c_hi = estimate_cost(base_vehicle, 80_000.0)
        assert c_hi.total > c_lo.total

    def test_higher_cost_propellant_costs_more(self, base_vehicle):
        from dataclasses import replace
        v_kero  = replace(base_vehicle, propellant=PROPELLANTS["kerolox"])
        v_store = replace(base_vehicle, propellant=PROPELLANTS["storable"])
        c_kero  = estimate_cost(v_kero,  50_000.0)
        c_store = estimate_cost(v_store, 50_000.0)
        assert c_store.propellant_cost > c_kero.propellant_cost
