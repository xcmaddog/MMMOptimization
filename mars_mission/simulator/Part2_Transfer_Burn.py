import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time, TimeDelta


# Core gravitational constants for the transfer-burn model.
SUN_MU_KM = 132712440018.0  # km^3 / s^2
EARTH_RADIUS_KM = 6378.0
EARTH_MU_KM = 398600.4418  # km^3 / s^2
EARTH_SOI_RADIUS_KM = 924000.0
MARS_RADIUS_KM = 3389.5
MARS_MU_KM = 42828.375214  # km^3 / s^2


"""
------------------------------------------------------------
Phase 2 Inputs

Phase 2 automatically pulls its starting rocket, Earth, Mars, and start
date-time values from the final output of Part 1.

For now, this file uses placeholder handoff values in PHASE1_HANDOFF_STATE.
Later, your phase-runner file can replace those placeholder values with the
real output from part 1 before running phase 2.

Those handoff values are intentionally separated from the burn settings.
Only change the burn and phase-2 run settings in PHASE2_INPUTS below.

Input details:
- requested_burn_duration_minutes:
  Requested motor burn time in minutes.
  If this is longer than the available fuel allows, thrust shuts off
  automatically when the fuel is exhausted.
- thrust_newtons:
  Motor thrust in newtons.
- initial_total_mass_kg:
  Initial total rocket mass at the start of phase 2, including fuel.
- burn_rate_kg_per_min:
  Fuel consumption rate during the burn.
- starting_fuel_mass_kg:
  Fuel mass available at the start of phase 2.
- remaining_stage_mass_kg:
  Remaining rocket mass after stage separation.
- stage_separation_relative_speed_m_s:
  Assumed backward ejection speed of the discarded mass relative to the rocket
  flight direction during separation. This is used to give the remaining stage
  a forward speed increase while approximately preserving momentum.
- phase3_collision_lead_hours:
  If the rocket would impact Mars, phase 3 begins this many hours before the
  predicted impact so it can attempt a prograde collision-avoidance burn.
- dt_seconds:
  Output time step in seconds.
  This controls how often the simulation stores points and updates the plot.
- max_step_seconds:
  Internal RK4 integration step size in seconds.
  Smaller values improve burn and near-planet accuracy.
- total_time_days:
  Total phase-2 simulation duration in days.
- playback_speed:
  Animation playback speed multiplier.
- fps:
  Animation frame rate target used by Matplotlib.
------------------------------------------------------------
"""
PHASE2_INPUTS = {
    "requested_burn_duration_minutes": 10.0,
    "thrust_newtons": 100000.0,
    "initial_total_mass_kg": 13000.0,
    "burn_rate_kg_per_min": 1346.0,
    "starting_fuel_mass_kg": 8000.0,
    "remaining_stage_mass_kg": 2500.0,
    "stage_separation_relative_speed_m_s": 50.0,
    "phase3_collision_lead_hours": 4.0,
    "dt_seconds": 60.0,
    "max_step_seconds": 30.0,
    "total_time_days": 180.0,
    "playback_speed": 5.0,
    "fps": 60,
}


# Placeholder phase-1 handoff state for standalone development of phase 2.
# Replace this dictionary from your multi-phase runner once part 1 and part 2
# are being executed together.
PHASE1_HANDOFF_STATE = {
    "simulation_start_time_utc": "2025-01-01T00:00:00.000",
    "end_datetime_utc": "2025-04-11T00:00:00.000",
    "rocket_position_km": {"x": 1.47095e8, "y": 2.0e4},
    "rocket_velocity_km_s": {"vx": -0.01, "vy": 30.29},
    "earth_position_km": {"x": 1.47095e8, "y": 0.0},
    "earth_velocity_km_s": {"vx": 0.0, "vy": 30.29},
    "mars_position_km": {"x": 2.05e8, "y": 7.5e7},
    "mars_velocity_km_s": {"vx": -8.0, "vy": 22.0},
}


def gravity_accel_from_body(object_x_km, object_y_km, body_x_km, body_y_km, mu_km):
    """Return the acceleration on an object caused by one gravitating body."""
    dx_km = object_x_km - body_x_km
    dy_km = object_y_km - body_y_km
    r_km = np.hypot(dx_km, dy_km)

    if r_km == 0.0:
        return 0.0, 0.0

    ax_km_s2 = -mu_km * dx_km / r_km**3
    ay_km_s2 = -mu_km * dy_km / r_km**3
    return ax_km_s2, ay_km_s2


def format_elapsed_time(seconds):
    """Convert seconds into a days / hours / minutes label."""
    total_seconds = int(round(seconds))
    days = total_seconds // 86400
    remainder = total_seconds % 86400
    hours = remainder // 3600
    remainder %= 3600
    minutes = remainder // 60
    return f"{days} d  {hours:02d} h  {minutes:02d} m"


def lerp(start_value, end_value, alpha):
    """Linearly interpolate between two values."""
    return (1.0 - alpha) * start_value + alpha * end_value


def planetary_states_heliocentric(t_seconds_array, simulation_start_time_utc):
    """
    Return Sun-centered Earth and Mars state arrays from Astropy ephemerides.

    Astropy supplies barycentric states. Subtracting the Sun's barycentric
    state converts those into heliocentric states.
    """
    start_time = Time(simulation_start_time_utc, scale="utc")
    sample_times = start_time + TimeDelta(t_seconds_array, format="sec")

    with solar_system_ephemeris.set("builtin"):
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", sample_times)
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", sample_times)
        mars_pos, mars_vel = get_body_barycentric_posvel("mars", sample_times)

    earth_rel_pos_km = (earth_pos.xyz - sun_pos.xyz).to_value(u.km)
    earth_rel_vel_km_s = (earth_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)
    mars_rel_pos_km = (mars_pos.xyz - sun_pos.xyz).to_value(u.km)
    mars_rel_vel_km_s = (mars_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)

    return {
        "earth_x_km": earth_rel_pos_km[0],
        "earth_y_km": earth_rel_pos_km[1],
        "earth_vx_km_s": earth_rel_vel_km_s[0],
        "earth_vy_km_s": earth_rel_vel_km_s[1],
        "mars_x_km": mars_rel_pos_km[0],
        "mars_y_km": mars_rel_pos_km[1],
        "mars_vx_km_s": mars_rel_vel_km_s[0],
        "mars_vy_km_s": mars_rel_vel_km_s[1],
    }


def thrust_accel_km_s2(reference_vx_km_s, reference_vy_km_s, thrust_newtons, mass_kg):
    """Return thrust acceleration components aligned with the chosen reference velocity."""
    speed_km_s = np.hypot(reference_vx_km_s, reference_vy_km_s)
    if speed_km_s == 0.0 or mass_kg <= 0.0 or thrust_newtons <= 0.0:
        return 0.0, 0.0

    direction_x = reference_vx_km_s / speed_km_s
    direction_y = reference_vy_km_s / speed_km_s
    accel_km_s2 = (thrust_newtons / mass_kg) / 1000.0
    return accel_km_s2 * direction_x, accel_km_s2 * direction_y


def transfer_state_derivative(
    state,
    earth_x_km,
    earth_y_km,
    earth_vx_km_s,
    earth_vy_km_s,
    mars_x_km,
    mars_y_km,
    thrust_on,
    thrust_newtons,
    mass_flow_kg_s,
):
    """
    Return time derivatives for [x, y, vx, vy, total_mass, fuel_mass].

    The rocket feels gravity from the Sun, Earth, and Mars.
    Thrust is modeled as a prograde burn.
    Near Earth, "prograde" is taken relative to Earth's local frame so the
    burn stays tangent to the parking orbit instead of the heliocentric path.
    """
    x_km, y_km, vx_km_s, vy_km_s, total_mass_kg, fuel_mass_kg = state

    sun_ax_km_s2, sun_ay_km_s2 = gravity_accel_from_body(x_km, y_km, 0.0, 0.0, SUN_MU_KM)
    earth_ax_km_s2, earth_ay_km_s2 = gravity_accel_from_body(
        x_km, y_km, earth_x_km, earth_y_km, EARTH_MU_KM
    )
    mars_ax_km_s2, mars_ay_km_s2 = gravity_accel_from_body(
        x_km, y_km, mars_x_km, mars_y_km, MARS_MU_KM
    )

    thrust_ax_km_s2 = 0.0
    thrust_ay_km_s2 = 0.0
    dmass_dt = 0.0
    dfuel_dt = 0.0

    if thrust_on and fuel_mass_kg > 0.0 and total_mass_kg > 0.0:
        rocket_to_earth_km = np.hypot(x_km - earth_x_km, y_km - earth_y_km)
        if rocket_to_earth_km <= EARTH_SOI_RADIUS_KM:
            thrust_reference_vx_km_s = vx_km_s - earth_vx_km_s
            thrust_reference_vy_km_s = vy_km_s - earth_vy_km_s
        else:
            thrust_reference_vx_km_s = vx_km_s
            thrust_reference_vy_km_s = vy_km_s

        thrust_ax_km_s2, thrust_ay_km_s2 = thrust_accel_km_s2(
            thrust_reference_vx_km_s,
            thrust_reference_vy_km_s,
            thrust_newtons,
            total_mass_kg,
        )
        dmass_dt = -mass_flow_kg_s
        dfuel_dt = -mass_flow_kg_s

    return np.array(
        [
            vx_km_s,
            vy_km_s,
            sun_ax_km_s2 + earth_ax_km_s2 + mars_ax_km_s2 + thrust_ax_km_s2,
            sun_ay_km_s2 + earth_ay_km_s2 + mars_ay_km_s2 + thrust_ay_km_s2,
            dmass_dt,
            dfuel_dt,
        ],
        dtype=float,
    )


def rk4_step_transfer(
    state,
    step_seconds,
    earth_start_x_km,
    earth_start_y_km,
    earth_start_vx_km_s,
    earth_start_vy_km_s,
    earth_end_x_km,
    earth_end_y_km,
    earth_end_vx_km_s,
    earth_end_vy_km_s,
    mars_start_x_km,
    mars_start_y_km,
    mars_end_x_km,
    mars_end_y_km,
    alpha_start,
    alpha_end,
    thrust_on,
    thrust_newtons,
    mass_flow_kg_s,
):
    """Advance the transfer state by one RK4 step while Earth and Mars move linearly."""
    alpha_mid = 0.5 * (alpha_start + alpha_end)

    earth_x_1 = lerp(earth_start_x_km, earth_end_x_km, alpha_start)
    earth_y_1 = lerp(earth_start_y_km, earth_end_y_km, alpha_start)
    earth_vx_1 = lerp(earth_start_vx_km_s, earth_end_vx_km_s, alpha_start)
    earth_vy_1 = lerp(earth_start_vy_km_s, earth_end_vy_km_s, alpha_start)
    earth_x_2 = lerp(earth_start_x_km, earth_end_x_km, alpha_mid)
    earth_y_2 = lerp(earth_start_y_km, earth_end_y_km, alpha_mid)
    earth_vx_2 = lerp(earth_start_vx_km_s, earth_end_vx_km_s, alpha_mid)
    earth_vy_2 = lerp(earth_start_vy_km_s, earth_end_vy_km_s, alpha_mid)
    earth_x_4 = lerp(earth_start_x_km, earth_end_x_km, alpha_end)
    earth_y_4 = lerp(earth_start_y_km, earth_end_y_km, alpha_end)
    earth_vx_4 = lerp(earth_start_vx_km_s, earth_end_vx_km_s, alpha_end)
    earth_vy_4 = lerp(earth_start_vy_km_s, earth_end_vy_km_s, alpha_end)

    mars_x_1 = lerp(mars_start_x_km, mars_end_x_km, alpha_start)
    mars_y_1 = lerp(mars_start_y_km, mars_end_y_km, alpha_start)
    mars_x_2 = lerp(mars_start_x_km, mars_end_x_km, alpha_mid)
    mars_y_2 = lerp(mars_start_y_km, mars_end_y_km, alpha_mid)
    mars_x_4 = lerp(mars_start_x_km, mars_end_x_km, alpha_end)
    mars_y_4 = lerp(mars_start_y_km, mars_end_y_km, alpha_end)

    k1 = transfer_state_derivative(
        state,
        earth_x_km=earth_x_1,
        earth_y_km=earth_y_1,
        earth_vx_km_s=earth_vx_1,
        earth_vy_km_s=earth_vy_1,
        mars_x_km=mars_x_1,
        mars_y_km=mars_y_1,
        thrust_on=thrust_on,
        thrust_newtons=thrust_newtons,
        mass_flow_kg_s=mass_flow_kg_s,
    )
    k2 = transfer_state_derivative(
        state + 0.5 * step_seconds * k1,
        earth_x_km=earth_x_2,
        earth_y_km=earth_y_2,
        earth_vx_km_s=earth_vx_2,
        earth_vy_km_s=earth_vy_2,
        mars_x_km=mars_x_2,
        mars_y_km=mars_y_2,
        thrust_on=thrust_on,
        thrust_newtons=thrust_newtons,
        mass_flow_kg_s=mass_flow_kg_s,
    )
    k3 = transfer_state_derivative(
        state + 0.5 * step_seconds * k2,
        earth_x_km=earth_x_2,
        earth_y_km=earth_y_2,
        earth_vx_km_s=earth_vx_2,
        earth_vy_km_s=earth_vy_2,
        mars_x_km=mars_x_2,
        mars_y_km=mars_y_2,
        thrust_on=thrust_on,
        thrust_newtons=thrust_newtons,
        mass_flow_kg_s=mass_flow_kg_s,
    )
    k4 = transfer_state_derivative(
        state + step_seconds * k3,
        earth_x_km=earth_x_4,
        earth_y_km=earth_y_4,
        earth_vx_km_s=earth_vx_4,
        earth_vy_km_s=earth_vy_4,
        mars_x_km=mars_x_4,
        mars_y_km=mars_y_4,
        thrust_on=thrust_on,
        thrust_newtons=thrust_newtons,
        mass_flow_kg_s=mass_flow_kg_s,
    )

    next_state = state + (step_seconds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[4] = max(next_state[4], 0.0)
    next_state[5] = max(next_state[5], 0.0)
    return next_state


def apply_stage_separation(state, remaining_stage_mass_kg, stage_separation_relative_speed_m_s):
    """
    Apply a stage-separation event.

    Assumption:
    - The discarded mass is ejected backward along the current flight direction.
    - The remaining stage gains forward speed through momentum exchange.
    """
    x_km, y_km, vx_km_s, vy_km_s, total_mass_kg, fuel_mass_kg = state
    discarded_mass_kg = max(0.0, total_mass_kg - remaining_stage_mass_kg)

    if discarded_mass_kg <= 0.0 or remaining_stage_mass_kg <= 0.0:
        next_state = state.copy()
        next_state[4] = remaining_stage_mass_kg
        next_state[5] = 0.0
        return next_state, 0.0, discarded_mass_kg

    speed_km_s = np.hypot(vx_km_s, vy_km_s)
    if speed_km_s == 0.0 or stage_separation_relative_speed_m_s <= 0.0:
        delta_v_km_s = 0.0
    else:
        ejection_speed_km_s = stage_separation_relative_speed_m_s / 1000.0
        delta_v_km_s = (discarded_mass_kg / remaining_stage_mass_kg) * ejection_speed_km_s
        vx_km_s += delta_v_km_s * (vx_km_s / speed_km_s)
        vy_km_s += delta_v_km_s * (vy_km_s / speed_km_s)

    return (
        np.array([x_km, y_km, vx_km_s, vy_km_s, remaining_stage_mass_kg, 0.0], dtype=float),
        delta_v_km_s * 1000.0,
        discarded_mass_kg,
    )


def build_phase3_handoff_state(
    simulation_start_time_utc,
    t_seconds,
    rocket_x_km,
    rocket_y_km,
    rocket_vx_km_s,
    rocket_vy_km_s,
    rocket_mass_kg,
    mars_x_km,
    mars_y_km,
    mars_vx_km_s,
    mars_vy_km_s,
    phase3_collision_lead_hours,
    status,
):
    """
    Build the phase-3 handoff state near Mars.

    Selection rule:
    - If the rocket impacts Mars, choose the sample some configurable number
      of hours before impact.
    - Otherwise choose the sample of closest approach to Mars.
    """
    rocket_to_mars_km = np.hypot(rocket_x_km - mars_x_km, rocket_y_km - mars_y_km)

    if status == "impacted Mars":
        target_time_seconds = max(0.0, t_seconds[-1] - phase3_collision_lead_hours * 3600.0)
        handoff_index = int(np.searchsorted(t_seconds, target_time_seconds, side="right") - 1)
        handoff_reason = f"{phase3_collision_lead_hours:.3f} hours before Mars impact"
        mars_approach_type = "collision_course"
        recommended_burn_direction = "prograde_then_retrograde"
    else:
        handoff_index = int(np.argmin(rocket_to_mars_km))
        handoff_reason = "closest approach to Mars"
        mars_approach_type = "near_pass"
        recommended_burn_direction = "retrograde_at_closest_pass"

    handoff_datetime_utc = (
        Time(simulation_start_time_utc, scale="utc") + TimeDelta(t_seconds[handoff_index], format="sec")
    ).utc.isot

    return {
        "handoff_reason": handoff_reason,
        "datetime_utc": handoff_datetime_utc,
        "elapsed_time_seconds": float(t_seconds[handoff_index]),
        "elapsed_time_pretty": format_elapsed_time(t_seconds[handoff_index]),
        "rocket_position_km": {
            "x": float(rocket_x_km[handoff_index]),
            "y": float(rocket_y_km[handoff_index]),
        },
        "rocket_velocity_km_s": {
            "vx": float(rocket_vx_km_s[handoff_index]),
            "vy": float(rocket_vy_km_s[handoff_index]),
        },
        "rocket_mass_kg": float(rocket_mass_kg[handoff_index]),
        "mars_position_km": {
            "x": float(mars_x_km[handoff_index]),
            "y": float(mars_y_km[handoff_index]),
        },
        "mars_velocity_km_s": {
            "vx": float(mars_vx_km_s[handoff_index]),
            "vy": float(mars_vy_km_s[handoff_index]),
        },
        "rocket_position_relative_to_mars_km": {
            "x": float(rocket_x_km[handoff_index] - mars_x_km[handoff_index]),
            "y": float(rocket_y_km[handoff_index] - mars_y_km[handoff_index]),
        },
        "rocket_velocity_relative_to_mars_km_s": {
            "vx": float(rocket_vx_km_s[handoff_index] - mars_vx_km_s[handoff_index]),
            "vy": float(rocket_vy_km_s[handoff_index] - mars_vy_km_s[handoff_index]),
        },
        "distance_to_mars_km": float(rocket_to_mars_km[handoff_index]),
        "mars_approach_type": mars_approach_type,
        "recommended_burn_direction": recommended_burn_direction,
        "phase3_collision_lead_hours": float(phase3_collision_lead_hours),
    }


def build_final_state(
    simulation_start_time_utc,
    end_datetime_utc,
    elapsed_time_seconds,
    rocket_x_km,
    rocket_y_km,
    rocket_vx_km_s,
    rocket_vy_km_s,
    earth_x_km,
    earth_y_km,
    earth_vx_km_s,
    earth_vy_km_s,
    mars_x_km,
    mars_y_km,
    mars_vx_km_s,
    mars_vy_km_s,
    rocket_mass_kg,
    fuel_remaining_kg,
    requested_burn_duration_minutes,
    actual_burn_duration_seconds,
    burn_shutdown_reason,
    stage_separated,
    stage_separation_datetime_utc,
    stage_separation_delta_v_m_s,
    discarded_stage_mass_kg,
    phase3_handoff_state,
    status,
):
    """Package the end-of-simulation values into one dictionary."""
    return {
        "simulation_start_time_utc": simulation_start_time_utc,
        "end_datetime_utc": end_datetime_utc,
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "elapsed_time_pretty": format_elapsed_time(elapsed_time_seconds),
        "rocket_position_km": {"x": float(rocket_x_km), "y": float(rocket_y_km)},
        "rocket_velocity_km_s": {"vx": float(rocket_vx_km_s), "vy": float(rocket_vy_km_s)},
        "earth_position_km": {"x": float(earth_x_km), "y": float(earth_y_km)},
        "earth_velocity_km_s": {"vx": float(earth_vx_km_s), "vy": float(earth_vy_km_s)},
        "mars_position_km": {"x": float(mars_x_km), "y": float(mars_y_km)},
        "mars_velocity_km_s": {"vx": float(mars_vx_km_s), "vy": float(mars_vy_km_s)},
        "rocket_mass_kg": float(rocket_mass_kg),
        "fuel_remaining_kg": float(fuel_remaining_kg),
        "requested_burn_duration_minutes": float(requested_burn_duration_minutes),
        "actual_burn_duration_minutes": float(actual_burn_duration_seconds / 60.0),
        "burn_shutdown_reason": burn_shutdown_reason,
        "stage_separated": stage_separated,
        "stage_separation_datetime_utc": stage_separation_datetime_utc,
        "stage_separation_delta_v_m_s": float(stage_separation_delta_v_m_s),
        "discarded_stage_mass_kg": float(discarded_stage_mass_kg),
        "phase3_handoff_state": phase3_handoff_state,
        "status": status,
    }


def validate_mass_inputs(initial_total_mass_kg, starting_fuel_mass_kg, remaining_stage_mass_kg):
    """Raise a clear error if the phase-2 mass inputs are physically inconsistent."""
    if starting_fuel_mass_kg < 0.0:
        raise ValueError("starting_fuel_mass_kg must be non-negative.")
    if initial_total_mass_kg <= 0.0:
        raise ValueError("initial_total_mass_kg must be positive.")
    if remaining_stage_mass_kg <= 0.0:
        raise ValueError("remaining_stage_mass_kg must be positive.")
    if starting_fuel_mass_kg > initial_total_mass_kg:
        raise ValueError("starting_fuel_mass_kg cannot exceed initial_total_mass_kg.")
    if remaining_stage_mass_kg > initial_total_mass_kg - starting_fuel_mass_kg:
        raise ValueError(
            "remaining_stage_mass_kg cannot exceed the non-fuel mass available before separation."
        )


def simulate_transfer_burn_phase2(
    requested_burn_duration_minutes=PHASE2_INPUTS["requested_burn_duration_minutes"],
    thrust_newtons=PHASE2_INPUTS["thrust_newtons"],
    initial_total_mass_kg=PHASE2_INPUTS["initial_total_mass_kg"],
    burn_rate_kg_per_min=PHASE2_INPUTS["burn_rate_kg_per_min"],
    starting_fuel_mass_kg=PHASE2_INPUTS["starting_fuel_mass_kg"],
    remaining_stage_mass_kg=PHASE2_INPUTS["remaining_stage_mass_kg"],
    stage_separation_relative_speed_m_s=PHASE2_INPUTS["stage_separation_relative_speed_m_s"],
    phase3_collision_lead_hours=PHASE2_INPUTS["phase3_collision_lead_hours"],
    dt_seconds=PHASE2_INPUTS["dt_seconds"],
    max_step_seconds=PHASE2_INPUTS["max_step_seconds"],
    total_time_days=PHASE2_INPUTS["total_time_days"],
):
    """
    Simulate phase 2 as a Sun-centered transfer with a finite burn.

    Starting conditions are pulled directly from the final output of phase 1.
    Phase-2 editable inputs are limited to burn, mass, and runtime settings.
    """
    validate_mass_inputs(initial_total_mass_kg, starting_fuel_mass_kg, remaining_stage_mass_kg)

    phase1_final_state = PHASE1_HANDOFF_STATE
    simulation_start_time_utc = phase1_final_state["end_datetime_utc"]

    total_time_seconds = total_time_days * 86400.0
    n_output_steps = int(total_time_seconds / dt_seconds) + 1
    t_seconds = np.linspace(0.0, total_time_seconds, n_output_steps)

    planet_states = planetary_states_heliocentric(
        t_seconds_array=t_seconds,
        simulation_start_time_utc=simulation_start_time_utc,
    )
    earth_x_km = planet_states["earth_x_km"]
    earth_y_km = planet_states["earth_y_km"]
    earth_vx_km_s = planet_states["earth_vx_km_s"]
    earth_vy_km_s = planet_states["earth_vy_km_s"]
    mars_x_km = planet_states["mars_x_km"]
    mars_y_km = planet_states["mars_y_km"]
    mars_vx_km_s = planet_states["mars_vx_km_s"]
    mars_vy_km_s = planet_states["mars_vy_km_s"]

    rocket_x_km = np.zeros(n_output_steps)
    rocket_y_km = np.zeros(n_output_steps)
    rocket_vx_km_s = np.zeros(n_output_steps)
    rocket_vy_km_s = np.zeros(n_output_steps)
    rocket_mass_kg = np.zeros(n_output_steps)
    fuel_mass_kg = np.zeros(n_output_steps)

    rocket_x_km[0] = phase1_final_state["rocket_position_km"]["x"]
    rocket_y_km[0] = phase1_final_state["rocket_position_km"]["y"]
    rocket_vx_km_s[0] = phase1_final_state["rocket_velocity_km_s"]["vx"]
    rocket_vy_km_s[0] = phase1_final_state["rocket_velocity_km_s"]["vy"]
    rocket_mass_kg[0] = initial_total_mass_kg
    fuel_mass_kg[0] = starting_fuel_mass_kg

    burn_rate_kg_s = burn_rate_kg_per_min / 60.0
    requested_burn_duration_seconds = max(0.0, requested_burn_duration_minutes * 60.0)

    if burn_rate_kg_s <= 0.0 or starting_fuel_mass_kg <= 0.0 or thrust_newtons <= 0.0:
        actual_burn_duration_seconds = 0.0
        burn_shutdown_reason = "no active burn"
    else:
        fuel_limited_burn_seconds = starting_fuel_mass_kg / burn_rate_kg_s
        actual_burn_duration_seconds = min(requested_burn_duration_seconds, fuel_limited_burn_seconds)
        if requested_burn_duration_seconds > fuel_limited_burn_seconds:
            burn_shutdown_reason = "fuel exhausted"
        else:
            burn_shutdown_reason = "requested burn duration reached"

    state = np.array(
        [
            rocket_x_km[0],
            rocket_y_km[0],
            rocket_vx_km_s[0],
            rocket_vy_km_s[0],
            rocket_mass_kg[0],
            fuel_mass_kg[0],
        ],
        dtype=float,
    )

    status = "in transfer"
    stop_index = n_output_steps
    burn_elapsed_seconds = 0.0
    stage_separated = False
    stage_separation_delta_v_m_s = 0.0
    discarded_stage_mass_kg = 0.0
    stage_separation_time_seconds = None

    for i in range(n_output_steps - 1):
        interval_seconds = t_seconds[i + 1] - t_seconds[i]
        interval_elapsed = 0.0
        interval_completed = True

        while interval_elapsed < interval_seconds - 1e-12:
            remaining_interval_seconds = interval_seconds - interval_elapsed
            step_seconds = min(max_step_seconds, remaining_interval_seconds)

            burn_active = burn_elapsed_seconds < actual_burn_duration_seconds - 1e-12
            if burn_active:
                remaining_burn_seconds = actual_burn_duration_seconds - burn_elapsed_seconds
                step_seconds = min(step_seconds, remaining_burn_seconds)

            alpha_start = interval_elapsed / interval_seconds
            alpha_end = (interval_elapsed + step_seconds) / interval_seconds

            earth_x_start = lerp(earth_x_km[i], earth_x_km[i + 1], alpha_start)
            earth_y_start = lerp(earth_y_km[i], earth_y_km[i + 1], alpha_start)
            earth_vx_start = lerp(earth_vx_km_s[i], earth_vx_km_s[i + 1], alpha_start)
            earth_vy_start = lerp(earth_vy_km_s[i], earth_vy_km_s[i + 1], alpha_start)
            earth_x_end = lerp(earth_x_km[i], earth_x_km[i + 1], alpha_end)
            earth_y_end = lerp(earth_y_km[i], earth_y_km[i + 1], alpha_end)
            earth_vx_end = lerp(earth_vx_km_s[i], earth_vx_km_s[i + 1], alpha_end)
            earth_vy_end = lerp(earth_vy_km_s[i], earth_vy_km_s[i + 1], alpha_end)
            mars_x_start = lerp(mars_x_km[i], mars_x_km[i + 1], alpha_start)
            mars_y_start = lerp(mars_y_km[i], mars_y_km[i + 1], alpha_start)
            mars_vx_start = lerp(mars_vx_km_s[i], mars_vx_km_s[i + 1], alpha_start)
            mars_vy_start = lerp(mars_vy_km_s[i], mars_vy_km_s[i + 1], alpha_start)

            rocket_to_earth_km = np.hypot(state[0] - earth_x_start, state[1] - earth_y_start)
            rocket_to_mars_km = np.hypot(state[0] - mars_x_start, state[1] - mars_y_start)

            if rocket_to_earth_km <= EARTH_RADIUS_KM:
                status = "impacted Earth"
                collision_time_seconds = t_seconds[i] + interval_elapsed
                t_seconds[i + 1] = collision_time_seconds
                earth_x_km[i + 1] = earth_x_start
                earth_y_km[i + 1] = earth_y_start
                earth_vx_km_s[i + 1] = earth_vx_start
                earth_vy_km_s[i + 1] = earth_vy_start
                mars_x_km[i + 1] = mars_x_start
                mars_y_km[i + 1] = mars_y_start
                mars_vx_km_s[i + 1] = mars_vx_start
                mars_vy_km_s[i + 1] = mars_vy_start
                rocket_x_km[i + 1] = state[0]
                rocket_y_km[i + 1] = state[1]
                rocket_vx_km_s[i + 1] = state[2]
                rocket_vy_km_s[i + 1] = state[3]
                rocket_mass_kg[i + 1] = state[4]
                fuel_mass_kg[i + 1] = state[5]
                stop_index = i + 2
                interval_completed = False
                break

            if rocket_to_mars_km <= MARS_RADIUS_KM:
                status = "impacted Mars"
                collision_time_seconds = t_seconds[i] + interval_elapsed
                t_seconds[i + 1] = collision_time_seconds
                earth_x_km[i + 1] = earth_x_start
                earth_y_km[i + 1] = earth_y_start
                earth_vx_km_s[i + 1] = earth_vx_start
                earth_vy_km_s[i + 1] = earth_vy_start
                mars_x_km[i + 1] = mars_x_start
                mars_y_km[i + 1] = mars_y_start
                mars_vx_km_s[i + 1] = mars_vx_start
                mars_vy_km_s[i + 1] = mars_vy_start
                rocket_x_km[i + 1] = state[0]
                rocket_y_km[i + 1] = state[1]
                rocket_vx_km_s[i + 1] = state[2]
                rocket_vy_km_s[i + 1] = state[3]
                rocket_mass_kg[i + 1] = state[4]
                fuel_mass_kg[i + 1] = state[5]
                stop_index = i + 2
                interval_completed = False
                break

            state = rk4_step_transfer(
                state=state,
                step_seconds=step_seconds,
                earth_start_x_km=earth_x_km[i],
                earth_start_y_km=earth_y_km[i],
                earth_start_vx_km_s=earth_vx_km_s[i],
                earth_start_vy_km_s=earth_vy_km_s[i],
                earth_end_x_km=earth_x_km[i + 1],
                earth_end_y_km=earth_y_km[i + 1],
                earth_end_vx_km_s=earth_vx_km_s[i + 1],
                earth_end_vy_km_s=earth_vy_km_s[i + 1],
                mars_start_x_km=mars_x_km[i],
                mars_start_y_km=mars_y_km[i],
                mars_end_x_km=mars_x_km[i + 1],
                mars_end_y_km=mars_y_km[i + 1],
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                thrust_on=burn_active,
                thrust_newtons=thrust_newtons,
                mass_flow_kg_s=burn_rate_kg_s,
            )

            if burn_active:
                burn_elapsed_seconds += step_seconds

            interval_elapsed += step_seconds

            if (
                not stage_separated
                and actual_burn_duration_seconds > 0.0
                and burn_elapsed_seconds >= actual_burn_duration_seconds - 1e-12
            ):
                state, stage_separation_delta_v_m_s, discarded_stage_mass_kg = apply_stage_separation(
                    state,
                    remaining_stage_mass_kg=remaining_stage_mass_kg,
                    stage_separation_relative_speed_m_s=stage_separation_relative_speed_m_s,
                )
                stage_separated = True
                stage_separation_time_seconds = t_seconds[i] + interval_elapsed

        if not interval_completed:
            break

        rocket_x_km[i + 1] = state[0]
        rocket_y_km[i + 1] = state[1]
        rocket_vx_km_s[i + 1] = state[2]
        rocket_vy_km_s[i + 1] = state[3]
        rocket_mass_kg[i + 1] = state[4]
        fuel_mass_kg[i + 1] = state[5]

    rocket_x_km = rocket_x_km[:stop_index]
    rocket_y_km = rocket_y_km[:stop_index]
    rocket_vx_km_s = rocket_vx_km_s[:stop_index]
    rocket_vy_km_s = rocket_vy_km_s[:stop_index]
    rocket_mass_kg = rocket_mass_kg[:stop_index]
    fuel_mass_kg = fuel_mass_kg[:stop_index]
    earth_x_km = earth_x_km[:stop_index]
    earth_y_km = earth_y_km[:stop_index]
    earth_vx_km_s = earth_vx_km_s[:stop_index]
    earth_vy_km_s = earth_vy_km_s[:stop_index]
    mars_x_km = mars_x_km[:stop_index]
    mars_y_km = mars_y_km[:stop_index]
    mars_vx_km_s = mars_vx_km_s[:stop_index]
    mars_vy_km_s = mars_vy_km_s[:stop_index]
    t_seconds = t_seconds[:stop_index]

    if status == "in transfer":
        status = "completed full simulation"

    end_datetime_utc = (
        Time(simulation_start_time_utc, scale="utc") + TimeDelta(t_seconds[-1], format="sec")
    ).utc.isot
    stage_separation_datetime_utc = None
    if stage_separation_time_seconds is not None:
        stage_separation_datetime_utc = (
            Time(simulation_start_time_utc, scale="utc")
            + TimeDelta(stage_separation_time_seconds, format="sec")
        ).utc.isot

    phase3_handoff_state = build_phase3_handoff_state(
        simulation_start_time_utc=simulation_start_time_utc,
        t_seconds=t_seconds,
        rocket_x_km=rocket_x_km,
        rocket_y_km=rocket_y_km,
        rocket_vx_km_s=rocket_vx_km_s,
        rocket_vy_km_s=rocket_vy_km_s,
        rocket_mass_kg=rocket_mass_kg,
        mars_x_km=mars_x_km,
        mars_y_km=mars_y_km,
        mars_vx_km_s=mars_vx_km_s,
        mars_vy_km_s=mars_vy_km_s,
        phase3_collision_lead_hours=phase3_collision_lead_hours,
        status=status,
    )

    final_state = build_final_state(
        simulation_start_time_utc=simulation_start_time_utc,
        end_datetime_utc=end_datetime_utc,
        elapsed_time_seconds=t_seconds[-1],
        rocket_x_km=rocket_x_km[-1],
        rocket_y_km=rocket_y_km[-1],
        rocket_vx_km_s=rocket_vx_km_s[-1],
        rocket_vy_km_s=rocket_vy_km_s[-1],
        earth_x_km=earth_x_km[-1],
        earth_y_km=earth_y_km[-1],
        earth_vx_km_s=earth_vx_km_s[-1],
        earth_vy_km_s=earth_vy_km_s[-1],
        mars_x_km=mars_x_km[-1],
        mars_y_km=mars_y_km[-1],
        mars_vx_km_s=mars_vx_km_s[-1],
        mars_vy_km_s=mars_vy_km_s[-1],
        rocket_mass_kg=rocket_mass_kg[-1],
        fuel_remaining_kg=fuel_mass_kg[-1],
        requested_burn_duration_minutes=requested_burn_duration_minutes,
        actual_burn_duration_seconds=actual_burn_duration_seconds,
        burn_shutdown_reason=burn_shutdown_reason,
        stage_separated=stage_separated,
        stage_separation_datetime_utc=stage_separation_datetime_utc,
        stage_separation_delta_v_m_s=stage_separation_delta_v_m_s,
        discarded_stage_mass_kg=discarded_stage_mass_kg,
        phase3_handoff_state=phase3_handoff_state,
        status=status,
    )

    return {
        "phase1_final_state": phase1_final_state,
        "t_seconds": t_seconds,
        "rocket_x_km": rocket_x_km,
        "rocket_y_km": rocket_y_km,
        "rocket_vx_km_s": rocket_vx_km_s,
        "rocket_vy_km_s": rocket_vy_km_s,
        "rocket_mass_kg": rocket_mass_kg,
        "fuel_mass_kg": fuel_mass_kg,
        "earth_x_km": earth_x_km,
        "earth_y_km": earth_y_km,
        "earth_vx_km_s": earth_vx_km_s,
        "earth_vy_km_s": earth_vy_km_s,
        "mars_x_km": mars_x_km,
        "mars_y_km": mars_y_km,
        "mars_vx_km_s": mars_vx_km_s,
        "mars_vy_km_s": mars_vy_km_s,
        "actual_burn_duration_seconds": actual_burn_duration_seconds,
        "burn_shutdown_reason": burn_shutdown_reason,
        "stage_separated": stage_separated,
        "stage_separation_time_seconds": stage_separation_time_seconds,
        "stage_separation_delta_v_m_s": stage_separation_delta_v_m_s,
        "discarded_stage_mass_kg": discarded_stage_mass_kg,
        "phase3_handoff_state": phase3_handoff_state,
        "status": status,
        "final_state": final_state,
    }


def animate_transfer_burn_phase2(
    requested_burn_duration_minutes=PHASE2_INPUTS["requested_burn_duration_minutes"],
    thrust_newtons=PHASE2_INPUTS["thrust_newtons"],
    initial_total_mass_kg=PHASE2_INPUTS["initial_total_mass_kg"],
    burn_rate_kg_per_min=PHASE2_INPUTS["burn_rate_kg_per_min"],
    starting_fuel_mass_kg=PHASE2_INPUTS["starting_fuel_mass_kg"],
    remaining_stage_mass_kg=PHASE2_INPUTS["remaining_stage_mass_kg"],
    stage_separation_relative_speed_m_s=PHASE2_INPUTS["stage_separation_relative_speed_m_s"],
    phase3_collision_lead_hours=PHASE2_INPUTS["phase3_collision_lead_hours"],
    dt_seconds=PHASE2_INPUTS["dt_seconds"],
    max_step_seconds=PHASE2_INPUTS["max_step_seconds"],
    total_time_days=PHASE2_INPUTS["total_time_days"],
    playback_speed=PHASE2_INPUTS["playback_speed"],
    fps=PHASE2_INPUTS["fps"],
):
    """Animate the Sun-centered transfer burn and return the simulation results."""
    simulation = simulate_transfer_burn_phase2(
        requested_burn_duration_minutes=requested_burn_duration_minutes,
        thrust_newtons=thrust_newtons,
        initial_total_mass_kg=initial_total_mass_kg,
        burn_rate_kg_per_min=burn_rate_kg_per_min,
        starting_fuel_mass_kg=starting_fuel_mass_kg,
        remaining_stage_mass_kg=remaining_stage_mass_kg,
        stage_separation_relative_speed_m_s=stage_separation_relative_speed_m_s,
        phase3_collision_lead_hours=phase3_collision_lead_hours,
        dt_seconds=dt_seconds,
        max_step_seconds=max_step_seconds,
        total_time_days=total_time_days,
    )

    t_seconds = simulation["t_seconds"]
    rocket_x_km = simulation["rocket_x_km"]
    rocket_y_km = simulation["rocket_y_km"]
    rocket_vx_km_s = simulation["rocket_vx_km_s"]
    rocket_vy_km_s = simulation["rocket_vy_km_s"]
    rocket_mass_kg = simulation["rocket_mass_kg"]
    fuel_mass_kg = simulation["fuel_mass_kg"]
    earth_x_km = simulation["earth_x_km"]
    earth_y_km = simulation["earth_y_km"]
    mars_x_km = simulation["mars_x_km"]
    mars_y_km = simulation["mars_y_km"]
    actual_burn_duration_seconds = simulation["actual_burn_duration_seconds"]
    status = simulation["status"]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x relative to Sun (km)")
    ax.set_ylabel("y relative to Sun (km)")
    ax.set_title("Phase 2: Transfer Burn")

    max_extent_km = max(
        np.max(np.abs(rocket_x_km)),
        np.max(np.abs(rocket_y_km)),
        np.max(np.abs(earth_x_km)),
        np.max(np.abs(earth_y_km)),
        np.max(np.abs(mars_x_km)),
        np.max(np.abs(mars_y_km)),
    )
    pad_km = 0.08 * max_extent_km + 5.0e6
    ax.set_xlim(-max_extent_km - pad_km, max_extent_km + pad_km)
    ax.set_ylim(-max_extent_km - pad_km, max_extent_km + pad_km)

    ax.plot([0.0], [0.0], marker="o", markersize=10, color="gold", label="Sun")
    earth_path_line, = ax.plot([], [], lw=1.5, color="royalblue", alpha=0.6, label="Earth path")
    mars_path_line, = ax.plot([], [], lw=1.5, color="orangered", alpha=0.6, label="Mars path")
    burn_path_line, = ax.plot([], [], lw=2.0, color="darkorange", label="Burn path")
    coast_path_line, = ax.plot([], [], lw=2.0, color="black", label="Post-burn path")
    earth_point, = ax.plot([], [], marker="o", color="royalblue", label="Earth")
    mars_point, = ax.plot([], [], marker="o", color="orangered", label="Mars")
    rocket_point, = ax.plot([], [], marker="o", color="black", label="Rocket")
    ax.legend(loc="upper right")

    info = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    playback_speed = max(playback_speed, 1e-6)
    target_display_frames = max(1, int(np.ceil(900 / playback_speed)))
    frame_step = max(1, len(t_seconds) // target_display_frames)
    frame_indices = np.arange(0, len(t_seconds), frame_step)
    if frame_indices[-1] != len(t_seconds) - 1:
        frame_indices = np.append(frame_indices, len(t_seconds) - 1)

    def init():
        earth_path_line.set_data([], [])
        mars_path_line.set_data([], [])
        burn_path_line.set_data([], [])
        coast_path_line.set_data([], [])
        earth_point.set_data([], [])
        mars_point.set_data([], [])
        rocket_point.set_data([], [])
        info.set_text("")
        return (
            earth_path_line,
            mars_path_line,
            burn_path_line,
            coast_path_line,
            earth_point,
            mars_point,
            rocket_point,
            info,
        )

    def update(frame_number):
        i = frame_indices[frame_number]
        earth_path_line.set_data(earth_x_km[: i + 1], earth_y_km[: i + 1])
        mars_path_line.set_data(mars_x_km[: i + 1], mars_y_km[: i + 1])

        burn_indices = t_seconds[: i + 1] <= actual_burn_duration_seconds + 1e-9
        if np.any(burn_indices):
            burn_path_line.set_data(rocket_x_km[: i + 1][burn_indices], rocket_y_km[: i + 1][burn_indices])
        else:
            burn_path_line.set_data([], [])

        if np.any(~burn_indices):
            coast_path_line.set_data(rocket_x_km[: i + 1][~burn_indices], rocket_y_km[: i + 1][~burn_indices])
        else:
            coast_path_line.set_data([], [])

        earth_point.set_data([earth_x_km[i]], [earth_y_km[i]])
        mars_point.set_data([mars_x_km[i]], [mars_y_km[i]])
        rocket_point.set_data([rocket_x_km[i]], [rocket_y_km[i]])

        rocket_to_earth_km = np.hypot(rocket_x_km[i] - earth_x_km[i], rocket_y_km[i] - earth_y_km[i])
        rocket_to_mars_km = np.hypot(rocket_x_km[i] - mars_x_km[i], rocket_y_km[i] - mars_y_km[i])
        thrust_status = "burning" if t_seconds[i] <= actual_burn_duration_seconds + 1e-9 else "coasting"

        info.set_text(
            f"time = {format_elapsed_time(t_seconds[i])}\n"
            f"rocket x = {rocket_x_km[i]:.1f} km\n"
            f"rocket y = {rocket_y_km[i]:.1f} km\n"
            f"rocket vx = {rocket_vx_km_s[i]:.4f} km/s\n"
            f"rocket vy = {rocket_vy_km_s[i]:.4f} km/s\n"
            f"mass = {rocket_mass_kg[i]:.2f} kg\n"
            f"fuel = {fuel_mass_kg[i]:.2f} kg\n"
            f"thrust status = {thrust_status}\n"
            f"distance to Earth = {rocket_to_earth_km:.1f} km\n"
            f"distance to Mars = {rocket_to_mars_km:.1f} km"
            + (f"\nstatus = {status}" if i == len(t_seconds) - 1 else "")
        )

        return (
            earth_path_line,
            mars_path_line,
            burn_path_line,
            coast_path_line,
            earth_point,
            mars_point,
            rocket_point,
            info,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000 / (fps * playback_speed),
        blit=True,
    )

    plt.show()
    return anim, simulation


def print_handoff_state(phase1_final_state):
    """Print the phase-1 handoff state used as the phase-2 start state."""
    print("Phase 2 start state (from Part 1)")
    print("-------------------------------")
    print(f"Start date-time (UTC): {phase1_final_state['end_datetime_utc']}")
    print(
        f"Rocket position relative to Sun (km): x = {phase1_final_state['rocket_position_km']['x']:.3f}, "
        f"y = {phase1_final_state['rocket_position_km']['y']:.3f}"
    )
    print(
        f"Rocket velocity relative to Sun (km/s): vx = {phase1_final_state['rocket_velocity_km_s']['vx']:.6f}, "
        f"vy = {phase1_final_state['rocket_velocity_km_s']['vy']:.6f}"
    )
    print(
        f"Earth position relative to Sun (km): x = {phase1_final_state['earth_position_km']['x']:.3f}, "
        f"y = {phase1_final_state['earth_position_km']['y']:.3f}"
    )
    print(
        f"Earth velocity relative to Sun (km/s): vx = {phase1_final_state['earth_velocity_km_s']['vx']:.6f}, "
        f"vy = {phase1_final_state['earth_velocity_km_s']['vy']:.6f}"
    )
    print(
        f"Mars position relative to Sun (km): x = {phase1_final_state['mars_position_km']['x']:.3f}, "
        f"y = {phase1_final_state['mars_position_km']['y']:.3f}"
    )
    print(
        f"Mars velocity relative to Sun (km/s): vx = {phase1_final_state['mars_velocity_km_s']['vx']:.6f}, "
        f"vy = {phase1_final_state['mars_velocity_km_s']['vy']:.6f}"
    )


def print_final_state(final_state):
    """Print the final phase-2 state and burn summary."""
    phase3_handoff_state = final_state["phase3_handoff_state"]

    print("Final phase 2 state")
    print("-------------------")
    print(f"Status: {final_state['status']}")
    print(f"Start date-time (UTC): {final_state['simulation_start_time_utc']}")
    print(f"End date-time (UTC): {final_state['end_datetime_utc']}")
    print(f"Elapsed time: {final_state['elapsed_time_pretty']} ({final_state['elapsed_time_seconds']:.1f} s)")
    print(
        f"Rocket position relative to Sun (km): x = {final_state['rocket_position_km']['x']:.3f}, "
        f"y = {final_state['rocket_position_km']['y']:.3f}"
    )
    print(
        f"Rocket velocity relative to Sun (km/s): vx = {final_state['rocket_velocity_km_s']['vx']:.6f}, "
        f"vy = {final_state['rocket_velocity_km_s']['vy']:.6f}"
    )
    print(
        f"Earth position relative to Sun (km): x = {final_state['earth_position_km']['x']:.3f}, "
        f"y = {final_state['earth_position_km']['y']:.3f}"
    )
    print(
        f"Earth velocity relative to Sun (km/s): vx = {final_state['earth_velocity_km_s']['vx']:.6f}, "
        f"vy = {final_state['earth_velocity_km_s']['vy']:.6f}"
    )
    print(
        f"Mars position relative to Sun (km): x = {final_state['mars_position_km']['x']:.3f}, "
        f"y = {final_state['mars_position_km']['y']:.3f}"
    )
    print(
        f"Mars velocity relative to Sun (km/s): vx = {final_state['mars_velocity_km_s']['vx']:.6f}, "
        f"vy = {final_state['mars_velocity_km_s']['vy']:.6f}"
    )
    print(f"Rocket mass: {final_state['rocket_mass_kg']:.3f} kg")
    print(f"Fuel remaining: {final_state['fuel_remaining_kg']:.3f} kg")
    print(f"Requested burn duration: {final_state['requested_burn_duration_minutes']:.3f} min")
    print(f"Actual burn duration: {final_state['actual_burn_duration_minutes']:.3f} min")
    print(f"Burn shutdown reason: {final_state['burn_shutdown_reason']}")
    print(f"Stage separated: {final_state['stage_separated']}")
    print(f"Stage separation date-time (UTC): {final_state['stage_separation_datetime_utc']}")
    print(f"Stage separation delta-v: {final_state['stage_separation_delta_v_m_s']:.3f} m/s")
    print(f"Discarded stage mass: {final_state['discarded_stage_mass_kg']:.3f} kg")
    print("Phase 3 handoff:")
    print(f"  Reason: {phase3_handoff_state['handoff_reason']}")
    print(f"  Date-time (UTC): {phase3_handoff_state['datetime_utc']}")
    print(f"  Approach type: {phase3_handoff_state['mars_approach_type']}")
    print(f"  Recommended burn: {phase3_handoff_state['recommended_burn_direction']}")
    print(
        f"  Rocket position (km): x = {phase3_handoff_state['rocket_position_km']['x']:.3f}, "
        f"y = {phase3_handoff_state['rocket_position_km']['y']:.3f}"
    )
    print(
        f"  Rocket velocity (km/s): vx = {phase3_handoff_state['rocket_velocity_km_s']['vx']:.6f}, "
        f"vy = {phase3_handoff_state['rocket_velocity_km_s']['vy']:.6f}"
    )
    print(f"  Rocket mass (kg): {phase3_handoff_state['rocket_mass_kg']:.3f}")
    print(
        f"  Mars position (km): x = {phase3_handoff_state['mars_position_km']['x']:.3f}, "
        f"y = {phase3_handoff_state['mars_position_km']['y']:.3f}"
    )
    print(f"  Distance to Mars (km): {phase3_handoff_state['distance_to_mars_km']:.3f}")


if __name__ == "__main__":
    animation, simulation = animate_transfer_burn_phase2(**PHASE2_INPUTS)
    print_handoff_state(simulation["phase1_final_state"])
    print_final_state(simulation["final_state"])
