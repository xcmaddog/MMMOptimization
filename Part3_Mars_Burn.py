import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time, TimeDelta


# Core constants for the phase-3 Mars-centric observation model.
MARS_RADIUS_KM = 3389.5
MARS_MU_KM = 42828.375214  # km^3 / s^2


"""
------------------------------------------------------------
Phase 3 Inputs

Phase 3 starts from the Mars handoff state produced by part 2.
For now, this file uses placeholder values in PHASE2_HANDOFF_STATE.
Later, your combined runner can replace those values with the real phase-2
handoff dictionary before running phase 3.

Input details:
- correction_fuel_mass_kg:
  Fuel available for the corrective burn at the start of phase 3.
- thrust_newtons:
  Corrective-thruster force in newtons.
- collision_conversion_burn_duration_minutes:
  Duration of the initial prograde burn used only when phase 2 says the rocket
  is on a collision course with Mars.
- requested_burn_duration_minutes:
  Duration of the retrograde burn applied at closest pass.
  In a near-pass case, this is the only burn.
- dt_seconds:
  Output time step in seconds.
- max_step_seconds:
  Internal RK4 integration step size in seconds.
- total_time_days:
  Total duration of the phase-3 observation window.
- playback_speed:
  Animation speed multiplier.
- fps:
  Animation frame rate target used by Matplotlib.
------------------------------------------------------------
"""
PHASE3_INPUTS = {
    "correction_fuel_mass_kg": 1640.0,
    "thrust_newtons": 1200.0,
    "collision_conversion_burn_duration_minutes": 30.0,
    "requested_burn_duration_minutes": 30.0,
    "dt_seconds": 180.0,
    "max_step_seconds": 30.0,
    "total_time_days": 5.0,
    "playback_speed": 1.0,
    "fps": 30,
}


# Placeholder phase-2 handoff state for standalone development of phase 3.
# Replace this dictionary from your multi-phase runner once phase 2 and phase 3
# are being executed together.
PHASE2_HANDOFF_STATE = {
    "handoff_reason": "closest approach to Mars",
    "mars_approach_type": "near_pass",
    "recommended_burn_direction": "retrograde_at_closest_pass",
    "phase3_collision_lead_hours": 4.0,
    "datetime_utc": "2025-10-01T00:00:00.000",
    "elapsed_time_seconds": 0.0,
    "elapsed_time_pretty": "0 d  00 h  00 m",
    "rocket_position_km": {"x": 2.05e8 + 8000.0, "y": 7.5e7},
    "rocket_velocity_km_s": {"vx": -8.0, "vy": 23.5},
    "rocket_mass_kg": 2500.0,
    "mars_position_km": {"x": 2.05e8, "y": 7.5e7},
    "mars_velocity_km_s": {"vx": -8.0, "vy": 22.0},
    "rocket_position_relative_to_mars_km": {"x": 8000.0, "y": 0.0},
    "rocket_velocity_relative_to_mars_km_s": {"vx": 0.0, "vy": 1.5},
    "distance_to_mars_km": 8000.0,
}


def gravity_accel_mars_centered(x_km, y_km, mu_km=MARS_MU_KM):
    """Return Mars' gravitational acceleration in a Mars-centered frame."""
    r_km = np.hypot(x_km, y_km)
    if r_km == 0.0:
        return 0.0, 0.0

    ax_km_s2 = -mu_km * x_km / r_km**3
    ay_km_s2 = -mu_km * y_km / r_km**3
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


def thrust_accel_mars_centered(vx_km_s, vy_km_s, thrust_newtons, total_mass_kg, direction_sign):
    """Return thrust acceleration aligned with the Mars-relative velocity direction."""
    speed_km_s = np.hypot(vx_km_s, vy_km_s)
    if speed_km_s == 0.0 or thrust_newtons <= 0.0 or total_mass_kg <= 0.0:
        return 0.0, 0.0

    accel_km_s2 = (thrust_newtons / total_mass_kg) / 1000.0
    direction_x = direction_sign * (vx_km_s / speed_km_s)
    direction_y = direction_sign * (vy_km_s / speed_km_s)
    return accel_km_s2 * direction_x, accel_km_s2 * direction_y


def radial_velocity_km_s(x_km, y_km, vx_km_s, vy_km_s):
    """Return radial velocity relative to Mars. Negative means inbound."""
    radius_km = np.hypot(x_km, y_km)
    if radius_km == 0.0:
        return 0.0
    return (x_km * vx_km_s + y_km * vy_km_s) / radius_km


def mars_states_heliocentric(t_seconds_array, simulation_start_time_utc):
    """
    Return Sun-centered Mars state arrays from Astropy ephemerides.

    Astropy supplies barycentric states. Subtracting the Sun's barycentric
    state converts those into heliocentric states.
    """
    start_time = Time(simulation_start_time_utc, scale="utc")
    sample_times = start_time + TimeDelta(t_seconds_array, format="sec")

    with solar_system_ephemeris.set("builtin"):
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", sample_times)
        mars_pos, mars_vel = get_body_barycentric_posvel("mars", sample_times)

    mars_rel_pos_km = (mars_pos.xyz - sun_pos.xyz).to_value(u.km)
    mars_rel_vel_km_s = (mars_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)

    return {
        "mars_x_km": mars_rel_pos_km[0],
        "mars_y_km": mars_rel_pos_km[1],
        "mars_vx_km_s": mars_rel_vel_km_s[0],
        "mars_vy_km_s": mars_rel_vel_km_s[1],
    }


def rocket_state_derivative_mars_centered(
    state,
    thrust_on,
    thrust_newtons,
    mass_flow_kg_s,
    burn_direction,
    mars_mu_km=MARS_MU_KM,
):
    """Return time derivatives for [x, y, vx, vy, total_mass, fuel_mass] in the Mars-centered frame."""
    x_km, y_km, vx_km_s, vy_km_s, total_mass_kg, fuel_mass_kg = state
    ax_km_s2, ay_km_s2 = gravity_accel_mars_centered(x_km, y_km, mars_mu_km)
    thrust_ax_km_s2 = 0.0
    thrust_ay_km_s2 = 0.0
    dmass_dt = 0.0
    dfuel_dt = 0.0

    if thrust_on and fuel_mass_kg > 0.0 and total_mass_kg > 0.0:
        direction_sign = 1.0 if burn_direction == "prograde" else -1.0
        thrust_ax_km_s2, thrust_ay_km_s2 = thrust_accel_mars_centered(
            vx_km_s,
            vy_km_s,
            thrust_newtons,
            total_mass_kg,
            direction_sign,
        )
        dmass_dt = -mass_flow_kg_s
        dfuel_dt = -mass_flow_kg_s

    return np.array(
        [vx_km_s, vy_km_s, ax_km_s2 + thrust_ax_km_s2, ay_km_s2 + thrust_ay_km_s2, dmass_dt, dfuel_dt],
        dtype=float,
    )


def rk4_step_mars_centered(
    state,
    step_seconds,
    thrust_on,
    thrust_newtons,
    mass_flow_kg_s,
    burn_direction,
    mars_mu_km=MARS_MU_KM,
):
    """Advance the rocket by one RK4 step in the Mars-centered frame."""
    k1 = rocket_state_derivative_mars_centered(
        state, thrust_on, thrust_newtons, mass_flow_kg_s, burn_direction, mars_mu_km=mars_mu_km
    )
    k2 = rocket_state_derivative_mars_centered(
        state + 0.5 * step_seconds * k1,
        thrust_on,
        thrust_newtons,
        mass_flow_kg_s,
        burn_direction,
        mars_mu_km=mars_mu_km,
    )
    k3 = rocket_state_derivative_mars_centered(
        state + 0.5 * step_seconds * k2,
        thrust_on,
        thrust_newtons,
        mass_flow_kg_s,
        burn_direction,
        mars_mu_km=mars_mu_km,
    )
    k4 = rocket_state_derivative_mars_centered(
        state + step_seconds * k3,
        thrust_on,
        thrust_newtons,
        mass_flow_kg_s,
        burn_direction,
        mars_mu_km=mars_mu_km,
    )
    return state + (step_seconds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def angular_position_deg(x_km, y_km):
    """Return angular position in degrees measured from the +x axis."""
    return np.mod(np.degrees(np.arctan2(y_km, x_km)), 360.0)


def build_final_state(
    elapsed_time_seconds,
    simulation_start_time_utc,
    end_datetime_utc,
    rocket_rel_x_km,
    rocket_rel_y_km,
    rocket_angle_deg,
    rocket_rel_vx_km_s,
    rocket_rel_vy_km_s,
    mars_x_km,
    mars_y_km,
    mars_vx_km_s,
    mars_vy_km_s,
    rocket_mass_kg,
    fuel_remaining_kg,
    collision_conversion_burn_duration_minutes,
    requested_burn_duration_minutes,
    initial_actual_burn_duration_seconds,
    closest_pass_actual_burn_duration_seconds,
    thrust_newtons,
    mars_approach_type,
    initial_burn_applied,
    initial_burn_direction,
    initial_burn_duration_seconds,
    closest_pass_burn_applied,
    closest_pass_burn_duration_seconds,
    stable_orbit_detected,
    first_closest_pass_distance_km,
    first_closest_pass_datetime_utc,
    returned_inward_after_first_pass,
    resumed_outward_after_return,
    orbit_confirmation_datetime_utc,
    status,
):
    """Package the phase-3 end state in both Mars-relative and Sun-relative form."""
    rocket_x_sun_km = mars_x_km + rocket_rel_x_km
    rocket_y_sun_km = mars_y_km + rocket_rel_y_km
    rocket_vx_sun_km_s = mars_vx_km_s + rocket_rel_vx_km_s
    rocket_vy_sun_km_s = mars_vy_km_s + rocket_rel_vy_km_s

    return {
        "simulation_start_time_utc": simulation_start_time_utc,
        "end_datetime_utc": end_datetime_utc,
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "elapsed_time_pretty": format_elapsed_time(elapsed_time_seconds),
        "rocket_position_km": {"x": float(rocket_x_sun_km), "y": float(rocket_y_sun_km)},
        "rocket_velocity_km_s": {"vx": float(rocket_vx_sun_km_s), "vy": float(rocket_vy_sun_km_s)},
        "mars_position_km": {"x": float(mars_x_km), "y": float(mars_y_km)},
        "mars_velocity_km_s": {"vx": float(mars_vx_km_s), "vy": float(mars_vy_km_s)},
        "rocket_relative_to_mars_km": {"x": float(rocket_rel_x_km), "y": float(rocket_rel_y_km)},
        "rocket_relative_to_mars_velocity_km_s": {
            "vx": float(rocket_rel_vx_km_s),
            "vy": float(rocket_rel_vy_km_s),
        },
        "rocket_angle_deg": float(rocket_angle_deg),
        "rocket_mass_kg": float(rocket_mass_kg),
        "fuel_remaining_kg": float(fuel_remaining_kg),
        "collision_conversion_burn_duration_minutes": float(collision_conversion_burn_duration_minutes),
        "requested_burn_duration_minutes": float(requested_burn_duration_minutes),
        "initial_actual_burn_duration_minutes": float(initial_actual_burn_duration_seconds / 60.0),
        "closest_pass_actual_burn_duration_minutes": float(closest_pass_actual_burn_duration_seconds / 60.0),
        "thrust_newtons": float(thrust_newtons),
        "mars_approach_type": mars_approach_type,
        "initial_burn_applied": initial_burn_applied,
        "initial_burn_direction": initial_burn_direction,
        "initial_burn_duration_minutes": float(initial_burn_duration_seconds / 60.0),
        "closest_pass_burn_applied": closest_pass_burn_applied,
        "closest_pass_burn_duration_minutes": float(closest_pass_burn_duration_seconds / 60.0),
        "stable_orbit_detected": stable_orbit_detected,
        "first_closest_pass_distance_km": None
        if first_closest_pass_distance_km is None
        else float(first_closest_pass_distance_km),
        "first_closest_pass_datetime_utc": first_closest_pass_datetime_utc,
        "returned_inward_after_first_pass": returned_inward_after_first_pass,
        "resumed_outward_after_return": resumed_outward_after_return,
        "orbit_confirmation_datetime_utc": orbit_confirmation_datetime_utc,
        "status": status,
    }


def simulate_mars_orbit_phase3(
    correction_fuel_mass_kg=PHASE3_INPUTS["correction_fuel_mass_kg"],
    thrust_newtons=PHASE3_INPUTS["thrust_newtons"],
    collision_conversion_burn_duration_minutes=PHASE3_INPUTS["collision_conversion_burn_duration_minutes"],
    requested_burn_duration_minutes=PHASE3_INPUTS["requested_burn_duration_minutes"],
    dt_seconds=PHASE3_INPUTS["dt_seconds"],
    max_step_seconds=PHASE3_INPUTS["max_step_seconds"],
    total_time_days=PHASE3_INPUTS["total_time_days"],
    mars_radius_km=MARS_RADIUS_KM,
    mars_mu_km=MARS_MU_KM,
):
    """
    Simulate phase 3 as a fixed-Mars observation orbit.

    Modeling assumptions:
    - Mars stays fixed at the origin for the local rocket dynamics.
    - The rocket feels Mars' gravity only.
    - Collision cases begin before impact and receive an initial prograde burn.
    - Near-pass cases coast at first.
    - Once closest pass is reached, a retrograde burn is applied to either case.
    - Mars heliocentric states are still tracked with Astropy.
    - Rocket heliocentric output is reconstructed from the Mars-relative state.
    """
    handoff_state = PHASE2_HANDOFF_STATE
    simulation_start_time_utc = handoff_state["datetime_utc"]
    mars_approach_type = handoff_state.get("mars_approach_type", "near_pass")
    handoff_total_mass_kg = handoff_state.get("rocket_mass_kg", 0.0)
    dry_mass_kg = max(0.0, handoff_total_mass_kg - correction_fuel_mass_kg)

    total_time_seconds = total_time_days * 86400.0
    n_output_steps = int(total_time_seconds / dt_seconds) + 1
    t_seconds = np.linspace(0.0, total_time_seconds, n_output_steps)

    mars_states = mars_states_heliocentric(
        t_seconds_array=t_seconds,
        simulation_start_time_utc=simulation_start_time_utc,
    )
    mars_x_km = mars_states["mars_x_km"]
    mars_y_km = mars_states["mars_y_km"]
    mars_vx_km_s = mars_states["mars_vx_km_s"]
    mars_vy_km_s = mars_states["mars_vy_km_s"]

    rocket_rel_x_km = np.zeros(n_output_steps)
    rocket_rel_y_km = np.zeros(n_output_steps)
    rocket_rel_vx_km_s = np.zeros(n_output_steps)
    rocket_rel_vy_km_s = np.zeros(n_output_steps)
    rocket_mass_kg = np.zeros(n_output_steps)
    fuel_mass_kg = np.zeros(n_output_steps)

    rocket_rel_x_km[0] = handoff_state["rocket_position_km"]["x"] - handoff_state["mars_position_km"]["x"]
    rocket_rel_y_km[0] = handoff_state["rocket_position_km"]["y"] - handoff_state["mars_position_km"]["y"]
    rocket_rel_vx_km_s[0] = handoff_state["rocket_velocity_km_s"]["vx"] - handoff_state["mars_velocity_km_s"]["vx"]
    rocket_rel_vy_km_s[0] = handoff_state["rocket_velocity_km_s"]["vy"] - handoff_state["mars_velocity_km_s"]["vy"]
    rocket_mass_kg[0] = handoff_total_mass_kg
    fuel_mass_kg[0] = min(correction_fuel_mass_kg, handoff_total_mass_kg)

    collision_conversion_burn_duration_seconds = max(0.0, collision_conversion_burn_duration_minutes * 60.0)
    requested_burn_duration_seconds = max(0.0, requested_burn_duration_minutes * 60.0)
    total_planned_burn_seconds = requested_burn_duration_seconds
    if mars_approach_type == "collision_course":
        total_planned_burn_seconds += collision_conversion_burn_duration_seconds

    if total_planned_burn_seconds > 0.0 and correction_fuel_mass_kg > 0.0 and thrust_newtons > 0.0:
        mass_flow_kg_s = correction_fuel_mass_kg / total_planned_burn_seconds
    else:
        mass_flow_kg_s = 0.0

    status = "observing Mars encounter"
    stop_index = n_output_steps
    initial_burn_enabled = (
        mars_approach_type == "collision_course"
        and collision_conversion_burn_duration_seconds > 0.0
        and thrust_newtons > 0.0
        and fuel_mass_kg[0] > 0.0
    )
    initial_burn_direction = "prograde" if initial_burn_enabled else "none"
    initial_burn_elapsed_seconds = 0.0
    closest_pass_burn_started = False
    closest_pass_burn_elapsed_seconds = 0.0
    first_closest_pass_distance_km = None
    first_closest_pass_time_seconds = None
    returned_inward_after_first_pass = False
    resumed_outward_after_return = False
    orbit_confirmation_time_seconds = None
    previous_radial_velocity = radial_velocity_km_s(
        rocket_rel_x_km[0],
        rocket_rel_y_km[0],
        rocket_rel_vx_km_s[0],
        rocket_rel_vy_km_s[0],
    )

    for i in range(n_output_steps - 1):
        interval_seconds = t_seconds[i + 1] - t_seconds[i]
        state = np.array(
            [
                rocket_rel_x_km[i],
                rocket_rel_y_km[i],
                rocket_rel_vx_km_s[i],
                rocket_rel_vy_km_s[i],
                rocket_mass_kg[i],
                fuel_mass_kg[i],
            ],
            dtype=float,
        )

        interval_completed = True
        interval_elapsed = 0.0

        while interval_elapsed < interval_seconds - 1e-12:
            remaining_interval_seconds = interval_seconds - interval_elapsed
            substep_seconds = min(max_step_seconds, remaining_interval_seconds)

            active_burn_direction = None
            active_burn_elapsed_seconds = 0.0
            if (
                initial_burn_enabled
                and initial_burn_elapsed_seconds < collision_conversion_burn_duration_seconds - 1e-12
                and state[5] > 0.0
            ):
                active_burn_direction = "prograde"
                active_burn_elapsed_seconds = initial_burn_elapsed_seconds
                active_burn_target_seconds = collision_conversion_burn_duration_seconds
            elif (
                closest_pass_burn_started
                and closest_pass_burn_elapsed_seconds < requested_burn_duration_seconds - 1e-12
                and state[5] > 0.0
            ):
                active_burn_direction = "retrograde"
                active_burn_elapsed_seconds = closest_pass_burn_elapsed_seconds
                active_burn_target_seconds = requested_burn_duration_seconds
            else:
                active_burn_target_seconds = 0.0

            burn_active = active_burn_direction is not None
            if burn_active:
                remaining_burn_seconds = active_burn_target_seconds - active_burn_elapsed_seconds
                substep_seconds = min(substep_seconds, remaining_burn_seconds)

            rocket_radius_km = np.hypot(state[0], state[1])

            if rocket_radius_km <= mars_radius_km:
                status = "impacted Mars"
                collision_time_seconds = t_seconds[i] + interval_elapsed
                t_seconds[i + 1] = collision_time_seconds
                rocket_rel_x_km[i + 1] = state[0]
                rocket_rel_y_km[i + 1] = state[1]
                rocket_rel_vx_km_s[i + 1] = state[2]
                rocket_rel_vy_km_s[i + 1] = state[3]
                rocket_mass_kg[i + 1] = state[4]
                fuel_mass_kg[i + 1] = state[5]
                stop_index = i + 2
                interval_completed = False
                break

            state = rk4_step_mars_centered(
                state,
                step_seconds=substep_seconds,
                thrust_on=burn_active,
                thrust_newtons=thrust_newtons,
                mass_flow_kg_s=mass_flow_kg_s,
                burn_direction=active_burn_direction or "retrograde",
                mars_mu_km=mars_mu_km,
            )
            state[5] = max(0.0, state[5])
            state[4] = max(dry_mass_kg, state[4])

            interval_elapsed += substep_seconds
            if active_burn_direction == "prograde":
                initial_burn_elapsed_seconds += substep_seconds
            elif active_burn_direction == "retrograde":
                closest_pass_burn_elapsed_seconds += substep_seconds

            current_radial_velocity = radial_velocity_km_s(state[0], state[1], state[2], state[3])
            if (
                previous_radial_velocity < 0.0
                and current_radial_velocity >= 0.0
            ):
                periapsis_distance_km = np.hypot(state[0], state[1])
                periapsis_time_seconds = t_seconds[i] + interval_elapsed

                if first_closest_pass_distance_km is None:
                    first_closest_pass_distance_km = periapsis_distance_km
                    first_closest_pass_time_seconds = periapsis_time_seconds
                    closest_pass_burn_started = (
                        requested_burn_duration_seconds > 0.0 and thrust_newtons > 0.0 and state[5] > 0.0
                    )
                elif returned_inward_after_first_pass and not resumed_outward_after_return:
                    resumed_outward_after_return = True
                    orbit_confirmation_time_seconds = periapsis_time_seconds
            elif (
                first_closest_pass_time_seconds is not None
                and not returned_inward_after_first_pass
                and current_radial_velocity < 0.0
            ):
                returned_inward_after_first_pass = True
            previous_radial_velocity = current_radial_velocity

        if not interval_completed:
            break

        rocket_rel_x_km[i + 1] = state[0]
        rocket_rel_y_km[i + 1] = state[1]
        rocket_rel_vx_km_s[i + 1] = state[2]
        rocket_rel_vy_km_s[i + 1] = state[3]
        rocket_mass_kg[i + 1] = state[4]
        fuel_mass_kg[i + 1] = state[5]

    rocket_rel_x_km = rocket_rel_x_km[:stop_index]
    rocket_rel_y_km = rocket_rel_y_km[:stop_index]
    rocket_rel_vx_km_s = rocket_rel_vx_km_s[:stop_index]
    rocket_rel_vy_km_s = rocket_rel_vy_km_s[:stop_index]
    rocket_mass_kg = rocket_mass_kg[:stop_index]
    fuel_mass_kg = fuel_mass_kg[:stop_index]
    rocket_angle_deg = angular_position_deg(rocket_rel_x_km, rocket_rel_y_km)
    mars_x_km = mars_x_km[:stop_index]
    mars_y_km = mars_y_km[:stop_index]
    mars_vx_km_s = mars_vx_km_s[:stop_index]
    mars_vy_km_s = mars_vy_km_s[:stop_index]
    t_seconds = t_seconds[:stop_index]

    if status == "observing Mars encounter":
        status = "completed full simulation"

    end_datetime_utc = (
        Time(simulation_start_time_utc, scale="utc") + TimeDelta(t_seconds[-1], format="sec")
    ).utc.isot

    rocket_x_sun_km = mars_x_km + rocket_rel_x_km
    rocket_y_sun_km = mars_y_km + rocket_rel_y_km
    rocket_vx_sun_km_s = mars_vx_km_s + rocket_rel_vx_km_s
    rocket_vy_sun_km_s = mars_vy_km_s + rocket_rel_vy_km_s

    final_state = build_final_state(
        elapsed_time_seconds=t_seconds[-1],
        simulation_start_time_utc=simulation_start_time_utc,
        end_datetime_utc=end_datetime_utc,
        rocket_rel_x_km=rocket_rel_x_km[-1],
        rocket_rel_y_km=rocket_rel_y_km[-1],
        rocket_angle_deg=rocket_angle_deg[-1],
        rocket_rel_vx_km_s=rocket_rel_vx_km_s[-1],
        rocket_rel_vy_km_s=rocket_rel_vy_km_s[-1],
        mars_x_km=mars_x_km[-1],
        mars_y_km=mars_y_km[-1],
        mars_vx_km_s=mars_vx_km_s[-1],
        mars_vy_km_s=mars_vy_km_s[-1],
        rocket_mass_kg=rocket_mass_kg[-1],
        fuel_remaining_kg=fuel_mass_kg[-1],
        collision_conversion_burn_duration_minutes=collision_conversion_burn_duration_minutes,
        requested_burn_duration_minutes=requested_burn_duration_minutes,
        initial_actual_burn_duration_seconds=initial_burn_elapsed_seconds,
        closest_pass_actual_burn_duration_seconds=closest_pass_burn_elapsed_seconds,
        thrust_newtons=thrust_newtons,
        mars_approach_type=mars_approach_type,
        initial_burn_applied=initial_burn_elapsed_seconds > 0.0,
        initial_burn_direction=initial_burn_direction,
        initial_burn_duration_seconds=initial_burn_elapsed_seconds,
        closest_pass_burn_applied=closest_pass_burn_elapsed_seconds > 0.0,
        closest_pass_burn_duration_seconds=closest_pass_burn_elapsed_seconds,
        stable_orbit_detected=(status != "impacted Mars" and returned_inward_after_first_pass and resumed_outward_after_return),
        first_closest_pass_distance_km=first_closest_pass_distance_km,
        first_closest_pass_datetime_utc=None
        if first_closest_pass_time_seconds is None
        else (Time(simulation_start_time_utc, scale="utc") + TimeDelta(first_closest_pass_time_seconds, format="sec")).utc.isot,
        returned_inward_after_first_pass=returned_inward_after_first_pass,
        resumed_outward_after_return=resumed_outward_after_return,
        orbit_confirmation_datetime_utc=None
        if orbit_confirmation_time_seconds is None
        else (Time(simulation_start_time_utc, scale="utc") + TimeDelta(orbit_confirmation_time_seconds, format="sec")).utc.isot,
        status=status,
    )

    return {
        "t_seconds": t_seconds,
        "rocket_rel_x_km": rocket_rel_x_km,
        "rocket_rel_y_km": rocket_rel_y_km,
        "rocket_angle_deg": rocket_angle_deg,
        "rocket_rel_vx_km_s": rocket_rel_vx_km_s,
        "rocket_rel_vy_km_s": rocket_rel_vy_km_s,
        "rocket_mass_kg": rocket_mass_kg,
        "fuel_mass_kg": fuel_mass_kg,
        "rocket_x_km": rocket_x_sun_km,
        "rocket_y_km": rocket_y_sun_km,
        "rocket_vx_km_s": rocket_vx_sun_km_s,
        "rocket_vy_km_s": rocket_vy_sun_km_s,
        "mars_x_km": mars_x_km,
        "mars_y_km": mars_y_km,
        "mars_vx_km_s": mars_vx_km_s,
        "mars_vy_km_s": mars_vy_km_s,
        "status": status,
        "final_state": final_state,
    }


def animate_mars_orbit_phase3(
    correction_fuel_mass_kg=PHASE3_INPUTS["correction_fuel_mass_kg"],
    thrust_newtons=PHASE3_INPUTS["thrust_newtons"],
    collision_conversion_burn_duration_minutes=PHASE3_INPUTS["collision_conversion_burn_duration_minutes"],
    requested_burn_duration_minutes=PHASE3_INPUTS["requested_burn_duration_minutes"],
    dt_seconds=PHASE3_INPUTS["dt_seconds"],
    max_step_seconds=PHASE3_INPUTS["max_step_seconds"],
    total_time_days=PHASE3_INPUTS["total_time_days"],
    playback_speed=PHASE3_INPUTS["playback_speed"],
    fps=PHASE3_INPUTS["fps"],
):
    """
    Animate phase 3 using a fixed-Mars observation view.

    Mars stays fixed on screen while the rocket motion is observed relative to it.
    """
    simulation = simulate_mars_orbit_phase3(
        correction_fuel_mass_kg=correction_fuel_mass_kg,
        thrust_newtons=thrust_newtons,
        collision_conversion_burn_duration_minutes=collision_conversion_burn_duration_minutes,
        requested_burn_duration_minutes=requested_burn_duration_minutes,
        dt_seconds=dt_seconds,
        max_step_seconds=max_step_seconds,
        total_time_days=total_time_days,
    )

    t_seconds = simulation["t_seconds"]
    rocket_rel_x_km = simulation["rocket_rel_x_km"]
    rocket_rel_y_km = simulation["rocket_rel_y_km"]
    rocket_rel_vx_km_s = simulation["rocket_rel_vx_km_s"]
    rocket_rel_vy_km_s = simulation["rocket_rel_vy_km_s"]
    rocket_mass_kg = simulation["rocket_mass_kg"]
    fuel_mass_kg = simulation["fuel_mass_kg"]
    status = simulation["status"]
    phase3_summary = simulation["final_state"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x relative to Mars (km)")
    ax.set_ylabel("y relative to Mars (km)")
    ax.set_title("Phase 3: Rocket Motion Near Mars")

    max_extent_km = max(
        np.max(np.abs(rocket_rel_x_km)),
        np.max(np.abs(rocket_rel_y_km)),
        MARS_RADIUS_KM,
    )
    pad_km = 0.12 * max_extent_km + 300.0
    ax.set_xlim(-max_extent_km - pad_km, max_extent_km + pad_km)
    ax.set_ylim(-max_extent_km - pad_km, max_extent_km + pad_km)

    mars = plt.Circle((0.0, 0.0), MARS_RADIUS_KM, color="orangered", alpha=0.85)
    ax.add_patch(mars)

    rocket_path_line, = ax.plot([], [], lw=2, color="black", label="Rocket path")
    rocket_point, = ax.plot([], [], marker="o", color="black", label="Rocket")
    ax.plot([rocket_rel_x_km[0]], [rocket_rel_y_km[0]], marker="x", markersize=8, color="green", label="Start")
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
        rocket_path_line.set_data([], [])
        rocket_point.set_data([], [])
        info.set_text("")
        return rocket_path_line, rocket_point, info

    def update(frame_number):
        i = frame_indices[frame_number]
        rocket_path_line.set_data(rocket_rel_x_km[: i + 1], rocket_rel_y_km[: i + 1])
        rocket_point.set_data([rocket_rel_x_km[i]], [rocket_rel_y_km[i]])

        speed_km_s = np.hypot(rocket_rel_vx_km_s[i], rocket_rel_vy_km_s[i])
        altitude_now_km = np.hypot(rocket_rel_x_km[i], rocket_rel_y_km[i]) - MARS_RADIUS_KM
        if phase3_summary["initial_burn_applied"] and not phase3_summary["closest_pass_burn_applied"]:
            burn_status = "collision-conversion strategy active"
        elif phase3_summary["initial_burn_applied"] and phase3_summary["closest_pass_burn_applied"]:
            burn_status = "two-burn strategy completed"
        elif phase3_summary["closest_pass_burn_applied"]:
            burn_status = "closest-pass capture burn completed"
        else:
            burn_status = "coasting"

        info.set_text(
            f"time = {format_elapsed_time(t_seconds[i])}\n"
            f"rocket x = {rocket_rel_x_km[i]:.1f} km\n"
            f"rocket y = {rocket_rel_y_km[i]:.1f} km\n"
            f"rocket vx = {rocket_rel_vx_km_s[i]:.4f} km/s\n"
            f"rocket vy = {rocket_rel_vy_km_s[i]:.4f} km/s\n"
            f"mass = {rocket_mass_kg[i]:.2f} kg\n"
            f"fuel = {fuel_mass_kg[i]:.2f} kg\n"
            f"burn = {burn_status}\n"
            f"altitude = {altitude_now_km:.1f} km\n"
            f"speed = {speed_km_s:.4f} km/s"
            + (f"\nstatus = {status}" if i == len(t_seconds) - 1 else "")
        )
        return rocket_path_line, rocket_point, info

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000 / (fps * playback_speed),
        blit=True,
    )

    plt.show()
    return anim, simulation["final_state"]


def print_final_state(final_state):
    """Print the final Sun-relative and Mars-relative phase-3 states."""
    print("Final phase 3 state")
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
        f"Rocket position relative to Mars (km): x = {final_state['rocket_relative_to_mars_km']['x']:.3f}, "
        f"y = {final_state['rocket_relative_to_mars_km']['y']:.3f}"
    )
    print(
        f"Rocket velocity relative to Mars (km/s): vx = {final_state['rocket_relative_to_mars_velocity_km_s']['vx']:.6f}, "
        f"vy = {final_state['rocket_relative_to_mars_velocity_km_s']['vy']:.6f}"
    )
    print(f"Rocket final angle around Mars: {final_state['rocket_angle_deg']:.3f} deg")
    print(f"Mars approach type from phase 2: {final_state['mars_approach_type']}")
    print(f"Thrust: {final_state['thrust_newtons']:.3f} N")
    print(
        f"Collision-conversion burn duration setting: "
        f"{final_state['collision_conversion_burn_duration_minutes']:.3f} min"
    )
    print(f"Requested burn duration: {final_state['requested_burn_duration_minutes']:.3f} min")
    print(f"Initial burn applied: {final_state['initial_burn_applied']}")
    print(f"Initial burn direction: {final_state['initial_burn_direction']}")
    print(f"Initial actual burn duration: {final_state['initial_actual_burn_duration_minutes']:.3f} min")
    print(f"Closest-pass burn applied: {final_state['closest_pass_burn_applied']}")
    print(
        f"Closest-pass actual burn duration: "
        f"{final_state['closest_pass_actual_burn_duration_minutes']:.3f} min"
    )
    print(f"Stable orbit detected: {final_state['stable_orbit_detected']}")
    print(f"First closest-pass date-time (UTC): {final_state['first_closest_pass_datetime_utc']}")
    print(f"First closest-pass distance (km): {final_state['first_closest_pass_distance_km']}")
    print(f"Returned inward after first pass: {final_state['returned_inward_after_first_pass']}")
    print(f"Resumed outward after return: {final_state['resumed_outward_after_return']}")
    print(f"Orbit confirmation date-time (UTC): {final_state['orbit_confirmation_datetime_utc']}")
    print(f"Rocket mass: {final_state['rocket_mass_kg']:.3f} kg")
    print(f"Fuel remaining: {final_state['fuel_remaining_kg']:.3f} kg")
    print(
        f"Mars position relative to Sun (km): x = {final_state['mars_position_km']['x']:.3f}, "
        f"y = {final_state['mars_position_km']['y']:.3f}"
    )
    print(
        f"Mars velocity relative to Sun (km/s): vx = {final_state['mars_velocity_km_s']['vx']:.6f}, "
        f"vy = {final_state['mars_velocity_km_s']['vy']:.6f}"
    )


if __name__ == "__main__":
    animation, final_state = animate_mars_orbit_phase3(**PHASE3_INPUTS)
    print_final_state(final_state)
