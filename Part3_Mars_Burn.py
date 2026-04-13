import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time, TimeDelta

# Core constants for the phase-3 Mars-centric observation model.
MARS_RADIUS_KM = 3389.5
MARS_MU_KM = 42828.375214  # km^3 / s^2

"""
------------------------------------------------------------
Phase 3 Inputs
------------------------------------------------------------
"""
PHASE3_INPUTS = {
    "correction_fuel_mass_kg": 1640.0,
    "thrust_newtons": 1200.0,
    "collision_conversion_burn_duration_minutes": 30.0,
    "requested_burn_duration_minutes": 30.0,
    "capture_start_radial_velocity_km_s": -0.05,
    "dt_seconds": 180.0,
    "max_step_seconds": 30.0,
    "total_time_days": 5.0,
    "playback_speed": 1.0,
    "fps": 30,
}

PHASE2_HANDOFF_STATE = {
    "handoff_reason": "closest approach to Mars",
    "mars_approach_type": "near_pass",
    "recommended_burn_direction": "retrograde_at_closest_pass",
    "phase3_collision_lead_hours": 4.0,
    "datetime_utc": "2025-10-01T00:00:00.000",
    "elapsed_time_seconds": 0.0,
    "elapsed_time_pretty": "0 d 00 h 00 m",
    "rocket_position_km": {"x": 2.05e8 + 8000.0, "y": 7.5e7},
    "rocket_velocity_km_s": {"vx": -8.0, "vy": 23.5},
    "rocket_mass_kg": 2500.0,
    "mars_position_km": {"x": 2.05e8, "y": 7.5e7},
    "mars_velocity_km_s": {"vx": -8.0, "vy": 22.0},
    "rocket_position_relative_to_mars_km": {"x": 8000.0, "y": 0.0},
    "rocket_velocity_relative_to_mars_km_s": {"vx": 0.0, "vy": 1.5},
    "distance_to_mars_km": 8000.0,
}

# ---------------------------------------------------------
# NUMBA ACCELERATED PHYSICS CORE
# ---------------------------------------------------------

@njit
def _radial_velocity_jit(x, y, vx, vy):
    """Calculates radial velocity (negative means approaching, positive means leaving)."""
    r = np.sqrt(x * x + y * y)
    if r == 0.0:
        return 0.0
    return (x * vx + y * vy) / r

@njit
def _rocket_derivative_phase3_jit(state, thrust_on, thrust_n, mass_flow, burn_dir_sign, mars_mu):
    """Derivatives in Mars-centered frame."""
    x, y, vx, vy, mass, fuel = state[0], state[1], state[2], state[3], state[4], state[5]
    r_sq = x * x + y * y
    r = np.sqrt(r_sq)

    ax, ay = 0.0, 0.0
    if r > 0.0:
        factor = -mars_mu / (r_sq * r)
        ax = x * factor
        ay = y * factor

    tax, tay, dm, df = 0.0, 0.0, 0.0, 0.0
    if thrust_on and fuel > 0.0 and mass > 0.0:
        speed = np.sqrt(vx * vx + vy * vy)
        if speed > 0.0:
            accel = (thrust_n / mass) / 1000.0
            # burn_dir_sign = -1 is Retrograde (slow down for capture)
            tax = accel * burn_dir_sign * (vx / speed)
            tay = accel * burn_dir_sign * (vy / speed)
            dm = -mass_flow
            df = -mass_flow

    return np.array([vx, vy, ax + tax, ay + tay, dm, df], dtype=np.float64)

@njit
def _rk4_step_phase3_jit(state, dt, thrust_on, thrust_n, mass_flow, burn_dir_sign, mars_mu):
    k1 = _rocket_derivative_phase3_jit(state, thrust_on, thrust_n, mass_flow, burn_dir_sign, mars_mu)
    k2 = _rocket_derivative_phase3_jit(state + 0.5 * dt * k1, thrust_on, thrust_n, mass_flow, burn_dir_sign, mars_mu)
    k3 = _rocket_derivative_phase3_jit(state + 0.5 * dt * k2, thrust_on, thrust_n, mass_flow, burn_dir_sign, mars_mu)
    k4 = _rocket_derivative_phase3_jit(state + dt * k3, thrust_on, thrust_n, mass_flow, burn_dir_sign, mars_mu)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@njit
def _propagate_phase3_jit(start_state, t_array, max_step, mars_radius, mars_mu,
                          thrust_n, mass_flow, col_burn_sec, req_burn_sec,
                          is_collision_course, dry_mass, coast_sec,
                          capture_start_radial_velocity_km_s):
    """Main compiled integration loop for Phase 3."""
    n_steps = len(t_array)
    out = np.zeros((n_steps, 6), dtype=np.float64)
    out[0] = start_state
    current_state = start_state.copy()

    init_burn_enabled = (is_collision_course and col_burn_sec > 0.0 and thrust_n > 0.0 and current_state[5] > 0.0)
    init_burn_elapsed = 0.0

    cp_burn_started = False
    cp_burn_armed = ((not is_collision_course) and (req_burn_sec > 0.0 and thrust_n > 0.0 and current_state[5] > 0.0))
    cp_burn_elapsed = 0.0

    first_cp_dist = -1.0
    first_cp_time = -1.0
    orbit_conf_time = -1.0
    ret_inward = False
    res_outward = False

    prev_rad_v = _radial_velocity_jit(current_state[0], current_state[1], current_state[2], current_state[3])

    for i in range(n_steps - 1):
        int_sec = t_array[i + 1] - t_array[i]
        int_elapsed = 0.0

        while int_elapsed < int_sec - 1e-12:
            rem_int_sec = int_sec - int_elapsed
            substep = min(max_step, rem_int_sec)

            # --- ENGINE IGNITION LOGIC FIX ---
            burn_dir_sign = 0
            active_target = 0.0
            active_elapsed = 0.0

            if init_burn_enabled and init_burn_elapsed < col_burn_sec:
                burn_dir_sign = 1  # Prograde dodge
                active_target = col_burn_sec
                active_elapsed = init_burn_elapsed
            elif cp_burn_started and cp_burn_elapsed < req_burn_sec:
                burn_dir_sign = -1 # Retrograde capture
                active_target = req_burn_sec
                active_elapsed = cp_burn_elapsed

            thrust_on = (burn_dir_sign != 0)
            if thrust_on:
                rem_burn = active_target - active_elapsed
                substep = min(substep, rem_burn)

            r = np.sqrt(current_state[0] ** 2 + current_state[1] ** 2)
            if r <= mars_radius:
                return (out, i + 1, 1, init_burn_elapsed, cp_burn_elapsed, first_cp_dist, first_cp_time, ret_inward, res_outward, orbit_conf_time)

            current_state = _rk4_step_phase3_jit(current_state, substep, thrust_on, thrust_n, mass_flow, burn_dir_sign, mars_mu)

            current_state[5] = max(0.0, current_state[5])
            current_state[4] = max(dry_mass, current_state[4])

            int_elapsed += substep
            if burn_dir_sign == 1:
                init_burn_elapsed += substep
            elif burn_dir_sign == -1:
                cp_burn_elapsed += substep

            # Calculate radial velocity to determine when we hit periapsis
            curr_rad_v = _radial_velocity_jit(current_state[0], current_state[1], current_state[2], current_state[3])

            # Trigger the capture burn just before we hit closest approach
            if cp_burn_armed and not cp_burn_started:
                if curr_rad_v < 0.0 and curr_rad_v > capture_start_radial_velocity_km_s:
                    cp_burn_started = True

            # --- ORBIT TELEMETRY TRACKING ---
            # 1. Crossing 0 from negative to positive -> Periapsis (Closest Pass)
            if prev_rad_v < 0.0 and curr_rad_v >= 0.0:
                if first_cp_time < 0.0:
                    first_cp_time = t_array[i] + int_elapsed
                    first_cp_dist = r

            # 2. Crossing 0 from positive to negative -> Apoapsis (Farthest Point)
            elif prev_rad_v > 0.0 and curr_rad_v <= 0.0:
                if first_cp_time > 0.0: # We already had a closest pass
                    res_outward = True
                    ret_inward = True # It is falling back inward (Stable Orbit Confirmed!)
                    orbit_conf_time = t_array[i] + int_elapsed

            prev_rad_v = curr_rad_v

        out[i + 1] = current_state

    return (out, n_steps, 0, init_burn_elapsed, cp_burn_elapsed, first_cp_dist, first_cp_time, ret_inward, res_outward, orbit_conf_time)
# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def format_elapsed_time(seconds):
    total_seconds = int(round(seconds))
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = ((total_seconds % 86400) % 3600) // 60
    return f"{days} d {hours:02d} h {minutes:02d} m"

def angular_position_deg(x_km, y_km):
    return np.mod(np.degrees(np.arctan2(y_km, x_km)), 360.0)

def mars_states_heliocentric(t_seconds_array, simulation_start_time_utc):
    start_time = Time(simulation_start_time_utc, scale="utc")
    sample_times = start_time + TimeDelta(t_seconds_array, format="sec")
    with solar_system_ephemeris.set("builtin"):
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", sample_times)
        mars_pos, mars_vel = get_body_barycentric_posvel("mars", sample_times)
    return {
        "mars_x_km": (mars_pos.xyz - sun_pos.xyz).to_value(u.km)[0],
        "mars_y_km": (mars_pos.xyz - sun_pos.xyz).to_value(u.km)[1],
        "mars_vx_km_s": (mars_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)[0],
        "mars_vy_km_s": (mars_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)[1],
    }

def build_final_state(
    elapsed_time_seconds, simulation_start_time_utc, end_datetime_utc, rocket_rel_x_km, rocket_rel_y_km,
    rocket_angle_deg, rocket_rel_vx_km_s, rocket_rel_vy_km_s, mars_x_km, mars_y_km, mars_vx_km_s, mars_vy_km_s,
    rocket_mass_kg, fuel_remaining_kg, collision_conversion_burn_duration_minutes, requested_burn_duration_minutes,
    initial_actual_burn_duration_seconds, closest_pass_actual_burn_duration_seconds, thrust_newtons, mars_approach_type,
    initial_burn_applied, initial_burn_direction, initial_burn_duration_seconds, closest_pass_burn_applied,
    closest_pass_burn_duration_seconds, stable_orbit_detected, first_closest_pass_distance_km,
    first_closest_pass_datetime_utc, returned_inward_after_first_pass, resumed_outward_after_return,
    orbit_confirmation_datetime_utc, status
):
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
        "rocket_relative_to_mars_velocity_km_s": {"vx": float(rocket_rel_vx_km_s), "vy": float(rocket_rel_vy_km_s)},
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
        "first_closest_pass_distance_km": None if first_closest_pass_distance_km < 0 else float(first_closest_pass_distance_km),
        "first_closest_pass_datetime_utc": first_closest_pass_datetime_utc,
        "returned_inward_after_first_pass": returned_inward_after_first_pass,
        "resumed_outward_after_return": resumed_outward_after_return,
        "orbit_confirmation_datetime_utc": orbit_confirmation_datetime_utc,
        "status": status,
    }

# ---------------------------------------------------------
# SIMULATION WRAPPER
# ---------------------------------------------------------

def simulate_mars_orbit_phase3(
    correction_fuel_mass_kg=PHASE3_INPUTS["correction_fuel_mass_kg"],
    thrust_newtons=PHASE3_INPUTS["thrust_newtons"],
    collision_conversion_burn_duration_minutes=PHASE3_INPUTS["collision_conversion_burn_duration_minutes"],
    requested_burn_duration_minutes=PHASE3_INPUTS["requested_burn_duration_minutes"],
    capture_start_radial_velocity_km_s=PHASE3_INPUTS["capture_start_radial_velocity_km_s"],
    dt_seconds=PHASE3_INPUTS["dt_seconds"],
    max_step_seconds=PHASE3_INPUTS["max_step_seconds"],
    total_time_days=PHASE3_INPUTS["total_time_days"],
    mars_radius_km=MARS_RADIUS_KM,
    mars_mu_km=MARS_MU_KM,
    coast_time_days=0.0,
):
    handoff_state = PHASE2_HANDOFF_STATE
    sim_start_utc = handoff_state["datetime_utc"]
    mars_approach_type = handoff_state.get("mars_approach_type", "near_pass")
    handoff_total_mass_kg = handoff_state.get("rocket_mass_kg", 0.0)
    dry_mass_kg = max(0.0, handoff_total_mass_kg - correction_fuel_mass_kg)

    t_seconds = np.linspace(0.0, total_time_days * 86400.0, int((total_time_days * 86400.0) / dt_seconds) + 1)

    try:
        from Porkchop_Searcher import load_interpolators
        _, _, m_r_f, m_v_f = load_interpolators("ephemeris_cache.npz")
        query_jds = Time(sim_start_utc, scale="utc").jd + (t_seconds / 86400.0)
        m_pos, m_vel = m_r_f(query_jds), m_v_f(query_jds)
        mars_x_km, mars_y_km = m_pos[:, 0], m_pos[:, 1]
        mars_vx_km_s, mars_vy_km_s = m_vel[:, 0], m_vel[:, 1]
    except Exception as e:
        print(f"Warning: Falling back to Astropy. Error: {e}")
        m_states = mars_states_heliocentric(t_seconds, sim_start_utc)
        mars_x_km, mars_y_km = m_states["mars_x_km"], m_states["mars_y_km"]
        mars_vx_km_s, mars_vy_km_s = m_states["mars_vx_km_s"], m_states["mars_vy_km_s"]

    col_burn_sec = max(0.0, collision_conversion_burn_duration_minutes * 60.0)
    req_burn_sec = max(0.0, requested_burn_duration_minutes * 60.0)
    coast_sec = max(0.0, coast_time_days * 86400.0)

    total_planned = req_burn_sec + (col_burn_sec if mars_approach_type == "collision_course" else 0.0)

    mass_flow = ((thrust_newtons / 50000) * 673.0) / 60.0

    start_state = np.array([
        handoff_state["rocket_position_relative_to_mars_km"]["x"],
        handoff_state["rocket_position_relative_to_mars_km"]["y"],
        handoff_state["rocket_velocity_relative_to_mars_km_s"]["vx"],
        handoff_state["rocket_velocity_relative_to_mars_km_s"]["vy"],
        handoff_total_mass_kg,
        min(correction_fuel_mass_kg, handoff_total_mass_kg)
    ], dtype=np.float64)

    is_collision = (mars_approach_type == "collision_course")

    res = _propagate_phase3_jit(
        start_state, t_seconds, max_step_seconds, mars_radius_km, mars_mu_km,
        thrust_newtons, mass_flow, col_burn_sec, req_burn_sec,
        is_collision, dry_mass_kg, coast_sec, capture_start_radial_velocity_km_s
    )

    states, stop_idx, stat_code, init_burn, cp_burn, f_cp_dist, f_cp_time, ret_in, res_out, orb_conf_time = res

    t_sec = t_seconds[:stop_idx]
    r_x, r_y, r_vx, r_vy = states[:stop_idx, 0], states[:stop_idx, 1], states[:stop_idx, 2], states[:stop_idx, 3]
    rmass, rfuel = states[:stop_idx, 4], states[:stop_idx, 5]
    m_x, m_y, m_vx, m_vy = mars_x_km[:stop_idx], mars_y_km[:stop_idx], mars_vx_km_s[:stop_idx], mars_vy_km_s[:stop_idx]

    rocket_angle_deg = angular_position_deg(r_x, r_y)
    status = "impacted Mars" if stat_code == 1 else "completed full simulation"

    end_utc = (Time(sim_start_utc, scale="utc") + TimeDelta(t_sec[-1], format="sec")).utc.isot

    stable = (status != "impacted Mars" and ret_in and res_out)
    f_cp_utc = None if f_cp_time < 0 else (Time(sim_start_utc, scale="utc") + TimeDelta(f_cp_time, format="sec")).utc.isot
    orb_utc = None if orb_conf_time < 0 else (Time(sim_start_utc, scale="utc") + TimeDelta(orb_conf_time, format="sec")).utc.isot
    init_dir = "prograde" if (is_collision and init_burn > 0) else "none"

    final_state = build_final_state(
        t_sec[-1], sim_start_utc, end_utc, r_x[-1], r_y[-1], rocket_angle_deg[-1], r_vx[-1], r_vy[-1],
        m_x[-1], m_y[-1], m_vx[-1], m_vy[-1], rmass[-1], rfuel[-1],
        collision_conversion_burn_duration_minutes, requested_burn_duration_minutes,
        init_burn, cp_burn, thrust_newtons, mars_approach_type,
        (init_burn > 0), init_dir, init_burn, (cp_burn > 0), cp_burn,
        stable, f_cp_dist, f_cp_utc, ret_in, res_out, orb_utc, status
    )

    rocket_x_sun_km = m_x + r_x
    rocket_y_sun_km = m_y + r_y
    rocket_vx_sun_km_s = m_vx + r_vx
    rocket_vy_sun_km_s = m_vy + r_vy

    return {
        "t_seconds": t_sec,
        "rocket_rel_x_km": r_x,
        "rocket_rel_y_km": r_y,
        "rocket_angle_deg": rocket_angle_deg,
        "rocket_rel_vx_km_s": r_vx,
        "rocket_rel_vy_km_s": r_vy,
        "rocket_mass_kg": rmass,
        "fuel_mass_kg": rfuel,
        "rocket_x_km": rocket_x_sun_km,
        "rocket_y_km": rocket_y_sun_km,
        "rocket_vx_km_s": rocket_vx_sun_km_s,
        "rocket_vy_km_s": rocket_vy_sun_km_s,
        "mars_x_km": m_x,
        "mars_y_km": m_y,
        "mars_vx_km_s": m_vx,
        "mars_vy_km_s": m_vy,
        "status": status,
        "final_state": final_state,
    }

def animate_mars_orbit_phase3(
    correction_fuel_mass_kg=PHASE3_INPUTS["correction_fuel_mass_kg"],
    thrust_newtons=PHASE3_INPUTS["thrust_newtons"],
    collision_conversion_burn_duration_minutes=PHASE3_INPUTS["collision_conversion_burn_duration_minutes"],
    requested_burn_duration_minutes=PHASE3_INPUTS["requested_burn_duration_minutes"],
    capture_start_radial_velocity_km_s=PHASE3_INPUTS["capture_start_radial_velocity_km_s"],
    dt_seconds=PHASE3_INPUTS["dt_seconds"],
    max_step_seconds=PHASE3_INPUTS["max_step_seconds"],
    total_time_days=PHASE3_INPUTS["total_time_days"],
    coast_time_days=0.0,
    playback_speed=PHASE3_INPUTS["playback_speed"],
    fps=PHASE3_INPUTS["fps"]
):
    simulation = simulate_mars_orbit_phase3(
        correction_fuel_mass_kg=correction_fuel_mass_kg,
        thrust_newtons=thrust_newtons,
        collision_conversion_burn_duration_minutes=collision_conversion_burn_duration_minutes,
        requested_burn_duration_minutes=requested_burn_duration_minutes,
        capture_start_radial_velocity_km_s=capture_start_radial_velocity_km_s,
        dt_seconds=dt_seconds,
        max_step_seconds=max_step_seconds,
        total_time_days=total_time_days,
        coast_time_days=coast_time_days
    )

    t_sec, rx, ry = simulation["t_seconds"], simulation["rocket_rel_x_km"], simulation["rocket_rel_y_km"]
    rvx, rvy = simulation["rocket_rel_vx_km_s"], simulation["rocket_rel_vy_km_s"]
    rmass, rfuel, status = simulation["rocket_mass_kg"], simulation["fuel_mass_kg"], simulation["status"]
    summary = simulation["final_state"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x relative to Mars (km)")
    ax.set_ylabel("y relative to Mars (km)")
    ax.set_title("Phase 3: Rocket Motion Near Mars")

    max_extent = max(np.max(np.abs(rx)), np.max(np.abs(ry)), MARS_RADIUS_KM)
    pad = 0.12 * max_extent + 300.0
    ax.set_xlim(-max_extent - pad, max_extent + pad)
    ax.set_ylim(-max_extent - pad, max_extent + pad)

    mars = plt.Circle((0.0, 0.0), MARS_RADIUS_KM, color="orangered", alpha=0.85)
    ax.add_patch(mars)

    path, = ax.plot([], [], lw=2, color="black", label="Rocket path")
    point, = ax.plot([], [], marker="o", color="black", label="Rocket")
    ax.plot([rx[0]], [ry[0]], marker="x", markersize=8, color="green", label="Start")
    ax.legend(loc="upper right")

    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    target_frames = max(1, int(np.ceil(900 / playback_speed)))
    step = max(1, len(t_sec) // target_frames)
    frame_indices = np.append(np.arange(0, len(t_sec), step), len(t_sec) - 1)
    frame_indices = np.unique(frame_indices)

    def init():
        path.set_data([], [])
        point.set_data([], [])
        info.set_text("")
        return path, point, info

    def update(f):
        i = frame_indices[f]
        path.set_data(rx[: i + 1], ry[: i + 1])
        point.set_data([rx[i]], [ry[i]])

        b_stat = "coasting"
        if summary["initial_burn_applied"] and not summary["closest_pass_burn_applied"]:
            b_stat = "collision-conversion strategy active"
        elif summary["initial_burn_applied"] and summary["closest_pass_burn_applied"]:
            b_stat = "two-burn strategy completed"
        elif summary["closest_pass_burn_applied"]:
            b_stat = "closest-pass capture burn completed"

        info.set_text(
            f"time = {format_elapsed_time(t_sec[i])}\n"
            f"speed = {np.hypot(rvx[i], rvy[i]):.4f} km/s\n"
            f"mass = {rmass[i]:.2f} kg\n"
            f"fuel = {rfuel[i]:.2f} kg\n"
            f"burn = {b_stat}\n"
            f"altitude = {np.hypot(rx[i], ry[i]) - MARS_RADIUS_KM:.1f} km"
            + (f"\nstatus = {status}" if i == len(t_sec) - 1 else "")
        )
        return path, point, info

    anim = FuncAnimation(
        fig, update, frames=len(frame_indices), init_func=init,
        interval=1000 / (fps * playback_speed), blit=True
    )
    plt.show()
    return anim, simulation["final_state"]

def print_final_state(final_state):
    print("\nFinal phase 3 state")
    print(f"Status: {final_state['status']}")
    print(f"Stable orbit detected: {final_state['stable_orbit_detected']}")
    print(f"First closest-pass distance (km): {final_state['first_closest_pass_distance_km']}")

if __name__ == "__main__":
    animation, final_state = animate_mars_orbit_phase3(**PHASE3_INPUTS)
    print_final_state(final_state)