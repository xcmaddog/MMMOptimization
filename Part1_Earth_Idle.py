import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time, TimeDelta

# Core constants for the phase-1 parking-orbit model.
EARTH_RADIUS_KM = 6378.0
EARTH_MU_KM = 398600.4418  # km^3 / s^2

"""
------------------------------------------------------------
Phase 1 Inputs
------------------------------------------------------------
"""
PHASE1_INPUTS = {
    "launch_altitude_km": 200.0,
    "initial_velocity_km_s": None,
    "launch_angle_deg": 90.0,
    "simulation_start_time_utc": "2020-08-30T00:00:00",
    "dt_seconds": 180.0,
    "max_step_seconds": 60.0,
    "total_time_days": 0.5,
    "playback_speed": 0.5,
    "fps": 30,
}

# ---------------------------------------------------------
# NUMBA ACCELERATED PHYSICS CORE
# ---------------------------------------------------------

@njit
def _gravity_accel_jit(x, y, mu):
    """Blazing fast gravity calculation."""
    r_sq = x*x + y*y
    r = np.sqrt(r_sq)
    if r == 0: return 0.0, 0.0
    factor = -mu / (r_sq * r)
    return x * factor, y * factor

@njit
def _rk4_step_jit(state, dt, mu):
    """RK4 step compiled to machine code."""
    x1, y1, vx1, vy1 = state[0], state[1], state[2], state[3]
    ax1, ay1 = _gravity_accel_jit(x1, y1, mu)
    
    x2 = x1 + 0.5 * dt * vx1
    y2 = y1 + 0.5 * dt * vy1
    vx2 = vx1 + 0.5 * dt * ax1
    vy2 = vy1 + 0.5 * dt * ay1
    ax2, ay2 = _gravity_accel_jit(x2, y2, mu)
    
    x3 = x1 + 0.5 * dt * vx2
    y3 = y1 + 0.5 * dt * vy2
    vx3 = vx1 + 0.5 * dt * ax2
    vy3 = vy1 + 0.5 * dt * ay2
    ax3, ay3 = _gravity_accel_jit(x3, y3, mu)
    
    x4 = x1 + dt * vx3
    y4 = y1 + dt * vy3
    vx4 = vx1 + dt * ax3
    vy4 = vy1 + dt * ay3
    ax4, ay4 = _gravity_accel_jit(x4, y4, mu)
    
    next_state = np.empty(4)
    next_state[0] = x1 + (dt/6.0) * (vx1 + 2*vx2 + 2*vx3 + vx4)
    next_state[1] = y1 + (dt/6.0) * (vy1 + 2*vy2 + 2*vy3 + vy4)
    next_state[2] = vx1 + (dt/6.0) * (ax1 + 2*ax2 + 2*ax3 + ax4)
    next_state[3] = vy1 + (dt/6.0) * (ay1 + 2*ay2 + 2*ay3 + ay4)
    return next_state

@njit
def _propagate_jit(start_state, t_array, max_step, mu, earth_radius):
    """
    Propagates the entire orbit in one compiled operation.
    Returns: (states_array, collision_index, status_code)
    """
    n_steps = len(t_array)
    out = np.zeros((n_steps, 4))
    out[0] = start_state
    
    current_state = start_state.copy()
    
    for i in range(n_steps - 1):
        dt_interval = t_array[i+1] - t_array[i]
        substeps = int(np.ceil(dt_interval / max_step))
        if substeps < 1:
            substeps = 1
        sub_dt = dt_interval / substeps
        
        for _ in range(substeps):
            r = np.sqrt(current_state[0]**2 + current_state[1]**2)
            if r <= earth_radius:
                return out, i + 1, 1
            current_state = _rk4_step_jit(current_state, sub_dt, mu)
        
        out[i+1] = current_state
        
    return out, n_steps, 0

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def circular_speed(radius_from_center_km, mu_km=EARTH_MU_KM):
    """Return the circular-orbit speed for the given orbital radius."""
    return np.sqrt(mu_km / radius_from_center_km)

def format_elapsed_time(seconds):
    """Convert seconds into a days / hours / minutes label."""
    total_seconds = int(round(seconds))
    days = total_seconds // 86400
    remainder = total_seconds % 86400
    hours = remainder // 3600
    remainder %= 3600
    minutes = remainder // 60
    return f"{days} d  {hours:02d} h  {minutes:02d} m"

def planetary_states_heliocentric(t_seconds_array, simulation_start_time_utc):
    """Astropy Fallback for Helocentric coordinates."""
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

def angular_position_deg(x_km, y_km):
    """Return angular position in degrees measured from the +x axis."""
    return np.mod(np.degrees(np.arctan2(y_km, x_km)), 360.0)

def build_final_state(
    elapsed_time_seconds, simulation_start_time_utc, end_datetime_utc,
    rocket_rel_x_km, rocket_rel_y_km, rocket_angle_deg,
    rocket_rel_vx_km_s, rocket_rel_vy_km_s,
    earth_x_km, earth_y_km, earth_vx_km_s, earth_vy_km_s,
    mars_x_km, mars_y_km, mars_vx_km_s, mars_vy_km_s, status,
):
    """Package the requested phase-1 end state in Sun-relative coordinates."""
    rocket_x_sun_km = earth_x_km + rocket_rel_x_km
    rocket_y_sun_km = earth_y_km + rocket_rel_y_km
    rocket_vx_sun_km_s = earth_vx_km_s + rocket_rel_vx_km_s
    rocket_vy_sun_km_s = earth_vy_km_s + rocket_rel_vy_km_s

    return {
        "simulation_start_time_utc": simulation_start_time_utc,
        "end_datetime_utc": end_datetime_utc,
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "elapsed_time_pretty": format_elapsed_time(elapsed_time_seconds),
        "rocket_position_km": {"x": float(rocket_x_sun_km), "y": float(rocket_y_sun_km)},
        "rocket_velocity_km_s": {"vx": float(rocket_vx_sun_km_s), "vy": float(rocket_vy_sun_km_s)},
        "earth_position_km": {"x": float(earth_x_km), "y": float(earth_y_km)},
        "earth_velocity_km_s": {"vx": float(earth_vx_km_s), "vy": float(earth_vy_km_s)},
        "mars_position_km": {"x": float(mars_x_km), "y": float(mars_y_km)},
        "mars_velocity_km_s": {"vx": float(mars_vx_km_s), "vy": float(mars_vy_km_s)},
        "rocket_relative_to_earth_km": {"x": float(rocket_rel_x_km), "y": float(rocket_rel_y_km)},
        "rocket_angle_deg": float(rocket_angle_deg),
        "rocket_relative_to_earth_velocity_km_s": {
            "vx": float(rocket_rel_vx_km_s),
            "vy": float(rocket_rel_vy_km_s),
        },
        "status": status,
    }

# ---------------------------------------------------------
# SIMULATION LOGIC
# ---------------------------------------------------------

def simulate_parking_orbit_phase1(
    launch_altitude_km=PHASE1_INPUTS["launch_altitude_km"],
    initial_velocity_km_s=PHASE1_INPUTS["initial_velocity_km_s"],
    launch_angle_deg=PHASE1_INPUTS["launch_angle_deg"],
    simulation_start_time_utc=PHASE1_INPUTS["simulation_start_time_utc"],
    dt_seconds=PHASE1_INPUTS["dt_seconds"],
    max_step_seconds=PHASE1_INPUTS["max_step_seconds"],
    total_time_days=PHASE1_INPUTS["total_time_days"],
    earth_radius_km=EARTH_RADIUS_KM,
    earth_mu_km=EARTH_MU_KM,
):
    """
    Simulate phase 1 using high-speed Numba JIT integration and pre-cached ephemeris.
    """
    total_time_seconds = total_time_days * 86400.0
    n_output_steps = int(total_time_seconds / dt_seconds) + 1
    t_seconds = np.linspace(0.0, total_time_seconds, n_output_steps)

    # 1. Setup Initial State
    launch_radius_km = earth_radius_km + launch_altitude_km
    launch_angle_rad = np.deg2rad(launch_angle_deg)

    if initial_velocity_km_s is None:
        initial_velocity_km_s = circular_speed(launch_radius_km, earth_mu_km)

    start_state = np.array([
        launch_radius_km * np.cos(launch_angle_rad),
        launch_radius_km * np.sin(launch_angle_rad),
        -initial_velocity_km_s * np.sin(launch_angle_rad),
        initial_velocity_km_s * np.cos(launch_angle_rad)
    ], dtype=float)

    # 2. Fast Propagation Loop
    states, stop_idx, status_code = _propagate_jit(
        start_state, t_seconds, max_step_seconds, earth_mu_km, earth_radius_km
    )

    t_seconds = t_seconds[:stop_idx]
    states = states[:stop_idx]
    status = "impacted Earth" if status_code == 1 else "completed full simulation"

    rocket_rel_x_km = states[:, 0]
    rocket_rel_y_km = states[:, 1]
    rocket_rel_vx_km_s = states[:, 2]
    rocket_rel_vy_km_s = states[:, 3]

    # 3. High-Speed Planetary States Lookup
    try:
        from Porkchop_Searcher import load_interpolators
        e_r_f, e_v_f, m_r_f, m_v_f = load_interpolators("ephemeris_cache.npz")
        
        start_jd = Time(simulation_start_time_utc, scale="utc").jd
        query_jds = start_jd + (t_seconds / 86400.0)
        
        earth_pos = e_r_f(query_jds)
        earth_vel = e_v_f(query_jds)
        mars_pos = m_r_f(query_jds)
        mars_vel = m_v_f(query_jds)
        
        earth_x_km, earth_y_km = earth_pos[:, 0], earth_pos[:, 1]
        earth_vx_km_s, earth_vy_km_s = earth_vel[:, 0], earth_vel[:, 1]
        mars_x_km, mars_y_km = mars_pos[:, 0], mars_pos[:, 1]
        mars_vx_km_s, mars_vy_km_s = mars_vel[:, 0], mars_vel[:, 1]
        
    except Exception as e:
        print(f"Warning: Falling back to slow Astropy. Error: {e}")
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

    # 4. Final Reconstructions
    mars_rel_x_km = mars_x_km - earth_x_km
    mars_rel_y_km = mars_y_km - earth_y_km
    mars_rel_vx_km_s = mars_vx_km_s - earth_vx_km_s
    mars_rel_vy_km_s = mars_vy_km_s - earth_vy_km_s

    rocket_angle_deg = angular_position_deg(rocket_rel_x_km, rocket_rel_y_km)

    end_datetime_utc = (
        Time(simulation_start_time_utc, scale="utc") + TimeDelta(t_seconds[-1], format="sec")
    ).utc.isot

    rocket_x_sun_km = earth_x_km + rocket_rel_x_km
    rocket_y_sun_km = earth_y_km + rocket_rel_y_km
    rocket_vx_sun_km_s = earth_vx_km_s + rocket_rel_vx_km_s
    rocket_vy_sun_km_s = earth_vy_km_s + rocket_rel_vy_km_s

    final_state = build_final_state(
        elapsed_time_seconds=t_seconds[-1],
        simulation_start_time_utc=simulation_start_time_utc,
        end_datetime_utc=end_datetime_utc,
        rocket_rel_x_km=rocket_rel_x_km[-1],
        rocket_rel_y_km=rocket_rel_y_km[-1],
        rocket_angle_deg=rocket_angle_deg[-1],
        rocket_rel_vx_km_s=rocket_rel_vx_km_s[-1],
        rocket_rel_vy_km_s=rocket_rel_vy_km_s[-1],
        earth_x_km=earth_x_km[-1],
        earth_y_km=earth_y_km[-1],
        earth_vx_km_s=earth_vx_km_s[-1],
        earth_vy_km_s=earth_vy_km_s[-1],
        mars_x_km=mars_x_km[-1],
        mars_y_km=mars_y_km[-1],
        mars_vx_km_s=mars_vx_km_s[-1],
        mars_vy_km_s=mars_vy_km_s[-1],
        status=status,
    )

    return {
        "t_seconds": t_seconds,
        "rocket_rel_x_km": rocket_rel_x_km,
        "rocket_rel_y_km": rocket_rel_y_km,
        "rocket_angle_deg": rocket_angle_deg,
        "rocket_rel_vx_km_s": rocket_rel_vx_km_s,
        "rocket_rel_vy_km_s": rocket_rel_vy_km_s,
        "rocket_x_km": rocket_x_sun_km,
        "rocket_y_km": rocket_y_sun_km,
        "rocket_vx_km_s": rocket_vx_sun_km_s,
        "rocket_vy_km_s": rocket_vy_sun_km_s,
        "earth_x_km": earth_x_km,
        "earth_y_km": earth_y_km,
        "earth_vx_km_s": earth_vx_km_s,
        "earth_vy_km_s": earth_vy_km_s,
        "mars_x_km": mars_x_km,
        "mars_y_km": mars_y_km,
        "mars_vx_km_s": mars_vx_km_s,
        "mars_vy_km_s": mars_vy_km_s,
        "mars_rel_x_km": mars_rel_x_km,
        "mars_rel_y_km": mars_rel_y_km,
        "mars_rel_vx_km_s": mars_rel_vx_km_s,
        "mars_rel_vy_km_s": mars_rel_vy_km_s,
        "status": status,
        "final_state": final_state,
    }

def animate_parking_orbit_phase1(
    launch_altitude_km=PHASE1_INPUTS["launch_altitude_km"],
    initial_velocity_km_s=PHASE1_INPUTS["initial_velocity_km_s"],
    launch_angle_deg=PHASE1_INPUTS["launch_angle_deg"],
    simulation_start_time_utc=PHASE1_INPUTS["simulation_start_time_utc"],
    dt_seconds=PHASE1_INPUTS["dt_seconds"],
    max_step_seconds=PHASE1_INPUTS["max_step_seconds"],
    total_time_days=PHASE1_INPUTS["total_time_days"],
    playback_speed=PHASE1_INPUTS["playback_speed"],
    fps=PHASE1_INPUTS["fps"],
):
    """Animate phase 1 using a fixed-Earth parking-orbit view."""
    simulation = simulate_parking_orbit_phase1(
        launch_altitude_km=launch_altitude_km,
        initial_velocity_km_s=initial_velocity_km_s,
        launch_angle_deg=launch_angle_deg,
        simulation_start_time_utc=simulation_start_time_utc,
        dt_seconds=dt_seconds,
        max_step_seconds=max_step_seconds,
        total_time_days=total_time_days,
    )

    t_seconds = simulation["t_seconds"]
    rocket_rel_x_km = simulation["rocket_rel_x_km"]
    rocket_rel_y_km = simulation["rocket_rel_y_km"]
    rocket_rel_vx_km_s = simulation["rocket_rel_vx_km_s"]
    rocket_rel_vy_km_s = simulation["rocket_rel_vy_km_s"]
    mars_rel_x_km = simulation["mars_rel_x_km"]
    mars_rel_y_km = simulation["mars_rel_y_km"]
    status = simulation["status"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x relative to Earth (km)")
    ax.set_ylabel("y relative to Earth (km)")
    ax.set_title("Phase 1: Parking Orbit Around Earth")

    max_extent_km = max(
        np.max(np.abs(rocket_rel_x_km)),
        np.max(np.abs(rocket_rel_y_km)),
        EARTH_RADIUS_KM + launch_altitude_km,
    )
    pad_km = 0.12 * max_extent_km + 300.0
    ax.set_xlim(-max_extent_km - pad_km, max_extent_km + pad_km)
    ax.set_ylim(-max_extent_km - pad_km, max_extent_km + pad_km)

    earth = plt.Circle((0.0, 0.0), EARTH_RADIUS_KM, color="royalblue", alpha=0.85)
    ax.add_patch(earth)

    rocket_path_line, = ax.plot([], [], lw=2, color="black", label="Rocket path")
    rocket_point, = ax.plot([], [], marker="o", color="black", label="Rocket")
    ax.plot([rocket_rel_x_km[0]], [rocket_rel_y_km[0]], marker="x", markersize=8, color="green", label="Start")
    ax.legend(loc="upper right")

    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
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
        altitude_now_km = np.hypot(rocket_rel_x_km[i], rocket_rel_y_km[i]) - EARTH_RADIUS_KM
        rocket_to_mars_km = np.hypot(rocket_rel_x_km[i] - mars_rel_x_km[i], rocket_rel_y_km[i] - mars_rel_y_km[i])

        info.set_text(
            f"time = {format_elapsed_time(t_seconds[i])}\n"
            f"rocket x = {rocket_rel_x_km[i]:.1f} km\n"
            f"rocket y = {rocket_rel_y_km[i]:.1f} km\n"
            f"rocket vx = {rocket_rel_vx_km_s[i]:.4f} km/s\n"
            f"rocket vy = {rocket_rel_vy_km_s[i]:.4f} km/s\n"
            f"altitude = {altitude_now_km:.1f} km\n"
            f"speed = {speed_km_s:.4f} km/s\n"
            f"distance to Mars = {rocket_to_mars_km:.1f} km"
            + (f"\nstatus = {status}" if i == len(t_seconds) - 1 else "")
        )
        return rocket_path_line, rocket_point, info

    anim = FuncAnimation(
        fig, update, frames=len(frame_indices), init_func=init,
        interval=1000 / (fps * playback_speed), blit=True,
    )

    plt.show()
    return anim, simulation["final_state"]

def print_final_state(final_state):
    """Print the final Sun-relative states needed to hand off into phase 2."""
    print("Final simulation state")
    print("----------------------")
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
    print(f"Rocket final angle around Earth: {final_state['rocket_angle_deg']:.3f} deg")
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

if __name__ == "__main__":
    animation, final_state = animate_parking_orbit_phase1(**PHASE1_INPUTS)
    print_final_state(final_state)