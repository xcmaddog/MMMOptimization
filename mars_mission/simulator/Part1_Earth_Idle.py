import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time, TimeDelta


# Core constants for the phase-1 parking-orbit model.
EARTH_RADIUS_KM = 6378.0
EARTH_MU_KM = 398600.4418  # km^3 / s^2


"""
------------------------------------------------------------
Phase 1 Inputs

Edit PHASE1_INPUTS below to change the phase-1 run.
This block is intended to be the one main place where you update inputs.

Input details:
- launch_altitude_km:
  Rocket starting altitude above Earth's surface in km.
- initial_velocity_km_s:
  Rocket starting speed relative to Earth in km/s.
  Set this to None to use the circular-orbit speed automatically.
- launch_angle_deg:
  Starting angle around Earth in degrees.
  `90.0` places the rocket above Earth on the +y axis like the original model.
- simulation_start_time_utc:
  Astropy epoch used to look up Earth and Mars heliocentric states.
- dt_seconds:
  Output time step in seconds.
  This controls how often the simulation stores points and updates the plotted data.
  Default is 180 seconds, which is 3 minutes.
- max_step_seconds:
  Internal RK4 integration step size in seconds.
  Smaller values improve parking-orbit accuracy.
  This can be smaller than dt_seconds.
- total_time_days:
  Total phase-1 simulation duration in days.
- playback_speed:
  Animation playback speed multiplier.
  1.0 is normal speed, 2.0 is twice as fast, and 0.5 is half speed.
- fps:
  Animation frame rate target used by Matplotlib.
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


def gravity_accel_earth_centered(x_km, y_km, mu_km=EARTH_MU_KM):
    """Return Earth's gravitational acceleration in an Earth-centered frame."""
    r_km = np.hypot(x_km, y_km)
    if r_km == 0.0:
        return 0.0, 0.0

    ax_km_s2 = -mu_km * x_km / r_km**3
    ay_km_s2 = -mu_km * y_km / r_km**3
    return ax_km_s2, ay_km_s2


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
    """
    Return Sun-centered Earth and Mars state arrays from Astropy ephemerides.

    Astropy supplies barycentric states. Subtracting the Sun's barycentric
    state converts those into heliocentric states so we can still report Earth,
    Mars, and rocket states relative to the Sun at the end of phase 1.
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


def rocket_state_derivative_earth_centered(state, earth_mu_km=EARTH_MU_KM):
    """Return time derivatives for [x, y, vx, vy] in the Earth-centered frame."""
    x_km, y_km, vx_km_s, vy_km_s = state
    ax_km_s2, ay_km_s2 = gravity_accel_earth_centered(x_km, y_km, earth_mu_km)
    return np.array([vx_km_s, vy_km_s, ax_km_s2, ay_km_s2], dtype=float)


def rk4_step_earth_centered(state, step_seconds, earth_mu_km=EARTH_MU_KM):
    """Advance the rocket by one RK4 step in the Earth-centered frame."""
    k1 = rocket_state_derivative_earth_centered(state, earth_mu_km=earth_mu_km)
    k2 = rocket_state_derivative_earth_centered(state + 0.5 * step_seconds * k1, earth_mu_km=earth_mu_km)
    k3 = rocket_state_derivative_earth_centered(state + 0.5 * step_seconds * k2, earth_mu_km=earth_mu_km)
    k4 = rocket_state_derivative_earth_centered(state + step_seconds * k3, earth_mu_km=earth_mu_km)
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
    earth_x_km,
    earth_y_km,
    earth_vx_km_s,
    earth_vy_km_s,
    mars_x_km,
    mars_y_km,
    mars_vx_km_s,
    mars_vy_km_s,
    status,
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
    Simulate phase 1 as a fixed-Earth parking orbit.

    Modeling assumptions:
    - Earth stays fixed at the origin for the rocket dynamics.
    - The rocket feels Earth's gravity only.
    - Earth and Mars heliocentric states are still tracked with Astropy.
    - Rocket heliocentric output is reconstructed by adding the Earth-centered
      rocket state to Earth's heliocentric state at the same time.
    """
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

    # Track Mars relative to Earth for phase-1 reference, even though Mars is
    # not part of the local parking-orbit dynamics.
    mars_rel_x_km = mars_x_km - earth_x_km
    mars_rel_y_km = mars_y_km - earth_y_km
    mars_rel_vx_km_s = mars_vx_km_s - earth_vx_km_s
    mars_rel_vy_km_s = mars_vy_km_s - earth_vy_km_s

    rocket_rel_x_km = np.zeros(n_output_steps)
    rocket_rel_y_km = np.zeros(n_output_steps)
    rocket_rel_vx_km_s = np.zeros(n_output_steps)
    rocket_rel_vy_km_s = np.zeros(n_output_steps)

    launch_radius_km = earth_radius_km + launch_altitude_km
    launch_angle_rad = np.deg2rad(launch_angle_deg)

    if initial_velocity_km_s is None:
        initial_velocity_km_s = circular_speed(launch_radius_km, earth_mu_km)

    radial_x = np.cos(launch_angle_rad)
    radial_y = np.sin(launch_angle_rad)
    tangent_x = -np.sin(launch_angle_rad)
    tangent_y = np.cos(launch_angle_rad)

    rocket_rel_x_km[0] = launch_radius_km * radial_x
    rocket_rel_y_km[0] = launch_radius_km * radial_y
    rocket_rel_vx_km_s[0] = initial_velocity_km_s * tangent_x
    rocket_rel_vy_km_s[0] = initial_velocity_km_s * tangent_y

    status = "in parking orbit"
    stop_index = n_output_steps

    for i in range(n_output_steps - 1):
        interval_seconds = t_seconds[i + 1] - t_seconds[i]
        substeps = max(1, int(np.ceil(interval_seconds / max_step_seconds)))
        substep_seconds = interval_seconds / substeps

        state = np.array(
            [
                rocket_rel_x_km[i],
                rocket_rel_y_km[i],
                rocket_rel_vx_km_s[i],
                rocket_rel_vy_km_s[i],
            ],
            dtype=float,
        )

        interval_completed = True

        for substep_index in range(substeps):
            rocket_radius_km = np.hypot(state[0], state[1])

            if rocket_radius_km <= earth_radius_km:
                status = "impacted Earth"
                collision_time_seconds = t_seconds[i] + (substep_index / substeps) * interval_seconds
                t_seconds[i + 1] = collision_time_seconds
                rocket_rel_x_km[i + 1] = state[0]
                rocket_rel_y_km[i + 1] = state[1]
                rocket_rel_vx_km_s[i + 1] = state[2]
                rocket_rel_vy_km_s[i + 1] = state[3]
                stop_index = i + 2
                interval_completed = False
                break

            state = rk4_step_earth_centered(
                state,
                step_seconds=substep_seconds,
                earth_mu_km=earth_mu_km,
            )

        if not interval_completed:
            break

        rocket_rel_x_km[i + 1] = state[0]
        rocket_rel_y_km[i + 1] = state[1]
        rocket_rel_vx_km_s[i + 1] = state[2]
        rocket_rel_vy_km_s[i + 1] = state[3]

    rocket_rel_x_km = rocket_rel_x_km[:stop_index]
    rocket_rel_y_km = rocket_rel_y_km[:stop_index]
    rocket_rel_vx_km_s = rocket_rel_vx_km_s[:stop_index]
    rocket_rel_vy_km_s = rocket_rel_vy_km_s[:stop_index]
    rocket_angle_deg = angular_position_deg(rocket_rel_x_km, rocket_rel_y_km)
    earth_x_km = earth_x_km[:stop_index]
    earth_y_km = earth_y_km[:stop_index]
    earth_vx_km_s = earth_vx_km_s[:stop_index]
    earth_vy_km_s = earth_vy_km_s[:stop_index]
    mars_x_km = mars_x_km[:stop_index]
    mars_y_km = mars_y_km[:stop_index]
    mars_vx_km_s = mars_vx_km_s[:stop_index]
    mars_vy_km_s = mars_vy_km_s[:stop_index]
    mars_rel_x_km = mars_rel_x_km[:stop_index]
    mars_rel_y_km = mars_rel_y_km[:stop_index]
    mars_rel_vx_km_s = mars_rel_vx_km_s[:stop_index]
    mars_rel_vy_km_s = mars_rel_vy_km_s[:stop_index]
    t_seconds = t_seconds[:stop_index]

    if status == "in parking orbit":
        status = "completed full simulation"

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
    """
    Animate phase 1 using a fixed-Earth parking-orbit view.

    Earth stays fixed on screen. Mars is tracked internally and reported in the
    returned data, but it is not plotted in this phase.
    """
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
