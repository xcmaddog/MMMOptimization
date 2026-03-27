import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Core planetary constants.
EARTH_RADIUS_KM = 6378.0
EARTH_MU_KM = 398600.4418  # km^3 / s^2
MARS_RADIUS_KM = 3389.5

# Circular heliocentric orbit values used to prescribe Mars' relative motion
# while Earth is held fixed at the origin.
AU_KM = 149597870.7
EARTH_ORBIT_RADIUS_KM = AU_KM
MARS_ORBIT_RADIUS_KM = 1.523679 * AU_KM
MARS_ORBIT_PERIOD_SECONDS = 686.98 * 86400.0


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


def mars_state_relative_to_stationary_earth(
    t_seconds,
    mars_phase_deg=0.0,
    earth_orbit_radius_km=EARTH_ORBIT_RADIUS_KM,
    mars_orbit_radius_km=MARS_ORBIT_RADIUS_KM,
    mars_orbit_period_seconds=MARS_ORBIT_PERIOD_SECONDS,
):
    """
    Return Mars' position and velocity in an Earth-centered frame.

    Earth is held fixed at the origin. Mars follows a circular heliocentric
    orbit translated into this Earth-centered frame. This gives us the relative
    Mars arc needed for a simplified transfer visualization.
    """
    theta_rad = np.deg2rad(mars_phase_deg) + (2.0 * np.pi / mars_orbit_period_seconds) * t_seconds
    mars_omega_rad_s = 2.0 * np.pi / mars_orbit_period_seconds

    mars_x_km = mars_orbit_radius_km * np.cos(theta_rad) - earth_orbit_radius_km
    mars_y_km = mars_orbit_radius_km * np.sin(theta_rad)
    mars_vx_km_s = -mars_orbit_radius_km * mars_omega_rad_s * np.sin(theta_rad)
    mars_vy_km_s = mars_orbit_radius_km * mars_omega_rad_s * np.cos(theta_rad)
    return mars_x_km, mars_y_km, mars_vx_km_s, mars_vy_km_s


def rocket_accel_from_earth_only(rocket_x_km, rocket_y_km, earth_mu_km=EARTH_MU_KM):
    """
    Return the rocket acceleration caused by Earth only.

    Mars is tracked as a moving target, but its gravity is intentionally ignored
    here because its effect is negligible for this simplified model.
    """
    return gravity_accel_from_body(
        rocket_x_km,
        rocket_y_km,
        0.0,
        0.0,
        earth_mu_km,
    )


def build_final_state(
    elapsed_time_seconds,
    rocket_x_km,
    rocket_y_km,
    rocket_vx_km_s,
    rocket_vy_km_s,
    mars_x_km,
    mars_y_km,
    mars_vx_km_s,
    mars_vy_km_s,
    status,
):
    """Package the requested end-of-simulation values into one dictionary."""
    return {
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "elapsed_time_pretty": format_elapsed_time(elapsed_time_seconds),
        "rocket_position_km": {"x": float(rocket_x_km), "y": float(rocket_y_km)},
        "rocket_velocity_km_s": {"vx": float(rocket_vx_km_s), "vy": float(rocket_vy_km_s)},
        "mars_position_km": {"x": float(mars_x_km), "y": float(mars_y_km)},
        "mars_velocity_km_s": {"vx": float(mars_vx_km_s), "vy": float(mars_vy_km_s)},
        "status": status,
    }


def simulate_transfer_earth_to_mars(
    launch_altitude_km=200.0,
    initial_velocity_km_s=None,
    launch_angle_deg=0.0,
    mars_phase_deg=0.0,
    dt_seconds=3600.0,
    total_time_days=300.0,
    earth_radius_km=EARTH_RADIUS_KM,
    earth_mu_km=EARTH_MU_KM,
    mars_radius_km=MARS_RADIUS_KM,
):
    """
    Simulate a 2D Earth-to-Mars transfer in an Earth-centered frame.

    Modeling assumptions:
    - Earth stays fixed at the origin.
    - Mars moves along a prescribed relative orbital arc.
    - The rocket feels only Earth's gravity.
    - The rocket starts tangent to Earth at the selected launch angle.
    """
    total_time_seconds = total_time_days * 86400.0
    n_steps = int(total_time_seconds / dt_seconds) + 1
    t_seconds = np.linspace(0.0, total_time_seconds, n_steps)

    rocket_x_km = np.zeros(n_steps)
    rocket_y_km = np.zeros(n_steps)
    rocket_vx_km_s = np.zeros(n_steps)
    rocket_vy_km_s = np.zeros(n_steps)
    mars_x_km = np.zeros(n_steps)
    mars_y_km = np.zeros(n_steps)
    mars_vx_km_s = np.zeros(n_steps)
    mars_vy_km_s = np.zeros(n_steps)

    launch_radius_km = earth_radius_km + launch_altitude_km
    launch_angle_rad = np.deg2rad(launch_angle_deg)

    # Default to the circular-orbit speed at the launch radius if the user does
    # not enter a custom launch speed.
    if initial_velocity_km_s is None:
        initial_velocity_km_s = circular_speed(launch_radius_km, earth_mu_km)

    # A tangential launch direction keeps the old "one speed value" behavior.
    tangent_x = -np.sin(launch_angle_rad)
    tangent_y = np.cos(launch_angle_rad)

    rocket_x_km[0] = launch_radius_km * np.cos(launch_angle_rad)
    rocket_y_km[0] = launch_radius_km * np.sin(launch_angle_rad)
    rocket_vx_km_s[0] = initial_velocity_km_s * tangent_x
    rocket_vy_km_s[0] = initial_velocity_km_s * tangent_y

    mars_x_km[0], mars_y_km[0], mars_vx_km_s[0], mars_vy_km_s[0] = mars_state_relative_to_stationary_earth(
        t_seconds=0.0,
        mars_phase_deg=mars_phase_deg,
    )

    status = "in flight"
    stop_index = n_steps

    for i in range(n_steps - 1):
        earth_distance_km = np.hypot(rocket_x_km[i], rocket_y_km[i])
        rocket_to_mars_km = np.hypot(rocket_x_km[i] - mars_x_km[i], rocket_y_km[i] - mars_y_km[i])

        if earth_distance_km <= earth_radius_km:
            status = "impacted Earth"
            stop_index = i + 1
            break

        if rocket_to_mars_km <= mars_radius_km:
            status = "impacted Mars"
            stop_index = i + 1
            break

        ax_km_s2, ay_km_s2 = rocket_accel_from_earth_only(
            rocket_x_km=rocket_x_km[i],
            rocket_y_km=rocket_y_km[i],
            earth_mu_km=earth_mu_km,
        )

        rocket_x_km[i + 1] = rocket_x_km[i] + rocket_vx_km_s[i] * dt_seconds + 0.5 * ax_km_s2 * dt_seconds**2
        rocket_y_km[i + 1] = rocket_y_km[i] + rocket_vy_km_s[i] * dt_seconds + 0.5 * ay_km_s2 * dt_seconds**2

        mars_x_km[i + 1], mars_y_km[i + 1], mars_vx_km_s[i + 1], mars_vy_km_s[i + 1] = mars_state_relative_to_stationary_earth(
            t_seconds=t_seconds[i + 1],
            mars_phase_deg=mars_phase_deg,
        )

        ax_new_km_s2, ay_new_km_s2 = rocket_accel_from_earth_only(
            rocket_x_km=rocket_x_km[i + 1],
            rocket_y_km=rocket_y_km[i + 1],
            earth_mu_km=earth_mu_km,
        )

        rocket_vx_km_s[i + 1] = rocket_vx_km_s[i] + 0.5 * (ax_km_s2 + ax_new_km_s2) * dt_seconds
        rocket_vy_km_s[i + 1] = rocket_vy_km_s[i] + 0.5 * (ay_km_s2 + ay_new_km_s2) * dt_seconds

    rocket_x_km = rocket_x_km[:stop_index]
    rocket_y_km = rocket_y_km[:stop_index]
    rocket_vx_km_s = rocket_vx_km_s[:stop_index]
    rocket_vy_km_s = rocket_vy_km_s[:stop_index]
    mars_x_km = mars_x_km[:stop_index]
    mars_y_km = mars_y_km[:stop_index]
    mars_vx_km_s = mars_vx_km_s[:stop_index]
    mars_vy_km_s = mars_vy_km_s[:stop_index]
    t_seconds = t_seconds[:stop_index]

    if status == "in flight":
        status = "completed full simulation"

    final_state = build_final_state(
        elapsed_time_seconds=t_seconds[-1],
        rocket_x_km=rocket_x_km[-1],
        rocket_y_km=rocket_y_km[-1],
        rocket_vx_km_s=rocket_vx_km_s[-1],
        rocket_vy_km_s=rocket_vy_km_s[-1],
        mars_x_km=mars_x_km[-1],
        mars_y_km=mars_y_km[-1],
        mars_vx_km_s=mars_vx_km_s[-1],
        mars_vy_km_s=mars_vy_km_s[-1],
        status=status,
    )

    return {
        "t_seconds": t_seconds,
        "rocket_x_km": rocket_x_km,
        "rocket_y_km": rocket_y_km,
        "rocket_vx_km_s": rocket_vx_km_s,
        "rocket_vy_km_s": rocket_vy_km_s,
        "mars_x_km": mars_x_km,
        "mars_y_km": mars_y_km,
        "mars_vx_km_s": mars_vx_km_s,
        "mars_vy_km_s": mars_vy_km_s,
        "status": status,
        "final_state": final_state,
    }


def animate_transfer_earth_to_mars(
    launch_altitude_km=200.0,
    initial_velocity_km_s=None,
    launch_angle_deg=0.0,
    mars_phase_deg=0.0,
    dt_seconds=3600.0,
    total_time_days=300.0,
    playback_speed=1.0,
    fps=30,
):
    """
    Animate the phase-1 transfer simulation and return the final state.

    In phase 1 we only plot the rocket near Earth. Mars is still tracked
    internally for mission-state reporting, but it is not drawn on screen.
    """
    simulation = simulate_transfer_earth_to_mars(
        launch_altitude_km=launch_altitude_km,
        initial_velocity_km_s=initial_velocity_km_s,
        launch_angle_deg=launch_angle_deg,
        mars_phase_deg=mars_phase_deg,
        dt_seconds=dt_seconds,
        total_time_days=total_time_days,
    )

    t_seconds = simulation["t_seconds"]
    rocket_x_km = simulation["rocket_x_km"]
    rocket_y_km = simulation["rocket_y_km"]
    rocket_vx_km_s = simulation["rocket_vx_km_s"]
    rocket_vy_km_s = simulation["rocket_vy_km_s"]
    mars_vx_km_s = simulation["mars_vx_km_s"]
    mars_vy_km_s = simulation["mars_vy_km_s"]
    mars_x_km = simulation["mars_x_km"]
    mars_y_km = simulation["mars_y_km"]
    status = simulation["status"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Phase 1: Rocket Motion Near Earth")

    # Keep the phase-1 view centered on Earth and scaled to the rocket motion only.
    max_extent_km = max(
        np.max(np.abs(rocket_x_km)),
        np.max(np.abs(rocket_y_km)),
        EARTH_RADIUS_KM + launch_altitude_km,
    )
    pad_km = 0.12 * max_extent_km + 300.0
    ax.set_xlim(-max_extent_km - pad_km, max_extent_km + pad_km)
    ax.set_ylim(-max_extent_km - pad_km, max_extent_km + pad_km)

    earth = plt.Circle((0.0, 0.0), EARTH_RADIUS_KM, color="royalblue", alpha=0.85)
    ax.add_patch(earth)

    rocket_path_line, = ax.plot([], [], lw=2, color="black", label="Rocket path")
    rocket_point, = ax.plot([], [], marker="o", color="black", label="Rocket")
    ax.plot([rocket_x_km[0]], [rocket_y_km[0]], marker="x", markersize=8, color="green", label="Launch")
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
        rocket_path_line.set_data(rocket_x_km[: i + 1], rocket_y_km[: i + 1])
        rocket_point.set_data([rocket_x_km[i]], [rocket_y_km[i]])

        rocket_to_mars_km = np.hypot(rocket_x_km[i] - mars_x_km[i], rocket_y_km[i] - mars_y_km[i])
        info.set_text(
            f"time = {format_elapsed_time(t_seconds[i])}\n"
            f"rocket x = {rocket_x_km[i]:.1f} km\n"
            f"rocket y = {rocket_y_km[i]:.1f} km\n"
            f"rocket vx = {rocket_vx_km_s[i]:.4f} km/s\n"
            f"rocket vy = {rocket_vy_km_s[i]:.4f} km/s\n"
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
    """Print the requested end-of-simulation state values."""
    print("Final simulation state")
    print("----------------------")
    print(f"Status: {final_state['status']}")
    print(f"Elapsed time: {final_state['elapsed_time_pretty']} ({final_state['elapsed_time_seconds']:.1f} s)")
    print(
        f"Rocket position (km): x = {final_state['rocket_position_km']['x']:.3f}, "
        f"y = {final_state['rocket_position_km']['y']:.3f}"
    )
    print(
        f"Rocket velocity (km/s): vx = {final_state['rocket_velocity_km_s']['vx']:.6f}, "
        f"vy = {final_state['rocket_velocity_km_s']['vy']:.6f}"
    )
    print(
        f"Mars position (km): x = {final_state['mars_position_km']['x']:.3f}, "
        f"y = {final_state['mars_position_km']['y']:.3f}"
    )
    print(
        f"Mars velocity (km/s): vx = {final_state['mars_velocity_km_s']['vx']:.6f}, "
        f"vy = {final_state['mars_velocity_km_s']['vy']:.6f}"
    )


"""
------------------------------------------------------------
User-editable inputs:
- launch_altitude_km is the starting height above Earth's surface.
- initial_velocity_km_s is the rocket launch speed.
  Set it to None to use the circular-orbit speed automatically.
- launch_angle_deg chooses where around Earth the rocket starts.
  The code launches tangentially at that point.
- mars_phase_deg chooses where Mars begins along its relative orbit arc.
  Mars is tracked in phase 1 but not plotted yet.
- dt_seconds controls the integration step size.
- total_time_days is the full simulated mission length.
- playback_speed affects only the animation playback.
------------------------------------------------------------
"""
launch_altitude_km = 200.0
initial_velocity_km_s = None
launch_angle_deg = 0.0
mars_phase_deg = 0.0
dt_seconds = 60.0
total_time_days = 365*5
playback_speed = 5


animation, final_state = animate_transfer_earth_to_mars(
    launch_altitude_km=launch_altitude_km,
    initial_velocity_km_s=initial_velocity_km_s,
    launch_angle_deg=launch_angle_deg,
    mars_phase_deg=mars_phase_deg,
    dt_seconds=dt_seconds,
    total_time_days=total_time_days,
    playback_speed=playback_speed,
)

print_final_state(final_state)
