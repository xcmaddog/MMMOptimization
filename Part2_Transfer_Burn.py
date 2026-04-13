import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
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

# ---------------------------------------------------------
# NUMBA ACCELERATED PHYSICS CORE
# ---------------------------------------------------------

@njit
def _lerp_jit(start_val, end_val, alpha):
    """Linearly interpolate between two values."""
    return (1.0 - alpha) * start_val + alpha * end_val

@njit
def _gravity_from_body_jit(obj_x, obj_y, body_x, body_y, mu):
    """Calculates gravity vector from a specific body."""
    dx = obj_x - body_x
    dy = obj_y - body_y
    r_sq = dx*dx + dy*dy
    r = np.sqrt(r_sq)
    if r == 0.0: return 0.0, 0.0
    factor = -mu / (r_sq * r)
    return dx * factor, dy * factor

@njit
def _transfer_derivative_jit(state, ex, ey, evx, evy, mx, my, thrust_on, thrust_n, mass_flow):
    """Derivatives considering Sun, Earth, Mars, and Thrust."""
    x, y, vx, vy, mass, fuel = state[0], state[1], state[2], state[3], state[4], state[5]

    sun_ax, sun_ay = _gravity_from_body_jit(x, y, 0.0, 0.0, SUN_MU_KM)
    earth_ax, earth_ay = _gravity_from_body_jit(x, y, ex, ey, EARTH_MU_KM)
    mars_ax, mars_ay = _gravity_from_body_jit(x, y, mx, my, MARS_MU_KM)

    thrust_ax, thrust_ay, dmass, dfuel = 0.0, 0.0, 0.0, 0.0

    if thrust_on and fuel > 0.0 and mass > 0.0:
        # Determine Reference Frame for Prograde direction (Earth SOI vs Sun)
        dist_to_earth = np.sqrt((x - ex)**2 + (y - ey)**2)
        if dist_to_earth <= EARTH_SOI_RADIUS_KM:
            ref_vx = vx - evx
            ref_vy = vy - evy
        else:
            ref_vx = vx
            ref_vy = vy

        speed = np.sqrt(ref_vx**2 + ref_vy**2)
        if speed > 0.0:
            accel = (thrust_n / mass) / 1000.0
            thrust_ax = accel * (ref_vx / speed)
            thrust_ay = accel * (ref_vy / speed)
            dmass = -mass_flow
            dfuel = -mass_flow

    return np.array([
        vx, vy,
        sun_ax + earth_ax + mars_ax + thrust_ax,
        sun_ay + earth_ay + mars_ay + thrust_ay,
        dmass, dfuel
    ], dtype=np.float64)

@njit
def _rk4_step_transfer_jit(state, dt, ex_s, ey_s, evx_s, evy_s, ex_e, ey_e, evx_e, evy_e, 
                           mx_s, my_s, mx_e, my_e, a_s, a_e, thrust_on, thrust_n, mass_flow):
    """RK4 step with lerped planet positions."""
    a_mid = 0.5 * (a_s + a_e)

    # Planet positions at points 1, 2, 4 (for k1, k2/k3, k4)
    ex1, ey1 = _lerp_jit(ex_s, ex_e, a_s), _lerp_jit(ey_s, ey_e, a_s)
    evx1, evy1 = _lerp_jit(evx_s, evx_e, a_s), _lerp_jit(evy_s, evy_e, a_s)
    mx1, my1 = _lerp_jit(mx_s, mx_e, a_s), _lerp_jit(my_s, my_e, a_s)

    ex2, ey2 = _lerp_jit(ex_s, ex_e, a_mid), _lerp_jit(ey_s, ey_e, a_mid)
    evx2, evy2 = _lerp_jit(evx_s, evx_e, a_mid), _lerp_jit(evy_s, evy_e, a_mid)
    mx2, my2 = _lerp_jit(mx_s, mx_e, a_mid), _lerp_jit(my_s, my_e, a_mid)

    ex4, ey4 = _lerp_jit(ex_s, ex_e, a_e), _lerp_jit(ey_s, ey_e, a_e)
    evx4, evy4 = _lerp_jit(evx_s, evx_e, a_e), _lerp_jit(evy_s, evy_e, a_e)
    mx4, my4 = _lerp_jit(mx_s, mx_e, a_e), _lerp_jit(my_s, my_e, a_e)

    k1 = _transfer_derivative_jit(state, ex1, ey1, evx1, evy1, mx1, my1, thrust_on, thrust_n, mass_flow)
    k2 = _transfer_derivative_jit(state + 0.5*dt*k1, ex2, ey2, evx2, evy2, mx2, my2, thrust_on, thrust_n, mass_flow)
    k3 = _transfer_derivative_jit(state + 0.5*dt*k2, ex2, ey2, evx2, evy2, mx2, my2, thrust_on, thrust_n, mass_flow)
    k4 = _transfer_derivative_jit(state + dt*k3, ex4, ey4, evx4, evy4, mx4, my4, thrust_on, thrust_n, mass_flow)

    next_state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    next_state[4] = max(next_state[4], 0.0)
    next_state[5] = max(next_state[5], 0.0)
    return next_state

@njit
def _propagate_phase2_jit(start_state, t_array, max_step, actual_burn_sec, thrust_n, mass_flow,
                          rem_stage_mass, sep_speed_m_s, 
                          e_x, e_y, e_vx, e_vy, m_x, m_y, m_vx, m_vy):
    """Main compiled integration loop for Phase 2."""
    n_steps = len(t_array)
    out = np.zeros((n_steps, 6), dtype=np.float64)
    out[0] = start_state
    
    current_state = start_state.copy()
    burn_elapsed = 0.0
    stage_separated = False
    
    # Return variables for separation event
    sep_time = -1.0
    sep_dv = 0.0
    discarded_mass = 0.0

    for i in range(n_steps - 1):
        int_sec = t_array[i+1] - t_array[i]
        int_elapsed = 0.0
        
        while int_elapsed < int_sec - 1e-12:
            rem_int_sec = int_sec - int_elapsed
            step_sec = min(max_step, rem_int_sec)
            
            burn_active = burn_elapsed < actual_burn_sec - 1e-12
            if burn_active:
                rem_burn = actual_burn_sec - burn_elapsed
                step_sec = min(step_sec, rem_burn)
                
            a_start = int_elapsed / int_sec
            a_end = (int_elapsed + step_sec) / int_sec
            
            # Collision checks
            e_x_s = _lerp_jit(e_x[i], e_x[i+1], a_start)
            e_y_s = _lerp_jit(e_y[i], e_y[i+1], a_start)
            m_x_s = _lerp_jit(m_x[i], m_x[i+1], a_start)
            m_y_s = _lerp_jit(m_y[i], m_y[i+1], a_start)
            
            d_earth = np.sqrt((current_state[0] - e_x_s)**2 + (current_state[1] - e_y_s)**2)
            d_mars = np.sqrt((current_state[0] - m_x_s)**2 + (current_state[1] - m_y_s)**2)
            
            if d_earth <= EARTH_RADIUS_KM:
                return out, i + 1, 1, sep_time, sep_dv, discarded_mass # 1 = Earth Impact
            if d_mars <= MARS_RADIUS_KM:
                return out, i + 1, 2, sep_time, sep_dv, discarded_mass # 2 = Mars Impact
            
            current_state = _rk4_step_transfer_jit(
                current_state, step_sec,
                e_x[i], e_y[i], e_vx[i], e_vy[i],
                e_x[i+1], e_y[i+1], e_vx[i+1], e_vy[i+1],
                m_x[i], m_y[i], m_x[i+1], m_y[i+1],
                a_start, a_end, burn_active, thrust_n, mass_flow
            )
            
            if burn_active: burn_elapsed += step_sec
            int_elapsed += step_sec
            
            # Stage Separation check
            if not stage_separated and actual_burn_sec > 0.0 and burn_elapsed >= actual_burn_sec - 1e-12:
                tot_mass = current_state[4]
                discarded_mass = max(0.0, tot_mass - rem_stage_mass)
                if discarded_mass > 0.0 and rem_stage_mass > 0.0:
                    speed = np.sqrt(current_state[2]**2 + current_state[3]**2)
                    if speed > 0.0:
                        ej_speed = sep_speed_m_s / 1000.0
                        sep_dv = (discarded_mass / rem_stage_mass) * ej_speed
                        current_state[2] += sep_dv * (current_state[2] / speed)
                        current_state[3] += sep_dv * (current_state[3] / speed)
                    
                    current_state[4] = rem_stage_mass
                    current_state[5] = 0.0
                
                stage_separated = True
                sep_time = t_array[i] + int_elapsed
                
        out[i+1] = current_state
        
    return out, n_steps, 0, sep_time, sep_dv * 1000.0, discarded_mass

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def format_elapsed_time(seconds):
    total_seconds = int(round(seconds))
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = ((total_seconds % 86400) % 3600) // 60
    return f"{days} d  {hours:02d} h  {minutes:02d} m"

def planetary_states_heliocentric(t_seconds_array, simulation_start_time_utc):
    """Fallback Astropy lookup."""
    start_time = Time(simulation_start_time_utc, scale="utc")
    sample_times = start_time + TimeDelta(t_seconds_array, format="sec")

    with solar_system_ephemeris.set("builtin"):
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", sample_times)
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", sample_times)
        mars_pos, mars_vel = get_body_barycentric_posvel("mars", sample_times)

    return {
        "earth_x_km": (earth_pos.xyz - sun_pos.xyz).to_value(u.km)[0],
        "earth_y_km": (earth_pos.xyz - sun_pos.xyz).to_value(u.km)[1],
        "earth_vx_km_s": (earth_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)[0],
        "earth_vy_km_s": (earth_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)[1],
        "mars_x_km": (mars_pos.xyz - sun_pos.xyz).to_value(u.km)[0],
        "mars_y_km": (mars_pos.xyz - sun_pos.xyz).to_value(u.km)[1],
        "mars_vx_km_s": (mars_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)[0],
        "mars_vy_km_s": (mars_vel.xyz - sun_vel.xyz).to_value(u.km / u.s)[1],
    }

def build_phase3_handoff_state(sim_start, t_sec, rx, ry, rvx, rvy, rmass, mx, my, mvx, mvy, lead_hours, status):
    dist_to_mars = np.hypot(rx - mx, ry - my)

    if status == "impacted Mars":
        target = max(0.0, t_sec[-1] - lead_hours * 3600.0)
        idx = int(np.searchsorted(t_sec, target, side="right") - 1)
        reason = f"{lead_hours:.3f} hours before Mars impact"
        app_type = "collision_course"
        rec_burn = "prograde_then_retrograde"
    else:
        # THE FIX: Find the closest approach, then back up the clock
        periapsis_index = int(np.argmin(dist_to_mars))
        
        # Back up by 2 hours (7200 seconds) to center a ~4 minute burn
        lead_seconds = 7200.0 
        target_time_seconds = max(0.0, t_sec[periapsis_index] - lead_seconds)
        
        # Find the array index that matches this earlier time
        idx = int(np.searchsorted(t_sec, target_time_seconds, side="right") - 1)
        
        reason = "approaching closest pass to Mars (Centered Burn Setup)"
        app_type = "near_pass"
        rec_burn = "retrograde_at_closest_pass"

    dt_utc = (Time(sim_start, scale="utc") + TimeDelta(t_sec[idx], format="sec")).utc.isot

    return {
        "handoff_reason": reason, "datetime_utc": dt_utc,
        "elapsed_time_seconds": float(t_sec[idx]), "elapsed_time_pretty": format_elapsed_time(t_sec[idx]),
        "rocket_position_km": {"x": float(rx[idx]), "y": float(ry[idx])},
        "rocket_velocity_km_s": {"vx": float(rvx[idx]), "vy": float(rvy[idx])},
        "rocket_mass_kg": float(rmass[idx]),
        "mars_position_km": {"x": float(mx[idx]), "y": float(my[idx])},
        "mars_velocity_km_s": {"vx": float(mvx[idx]), "vy": float(mvy[idx])},
        "rocket_position_relative_to_mars_km": {"x": float(rx[idx] - mx[idx]), "y": float(ry[idx] - my[idx])},
        "rocket_velocity_relative_to_mars_km_s": {"vx": float(rvx[idx] - mvx[idx]), "vy": float(rvy[idx] - mvy[idx])},
        "distance_to_mars_km": float(dist_to_mars[idx]),
        "mars_approach_type": app_type, "recommended_burn_direction": rec_burn,
        "phase3_collision_lead_hours": float(lead_hours),
    }

def build_final_state(
    start_utc, end_utc, elap_sec, rx, ry, rvx, rvy, ex, ey, evx, evy, mx, my, mvx, mvy, mass, fuel,
    req_burn_min, act_burn_sec, shutdown_reason, sep, sep_utc, sep_dv, disc_mass, handoff, status
):
    return {
        "simulation_start_time_utc": start_utc, "end_datetime_utc": end_utc,
        "elapsed_time_seconds": float(elap_sec), "elapsed_time_pretty": format_elapsed_time(elap_sec),
        "rocket_position_km": {"x": float(rx), "y": float(ry)}, "rocket_velocity_km_s": {"vx": float(rvx), "vy": float(rvy)},
        "earth_position_km": {"x": float(ex), "y": float(ey)}, "earth_velocity_km_s": {"vx": float(evx), "vy": float(evy)},
        "mars_position_km": {"x": float(mx), "y": float(my)}, "mars_velocity_km_s": {"vx": float(mvx), "vy": float(mvy)},
        "rocket_mass_kg": float(mass), "fuel_remaining_kg": float(fuel),
        "requested_burn_duration_minutes": float(req_burn_min), "actual_burn_duration_minutes": float(act_burn_sec / 60.0),
        "burn_shutdown_reason": shutdown_reason, "stage_separated": sep, "stage_separation_datetime_utc": sep_utc,
        "stage_separation_delta_v_m_s": float(sep_dv), "discarded_stage_mass_kg": float(disc_mass),
        "phase3_handoff_state": handoff, "status": status,
    }

def validate_mass_inputs(tot, fuel, rem):
    if fuel < 0.0: raise ValueError("starting_fuel_mass_kg must be non-negative.")
    if tot <= 0.0: raise ValueError("initial_total_mass_kg must be positive.")
    if rem <= 0.0: raise ValueError("remaining_stage_mass_kg must be positive.")
    if fuel > tot: raise ValueError("starting_fuel_mass_kg cannot exceed initial_total_mass_kg.")
    if rem > tot - fuel: raise ValueError("remaining_stage_mass_kg cannot exceed non-fuel mass.")

# ---------------------------------------------------------
# MAIN SIMULATION WRAPPER
# ---------------------------------------------------------

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
    validate_mass_inputs(initial_total_mass_kg, starting_fuel_mass_kg, remaining_stage_mass_kg)

    phase1_final_state = PHASE1_HANDOFF_STATE
    sim_start_utc = phase1_final_state["end_datetime_utc"]

    t_seconds = np.linspace(0.0, total_time_days * 86400.0, int((total_time_days * 86400.0) / dt_seconds) + 1)

    # Fast Planet Lookup
    try:
        from Porkchop_Searcher import load_interpolators
        e_r_f, e_v_f, m_r_f, m_v_f = load_interpolators("ephemeris_cache.npz")
        query_jds = Time(sim_start_utc, scale="utc").jd + (t_seconds / 86400.0)
        e_pos, e_vel = e_r_f(query_jds), e_v_f(query_jds)
        m_pos, m_vel = m_r_f(query_jds), m_v_f(query_jds)
        e_x, e_y, e_vx, e_vy = e_pos[:, 0], e_pos[:, 1], e_vel[:, 0], e_vel[:, 1]
        m_x, m_y, m_vx, m_vy = m_pos[:, 0], m_pos[:, 1], m_vel[:, 0], m_vel[:, 1]
    except Exception as e:
        print(f"Warning: Falling back to Astropy. Error: {e}")
        p_states = planetary_states_heliocentric(t_seconds, sim_start_utc)
        e_x, e_y = p_states["earth_x_km"], p_states["earth_y_km"]
        e_vx, e_vy = p_states["earth_vx_km_s"], p_states["earth_vy_km_s"]
        m_x, m_y = p_states["mars_x_km"], p_states["mars_y_km"]
        m_vx, m_vy = p_states["mars_vx_km_s"], p_states["mars_vy_km_s"]

    # Burn calculations
    burn_rate_kg_s = burn_rate_kg_per_min / 60.0
    req_burn_sec = max(0.0, requested_burn_duration_minutes * 60.0)
    
    if burn_rate_kg_s <= 0.0 or starting_fuel_mass_kg <= 0.0 or thrust_newtons <= 0.0:
        actual_burn_sec, shutdown_reason = 0.0, "no active burn"
    else:
        fuel_lim_sec = starting_fuel_mass_kg / burn_rate_kg_s
        actual_burn_sec = min(req_burn_sec, fuel_lim_sec)
        shutdown_reason = "fuel exhausted" if req_burn_sec > fuel_lim_sec else "requested burn duration reached"

    start_state = np.array([
        phase1_final_state["rocket_position_km"]["x"], phase1_final_state["rocket_position_km"]["y"],
        phase1_final_state["rocket_velocity_km_s"]["vx"], phase1_final_state["rocket_velocity_km_s"]["vy"],
        initial_total_mass_kg, starting_fuel_mass_kg
    ], dtype=np.float64)

    # CALL NUMBA JIT FUNCTION
    states, stop_idx, stat_code, sep_time, sep_dv, disc_mass = _propagate_phase2_jit(
        start_state, t_seconds, max_step_seconds, actual_burn_sec, thrust_newtons, burn_rate_kg_s,
        remaining_stage_mass_kg, stage_separation_relative_speed_m_s,
        e_x, e_y, e_vx, e_vy, m_x, m_y, m_vx, m_vy
    )

    # Reconstruct data
    t_sec = t_seconds[:stop_idx]
    rx, ry, rvx, rvy = states[:stop_idx, 0], states[:stop_idx, 1], states[:stop_idx, 2], states[:stop_idx, 3]
    rmass, rfuel = states[:stop_idx, 4], states[:stop_idx, 5]
    ex, ey, evx, evy = e_x[:stop_idx], e_y[:stop_idx], e_vx[:stop_idx], e_vy[:stop_idx]
    mx, my, mvx, mvy = m_x[:stop_idx], m_y[:stop_idx], m_vx[:stop_idx], m_vy[:stop_idx]

    if stat_code == 1: status = "impacted Earth"
    elif stat_code == 2: status = "impacted Mars"
    else: status = "completed full simulation"

    end_utc = (Time(sim_start_utc, scale="utc") + TimeDelta(t_sec[-1], format="sec")).utc.isot
    sep_utc = None
    if sep_time >= 0.0:
        sep_utc = (Time(sim_start_utc, scale="utc") + TimeDelta(sep_time, format="sec")).utc.isot

    phase3_handoff = build_phase3_handoff_state(
        sim_start_utc, t_sec, rx, ry, rvx, rvy, rmass, mx, my, mvx, mvy, phase3_collision_lead_hours, status
    )

    final_state = build_final_state(
        sim_start_utc, end_utc, t_sec[-1], rx[-1], ry[-1], rvx[-1], rvy[-1],
        ex[-1], ey[-1], evx[-1], evy[-1], mx[-1], my[-1], mvx[-1], mvy[-1],
        rmass[-1], rfuel[-1], requested_burn_duration_minutes, actual_burn_sec,
        shutdown_reason, (sep_time >= 0), sep_utc, sep_dv, disc_mass, phase3_handoff, status
    )

    return {
        "phase1_final_state": phase1_final_state,
        "t_seconds": t_sec, "rocket_x_km": rx, "rocket_y_km": ry, "rocket_vx_km_s": rvx, "rocket_vy_km_s": rvy,
        "rocket_mass_kg": rmass, "fuel_mass_kg": rfuel,
        "earth_x_km": ex, "earth_y_km": ey, "earth_vx_km_s": evx, "earth_vy_km_s": evy,
        "mars_x_km": mx, "mars_y_km": my, "mars_vx_km_s": mvx, "mars_vy_km_s": mvy,
        "actual_burn_duration_seconds": actual_burn_sec, "burn_shutdown_reason": shutdown_reason,
        "stage_separated": (sep_time >= 0), "stage_separation_time_seconds": (sep_time if sep_time >= 0 else None),
        "stage_separation_delta_v_m_s": sep_dv, "discarded_stage_mass_kg": disc_mass,
        "phase3_handoff_state": phase3_handoff, "status": status, "final_state": final_state,
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
    simulation = simulate_transfer_burn_phase2(
        requested_burn_duration_minutes=requested_burn_duration_minutes, thrust_newtons=thrust_newtons,
        initial_total_mass_kg=initial_total_mass_kg, burn_rate_kg_per_min=burn_rate_kg_per_min,
        starting_fuel_mass_kg=starting_fuel_mass_kg, remaining_stage_mass_kg=remaining_stage_mass_kg,
        stage_separation_relative_speed_m_s=stage_separation_relative_speed_m_s,
        phase3_collision_lead_hours=phase3_collision_lead_hours, dt_seconds=dt_seconds,
        max_step_seconds=max_step_seconds, total_time_days=total_time_days,
    )

    t_seconds, rx, ry = simulation["t_seconds"], simulation["rocket_x_km"], simulation["rocket_y_km"]
    rvx, rvy = simulation["rocket_vx_km_s"], simulation["rocket_vy_km_s"]
    rmass, rfuel = simulation["rocket_mass_kg"], simulation["fuel_mass_kg"]
    ex, ey, mx, my = simulation["earth_x_km"], simulation["earth_y_km"], simulation["mars_x_km"], simulation["mars_y_km"]
    actual_burn = simulation["actual_burn_duration_seconds"]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x relative to Sun (km)")
    ax.set_ylabel("y relative to Sun (km)")
    ax.set_title("Phase 2: Transfer Burn")

    max_extent = max(np.max(np.abs(rx)), np.max(np.abs(ry)), np.max(np.abs(ex)), 
                     np.max(np.abs(ey)), np.max(np.abs(mx)), np.max(np.abs(my)))
    pad = 0.08 * max_extent + 5.0e6
    ax.set_xlim(-max_extent - pad, max_extent + pad)
    ax.set_ylim(-max_extent - pad, max_extent + pad)

    ax.plot([0.0], [0.0], marker="o", markersize=10, color="gold", label="Sun")
    earth_path, = ax.plot([], [], lw=1.5, color="royalblue", alpha=0.6, label="Earth path")
    mars_path, = ax.plot([], [], lw=1.5, color="orangered", alpha=0.6, label="Mars path")
    burn_path, = ax.plot([], [], lw=2.0, color="darkorange", label="Burn path")
    coast_path, = ax.plot([], [], lw=2.0, color="black", label="Post-burn path")
    e_pt, = ax.plot([], [], marker="o", color="royalblue", label="Earth")
    m_pt, = ax.plot([], [], marker="o", color="orangered", label="Mars")
    r_pt, = ax.plot([], [], marker="o", color="black", label="Rocket")
    ax.legend(loc="upper right")

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    target_frames = max(1, int(np.ceil(900 / playback_speed)))
    frame_step = max(1, len(t_seconds) // target_frames)
    frame_indices = np.append(np.arange(0, len(t_seconds), frame_step), len(t_seconds) - 1)
    frame_indices = np.unique(frame_indices)

    def init():
        earth_path.set_data([], []); mars_path.set_data([], []); burn_path.set_data([], [])
        coast_path.set_data([], []); e_pt.set_data([], []); m_pt.set_data([], []); r_pt.set_data([], [])
        info.set_text("")
        return earth_path, mars_path, burn_path, coast_path, e_pt, m_pt, r_pt, info

    def update(f):
        i = frame_indices[f]
        earth_path.set_data(ex[: i + 1], ey[: i + 1])
        mars_path.set_data(mx[: i + 1], my[: i + 1])

        b_idx = t_seconds[: i + 1] <= actual_burn + 1e-9
        if np.any(b_idx): burn_path.set_data(rx[: i + 1][b_idx], ry[: i + 1][b_idx])
        else: burn_path.set_data([], [])

        if np.any(~b_idx): coast_path.set_data(rx[: i + 1][~b_idx], ry[: i + 1][~b_idx])
        else: coast_path.set_data([], [])

        e_pt.set_data([ex[i]], [ey[i]])
        m_pt.set_data([mx[i]], [my[i]])
        r_pt.set_data([rx[i]], [ry[i]])

        info.set_text(
            f"time = {format_elapsed_time(t_seconds[i])}\n"
            f"speed = {np.hypot(rvx[i], rvy[i]):.4f} km/s\n"
            f"mass = {rmass[i]:.2f} kg\n"
            f"thrust = {'burning' if t_seconds[i] <= actual_burn + 1e-9 else 'coasting'}\n"
            + (f"\nstatus = {simulation['status']}" if i == len(t_seconds) - 1 else "")
        )
        return earth_path, mars_path, burn_path, coast_path, e_pt, m_pt, r_pt, info

    anim = FuncAnimation(fig, update, frames=len(frame_indices), init_func=init,
                         interval=1000 / (fps * playback_speed), blit=True)
    plt.show()
    return anim, simulation

def print_handoff_state(phase1_final_state):
    print("Phase 2 start state (from Part 1)")
    print(f"Start date-time (UTC): {phase1_final_state['end_datetime_utc']}")

def print_final_state(final_state):
    print("\nFinal phase 2 state")
    print(f"Status: {final_state['status']}")
    print(f"Elapsed time: {final_state['elapsed_time_pretty']}")
    print(f"Burn actual: {final_state['actual_burn_duration_minutes']:.2f} min ({final_state['burn_shutdown_reason']})")
    print(f"Stage separated: {final_state['stage_separated']} (dv: {final_state['stage_separation_delta_v_m_s']:.2f} m/s)")
    print(f"Handoff reason: {final_state['phase3_handoff_state']['handoff_reason']}")

if __name__ == "__main__":
    animation, simulation = animate_transfer_burn_phase2(**PHASE2_INPUTS)
    print_handoff_state(simulation["phase1_final_state"])
    print_final_state(simulation["final_state"])