import Part1_Earth_Idle as phase1
import Part2_Transfer_Burn as phase2
import Part3_Mars_Burn as phase3

import numpy as np
import time
from scipy.optimize import minimize
from astropy.time import Time
import warnings
from erfa.core import ErfaWarning

# Mute Astropy's "dubious year" warnings for future dates
warnings.simplefilter('ignore', category=ErfaWarning)

"""
------------------------------------------------------------
Global Animation Toggles
------------------------------------------------------------
"""
RUN_PHASE1_ANIMATION = True
RUN_PHASE2_ANIMATION = True
RUN_PHASE3_ANIMATION = True

"""
------------------------------------------------------------
Default Settings
------------------------------------------------------------
"""
PHASE1_SETTINGS = {
    "launch_altitude_km": 200.0,
    "initial_velocity_km_s": None,
    "launch_angle_deg": 133.02, 
    "simulation_start_time_utc": "2020-09-02T00:00:00", 
    "dt_seconds": 180.0, 
    "max_step_seconds": 30.0, 
    "total_time_days": 1.0, 
    "playback_speed": 0.5,
    "fps": 30,
}

PHASE2_SETTINGS = {
    "requested_burn_duration_minutes": 60.0, 
    "thrust_newtons": 100000.0, 
    "initial_total_mass_kg": 17500.0, 
    "burn_rate_kg_per_min": 1346.0, 
    "starting_fuel_mass_kg": 15000.0, 
    "remaining_stage_mass_kg": 2500.0, 
    "stage_separation_relative_speed_m_s": 50.0, 
    "phase3_collision_lead_hours": 10.0, 
    "dt_seconds": 180.0, 
    "max_step_seconds": 60.0, 
    "total_time_days": 400.0, 
    "playback_speed": 5.0,
    "fps": 60,
}
PHASE3_SETTINGS = {
    "correction_fuel_mass_kg": 2000.0,
    "thrust_newtons": 50000.0,
    "collision_conversion_burn_duration_minutes": 25.0,
    "requested_burn_duration_minutes": 3.0,
    "capture_start_radial_velocity_km_s": -0.05,
    "dt_seconds": 60.0,
    "max_step_seconds": 10.0,
    "total_time_days": 2.0,
    "playback_speed": 5.0,
    "fps": 30,
}

def _split_simulation_and_animation_settings(settings):
    """Separate pure simulation inputs from animation-only controls."""
    simulation_settings = {key: value for key, value in settings.items() if key not in {"playback_speed", "fps"}}
    animation_settings = dict(settings)
    return simulation_settings, animation_settings

# ---------------------------------------------------------
# PHASE RUNNERS
# ---------------------------------------------------------
def run_phase1(phase1_settings=None, run_animation=False):
    effective_phase1_settings = {**PHASE1_SETTINGS, **(phase1_settings or {})}
    phase1_simulation_settings, phase1_animation_settings = _split_simulation_and_animation_settings(effective_phase1_settings)

    if run_animation:
        phase1_animation, phase1_final_state = phase1.animate_parking_orbit_phase1(**phase1_animation_settings)
        #print("Saving Phase 1 animation as 'phase1_animation.gif'...")
        #phase1_animation.save("phase1_animation.gif", writer='pillow', fps=phase1_animation_settings.get("fps", 30))
        
        phase1_simulation = None
    else:
        phase1_simulation = phase1.simulate_parking_orbit_phase1(**phase1_simulation_settings)
        phase1_animation = None
        phase1_final_state = phase1_simulation["final_state"]

    return {"phase1_animation": phase1_animation, "phase1_settings": effective_phase1_settings, "phase1_simulation": phase1_simulation, "phase1_final_state": phase1_final_state}

def run_phase2(phase1_final_state, phase2_settings=None, run_animation=False):
    effective_phase2_settings = {**PHASE2_SETTINGS, **(phase2_settings or {})}
    phase2_simulation_settings, phase2_animation_settings = _split_simulation_and_animation_settings(effective_phase2_settings)

    phase2.PHASE1_HANDOFF_STATE = phase1_final_state

    if run_animation:
        phase2_animation, phase2_simulation = phase2.animate_transfer_burn_phase2(**phase2_animation_settings)
        #print("Saving Phase 2 animation as 'phase2_animation.gif'...")
        #phase2_animation.save("phase2_animation.gif", writer='pillow', fps=phase2_animation_settings.get("fps", 30))
    else:
        phase2_simulation = phase2.simulate_transfer_burn_phase2(**phase2_simulation_settings)
        phase2_animation = None

    return {"phase2_animation": phase2_animation, "phase2_settings": effective_phase2_settings, "phase2_simulation": phase2_simulation, "phase2_final_state": phase2_simulation["final_state"], "phase3_handoff_state": phase2_simulation["phase3_handoff_state"]}

def run_phase3(phase2_handoff_state, phase3_settings=None, run_animation=False):
    effective_phase3_settings = {**PHASE3_SETTINGS, **(phase3_settings or {})}
    phase3_simulation_settings, phase3_animation_settings = _split_simulation_and_animation_settings(effective_phase3_settings)

    phase3.PHASE2_HANDOFF_STATE = phase2_handoff_state

    if run_animation:
        phase3_animation, phase3_final_state = phase3.animate_mars_orbit_phase3(**phase3_animation_settings)
        #print("Saving Phase 3 animation as 'phase3_animation.gif'...")
        #phase3_animation.save("phase3_animation.gif", writer='pillow', fps=phase3_animation_settings.get("fps", 30))
        phase3_simulation = None
    else:
        phase3_simulation = phase3.simulate_mars_orbit_phase3(**phase3_simulation_settings)
        phase3_animation = None
        phase3_final_state = phase3_simulation["final_state"]

    return {"phase3_animation": phase3_animation, "phase3_settings": effective_phase3_settings, "phase3_simulation": phase3_simulation, "phase3_final_state": phase3_final_state}

def run_transfer_simulator(
    phase1_settings=None, phase2_settings=None, phase3_settings=None,
    run_phase1_animation=RUN_PHASE1_ANIMATION, run_phase2_animation=RUN_PHASE2_ANIMATION, run_phase3_animation=RUN_PHASE3_ANIMATION
):
    """Run phase 1, hand its final state into phase 2, then hand phase 2 into phase 3."""
    phase1_result = run_phase1(phase1_settings=phase1_settings, run_animation=run_phase1_animation)
    phase1_final_state = phase1_result["phase1_final_state"]
    print("\n=== Phase 1 Complete ===")
    phase1.print_final_state(phase1_final_state)

    phase2_result = run_phase2(phase1_final_state=phase1_final_state, phase2_settings=phase2_settings, run_animation=run_phase2_animation)
    phase2_simulation = phase2_result["phase2_simulation"]
    print("\n=== Phase 2 Complete ===")
    phase2.print_final_state(phase2_simulation["final_state"])

    phase3_result = run_phase3(phase2_handoff_state=phase2_simulation["phase3_handoff_state"], phase3_settings=phase3_settings, run_animation=run_phase3_animation)
    print("\n=== Phase 3 Complete ===")
    phase3.print_final_state(phase3_result["phase3_final_state"])

    p1_end_utc = phase1_final_state["end_datetime_utc"]
    p3_final = phase3_result["phase3_final_state"]
    p3_orbit_utc = p3_final.get("orbit_confirmation_datetime_utc")

    print("\n" + "="*40)
    print("TOTAL MISSION TIMELINE")
    print("="*40)
    print(f"Departure from Earth:   {p1_end_utc}")
    
    if p3_orbit_utc:
        print(f"Stable Mars Orbit:      {p3_orbit_utc}")
        
        # Calculate duration
        t_start = Time(p1_end_utc)
        t_end = Time(p3_orbit_utc)
        duration = t_end - t_start
        
        days = int(duration.value)
        hours = int((duration.value - days) * 24)
        minutes = int(((duration.value - days) * 24 - hours) * 60)
        
        print(f"Total Transit Time:     {days} days, {hours} hours, {minutes} minutes")
    else:
        print("Stable Mars Orbit:      NOT ACHIEVED")
    print("="*40)

    return phase3_result

# ---------------------------------------------------------
# THE 3-STEP HIGH-FIDELITY OPTIMIZER
# ---------------------------------------------------------
def optimize_transfer(target_date_utc):
    """
    1. Coarse Sweep: Scans 360 degrees to find the right ejection vector.
    2. Interplanetary Polish: Finds exact angle and Phase 2 burn to hit a 1,000km flyby.
    3. Capture Polish: Finds exact Phase 3 burn to circularize the orbit perfectly.
    """
    print(f"\n[ Optimizer Started ] Target Date: {target_date_utc}")
    
    # --- PHASE 2 COST FUNCTION ---
    iteration_count_p2 = [0]
    def p2_cost_function(x, quiet=False):
        launch_angle, burn_mins = x

        max_burn_minutes = PHASE2_SETTINGS["starting_fuel_mass_kg"] / PHASE2_SETTINGS["burn_rate_kg_per_min"]
        burn_mins = min(burn_mins, max_burn_minutes)

        if burn_mins <= 0:
            return 1e9 + abs(burn_mins) * 1e6

        p1_set = {
            "simulation_start_time_utc": target_date_utc,
            "launch_angle_deg": float(launch_angle % 360.0),
            "total_time_days": 1.0,   # FIX #3: enough time to reach SOI edge
        }
        p2_set = {"requested_burn_duration_minutes": float(burn_mins)}

        try:
            res1 = run_phase1(p1_set, run_animation=False)
            res2 = run_phase2(res1["phase1_final_state"], p2_set, run_animation=False)

            sim    = res2["phase2_simulation"]
            h_state = res2["phase3_handoff_state"]   # FIX #2: use handoff, not final_state

            if sim["status"] == "impacted Earth":
                return 1e9

            # ── 1. Orbit altitude error ──────────────────────────────
            rx = sim["rocket_x_km"]
            ry = sim["rocket_y_km"]
            mx = sim["mars_x_km"]
            my = sim["mars_y_km"]
            
            all_distances = np.hypot(rx - mx, ry - my)
            true_min_distance = np.min(all_distances)

            target_distance = phase2.MARS_RADIUS_KM + 1000.0
            orbit_error_norm = abs(true_min_distance - target_distance) / 1000.0

            if sim["status"] == "impacted Mars":
                orbit_error_norm += 50.0

            # ── 2. Tangency penalty — pulled from handoff state ──────
            # FIX #2: velocities at actual Mars encounter, not 250 days later
            rvx = h_state["rocket_velocity_km_s"]["vx"]
            rvy = h_state["rocket_velocity_km_s"]["vy"]
            mvx = h_state["mars_velocity_km_s"]["vx"]
            mvy = h_state["mars_velocity_km_s"]["vy"]

            dv_x = rvx - mvx
            dv_y = rvy - mvy

            mars_speed = np.hypot(mvx, mvy)
            if mars_speed > 0.0:
                tang_x = mvx / mars_speed
                tang_y = mvy / mars_speed
            else:
                tang_x, tang_y = 1.0, 0.0

            # Radial component of relative velocity (zero = perfect tangent)
            radial_dv = abs(dv_x * tang_y - dv_y * tang_x)

            TANGENCY_WEIGHT = 20.0
            total_cost = orbit_error_norm + TANGENCY_WEIGHT * radial_dv

            if not quiet:
                iteration_count_p2[0] += 1
                if iteration_count_p2[0] % 15 == 0:
                    print(
                        f"  P2 Iter {iteration_count_p2[0]:03d} | "
                        f"Angle: {launch_angle % 360.0:6.2f}° | "
                        f"Burn: {burn_mins:5.2f} min | "
                        f"Orbit Err: {orbit_error_norm:,.0f} | "
                        f"Radial ΔV: {radial_dv:.3f} km/s | "
                        f"Cost: {total_cost:,.1f}"
                    )

            return total_cost

        except Exception:
            return 1e9

    # --- STEP 1: COARSE SWEEP — now sweeps BOTH angle AND burn together ---
    # FIX #1: co-sweep burn duration so the angle search isn't stranded in
    # the wrong basin by a mismatched fixed burn time.
    print("\nStep 1: Coarse Sweep (angle × burn grid)...")
    best_guess_angle, best_guess_burn, best_guess_error = 0.0, 8.5, float('inf')

    max_burn_mins = PHASE2_SETTINGS["starting_fuel_mass_kg"] / PHASE2_SETTINGS["burn_rate_kg_per_min"]
    burn_candidates = [b for b in [5.0, 6.5, 7.5, max_burn_mins] if b <= max_burn_mins]
    
    for test_angle in range(0, 360, 15):
        for test_burn in burn_candidates:
            error = p2_cost_function([test_angle, test_burn], quiet=True)
            if error < best_guess_error:
                best_guess_error = error
                best_guess_angle = test_angle
                best_guess_burn  = test_burn

    print(f"-> Sweep complete. Best start: angle={best_guess_angle}°, burn={best_guess_burn} min, error={best_guess_error:,.0f}")

    # --- STEP 2: INTERPLANETARY POLISH ---
    print("\nStep 2: Fine-Tuning Transfer Burn (Threading the needle to Mars)...")
    start_time = time.time()

    result_p2 = minimize(
        p2_cost_function, x0=[best_guess_angle, best_guess_burn], args=(False,),
        method='Nelder-Mead', options={'xatol': 0.05, 'fatol': 10.0, 'maxiter': 300}
    )

    best_angle  = result_p2.x[0] % 360.0
    best_p2_burn = result_p2.x[1]
    print(f"-> Transfer locked. Angle: {best_angle:.2f}°, Burn: {best_p2_burn:.2f} min. ({time.time() - start_time:.1f}s)")

    # --- STEP 3: CAPTURE POLISH ---
    print("\nStep 3: Optimizing Mars Capture Burn (Circularizing the Orbit)...")
    
    # 1. First, we lock in the exact handoff state from our perfect Phase 2 run
    res1 = run_phase1({"simulation_start_time_utc": target_date_utc, "launch_angle_deg": best_angle, "total_time_days": 1.0}, run_animation=False)
    res2 = run_phase2(res1["phase1_final_state"], {"requested_burn_duration_minutes": best_p2_burn}, run_animation=False)
    best_handoff = res2["phase3_handoff_state"]
    
    # 2. Optimize Phase 3 to minimize Orbital Eccentricity (e=0 is a perfect circle)
    iteration_count_p3 = [0]
    def p3_cost_function(x, quiet=False):
        p3_burn_mins = float(x[0])
        capture_trigger = float(x[1])

        p3_burn_mins = max(0.1, min(p3_burn_mins, 20.0))
        capture_trigger = max(-0.30, min(capture_trigger, -0.001))

        test_fuel_kg = p3_burn_mins * 673.0

        try:
            settings = {
                "coast_time_days": 0.0,
                "requested_burn_duration_minutes": p3_burn_mins,
                "correction_fuel_mass_kg": test_fuel_kg,
                "capture_start_radial_velocity_km_s": capture_trigger,
            }

            res3 = run_phase3(best_handoff, settings, run_animation=False)
            fstate = res3["phase3_final_state"]

            if fstate.get("status") == "impacted Mars":
                return 1e9 + 1000.0

            rx = fstate["rocket_relative_to_mars_km"]["x"]
            ry = fstate["rocket_relative_to_mars_km"]["y"]
            vx = fstate["rocket_relative_to_mars_velocity_km_s"]["vx"]
            vy = fstate["rocket_relative_to_mars_velocity_km_s"]["vy"]

            r = np.hypot(rx, ry)
            altitude_km = r - phase3.MARS_RADIUS_KM

            if altitude_km < 300.0:
                return 1000.0 + (300.0 - altitude_km)

            mu = phase3.MARS_MU_KM
            v2 = vx * vx + vy * vy
            specific_energy = 0.5 * v2 - mu / r

            h = rx * vy - ry * vx
            ex = (vy * h) / mu - (rx / r)
            ey = (-vx * h) / mu - (ry / r)
            eccentricity = np.hypot(ex, ey)

            if specific_energy >= 0.0:
                return 500.0 + eccentricity + abs(altitude_km - 1000.0) / 1000.0

            total_cost = eccentricity + abs(altitude_km - 1000.0) / 10000.0

            if not quiet:
                iteration_count_p3[0] += 1
                if iteration_count_p3[0] % 5 == 0:
                    print(
                        f" P3 Iter {iteration_count_p3[0]:03d} | "
                        f"Burn: {p3_burn_mins:5.2f}m | "
                        f"Trigger: {capture_trigger:7.4f} km/s | "
                        f"Alt: {altitude_km:.0f}km | "
                        f"Ecc: {eccentricity:.4f}"
                    )

            if eccentricity >= 1.0:
                return 1000.0 + eccentricity

            return total_cost

        except Exception:
            return 1e9

    result_p3 = minimize(
        p3_cost_function,
        x0=[1.0, -0.05],        # 1.0, -0.05; 8.0, -1.0
        args=(False,),
        method='Nelder-Mead',
        options={'xatol': 0.01, 'fatol': 0.005, 'maxiter': 500}
    )

    best_p3_burn = float(result_p3.x[0])
    best_p3_trigger = float(result_p3.x[1])
    best_p3_coast = 0.0

    print("\n" + "=" * 40)
    print("FULL MISSION OPTIMIZATION COMPLETE")
    print(f"Optimal Launch Angle: {best_angle:.2f}°")
    print(f"Optimal Earth Departure Burn: {best_p2_burn:.2f} min")
    print(f"Optimal Mars Coast Time: {best_p3_coast:.2f} days")
    print(f"Optimal Mars Capture Burn: {best_p3_burn:.2f} min")
    print(f"Optimal Capture Trigger: {best_p3_trigger:.4f} km/s")
    print(f"Final Mars Orbit Cost: {result_p3.fun:.4f}")
    print("=" * 40)

    return best_angle, best_p2_burn, best_p3_burn, best_p3_coast, best_p3_trigger

def get_optimal_date_from_porkchop(filename="porkchop_data_highres.npz", strategy="efficient", max_dv=6.5):
    """Reads the saved Porkchop grid and returns the best UTC date string."""
    try:
        data = np.load(filename)
        L, T, DV = data['L'], data['T'], data['DV']
        
        if strategy == "efficient":
            min_idx = np.unravel_index(np.argmin(DV), DV.shape)
        elif strategy == "fast":
            valid_T = np.where(DV <= max_dv, T, np.inf)
            if np.all(valid_T == np.inf):
                print(f"WARNING: No trajectories found under {max_dv} km/s. Defaulting to most efficient.")
                min_idx = np.unravel_index(np.argmin(DV), DV.shape)
            else:
                min_idx = np.unravel_index(np.argmin(valid_T), valid_T.shape)
        else:
            raise ValueError("Strategy must be 'efficient' or 'fast'")
            
        best_utc = Time(L[min_idx], format='jd').utc.isot
        
        print("\n" + "-"*40)
        print(f"PORKCHOP EXTRACTION: '{strategy.upper()}' STRATEGY")
        print("-" * 40)
        print(f"Target Date:    {best_utc}")
        print(f"Expected Time:  {T[min_idx]:.1f} days")
        print(f"Expected Cost:  {DV[min_idx]:.2f} km/s")
        print("-" * 40)
        
        return best_utc
        
    except FileNotFoundError:
        print(f"Error: {filename} not found! Run Porkchop_Searcher.py first.")
        return None

# ---------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Automatically pull the best date from the saved grid
    BEST_UTC_DATE = get_optimal_date_from_porkchop(strategy="efficient", max_dv=6.5)

    if BEST_UTC_DATE is not None:
        # 2. Hand control over to the Optimizer 
        opt_angle, opt_p2_burn, opt_p3_burn, opt_p3_coast, opt_p3_trigger = optimize_transfer(target_date_utc=BEST_UTC_DATE)        
        
        # 3. CALCULATE EXACT FUEL NEEDED (No dead weight, no over-burning)
        # Phase 3 Thrust is 50,000 N, which yields a burn rate of ~673 kg/min
        exact_p3_fuel_kg = opt_p3_burn * 673.0
        
        print("\nStarting Final Visualizations...")
        
        # 4. RUN THE SIMULATOR
        run_transfer_simulator(
            phase1_settings={
                "simulation_start_time_utc": BEST_UTC_DATE,
                "launch_angle_deg": opt_angle,  
                "total_time_days": 1.0 
            },
            phase2_settings={
                "requested_burn_duration_minutes": opt_p2_burn, 
                "total_time_days": 400.0 
            },
            phase3_settings={
                "coast_time_days": opt_p3_coast,
                "requested_burn_duration_minutes": opt_p3_burn,
                "capture_start_radial_velocity_km_s": opt_p3_trigger,
                "correction_fuel_mass_kg": exact_p3_fuel_kg,
                "total_time_days": 2.0,
            },
            run_phase1_animation=True,
            run_phase2_animation=True,
            run_phase3_animation=True
        )