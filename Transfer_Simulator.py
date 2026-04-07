import Part1_Earth_Idle as phase1
import Part2_Transfer_Burn as phase2
import Part3_Mars_Burn as phase3


"""
------------------------------------------------------------
Transfer Simulator Inputs

This file is the main control center for the full mission run.
Edit the settings below instead of changing the individual phase files.

How it works:
1. Phase 1 runs using PHASE1_SETTINGS.
2. The final state from phase 1 is passed into phase 2 automatically.
3. Phase 2 runs using PHASE2_SETTINGS.
4. Phase 3 runs using PHASE3_SETTINGS.

Optimizer usage:
- Another file can import run_transfer_simulator(...) and pass dictionaries of
  overrides for phase1_settings, phase2_settings, and phase3_settings on a
  per-run basis.

Animation controls:
- Set RUN_PHASE1_ANIMATION to True to show the parking-orbit animation.
- Set RUN_PHASE2_ANIMATION to True to show the transfer-burn animation.
- Set RUN_PHASE3_ANIMATION to True to show the Mars-centric observation
  animation.
- Any phase can still run without animation when its switch is False.
------------------------------------------------------------
"""
RUN_PHASE1_ANIMATION = True
RUN_PHASE2_ANIMATION = True
RUN_PHASE3_ANIMATION = True


"""
------------------------------------------------------------
Phase 1 Settings

These settings override the defaults inside Part1_Earth_Idle.py when the
combined simulator is run from this file.

Input details:
- launch_altitude_km:
  Rocket starting altitude above Earth's surface in km.
- initial_velocity_km_s:
  Rocket starting speed relative to Earth in km/s.
  Set to None to use the circular-orbit speed automatically.
- launch_angle_deg:
  Starting angle around Earth in degrees.
  90.0 places the rocket above Earth on the +y axis like the original model.
- simulation_start_time_utc:
  Astropy epoch used for Earth and Mars heliocentric states in phase 1.
- dt_seconds:
  Stored/output time step in seconds.
- max_step_seconds:
  Internal RK4 step size in seconds.
- total_time_days:
  Total duration of phase 1.
- playback_speed:
  Animation speed multiplier for phase 1.
- fps:
  Animation frame rate target for phase 1.
------------------------------------------------------------
"""
PHASE1_SETTINGS = {
    "launch_altitude_km": 200.0,
    "initial_velocity_km_s": None,
    "launch_angle_deg": 133.01, #Design variable
    "simulation_start_time_utc": "2020-09-02T00:00:00", #Design variable
    "dt_seconds": 180.0,
    "max_step_seconds": 60.0,
    "total_time_days": 1.0, #Design variable
    "playback_speed": 0.5,
    "fps": 30,
}


"""
------------------------------------------------------------
Phase 2 Settings

These settings override the defaults inside Part2_Transfer_Burn.py when the
combined simulator is run from this file.

The initial rocket, Earth, Mars, and starting date-time values are not set
here, because they come automatically from the final output of phase 1.

Input details:
- requested_burn_duration_minutes:
  Requested motor burn time in minutes.
  Thrust auto shuts off if fuel runs out first.
- thrust_newtons:
  Motor thrust in newtons.
- initial_total_mass_kg:
  Rocket mass at the start of phase 2, including fuel.
- burn_rate_kg_per_min:
  Fuel consumption rate during the burn.
- starting_fuel_mass_kg:
  Fuel available at the start of phase 2.
- remaining_stage_mass_kg:
  Remaining rocket mass after stage separation.
- stage_separation_relative_speed_m_s:
  Backward relative ejection speed used in the stage-separation momentum model.
- phase3_collision_lead_hours:
  If phase 2 predicts a Mars impact, phase 3 begins this many hours before the
  impact to allow a collision-conversion burn.
- dt_seconds:
  Stored/output time step in seconds.
- max_step_seconds:
  Internal RK4 step size in seconds.
- total_time_days:
  Total duration of phase 2.
- playback_speed:
  Animation speed multiplier for phase 2.
- fps:
  Animation frame rate target for phase 2.
------------------------------------------------------------
"""
PHASE2_SETTINGS = {
    "requested_burn_duration_minutes": 60.0, #Design variable
    "thrust_newtons": 100000.0, #Design variable
    "initial_total_mass_kg": 12000.0,
    "burn_rate_kg_per_min": 1346.0,
    "starting_fuel_mass_kg": 8500.0,
    "remaining_stage_mass_kg": 2500.0,
    "stage_separation_relative_speed_m_s": 50.0,
    "phase3_collision_lead_hours": 10.0,
    "dt_seconds": 180.0,
    "max_step_seconds": 30.0,
    "total_time_days": 150.0,
    "playback_speed": 5.0,
    "fps": 60,
}


"""
------------------------------------------------------------
Phase 3 Settings

These settings override the defaults inside Part3_Mars_Burn.py when the
combined simulator is run from this file.

The initial rocket, Mars, and starting date-time values are not set here,
because they come automatically from the phase-3 handoff output of phase 2.

Input details:
- correction_fuel_mass_kg:
  Fuel available for the corrective burn at the start of phase 3.
- thrust_newtons:
  Corrective-thruster force in newtons.
- collision_conversion_burn_duration_minutes:
  Duration of the initial prograde burn used only for collision-course cases.
- requested_burn_duration_minutes:
  Duration of the retrograde burn applied at closest pass.
- dt_seconds:
  Stored/output time step in seconds.
- max_step_seconds:
  Internal RK4 step size in seconds.
- total_time_days:
  Total duration of phase 3.
- playback_speed:
  Animation speed multiplier for phase 3.
- fps:
  Animation frame rate target for phase 3.
------------------------------------------------------------
"""
PHASE3_SETTINGS = {
    "correction_fuel_mass_kg": 1640.0,
    "thrust_newtons": 1200.0,
    "collision_conversion_burn_duration_minutes": 25.0, #Design variable
    "requested_burn_duration_minutes": 32, #Design variable
    "dt_seconds": 180.0,
    "max_step_seconds": 30.0,
    "total_time_days": 10.0,
    "playback_speed": 1.5,
    "fps": 30,
}


def _split_simulation_and_animation_settings(settings):
    """Separate pure simulation inputs from animation-only controls."""
    simulation_settings = {key: value for key, value in settings.items() if key not in {"playback_speed", "fps"}}
    animation_settings = dict(settings)
    return simulation_settings, animation_settings


def run_phase1(
    phase1_settings=None,
    run_animation=False,
):
    """
    Run phase 1 by itself.

    This is intended for optimizer-style imports where only the phase-1 state
    is needed and the later phases should not run.
    """
    effective_phase1_settings = {**PHASE1_SETTINGS, **(phase1_settings or {})}
    phase1_simulation_settings, phase1_animation_settings = _split_simulation_and_animation_settings(
        effective_phase1_settings
    )

    if run_animation:
        phase1_animation, phase1_final_state = phase1.animate_parking_orbit_phase1(**phase1_animation_settings)
        phase1_simulation = None
    else:
        phase1_simulation = phase1.simulate_parking_orbit_phase1(**phase1_simulation_settings)
        phase1_animation = None
        phase1_final_state = phase1_simulation["final_state"]

    return {
        "phase1_animation": phase1_animation,
        "phase1_settings": effective_phase1_settings,
        "phase1_simulation": phase1_simulation,
        "phase1_final_state": phase1_final_state,
    }


def run_phase2(
    phase1_final_state,
    phase2_settings=None,
    run_animation=False,
):
    """
    Run phase 2 by itself from a supplied phase-1 final state.

    phase1_final_state should match the structure returned by phase 1.
    """
    effective_phase2_settings = {**PHASE2_SETTINGS, **(phase2_settings or {})}
    phase2_simulation_settings, phase2_animation_settings = _split_simulation_and_animation_settings(
        effective_phase2_settings
    )

    phase2.PHASE1_HANDOFF_STATE = phase1_final_state

    if run_animation:
        phase2_animation, phase2_simulation = phase2.animate_transfer_burn_phase2(**phase2_animation_settings)
    else:
        phase2_simulation = phase2.simulate_transfer_burn_phase2(**phase2_simulation_settings)
        phase2_animation = None

    return {
        "phase2_animation": phase2_animation,
        "phase2_settings": effective_phase2_settings,
        "phase2_simulation": phase2_simulation,
        "phase2_final_state": phase2_simulation["final_state"],
        "phase3_handoff_state": phase2_simulation["phase3_handoff_state"],
    }


def run_phase3(
    phase2_handoff_state,
    phase3_settings=None,
    run_animation=False,
):
    """
    Run phase 3 by itself from a supplied phase-2 Mars handoff state.

    phase2_handoff_state should match the phase3_handoff_state structure
    returned by phase 2.
    """
    effective_phase3_settings = {**PHASE3_SETTINGS, **(phase3_settings or {})}
    phase3_simulation_settings, phase3_animation_settings = _split_simulation_and_animation_settings(
        effective_phase3_settings
    )

    phase3.PHASE2_HANDOFF_STATE = phase2_handoff_state

    if run_animation:
        phase3_animation, phase3_final_state = phase3.animate_mars_orbit_phase3(**phase3_animation_settings)
        phase3_simulation = None
    else:
        phase3_simulation = phase3.simulate_mars_orbit_phase3(**phase3_simulation_settings)
        phase3_animation = None
        phase3_final_state = phase3_simulation["final_state"]

    return {
        "phase3_animation": phase3_animation,
        "phase3_settings": effective_phase3_settings,
        "phase3_simulation": phase3_simulation,
        "phase3_final_state": phase3_final_state,
    }


def run_transfer_simulator(
    phase1_settings=None,
    phase2_settings=None,
    phase3_settings=None,
    run_phase1_animation=RUN_PHASE1_ANIMATION,
    run_phase2_animation=RUN_PHASE2_ANIMATION,
    run_phase3_animation=RUN_PHASE3_ANIMATION,
):
    """
    Run phase 1, hand its final state into phase 2, then hand phase 2 into phase 3.

    Optional arguments let other files supply per-run settings without editing
    the defaults in this module.
    """
    phase1_result = run_phase1(
        phase1_settings=phase1_settings,
        run_animation=run_phase1_animation,
    )
    phase1_animation = phase1_result["phase1_animation"]
    phase1_final_state = phase1_result["phase1_final_state"]

    print("\n=== Phase 1 Complete ===")
    phase1.print_final_state(phase1_final_state)

    phase2_result = run_phase2(
        phase1_final_state=phase1_final_state,
        phase2_settings=phase2_settings,
        run_animation=run_phase2_animation,
    )
    phase2_animation = phase2_result["phase2_animation"]
    phase2_simulation = phase2_result["phase2_simulation"]

    print("\n=== Phase 2 Start State ===")
    phase2.print_handoff_state(phase2_simulation["phase1_final_state"])

    print("\n=== Phase 2 Complete ===")
    phase2.print_final_state(phase2_simulation["final_state"])

    phase3_result = run_phase3(
        phase2_handoff_state=phase2_simulation["phase3_handoff_state"],
        phase3_settings=phase3_settings,
        run_animation=run_phase3_animation,
    )
    phase3_animation = phase3_result["phase3_animation"]
    phase3_final_state = phase3_result["phase3_final_state"]

    print("\n=== Phase 3 Start State ===")
    print(phase2_simulation["phase3_handoff_state"])

    print("\n=== Phase 3 Complete ===")
    phase3.print_final_state(phase3_final_state)

    return {
        "phase1_animation": phase1_animation,
        "phase1_settings": phase1_result["phase1_settings"],
        "phase1_final_state": phase1_final_state,
        "phase2_animation": phase2_animation,
        "phase2_settings": phase2_result["phase2_settings"],
        "phase2_simulation": phase2_simulation,
        "phase3_animation": phase3_animation,
        "phase3_settings": phase3_result["phase3_settings"],
        "phase3_final_state": phase3_final_state,
    }


if __name__ == "__main__":
    run_transfer_simulator()
