import Part1_Earth_Idle as phase1
import Part2_Transfer_Burn as phase2


"""
------------------------------------------------------------
Transfer Simulator Controls

This file runs the phases together in sequence:
1. Phase 1 parking orbit
2. Phase 2 transfer burn

Normal workflow:
- Edit phase-1 settings in Part1_Earth_Idle.py using PHASE1_INPUTS.
- Edit phase-2 burn settings in Part2_Transfer_Burn.py using PHASE2_INPUTS.
- Use the switches below to decide which animations should appear when this
  combined runner is executed.
------------------------------------------------------------
"""
RUN_PHASE1_ANIMATION = True
RUN_PHASE2_ANIMATION = True


def run_transfer_simulator():
    """Run phase 1, hand its final state into phase 2, then run phase 2."""
    if RUN_PHASE1_ANIMATION:
        phase1_animation, phase1_final_state = phase1.animate_parking_orbit_phase1(**phase1.PHASE1_INPUTS)
    else:
        phase1_simulation = phase1.simulate_parking_orbit_phase1(**phase1.PHASE1_INPUTS)
        phase1_animation = None
        phase1_final_state = phase1_simulation["final_state"]

    print("\n=== Phase 1 Complete ===")
    phase1.print_final_state(phase1_final_state)

    # Replace the placeholder handoff state in phase 2 with the actual phase-1 output.
    phase2.PHASE1_HANDOFF_STATE = phase1_final_state

    if RUN_PHASE2_ANIMATION:
        phase2_animation, phase2_simulation = phase2.animate_transfer_burn_phase2(**phase2.PHASE2_INPUTS)
    else:
        phase2_simulation = phase2.simulate_transfer_burn_phase2(**phase2.PHASE2_INPUTS)
        phase2_animation = None

    print("\n=== Phase 2 Start State ===")
    phase2.print_handoff_state(phase2_simulation["phase1_final_state"])

    print("\n=== Phase 2 Complete ===")
    phase2.print_final_state(phase2_simulation["final_state"])

    return {
        "phase1_animation": phase1_animation,
        "phase1_final_state": phase1_final_state,
        "phase2_animation": phase2_animation,
        "phase2_simulation": phase2_simulation,
    }


if __name__ == "__main__":
    run_transfer_simulator()
