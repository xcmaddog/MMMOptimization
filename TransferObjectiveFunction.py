import mars_mission.simulator.Part1_Earth_Idle as phase1
import mars_mission.simulator.Part2_Transfer_Burn as phase2
import mars_mission.simulator.Part3_Mars_Burn as phase3

"""
==============================================
phase 1 design variables
----------------------------------------------
"launch_angle_deg"
"simulation_start_time_utc"
"total_time_days"
==============================================
phase 2 design variables
----------------------------------------------
"requested_burn_duration_minutes"
"thrust_newtons"
"burn_rate_kg_per_min"
"starting_fuel_mass_kg"
"remaining_stage_mass_kg"
==============================================
phase 3 design variables
----------------------------------------------
"collision_conversion_burn_duration_minutes"
"requested_burn_duration_minutes"
==============================================
"""

def transObjFunc(x):
    # initialize static dict that stores the inputs and the outputs

    # check that the input isn't already in the static dict before doing the calculations

    # convert the start time from a float to UTC

    # create the dict of phase_one_settings

    # create the dict of phase_two_settings

    # create the dict of phase_three_settings

    # run all phases

    # calculate duration of transfer

    # calculate cost of transfer

    # calculate fuel used in transfer

    # bundle duration, cost, and fuel into a dict to store solutions and push that dict to the static dict with x as the key

    # return the dict

    return 0

def durationObj():
    # run transObjFunc and return the duration
    return 0

def costObj():
    # run transObjFunc and return the cost
    return 0

def fuelObj():
    # run transObjFunc and return the fuel used
    return 0