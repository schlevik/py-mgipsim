import os
from pymgipsim.Utilities.units_conversions_constants import UnitConversion, DEFAULT_RANDOM_SEED
from pymgipsim.Utilities.paths import results_path, default_settings_path
from pymgipsim.Utilities.Scenario import save_scenario
import json

# Initialize a dictionary to store default settings

scenario = {}

""" Simulation Settings """
# Set parameters for simulation
scenario["settings"] = {}
scenario["settings"]["save_directory"] = results_path
scenario["settings"]["start_time"] = 0
scenario["settings"]["end_time"] = 1440
scenario["settings"]['random_seed'] = DEFAULT_RANDOM_SEED
scenario["settings"]['random_state'] = {}
scenario["settings"]['sampling_time'] = 5 # minutes
scenario["settings"]["solver_name"] = "RK4"
scenario["settings"]['simulator_name'] = 'SingleScaleSolver'



""" Carb Settings """
scenario["input_generation"] = {}
# Set parameters related to carbohydrate intake
scenario['input_generation']['fraction_cho_intake'] = [0.4, 0.6]
scenario['input_generation']['fraction_cho_as_snack'] = [0.1, 0.2]
scenario['input_generation']['net_calorie_balance'] = [0]

# Set parameters related to carbohydrate intake
scenario["input_generation"]['meal_duration'] = [30, 60]
scenario["input_generation"]['snack_duration'] = [5, 15]

scenario['input_generation']['breakfast_carb_range'] = [80, 120]
scenario['input_generation']['lunch_carb_range'] = [80, 120]
scenario['input_generation']['dinner_carb_range'] = [80, 120]

scenario['input_generation']['am_snack_carb_range'] = [10, 20]
scenario['input_generation']['pm_snack_carb_range'] = [10, 20]

scenario['input_generation']['total_carb_range'] = [
                                                                    scenario['input_generation']['breakfast_carb_range'][i] +
                                                                    scenario['input_generation']['lunch_carb_range'][i] +
                                                                    scenario['input_generation']['dinner_carb_range'][i] +
                                                                    scenario['input_generation']['am_snack_carb_range'][i] +
                                                                    scenario['input_generation']['pm_snack_carb_range'][i] for i in range(2)
                                                                    ]

# Default Meal and Snack Times
scenario["input_generation"]['breakfast_time_range'] = [UnitConversion.time.convert_hour_to_min(6), UnitConversion.time.convert_hour_to_min(8)]
scenario["input_generation"]['lunch_time_range'] =  [UnitConversion.time.convert_hour_to_min(12), UnitConversion.time.convert_hour_to_min(14)]
scenario["input_generation"]['dinner_time_range'] =  [UnitConversion.time.convert_hour_to_min(18), UnitConversion.time.convert_hour_to_min(20)]

scenario["input_generation"]['am_snack_time_range'] = [UnitConversion.time.convert_hour_to_min(9), UnitConversion.time.convert_hour_to_min(11)]
scenario["input_generation"]['pm_snack_time_range'] = [UnitConversion.time.convert_hour_to_min(15), UnitConversion.time.convert_hour_to_min(17)]

scenario["input_generation"]["running_start_time"] = []
scenario["input_generation"]["running_duration"] = []
scenario["input_generation"]["running_incline"] = []
scenario["input_generation"]["running_speed"] = []
scenario["input_generation"]["cycling_start_time"] = []
scenario["input_generation"]["cycling_duration"] = []
scenario["input_generation"]["cycling_power"] = []

""" Drug Settings """
# SGLT2I

scenario["input_generation"]['sglt2i_dose_magnitude'] = 0
scenario["input_generation"]['sglt2i_dose_time_range'] = [UnitConversion.time.convert_hour_to_min(6), UnitConversion.time.convert_hour_to_min(9)]

""" Create blank inputs field """
scenario["inputs"] = None



""" Create blank controller field """
scenario["controller"] = {}
scenario["controller"]['name'] = 'HCL0'
scenario['controller']['parameters'] = []

""" Create blank patient field """
scenario["patient"] = {}
scenario["patient"]["demographic_info"] = None
scenario["patient"]['number_of_subjects'] = 20

scenario["patient"]["demographic_info"] = {}
scenario["patient"]["demographic_info"]["renal_function_category"] = 1
scenario["patient"]["demographic_info"]["body_weight_range"] = [60, 80]

scenario["patient"]["model"] = {}
scenario["patient"]["model"]["name"] = "T1DM.ExtHovorka"
scenario["patient"]["model"]["parameters"] = None
scenario["patient"]["model"]["initial_conditions"] = None
scenario["patient"]["mscale"] = {}
scenario['patient']["mscale"]["models"] = ["Multiscale.BodyWeight"]

save_scenario(os.path.join(default_settings_path, "scenario_default.json"), scenario)