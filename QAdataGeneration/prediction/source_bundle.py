from __future__ import annotations

import argparse
import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path

REQUIRED_PREDICTION_SCENARIOS = {
    "normal_day": "insulin_input_normal.csv",
    "late_night_snack": "insulin_input_late_night_snack.csv",
    "overeating_lunch": "insulin_input_overeating_lunch.csv",
}
PREDICTION_SCENARIO_OVERRIDES_PATH = Path(__file__).with_name("prediction_scenario_overrides.json")
INPUT_GENERATION_ARG_FIELDS = (
    "fraction_cho_intake",
    "fraction_cho_as_snack",
    "net_calorie_balance",
    "daily_energy_intake",
    "meal_duration",
    "snack_duration",
    "breakfast_time_range",
    "lunch_time_range",
    "dinner_time_range",
    "total_carb_range",
    "am_snack_time_range",
    "pm_snack_time_range",
    "sglt2i_dose_magnitude",
    "sglt2i_dose_time_range",
    "breakfast_carb_range",
    "lunch_carb_range",
    "dinner_carb_range",
    "am_snack_carb_range",
    "pm_snack_carb_range",
    "running_start_time",
    "running_duration",
    "running_incline",
    "running_speed",
    "cycling_start_time",
    "cycling_duration",
    "cycling_power",
)
PREDICTION_BUNDLE_SAMPLING_TIME_MINUTES = 5
PREDICTION_BUNDLE_NUMBER_OF_DAYS = 30


def load_saved_base_scenario(results_folder_path: Path):
    from pymgipsim.Utilities.Scenario import load_scenario

    simulation_settings_path = results_folder_path / "simulation_settings.json"
    if not simulation_settings_path.exists():
        raise FileNotFoundError(
            f"Missing base scenario artifact required to build prediction scenarios: "
            f"{simulation_settings_path}"
        )

    scenario_instance = load_scenario(str(simulation_settings_path))
    validate_prediction_base_scenario(scenario_instance, simulation_settings_path)
    return scenario_instance


def validate_prediction_base_scenario(scenario_instance, simulation_settings_path: Path) -> None:
    missing_fields = []

    if scenario_instance.settings is None:
        missing_fields.append("settings")
    else:
        if scenario_instance.settings.random_seed is None:
            missing_fields.append("settings.random_seed")
        if scenario_instance.settings.random_state is None:
            missing_fields.append("settings.random_state")
        if scenario_instance.settings.start_time is None or scenario_instance.settings.end_time is None:
            missing_fields.append("settings.start_time/settings.end_time")

    if scenario_instance.input_generation is None:
        missing_fields.append("input_generation")

    if scenario_instance.patient is None:
        missing_fields.append("patient")
    else:
        if scenario_instance.patient.number_of_subjects is None:
            missing_fields.append("patient.number_of_subjects")
        if scenario_instance.patient.model is None:
            missing_fields.append("patient.model")
        else:
            if not scenario_instance.patient.model.name:
                missing_fields.append("patient.model.name")
            if not scenario_instance.patient.model.parameters:
                missing_fields.append("patient.model.parameters")
        if scenario_instance.patient.demographic_info is None:
            missing_fields.append("patient.demographic_info")

    if missing_fields:
        formatted_fields = "\n- ".join(missing_fields)
        raise ValueError(
            f"Saved simulation settings at {simulation_settings_path} do not contain the scenario/cohort "
            f"artifacts required to regenerate prediction scenarios:\n- {formatted_fields}"
        )


def get_saved_patient_count(scenario_instance, simulation_settings_path: Path) -> int:
    patient_count = getattr(scenario_instance.patient, "number_of_subjects", None)
    if not isinstance(patient_count, int) or patient_count <= 0:
        raise ValueError(
            f"Invalid patient count in saved simulation settings at {simulation_settings_path}: "
            f"{patient_count!r}"
        )
    return patient_count


def load_prediction_scenario_overrides() -> dict[str, dict]:
    if not PREDICTION_SCENARIO_OVERRIDES_PATH.exists():
        raise FileNotFoundError(
            f"Missing prediction scenario overrides file: {PREDICTION_SCENARIO_OVERRIDES_PATH}"
        )

    with PREDICTION_SCENARIO_OVERRIDES_PATH.open("r", encoding="utf-8") as handle:
        overrides = json.load(handle)

    missing_scenarios = sorted(
        scenario_name
        for scenario_name in REQUIRED_PREDICTION_SCENARIOS
        if scenario_name not in overrides
    )
    if missing_scenarios:
        raise ValueError(
            f"Prediction scenario overrides are missing entries for: {', '.join(missing_scenarios)}"
        )

    return overrides


def mirror_saved_scenario_to_args(args: argparse.Namespace, scenario_instance, patient_count: int) -> None:
    args.number_of_subjects = patient_count
    args.model_name = scenario_instance.patient.model.name
    args.patient_names = deepcopy(scenario_instance.patient.files)
    args.random_seed = scenario_instance.settings.random_seed
    args.sampling_time = scenario_instance.settings.sampling_time
    args.multi_scale = scenario_instance.settings.simulator_name == "MultiScaleSolver"
    args.number_of_days = int(
        (scenario_instance.settings.end_time - scenario_instance.settings.start_time) / 1440
    )

    if scenario_instance.patient.demographic_info is not None:
        body_weight_range = getattr(scenario_instance.patient.demographic_info, "body_weight_range", None)
        renal_function_category = getattr(
            scenario_instance.patient.demographic_info, "renal_function_category", None
        )
        if body_weight_range is not None:
            args.body_weight_range = deepcopy(body_weight_range)
        if renal_function_category is not None:
            args.renal_function_category = deepcopy(renal_function_category)

    if scenario_instance.controller is not None:
        args.controller_name = scenario_instance.controller.name
        args.controller_parameters = deepcopy(scenario_instance.controller.parameters)

    for field_name in INPUT_GENERATION_ARG_FIELDS:
        value = getattr(scenario_instance.input_generation, field_name, None)
        if value is not None:
            setattr(args, field_name, deepcopy(value))


def apply_prediction_input_overrides(scenario_instance, args: argparse.Namespace, overrides: dict) -> None:
    valid_fields = set(type(scenario_instance.input_generation).__slots__)

    for field_name, value in overrides.items():
        if field_name not in valid_fields:
            raise ValueError(f"Unsupported prediction override field: {field_name}")

        copied_value = deepcopy(value)
        setattr(scenario_instance.input_generation, field_name, copied_value)
        setattr(args, field_name, deepcopy(value))


def apply_prediction_time_horizon(scenario_instance, args: argparse.Namespace) -> None:
    args.sampling_time = PREDICTION_BUNDLE_SAMPLING_TIME_MINUTES
    args.number_of_days = PREDICTION_BUNDLE_NUMBER_OF_DAYS

    scenario_instance.settings.sampling_time = PREDICTION_BUNDLE_SAMPLING_TIME_MINUTES
    scenario_instance.settings.end_time = (
        scenario_instance.settings.start_time + (PREDICTION_BUNDLE_NUMBER_OF_DAYS * 24 * 60)
    )


def transpose_insulin_input(source_csv_path: Path, destination_csv_path: Path, patient_count: int) -> None:
    with source_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [row for row in reader if row]

    if header is None or not rows:
        raise ValueError(f"Prediction insulin CSV is empty: {source_csv_path}")
    if len(header) != patient_count:
        raise ValueError(
            f"Prediction insulin CSV at {source_csv_path} contains {len(header)} patients, "
            f"expected {patient_count}."
        )
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"Prediction insulin CSV at {source_csv_path} is not rectangular.")

    transposed_rows = [
        [row[column_index] for row in rows]
        for column_index in range(len(header))
    ]

    with destination_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([str(index) for index in range(len(rows))])
        writer.writerows(transposed_rows)


def prefix_prediction_jsonl_outputs(output_dir: Path, scenario_name: str, patient_count: int) -> None:
    for patient_index in range(patient_count):
        source_path = output_dir / f"Patient_{patient_index}_simulation_data.jsonl"
        destination_path = output_dir / f"{scenario_name}_{patient_index}_simulation_data.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing prediction simulation data output for {scenario_name}: {source_path}"
            )
        source_path.rename(destination_path)


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for row in reader if row)


def has_prediction_source_bundle(root: Path) -> bool:
    try:
        validate_prediction_source_bundle(root)
    except FileNotFoundError:
        return False
    return True


def validate_prediction_source_bundle(bundle_root: Path) -> None:
    missing_paths = []
    for scenario_name, insulin_csv in REQUIRED_PREDICTION_SCENARIOS.items():
        scenario_data = bundle_root / "SimulationData" / scenario_name
        scenario_results = bundle_root / "SimulationResults" / scenario_name
        insulin_input = bundle_root / insulin_csv

        if not scenario_data.exists():
            missing_paths.append(str(scenario_data))
        if not scenario_results.exists():
            missing_paths.append(str(scenario_results))
        if not insulin_input.exists():
            missing_paths.append(str(insulin_input))

    if missing_paths:
        formatted_paths = "\n- ".join(missing_paths)
        raise FileNotFoundError(
            f"Prediction source bundle is incomplete under {bundle_root}. "
            f"Expected the following paths to exist:\n- {formatted_paths}"
        )


def available_prediction_patients(bundle_root: Path, scenario_name: str = "normal_day") -> int:
    scenario_dir = bundle_root / "SimulationData" / scenario_name
    return sum(1 for path in scenario_dir.glob(f"{scenario_name}_*_simulation_data.jsonl") if path.is_file())


def validate_prediction_bundle_outputs(bundle_root: Path, expected_patient_count: int) -> None:
    missing_paths = []
    count_errors = []

    for scenario_name, insulin_csv_name in REQUIRED_PREDICTION_SCENARIOS.items():
        scenario_results_dir = bundle_root / "SimulationResults" / scenario_name
        scenario_data_dir = bundle_root / "SimulationData" / scenario_name
        scenario_settings_path = scenario_results_dir / "simulation_settings.json"
        scenario_bg_path = scenario_results_dir / "model_state_results.xlsx"
        scenario_insulin_path = scenario_results_dir / "insulin_input.csv"
        bundled_insulin_path = bundle_root / insulin_csv_name

        for path in (
            scenario_results_dir,
            scenario_data_dir,
            scenario_settings_path,
            scenario_bg_path,
            scenario_insulin_path,
            bundled_insulin_path,
        ):
            if not path.exists():
                missing_paths.append(str(path))

        if scenario_settings_path.exists():
            from pymgipsim.Utilities.Scenario import load_scenario

            scenario_instance = load_scenario(str(scenario_settings_path))
            scenario_patient_count = getattr(scenario_instance.patient, "number_of_subjects", None)
            if scenario_patient_count != expected_patient_count:
                count_errors.append(
                    f"{scenario_settings_path} declares {scenario_patient_count} patients, "
                    f"expected {expected_patient_count}."
                )

        if scenario_data_dir.exists():
            data_paths = sorted(scenario_data_dir.glob(f"{scenario_name}_*_simulation_data.jsonl"))
            if len(data_paths) != expected_patient_count:
                count_errors.append(
                    f"{scenario_data_dir} contains {len(data_paths)} prediction JSONL files, "
                    f"expected {expected_patient_count}."
                )

        if bundled_insulin_path.exists():
            insulin_row_count = count_csv_rows(bundled_insulin_path)
            if insulin_row_count != expected_patient_count:
                count_errors.append(
                    f"{bundled_insulin_path} contains {insulin_row_count} patient rows, "
                    f"expected {expected_patient_count}."
                )

    if missing_paths:
        formatted_paths = "\n- ".join(missing_paths)
        raise FileNotFoundError(
            f"Prediction source bundle under {bundle_root} is missing required outputs:\n- {formatted_paths}"
        )

    if count_errors:
        formatted_errors = "\n- ".join(count_errors)
        raise ValueError(
            f"Prediction source bundle under {bundle_root} has patient-count mismatches:\n- {formatted_errors}"
        )


def build_prediction_scenario(
    base_scenario,
    base_args: argparse.Namespace,
    bundle_root: Path,
    scenario_name: str,
    input_overrides: dict,
    bundled_insulin_csv_name: str,
    patient_count: int,
) -> None:
    from preprocess_data import preprocess_data
    from pymgipsim.generate_inputs import generate_inputs_main
    from pymgipsim.generate_results import generate_results_main

    scenario_results_dir = bundle_root / "SimulationResults" / scenario_name
    scenario_data_dir = bundle_root / "SimulationData" / scenario_name

    if scenario_results_dir.exists():
        shutil.rmtree(scenario_results_dir)
    if scenario_data_dir.exists():
        shutil.rmtree(scenario_data_dir)

    scenario_results_dir.mkdir(parents=True, exist_ok=True)
    scenario_data_dir.mkdir(parents=True, exist_ok=True)

    scenario_instance = deepcopy(base_scenario)
    prediction_args = deepcopy(base_args)
    mirror_saved_scenario_to_args(prediction_args, scenario_instance, patient_count)
    apply_prediction_time_horizon(scenario_instance, prediction_args)
    apply_prediction_input_overrides(scenario_instance, prediction_args, input_overrides)
    prediction_args.to_excel = True

    scenario_instance = generate_inputs_main(
        scenario_instance=scenario_instance,
        args=prediction_args,
        results_folder_path=str(scenario_results_dir),
    )
    generate_results_main(
        scenario_instance=scenario_instance,
        args=vars(prediction_args),
        results_folder_path=str(scenario_results_dir),
        faults_array=None,
    )

    preprocess_data(
        simulation_path=str(scenario_results_dir / "simulation_settings.json"),
        bg_path=str(scenario_results_dir / "model_state_results.xlsx"),
        insulin_csv_path=str(scenario_results_dir / "insulin_input.csv"),
        output_path=str(scenario_data_dir),
        num_people=patient_count,
        scenario_name=scenario_name,
    )
    prefix_prediction_jsonl_outputs(scenario_data_dir, scenario_name, patient_count)
    transpose_insulin_input(
        source_csv_path=scenario_results_dir / "insulin_input.csv",
        destination_csv_path=bundle_root / bundled_insulin_csv_name,
        patient_count=patient_count,
    )


def build_prediction_source_bundle(
    results_folder_path: Path,
    base_scenario,
    base_args: argparse.Namespace,
    patient_count: int,
) -> Path:
    bundle_root = results_folder_path / "PredictionSource"
    scenario_overrides = load_prediction_scenario_overrides()

    if bundle_root.exists():
        shutil.rmtree(bundle_root)

    (bundle_root / "SimulationResults").mkdir(parents=True, exist_ok=True)
    (bundle_root / "SimulationData").mkdir(parents=True, exist_ok=True)

    for scenario_name, bundled_insulin_csv_name in REQUIRED_PREDICTION_SCENARIOS.items():
        print(f"Building prediction source bundle for scenario: {scenario_name}")
        build_prediction_scenario(
            base_scenario=base_scenario,
            base_args=base_args,
            bundle_root=bundle_root,
            scenario_name=scenario_name,
            input_overrides=scenario_overrides[scenario_name],
            bundled_insulin_csv_name=bundled_insulin_csv_name,
            patient_count=patient_count,
        )

    validate_prediction_source_bundle(bundle_root)
    validate_prediction_bundle_outputs(bundle_root, patient_count)
    return bundle_root


def prepare_prediction_source_bundle(
    results_folder_path: Path | str,
    base_args: argparse.Namespace,
) -> tuple[Path, int]:
    results_folder_path = Path(results_folder_path)
    base_scenario = load_saved_base_scenario(results_folder_path)
    base_settings_path = results_folder_path / "simulation_settings.json"
    patient_count = get_saved_patient_count(base_scenario, base_settings_path)
    base_args.number_of_subjects = patient_count

    bundle_root = build_prediction_source_bundle(
        results_folder_path=results_folder_path,
        base_scenario=base_scenario,
        base_args=base_args,
        patient_count=patient_count,
    )
    return bundle_root, patient_count
