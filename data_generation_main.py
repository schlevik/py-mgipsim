import argparse
import csv
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

from preprocess_data import preprocess_data
from pymgipsim.Utilities.paths import results_path
from pymgipsim.Utilities import simulation_folder
from pymgipsim.Utilities.Scenario import load_scenario

from pymgipsim.Interface.parser import generate_parser_cli
from pymgipsim.InputGeneration.activity_settings import activity_args_to_scenario
from pymgipsim.generate_settings import generate_simulation_settings_main
from pymgipsim.generate_inputs import generate_inputs_main
from pymgipsim.generate_subjects import generate_virtual_subjects_main
from pymgipsim.generate_plots import generate_plots_main
from pymgipsim.generate_results import generate_results_main
from pymgipsim.faultsGeneration import generate_faults

from QAdataGeneration.AnomalyDetection.build_anomaly_dataset import generate_anomaly_detection_qa
from QAdataGeneration.pattern.build_pattern_dataset import generate_pattern_recognition_qa
from QAdataGeneration.prediction.generate_prediction_qa_pairs import (
    REQUIRED_PREDICTION_SCENARIOS,
    generate_prediction_qa,
    validate_prediction_source_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTION_SCENARIO_OVERRIDES_PATH = (
    PROJECT_ROOT / "QAdataGeneration" / "prediction" / "prediction_scenario_overrides.json"
)
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


def normalize_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def resolve_active_results_folder(args: argparse.Namespace, new_results_folder_path: str | None = None) -> Path:
    active_path = new_results_folder_path if new_results_folder_path is not None else args.data_path
    if active_path in (None, "None"):
        raise FileNotFoundError("No active simulation results folder is available.")

    results_folder_path = normalize_path(active_path)
    if not results_folder_path.exists():
        raise FileNotFoundError(
            f"Simulation results folder does not exist: {results_folder_path}"
        )

    return results_folder_path


def load_saved_base_scenario(results_folder_path: Path):
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


def main(argv=None):
    """ Main execution function for the simulation pipeline. """

    parser = generate_parser_cli()
    parser.add_argument(
        '--data_path',
        type=str,
        default="None",
        help='Path to simulation data folder. If provided, QA data will be generated from this path.',
    )
    parser.add_argument(
        '--qa',
        nargs='+',
        choices=['pm', 'ad', 'pd'],
        default=None,
        help='Specify QA data types (pm, ad, pd). Can pass one or multiple. If not provided, only simulation data will be generated.',
    )
    parser.add_argument(
        '--ad_ids',
        type=int,
        nargs='+',
        default=None,
        help="List of anomaly detection question IDs to generate (e.g., 1 3 5)"
    )

    args = parser.parse_args(argv)
    print("Arguments parsed successfully.")

    created_results_folder_path = None
    faults_input = None
    random_scenario = None

    if not os.path.exists(args.data_path):
        print("Simulation data path does not exist, preparing generating new data...")
        subprocess.run(['python', 'initialization.py'], check=True)
        _, _, _, created_results_folder_path = simulation_folder.create_simulation_results_folder(results_path)
        settings_file = simulation_folder.load_settings_file(args, created_results_folder_path)
        activity_args_to_scenario(settings_file, args)

        args.to_excel = True

        if args.faults_file:
            print(f"   - Loading faults from file: {args.faults_file}")
            faults_input = generate_faults.generate_faults_from_file(
                faults_file=args.faults_file,
                simulation_days=int(args.number_of_days)
            )
        elif args.random_fault_intensity:
            print(f"   - Generating random faults with intensity: {args.random_fault_intensity}")
            faults_input = generate_faults.generate_random_faults(
                simulation_days=int(args.number_of_days),
                intensity=args.random_fault_intensity,
                random_state=args.random_seed,
                faulty_type=args.fault_type
            )

        if args.random_scenario:
            print("   - Using random scenario variations on inputs.")
            random_scenario = {
                'target': args.random_scenario,
                'method': args.random_scenario_methods,
                'intensity': args.random_scenario_intensity
            }

        if not args.scenario_name:
            print("\n Generating new scenario from provided arguments...")
            settings_file = generate_simulation_settings_main(
                scenario_instance=settings_file,
                args=args,
                results_folder_path=created_results_folder_path,
            )
            settings_file = generate_virtual_subjects_main(
                scenario_instance=settings_file,
                args=args,
                results_folder_path=created_results_folder_path,
            )
            settings_file = generate_inputs_main(
                scenario_instance=settings_file,
                args=args,
                results_folder_path=created_results_folder_path,
                random_scenario=random_scenario,
            )
        else:
            print(f"\n Loading pre-defined scenario: {args.scenario_name}")

        generate_results_main(
            scenario_instance=settings_file,
            args=vars(args),
            results_folder_path=created_results_folder_path,
            faults_array=faults_input,
        )

        results_folder_path = resolve_active_results_folder(args, created_results_folder_path)
        generate_plots_main(str(results_folder_path), args, faults_input)
        args.data_path = str(results_folder_path)
        print(f"\n Simulation pipeline completed successfully! Results saved to {results_folder_path}")
    else:
        results_folder_path = resolve_active_results_folder(args)
        args.data_path = str(results_folder_path)
        print(f"Loading simulation data from {results_folder_path}")

    base_scenario = load_saved_base_scenario(results_folder_path)
    base_settings_path = results_folder_path / "simulation_settings.json"
    patient_count = get_saved_patient_count(base_scenario, base_settings_path)
    args.number_of_subjects = patient_count

    if 'ad' in args.qa:
        print("Generating anomaly detection question answering dataset...")
        generate_anomaly_detection_qa(args.data_path, args.ad_ids)
    if 'pm' in args.qa:
        print("Generating pattern recognition question answering dataset...")
        generate_pattern_recognition_qa(args.data_path)
        print("Building prediction source bundle...")
    if 'pd' in args.qa:
        # if fault is there we dont want to include it in the prediction source bundle and therefore generate a new bundle without faults
        build_prediction_source_bundle(
            results_folder_path=results_folder_path,
            base_scenario=base_scenario,
            base_args=args,
            patient_count=patient_count,
        )
        print("Generating prediction question answering dataset...")
        generate_prediction_qa(args.data_path, patient_count=patient_count)

    # all_figures = []
    # for patient_index in range(patient_count):
    #     args.plot_patient = patient_index
    #     print(f"Generating plots for Patient (index {patient_index})")
    #     figures = generate_plots_main(str(results_folder_path), args)
    #     all_figures.append(figures)


if __name__ == '__main__':
    test_arguments = [
        '-pat', '0',
        '-rs', '146',
        '-d', '7',
        '-ns', '20',
        '-ctrl', 'OpenAPS',
        '-ft', 'repeated_episode',
        '--data_path', 'SimulationResults/Simulation_ad_testing_30day_withNaNInstruction',
        '-st', '5',
        '--qa', 'ad',
        '--ad_ids', '1', '5', '9', '46'
    ]
    main()
