import argparse
import os
import subprocess
from pathlib import Path

from pymgipsim.Utilities.paths import results_path
from pymgipsim.Utilities import simulation_folder

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
from QAdataGeneration.prediction.generate_prediction_qa_pairs import generate_prediction_qa
from QAdataGeneration.prediction.source_bundle import prepare_prediction_source_bundle, validate_prediction_source_bundle

PROJECT_ROOT = Path(__file__).resolve().parent


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
    requested_qa = set(args.qa or [])
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

    if 'ad' in requested_qa:
        print("Generating anomaly detection question answering dataset...")
        generate_anomaly_detection_qa(args.data_path, args.ad_ids, args.number_of_subjects)
    if 'pm' in requested_qa:
        print("Generating pattern recognition question answering dataset...")
        generate_pattern_recognition_qa(args.data_path)
    if 'pd' in requested_qa:
        # if fault is there we dont want to include it in the prediction source bundle and therefore generate a new bundle without faults
        try:
            source_root = Path(args.data_path) / "PredictionSource"
            validate_prediction_source_bundle(source_root)
        except FileNotFoundError as e:
            print(e)
            print("Building prediction source bundle...")
            _, patient_count = prepare_prediction_source_bundle(
                results_folder_path=results_folder_path,
                base_args=args,
            )
        print("Generating prediction question answering dataset...")
        generate_prediction_qa(args.data_path, patient_count=args.number_of_subjects)

    # for check simulation validation
    # Generate plots for all patients
    all_figures = []
    for i in range(args.number_of_subjects):
        args.plot_patient = i
        print(f"Generating plots for Patient (index {i})")
        figures = generate_plots_main(results_folder_path, args)
        all_figures.append(figures)

 
if __name__ == '__main__':
    test_arguments = [
        '-pat', '0',
        '-rs', '146',
        '-d', '30',
        '-ns', '20',
        '-ctrl', 'OpenAPS',
        # '-rfi', '0.1',
        # '-ft', 'missing_signal',
        '-ff', 'pymgipsim/faultsGeneration/faults_specification.csv',
        '-st', '5',
        '-bcr', '50', '80',
        '-lcr', '60', '90',
        '-dcr', '50', '80',
        '--qa', 'ad',
        '--data_path', 'Datasets/simulation_30days_faulty',
        '-rsp', '0.0',
        '-cpwr', '60', '90',
        '-cst', '15:00', '17:00'
    ]
    main()
