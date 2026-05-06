from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QAdataGeneration.pattern.preprocess_data import preprocess_data
from pymgipsim.Utilities.Scenario import load_scenario

from QAdataGeneration.pattern.generate_insulin_qa import (
    generate_questions_and_answers as generate_insulin_questions_and_answers,
)
from QAdataGeneration.pattern.generate_qa_pairs import (
    convert_np,
    generate_questions_and_answers as generate_glucose_questions_and_answers,
)

REQUIRED_INPUT_FILES = (
    "simulation_settings.json",
    "model_state_results.xlsx",
    "insulin_input.csv",
)
PATTERN_INPUT_CONTEXT_KEYS = (
    "carb_events",
    "insulin_events",
    "exercise_events",
    "bg_mgdl",
)
GLUCOSE_BUILDER_OFFSET = 0
INSULIN_BUILDER_OFFSET = 10_000


def normalize_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def resolve_active_results_folder(data_dir: str | Path) -> Path:
    if data_dir in (None, "None"):
        raise FileNotFoundError("No active simulation results folder is available.")

    results_folder_path = normalize_path(data_dir)
    if not results_folder_path.exists():
        raise FileNotFoundError(
            f"Simulation results folder does not exist: {results_folder_path}"
        )

    return results_folder_path


def validate_required_inputs(results_folder_path: Path) -> dict[str, Path]:
    resolved_paths = {
        filename: results_folder_path / filename
        for filename in REQUIRED_INPUT_FILES
    }
    missing_paths = [
        str(path)
        for path in resolved_paths.values()
        if not path.exists()
    ]
    if missing_paths:
        formatted_paths = "\n- ".join(missing_paths)
        raise FileNotFoundError(
            f"Pattern recognition QA requires the following source artifacts under "
            f"{results_folder_path}:\n- {formatted_paths}"
        )
    return resolved_paths


def load_saved_pattern_metadata(results_folder_path: Path) -> tuple[int, int]:
    simulation_settings_path = results_folder_path / "simulation_settings.json"
    scenario_instance = load_scenario(str(simulation_settings_path))

    missing_fields = []
    if scenario_instance.settings is None:
        missing_fields.append("settings")
    elif scenario_instance.settings.random_seed is None:
        missing_fields.append("settings.random_seed")

    if scenario_instance.patient is None:
        missing_fields.append("patient")
    elif scenario_instance.patient.number_of_subjects is None:
        missing_fields.append("patient.number_of_subjects")

    if missing_fields:
        formatted_fields = "\n- ".join(missing_fields)
        raise ValueError(
            f"Saved simulation settings at {simulation_settings_path} do not contain the "
            f"metadata required to build the pattern recognition dataset:\n- {formatted_fields}"
        )

    patient_count = int(scenario_instance.patient.number_of_subjects)
    simulation_seed = int(scenario_instance.settings.random_seed)

    if patient_count <= 0:
        raise ValueError(
            f"Invalid patient count in saved simulation settings at {simulation_settings_path}: "
            f"{patient_count!r}"
        )

    return patient_count, simulation_seed


@contextmanager
def deterministic_builder_seed(seed_value: int):
    state = random.getstate()
    random.seed(seed_value)
    try:
        yield
    finally:
        random.setstate(state)


def load_patient_simulation_data(simulation_data_dir: Path, patient_index: int) -> dict[str, Any]:
    patient_path = simulation_data_dir / f"Patient_{patient_index}_simulation_data.jsonl"
    if not patient_path.exists():
        raise FileNotFoundError(
            f"Pattern source data for Patient_{patient_index} was not generated: {patient_path}"
        )

    with patient_path.open("r", encoding="utf-8") as handle:
        line = handle.readline().strip()

    if not line:
        raise ValueError(f"Pattern source data file is empty: {patient_path}")

    return json.loads(line)


def add_question_ids(qa_pairs: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    indexed_pairs = []
    for index, qa in enumerate(qa_pairs):
        qa_pair = dict(qa)
        qa_pair["question_id"] = f"{prefix}{index}"
        indexed_pairs.append(qa_pair)
    return indexed_pairs


def build_input_context(patient_data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: patient_data.get(key)
        for key in PATTERN_INPUT_CONTEXT_KEYS
    }


def build_pattern_patient_record(
    patient_data: dict[str, Any],
    patient_index: int,
    simulation_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with deterministic_builder_seed(simulation_seed + patient_index + GLUCOSE_BUILDER_OFFSET):
        glucose_qa_pairs = add_question_ids(
            generate_glucose_questions_and_answers(patient_data),
            "pm_",
        )

    with deterministic_builder_seed(simulation_seed + patient_index + INSULIN_BUILDER_OFFSET):
        insulin_qa_pairs = add_question_ids(
            generate_insulin_questions_and_answers(patient_data),
            "pm_insulin_",
        )

    merged_qa_pairs = glucose_qa_pairs + insulin_qa_pairs
    patient_id = patient_data["patient_id"]

    qa_record = {
        "patient_id": patient_id,
        "qa_pairs": merged_qa_pairs,
    }
    qa_record_with_context = {
        "patient_id": patient_id,
        "input_context": build_input_context(patient_data),
        "qa_pairs": merged_qa_pairs,
    }
    return qa_record, qa_record_with_context


def write_json_records(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, default=convert_np)


def write_jsonl_records(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=convert_np))
            handle.write("\n")


def generate_pattern_recognition_qa(data_dir: str) -> None:
    results_folder_path = resolve_active_results_folder(data_dir)
    required_inputs = validate_required_inputs(results_folder_path)
    patient_count, simulation_seed = load_saved_pattern_metadata(results_folder_path)

    pattern_source_dir = results_folder_path / "PatternSource"
    simulation_data_dir = pattern_source_dir / "SimulationData"
    qa_output_dir = results_folder_path / "QAData"

    simulation_data_dir.mkdir(parents=True, exist_ok=True)
    qa_output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_data(
        simulation_path=str(required_inputs["simulation_settings.json"]),
        bg_path=str(required_inputs["model_state_results.xlsx"]),
        insulin_csv_path=str(required_inputs["insulin_input.csv"]),
        output_path=str(simulation_data_dir),
        num_people=patient_count,
        scenario_name=results_folder_path.name,
    )

    qa_records = []
    qa_records_with_context = []

    for patient_index in range(patient_count):
        patient_data = load_patient_simulation_data(simulation_data_dir, patient_index)
        qa_record, qa_record_with_context = build_pattern_patient_record(
            patient_data=patient_data,
            patient_index=patient_index,
            simulation_seed=simulation_seed,
        )
        qa_records.append(qa_record)
        qa_records_with_context.append(qa_record_with_context)

    write_json_records(qa_records, qa_output_dir / "QA_pattern.json")
    write_jsonl_records(qa_records_with_context, qa_output_dir / "QA_pattern_with_context.jsonl")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the merged pattern recognition QA dataset for an existing results folder."
    )
    parser.add_argument(
        "data_dir",
        help="Path to the simulation results folder to process.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    generate_pattern_recognition_qa(cli_args.data_dir)
