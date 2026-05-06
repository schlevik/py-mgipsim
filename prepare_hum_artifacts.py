#!/usr/bin/env python3
"""Prepare downstream QA artifacts directly from one HUM parquet file.

This script does not run py-mgipsim simulation and does not generate QA pairs.
It adapts one HUM observation parquet into the artifact shapes consumed by the
existing anomaly-detection and prediction-data inspection tooling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 402
PATIENT_ID = "Patient_0"
PREDICTION_JSONL_RELATIVE_PATH = (
    Path("PredictionSource")
    / "SimulationData"
    / "normal_day"
    / "normal_only_0_real_data.jsonl"
)


@dataclass(frozen=True)
class PreparedHumData:
    source_path: Path
    timeline: pd.DatetimeIndex
    frame: pd.DataFrame
    sampling_minutes: float
    hum_subject_id: str | list[str] | None
    hum_subject_ids: list[str]
    raw_cgm_mgdl: pd.Series
    interpolated_cgm_mgdl: pd.Series
    faults_label: list[str]
    insulin_mumin: pd.Series
    carb_events: list[dict[str, Any]]
    insulin_events: list[dict[str, Any]]
    exercise_events: list[dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare QA-compatible artifacts from one HUM parquet file."
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to one HUM parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to SimulationResults/HUM_<parquet-stem>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously generated artifacts in the output directory.",
    )
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--cgm-column", default="CGM")
    parser.add_argument("--carbs-column", default="carbs")
    parser.add_argument("--insulin-column", default="insulin")
    parser.add_argument("--basal-column", default="basal")
    parser.add_argument("--bolus-column", default="bolus")
    parser.add_argument(
        "--sampling-minutes",
        type=float,
        default=5,
        help="Sampling interval in minutes. Defaults to the median timestamp spacing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed metadata to store in simulation_settings.json.",
    )
    return parser.parse_args(argv)


def default_output_dir(parquet_path: Path) -> Path:
    return PROJECT_ROOT / "SimulationResults" / f"HUM_{parquet_path.stem}"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        return default_output_dir(args.parquet_path).resolve()
    return args.output_dir.expanduser().resolve()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    known_files = (
        "model_state_results.xlsx",
        "insulin_input.csv",
        "iob.csv",
        "simulation_settings.json",
    )
    known_dirs = ("PredictionSource",)

    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to replace generated artifacts."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        return

    for file_name in known_files:
        path = output_dir / file_name
        if path.exists():
            path.unlink()
    for dir_name in known_dirs:
        path = output_dir / dir_name
        if path.exists():
            shutil.rmtree(path)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")

    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise SystemExit(
            "Reading parquet requires a pandas parquet engine such as pyarrow. "
            "Install pyarrow in this environment and rerun the script."
        ) from exc


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required HUM column(s): {', '.join(missing)}")


def numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def first_non_null(values: pd.Series) -> Any:
    non_null = values.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def aggregate_duplicate_timestamps(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not frame[args.date_column].duplicated().any():
        return frame

    sum_columns = {
        args.carbs_column,
        args.insulin_column,
        args.basal_column,
        args.bolus_column,
    }
    aggregations: dict[str, Any] = {}
    for column in frame.columns:
        if column == args.date_column:
            continue
        if column == args.cgm_column:
            aggregations[column] = lambda values: pd.to_numeric(values, errors="coerce").mean()
        elif column in sum_columns:
            aggregations[column] = lambda values: pd.to_numeric(values, errors="coerce").fillna(0.0).sum()
        else:
            aggregations[column] = first_non_null

    return (
        frame.groupby(args.date_column, as_index=False, sort=True)
        .agg(aggregations)
        .sort_values(args.date_column)
    )


def infer_sampling_minutes(dates: pd.Series) -> float:
    ordered = dates.sort_values().drop_duplicates()
    deltas = ordered.diff().dropna().dt.total_seconds().div(60.0)
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 5.0
    return float(deltas.median())


def build_timeline(start: pd.Timestamp, end: pd.Timestamp, sampling_minutes: float) -> pd.DatetimeIndex:
    if sampling_minutes <= 0 or not math.isfinite(sampling_minutes):
        raise ValueError(f"Sampling minutes must be positive; got {sampling_minutes!r}")

    freq = pd.to_timedelta(sampling_minutes, unit="m")
    timeline = pd.date_range(start=start, end=end, freq=freq)
    if timeline.empty:
        timeline = pd.DatetimeIndex([start])
    if timeline[-1] != end and end > timeline[-1]:
        timeline = timeline.append(pd.DatetimeIndex([end]))
    return timeline


def sorted_subject_ids(frame: pd.DataFrame, id_column: str) -> list[str]:
    if id_column not in frame.columns:
        return []
    subject_values = [value for value in frame[id_column].dropna().unique().tolist()]
    return sorted(str(value) for value in subject_values)


def display_subject_id(subject_ids: list[str]) -> str | list[str] | None:
    if not subject_ids:
        return None
    if len(subject_ids) == 1:
        return subject_ids[0]
    return subject_ids


def format_number(value: float, digits: int = 3) -> float:
    rounded = round(float(value), digits)
    if rounded == -0.0:
        return 0.0
    return rounded


def json_number_or_none(value: Any, digits: int = 3) -> float | None:
    if value is None or pd.isna(value):
        return None
    return format_number(float(value), digits)


def json_list_from_series(series: pd.Series, digits: int = 3) -> list[float | None]:
    return [json_number_or_none(value, digits=digits) for value in series.tolist()]


def minutes_for_index(index: int, sampling_minutes: float) -> float:
    return float(index) * float(sampling_minutes)


def format_time_info(total_minutes: float) -> tuple[int, str]:
    day = int(total_minutes // 1440) + 1
    minute_of_day = total_minutes % 1440
    hours = int(minute_of_day // 60)
    minutes = int(minute_of_day % 60)
    return day, f"{hours:02d}:{minutes:02d}"


def non_empty_label(value: Any, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    label = str(value).strip()
    return label if label else default


def event_base(index: int, sampling_minutes: float) -> dict[str, Any]:
    total_minutes = minutes_for_index(index, sampling_minutes)
    day, time_str = format_time_info(total_minutes)
    return {
        "time": format_number(total_minutes),
        "day": day,
        "time_str": time_str,
    }


def extract_carb_events(frame: pd.DataFrame, args: argparse.Namespace, sampling_minutes: float) -> list[dict[str, Any]]:
    carbs = numeric_series(frame, args.carbs_column).fillna(0.0)
    events: list[dict[str, Any]] = []

    for index, value in enumerate(carbs.tolist()):
        if value <= 0:
            continue
        event = event_base(index, sampling_minutes)
        event["carbs"] = format_number(value)
        if "meal_label" in frame.columns:
            event["meal_type"] = non_empty_label(frame.iloc[index]["meal_label"], "meal")
        else:
            event["meal_type"] = "meal"
        events.append(event)

    return events


def extract_insulin_events(frame: pd.DataFrame, args: argparse.Namespace, sampling_minutes: float) -> list[dict[str, Any]]:
    basal = numeric_series(frame, args.basal_column).fillna(0.0)
    bolus = numeric_series(frame, args.bolus_column).fillna(0.0)
    combined = numeric_series(frame, args.insulin_column).fillna(0.0)
    has_split_columns = args.basal_column in frame.columns or args.bolus_column in frame.columns
    events: list[dict[str, Any]] = []

    for index in range(len(frame)):
        row_events: list[tuple[str, float]] = []
        if has_split_columns:
            if basal.iloc[index] > 0:
                row_events.append(("basal_insulin", float(basal.iloc[index])))
            if bolus.iloc[index] > 0:
                row_events.append(("bolus_insulin", float(bolus.iloc[index])))
        elif combined.iloc[index] > 0:
            row_events.append(("combined_insulin", float(combined.iloc[index])))

        for insulin_type, dosage in row_events:
            event = event_base(index, sampling_minutes)
            event.update(
                {
                    "dosage": format_number(dosage),
                    "insulin_type": insulin_type,
                    "insulin_mUmin": format_number(dosage * 1000.0 / sampling_minutes),
                }
            )
            events.append(event)

    return events


def extract_exercise_events(frame: pd.DataFrame, sampling_minutes: float) -> list[dict[str, Any]]:
    if "workout_duration" not in frame.columns and "workout_label" not in frame.columns:
        return []

    durations = numeric_series(frame, "workout_duration").fillna(0.0)
    intensities = numeric_series(frame, "workout_intensity", default=np.nan)
    events: list[dict[str, Any]] = []

    for index in range(len(frame)):
        label = non_empty_label(
            frame.iloc[index]["workout_label"] if "workout_label" in frame.columns else None,
            "",
        )
        duration = float(durations.iloc[index])
        if duration <= 0 and not label:
            continue
        if duration <= 0:
            duration = sampling_minutes

        intensity_value = intensities.iloc[index]
        magnitude = float(intensity_value) if not pd.isna(intensity_value) and intensity_value > 0 else 1.0
        exercise_type = label if label else "exercise"

        event = event_base(index, sampling_minutes)
        event.update(
            {
                "duration": format_number(duration),
                "magnitude": format_number(magnitude),
                "exercise_type": exercise_type,
            }
        )
        events.append(event)

    return events


def build_insulin_mumin(frame: pd.DataFrame, args: argparse.Namespace, sampling_minutes: float) -> pd.Series:
    if args.insulin_column in frame.columns:
        insulin_units = numeric_series(frame, args.insulin_column).fillna(0.0)
    else:
        insulin_units = (
            numeric_series(frame, args.basal_column).fillna(0.0)
            + numeric_series(frame, args.bolus_column).fillna(0.0)
        )
    return insulin_units * 1000.0 / sampling_minutes


def prepare_hum_data(parquet_path: Path, args: argparse.Namespace) -> PreparedHumData:
    source_path = parquet_path.expanduser().resolve()
    frame = read_parquet(source_path)
    require_columns(frame, [args.date_column, args.cgm_column])

    frame = frame.copy()
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    frame = frame[frame[args.date_column].notna()].sort_values(args.date_column)
    if frame.empty:
        raise ValueError("No rows with valid timestamps were found in the HUM parquet.")

    subject_ids = sorted_subject_ids(frame, args.id_column)

    for column in (
        args.cgm_column,
        args.carbs_column,
        args.insulin_column,
        args.basal_column,
        args.bolus_column,
        "workout_duration",
        "workout_intensity",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = aggregate_duplicate_timestamps(frame, args)
    sampling_minutes = args.sampling_minutes or infer_sampling_minutes(frame[args.date_column])
    start = frame[args.date_column].min()
    end = frame[args.date_column].max()
    timeline = build_timeline(start, end, sampling_minutes)

    indexed = frame.set_index(args.date_column).reindex(timeline)
    indexed.index.name = args.date_column

    raw_cgm = numeric_series(indexed, args.cgm_column)
    interpolated_cgm = raw_cgm.interpolate(method="linear", limit_direction="both")
    faults_label = ["missing_signal" if pd.isna(value) else "None" for value in raw_cgm.tolist()]
    insulin_mumin = build_insulin_mumin(indexed, args, sampling_minutes)

    return PreparedHumData(
        source_path=source_path,
        timeline=timeline,
        frame=indexed,
        sampling_minutes=float(sampling_minutes),
        hum_subject_id=display_subject_id(subject_ids),
        hum_subject_ids=subject_ids,
        raw_cgm_mgdl=raw_cgm,
        interpolated_cgm_mgdl=interpolated_cgm,
        faults_label=faults_label,
        insulin_mumin=insulin_mumin,
        carb_events=extract_carb_events(indexed, args, sampling_minutes),
        insulin_events=extract_insulin_events(indexed, args, sampling_minutes),
        exercise_events=extract_exercise_events(indexed, sampling_minutes),
    )


def event_dict(
    events: list[tuple[float, float, float | None]],
    with_duration: bool,
) -> dict[str, list[list[float]]]:
    magnitudes = [[format_number(event[0]) for event in events]]
    start_times = [[format_number(event[1]) for event in events]]
    durations = [[format_number(event[2]) for event in events if event[2] is not None]]
    return {
        "magnitude": magnitudes,
        "start_time": start_times,
        "duration": durations if with_duration else [],
    }


def empty_event(with_duration: bool) -> dict[str, list[list[float]]]:
    return {
        "magnitude": [[]],
        "start_time": [[]],
        "duration": [[]] if with_duration else [],
    }


def constant_event(value: float) -> dict[str, list[list[float]]]:
    return {
        "magnitude": [[format_number(value)]],
        "start_time": [[0.0]],
        "duration": [],
    }


def build_simulation_settings(prepared: PreparedHumData, output_dir: Path, seed: int) -> dict[str, Any]:
    carb_events = [
        (event["carbs"], event["time"], prepared.sampling_minutes)
        for event in prepared.carb_events
    ]
    basal_events = [
        (event["dosage"], event["time"], None)
        for event in prepared.insulin_events
        if event["insulin_type"] == "basal_insulin"
    ]
    bolus_events = [
        (event["dosage"], event["time"], prepared.sampling_minutes)
        for event in prepared.insulin_events
        if event["insulin_type"] in {"bolus_insulin", "combined_insulin"}
    ]

    running_events: list[tuple[float, float, float | None]] = []
    cycling_events: list[tuple[float, float, float | None]] = []
    for event in prepared.exercise_events:
        destination = cycling_events if "cycl" in event["exercise_type"].lower() else running_events
        destination.append((event["magnitude"], event["time"], event["duration"]))

    cgm_observed = int(prepared.raw_cgm_mgdl.notna().sum())
    cgm_total = len(prepared.raw_cgm_mgdl)
    cgm_fraction = cgm_observed / cgm_total if cgm_total else 0.0

    return {
        "settings": {
            "save_directory": str(output_dir),
            "start_time": 0,
            "end_time": format_number(len(prepared.timeline) * prepared.sampling_minutes),
            "random_seed": int(seed),
            "random_state": {
                "source": "prepare_hum_artifacts.py",
                "source_parquet": str(prepared.source_path),
                "hum_subject_id": prepared.hum_subject_id,
                "hum_subject_ids": prepared.hum_subject_ids,
                "date_range": {
                    "start": prepared.timeline[0].isoformat(),
                    "end": prepared.timeline[-1].isoformat(),
                },
                "cgm_coverage": {
                    "observed": cgm_observed,
                    "total": cgm_total,
                    "fraction": format_number(cgm_fraction, digits=6),
                },
                "insulin_input_units": (
                    "mU/min derived from HUM insulin units per sample as units * 1000 / sampling_minutes"
                ),
            },
            "sampling_time": format_number(prepared.sampling_minutes),
            "solver_name": "N/A",
            "simulator_name": "SingleScaleSolver",
        },
        "input_generation": {},
        "inputs": {
            "meal_carb": event_dict(carb_events, with_duration=True),
            "snack_carb": empty_event(with_duration=True),
            "sgl2i": None,
            "basal_insulin": event_dict(basal_events, with_duration=False),
            "bolus_insulin": event_dict(bolus_events, with_duration=True),
            "bodyweighteffect": None,
            "iob": constant_event(0.0),
            "heart_rate": constant_event(70.0),
            "taud": None,
            "running_speed": event_dict(running_events, with_duration=True),
            "running_incline": empty_event(with_duration=True),
            "cycling_power": event_dict(cycling_events, with_duration=True),
            "METACSM": constant_event(0.0),
            "energy_expenditure": constant_event(0.0),
            "daily_energy_intake": None,
            "daily_energy_expenditure": None,
            "daily_urinary_glucose_excretion": None,
        },
        "controller": {
            "name": "HUMRealData",
            "parameters": [],
        },
        "patient": {
            "demographic_info": {},
            "number_of_subjects": 1,
            "model": {
                "name": "real_hum",
                "parameters": None,
                "initial_conditions": None,
            },
            "mscale": {
                "models": [],
                "parameters": None,
            },
            "files": [],
        },
    }


def build_prediction_record(prepared: PreparedHumData) -> dict[str, Any]:
    cgm_observed = int(prepared.raw_cgm_mgdl.notna().sum())
    cgm_total = len(prepared.raw_cgm_mgdl)
    cgm_fraction = cgm_observed / cgm_total if cgm_total else 0.0

    return {
        "patient_id": PATIENT_ID,
        "hum_subject_id": prepared.hum_subject_id,
        "hum_subject_ids": prepared.hum_subject_ids,
        "source_parquet": str(prepared.source_path),
        "date_range": {
            "start": prepared.timeline[0].isoformat(),
            "end": prepared.timeline[-1].isoformat(),
        },
        "sampling_minutes": format_number(prepared.sampling_minutes),
        "cgm_coverage": {
            "observed": cgm_observed,
            "total": cgm_total,
            "fraction": format_number(cgm_fraction, digits=6),
        },
        "bg_mgdl": json_list_from_series(prepared.interpolated_cgm_mgdl, digits=3),
        "bg_mgdl_raw": json_list_from_series(prepared.raw_cgm_mgdl, digits=3),
        "faults_label": prepared.faults_label,
        "carb_events": prepared.carb_events,
        "insulin_events": prepared.insulin_events,
        "exercise_events": prepared.exercise_events,
        "insulin_mUmin": {
            "magnitude": json_list_from_series(prepared.insulin_mumin, digits=3),
        },
    }


def write_model_state_results(prepared: PreparedHumData, output_dir: Path) -> None:
    model_state = pd.DataFrame(
        {
            "IG (mmol/L)": prepared.raw_cgm_mgdl / 18.0,
            "faults_label": prepared.faults_label,
        }
    )
    if importlib.util.find_spec("xlsxwriter") is not None:
        engine = "xlsxwriter"
    elif importlib.util.find_spec("openpyxl") is not None:
        engine = "openpyxl"
    else:
        raise SystemExit(
            "Writing model_state_results.xlsx requires xlsxwriter or openpyxl. "
            "Install one Excel writer engine in this environment and rerun the script."
        )

    with pd.ExcelWriter(output_dir / "model_state_results.xlsx", engine=engine) as writer:
        model_state.to_excel(writer, sheet_name=PATIENT_ID)


def write_csv_artifacts(prepared: PreparedHumData, output_dir: Path) -> None:
    pd.DataFrame({"0": prepared.insulin_mumin}).to_csv(
        output_dir / "insulin_input.csv",
        index=False,
    )
    pd.DataFrame({"0": np.zeros(len(prepared.timeline), dtype=float)}).to_csv(
        output_dir / "iob.csv",
        index=False,
    )


def write_json_artifacts(prepared: PreparedHumData, output_dir: Path, seed: int) -> None:
    settings = build_simulation_settings(prepared, output_dir, seed)
    with (output_dir / "simulation_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=4, allow_nan=False)
        handle.write("\n")

    jsonl_path = output_dir / PREDICTION_JSONL_RELATIVE_PATH
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(build_prediction_record(prepared), allow_nan=False) + "\n")


def write_artifacts(prepared: PreparedHumData, output_dir: Path, seed: int) -> None:
    write_model_state_results(prepared, output_dir)
    write_csv_artifacts(prepared, output_dir)
    write_json_artifacts(prepared, output_dir, seed)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args)
    prepare_output_dir(output_dir, args.overwrite)
    prepared = prepare_hum_data(args.parquet_path, args)
    write_artifacts(prepared, output_dir, args.seed)
    print(f"Wrote HUM artifacts to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
