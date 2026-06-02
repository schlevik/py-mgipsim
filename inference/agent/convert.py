#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_FILE = Path(
    "/home/srini/time_series/py-mgipsim/final_sampled_benchmark/sampled/"
    "sampled_questions.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_FILE.parent / "data"
SAMPLE_INTERVAL_MINUTES = 5
MINUTES_PER_DAY = 24 * 60
DAYS_PER_WEEK = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert each sampled question input_context into a CSV file."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"JSONL file to read. Defaults to {DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated CSV files. Defaults to {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N examples.",
    )
    return parser.parse_args()


def nearest_sample_time(raw_time):
    return int(round(float(raw_time) / SAMPLE_INTERVAL_MINUTES) * SAMPLE_INTERVAL_MINUTES)


def week_day_time(timestamp):
    day = timestamp // MINUTES_PER_DAY + 1
    week = (day - 1) // DAYS_PER_WEEK + 1
    day_in_week = (day - 1) % DAYS_PER_WEEK + 1
    minute_of_day = timestamp % MINUTES_PER_DAY
    hour = minute_of_day // 60
    minute = minute_of_day % 60
    return f"W{week}D{day_in_week} {hour:02d}:{minute:02d}"


def format_insulin(value):
    if value == 0:
        return "0.0"
    return f"{value:.1E}"


def format_carbs(value):
    return f"{float(value):.1f}"


def format_plain_number(value):
    return str(value)


def build_insulin_by_time(insulin_events):
    insulin_by_time = defaultdict(float)

    if isinstance(insulin_events, dict):
        for index, dosage in enumerate(insulin_events.get("magnitude", [])):
            timestamp = index * SAMPLE_INTERVAL_MINUTES
            insulin_by_time[timestamp] += float(dosage)
        return insulin_by_time

    for index, event in enumerate(insulin_events or []):
        if isinstance(event, (int, float)):
            timestamp = index * SAMPLE_INTERVAL_MINUTES
            insulin_by_time[timestamp] += float(event)
            continue

        timestamp = nearest_sample_time(event["time"])
        insulin_by_time[timestamp] += float(
            event.get("insulin_mUmin", event.get("dosage", 0.0))
        )
    return insulin_by_time


def build_other_events_by_time(carb_events, exercise_events):
    other_events_by_time = defaultdict(list)

    for event in carb_events:
        timestamp = nearest_sample_time(event["time"])
        meal_type = event.get("meal_type", "carb_event")
        carbs = format_carbs(event.get("carbs", 0.0))
        other_events_by_time[timestamp].append(f"{meal_type} {carbs}g Carbs")

    for event in exercise_events:
        timestamp = nearest_sample_time(event["time"])
        exercise_type = event.get("exercise_type", "exercise")
        duration = format_plain_number(event.get("duration", 0.0))
        magnitude = format_plain_number(event.get("magnitude", 0.0))
        other_events_by_time[timestamp].append(
            f"{exercise_type} {duration}min magnitude {magnitude}"
        )

    return other_events_by_time


def write_context_csv(context, output_file):
    bg_values = context["bg_mgdl"]
    insulin_source = context.get("insulin_mUmin", context.get("insulin_events", []))
    insulin_by_time = build_insulin_by_time(insulin_source)
    exercise_events = context.get("exercise_events")
    if exercise_events is None:
        exercise_events = context.get("running_events", []) + context.get(
            "cycling_events", []
        )
    other_events_by_time = build_other_events_by_time(
        context.get("carb_events", []),
        exercise_events,
    )

    with output_file.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["timestamp", "Week_DAY_TIME", "bg_mgdl", "insulin_mUmin", "other_events"]
        )

        for index, bg_mgdl in enumerate(bg_values):
            timestamp = index * SAMPLE_INTERVAL_MINUTES
            other_events = other_events_by_time.get(timestamp)
            writer.writerow(
                [
                    f"{timestamp:4d}",
                    week_day_time(timestamp),
                    format_plain_number(bg_mgdl),
                    format_insulin(insulin_by_time.get(timestamp, 0.0)),
                    "; ".join(other_events) if other_events else "NA",
                ]
            )


def convert(input_file, output_dir, limit=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    qa_count = 0
            
    with input_file.open("r") as jsonl_file:
        for index, line in enumerate(jsonl_file):
            if limit is not None and converted >= limit:
                break

            record = json.loads(line)
            patient_id = record["patient_id"]
            if "qa_pairs" in record:
                print("QA pairs found")
                for qa in record["qa_pairs"]:
                    if limit is not None and converted >= limit:
                        break
                    question_id = qa["question_id"]
                    output_file = output_dir / f"{patient_id}_question_{question_id}_{qa_count}.csv"
                    write_context_csv(record["input_context"], output_file)
                    converted += 1
                    qa_count += 1
            else:
                question_id = record["question_id"]
                output_file = output_dir / f"{patient_id}_question_{question_id}.csv"
                write_context_csv(record["input_context"], output_file)
                converted += 1

    return converted


def main():
    args = parse_args()
    converted = convert(args.input_file, args.output_dir, args.limit)
    print(f"Converted {converted} examples into {args.output_dir}")


if __name__ == "__main__":
    main()
