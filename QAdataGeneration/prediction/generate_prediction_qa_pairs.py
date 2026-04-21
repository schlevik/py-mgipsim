#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SAMPLES_PER_DAY = 24 * 12
DEFAULT_PATIENT_COUNT = 20
QUESTION_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
PROMPT_FILE_INDEX = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
}

QUESTION_METADATA: dict[int, dict[str, str]] = {
    0: {
        "function_name": "get_bg_values",
        "question_id": "pm_0",
        "question_text": "Predict during what time of the day will my blood glucose level be highest?",
        "answer_generation_rule": "The answer should be morning, afternoon, evening or night where morning is from 6AM to 12PM, afternoon is from 12PM to 6PM, evening is from 6PM to 12AM and night is from 12AM to 6AM",
        "answer_instruction": "The answer should be morning, afternoon, evening or night where morning is from 6AM to 12PM, afternoon is from 12PM to 6PM, evening is from 6PM to 12AM and night is from 12AM to 6AM",
        "answer_type": "string",
        "metric": "accuracy",
    },
    1: {
        "function_name": "get_future_bg_values",
        "question_id": "pm_1",
        "question_text": "Predict what my blood glucose level will be in 30 minutes?",
        "answer_generation_rule": "Use the relevant information given and predict a valid blood glucose value",
        "answer_instruction": "Use the relevant information given and predict a valid blood glucose value",
        "answer_type": "float",
        "metric": "MAE",
    },
    2: {
        "function_name": "get_next_week_values",
        "question_id": "pm_2",
        "question_text": "How much insulin will I consume next monday?",
        "answer_generation_rule": "Use the insulin consumed on the 1st, 8th and 15th of the month to predict the total insulin consumption for monday next week.",
        "answer_instruction": "Use the insulin consumed on the 1st, 8th and 15th of the month to predict the total insulin consumption for monday next week.",
        "answer_type": "float",
        "metric": "MAE",
    },
    3: {
        "function_name": "get_spike_late_night",
        "question_id": "pm_3",
        "question_text": "Will having a late night snack now push my sugar to higher limits (>180mg/dL)?",
        "answer_generation_rule": "Consider 5 hours post snack time for checking if the sugar will be higher than 180mg/dL at any point. If you think the sugar will be higher than 180mg/dL, answer 'Yes', otherwise answer 'No'",
        "answer_instruction": "Consider 5 hours post snack time for checking if the sugar will be higher than 180mg/dL at any point. If you think the sugar will be higher than 180mg/dL, answer 'Yes', otherwise answer 'No'",
        "answer_type": "string",
        "metric": "accuracy",
    },
    4: {
        "function_name": "get_next_day_normal_values",
        "question_id": "pm_4",
        "question_text": "what percentage of the day tomorrow will I stay in range (70-180) assuming i take my insulin correctly?",
        "answer_generation_rule": "Make a prediction based on the carb intake, insulin, blood glucose values and exercise events",
        "answer_instruction": "Make a prediction based on the carb intake, insulin, blood glucose values and exercise events",
        "answer_type": "float",
        "metric": "MAE",
    },
    5: {
        "function_name": "check_hypoglycemia",
        "question_id": "pm_5",
        "question_text": "Will my running exercise cause me to become hypoglycemic in the next 90 minutes if i start training now where I will run for 30 minutes?",
        "answer_generation_rule": "Take the current blood glucose values and assume the running exercise event immediately starts and predict if the blood glucose values will be lower than 70mg/dL in the next 90 minutes.If you think the blood glucose values will be lower than 70mg/dL, answer 'Yes', otherwise answer 'No'",
        "answer_instruction": "Take the current blood glucose values and assume the running exercise event immediately starts and predict if the blood glucose values will be lower than 70mg/dL in the next 90 minutes.If you think the blood glucose values will be lower than 70mg/dL, answer 'Yes', otherwise answer 'No'",
        "answer_type": "string",
        "metric": "accuracy",
    },
    6: {
        "function_name": "check_hypoglycemia",
        "question_id": "pm_6",
        "question_text": "What will the glucose level be 1 hour after my breakfast? (Assume I am going to eat my breakfast right away)",
        "answer_generation_rule": "Take the current blood glucose values and assume the carbohydrate intake event immediately starts and predict the glucose level 1 hour after the carbohydrate intake. Return a float number for the glucose level 1 hour after the carbohydrate intake.",
        "answer_instruction": "Take the current blood glucose values and assume the carbohydrate intake event immediately starts and predict the glucose level 1 hour after the carbohydrate intake. Return a float number for the glucose level 1 hour after the carbohydrate intake.",
        "answer_type": "float",
        "metric": "MAE",
    },
    7: {
        "function_name": "check_change_in_glucose",
        "question_id": "pm_7",
        "question_text": "What’s the expected glucose change 15 minutes after a morning run session?",
        "answer_generation_rule": "Assume the running exercise event immediately starts and the running duration is 30 minutes. Predict the change in blood glucose values in the next 15 minutes after the run is completed. Return the change in blood glucose value as a float value",
        "answer_instruction": "Assume the running exercise event immediately starts and the running duration is 30 minutes. Predict the change in blood glucose values in the next 15 minutes after the run is completed. Return the change in blood glucose value as a float value",
        "answer_type": "float",
        "metric": "MAE",
    },
    8: {
        "function_name": "check_change_in_glucose",
        "question_id": "pm_8",
        "question_text": "What’s the expected glucose change 1 hour after a evening cycling session?",
        "answer_generation_rule": "Assume the cycling exercise event starts immedietly and the cycling duration is 20 minutes. Predict the change in blood glucose values in the next 60 minutes after the cycling session.",
        "answer_instruction": "Assume the cycling exercise event starts immedietly and the cycling duration is 20 minutes. Predict the change in blood glucose values in the next 60 minutes after the cycling session.",
        "answer_type": "float",
        "metric": "MAE",
    },
    9: {
        "function_name": "check_better_exercise",
        "question_id": "pm_9",
        "question_text": "Which exercise is better for me to bring down blood sugar tomorrow - cycling or running?",
        "answer_generation_rule": "Assume cycling is for 20 minutes and running is for 30 minutes. Return the exercise type that will bring down blood sugar the most.",
        "answer_instruction": "Assume cycling is for 20 minutes and running is for 30 minutes. Return the exercise type that will bring down blood sugar the most.",
        "answer_type": "string",
        "metric": "accuracy",
    },
    10: {
        "function_name": "check_normal_values",
        "question_id": "pm_10",
        "question_text": "Is the patient likely to remain within 70–180 mg/dL for the rest of the day?",
        "answer_generation_rule": "Take the current glucose, insulin values, food intake and exercise events and predict if the blood glucose values will be lower than 70mg/dL or higher than 180mg/dL at any point from now until the end of the day. If you think the blood glucose values will be between 70mg/dL and 180mg/dL at all points from now until the end of the day, answer 'Yes', otherwise answer 'No'",
        "answer_instruction": "Take the current glucose, insulin values, food intake and exercise events and predict if the blood glucose values will be lower than 70mg/dL or higher than 180mg/dL at any point from now until the end of the day. If you think the blood glucose values will be between 70mg/dL and 180mg/dL at all points from now until the end of the day, answer 'Yes', otherwise answer 'No'",
        "answer_type": "string",
        "metric": "accuracy",
    },
    11: {
        "function_name": "check_max_spike",
        "question_id": "pm_11",
        "question_text": "Given recent habits of having a heavy lunch (160-240) carbs, what is the most likely time of day the patient will experience a glucose spike tomorrow?",
        "answer_generation_rule": "Predict the time index assuming time is between 12AM to 12AM where index goes from 0 to 288 where time is sampled every 5 minutes",
        "answer_instruction": "Predict the time index assuming time is between 12AM to 12AM where index goes from 0 to 288 where time is sampled every 5 minutes",
        "answer_type": "int",
        "metric": "MAE",
    },
    12: {
        "function_name": "check_glucose_trend",
        "question_id": "pm_12",
        "question_text": "Will today’s glucose average be higher than tomorrows?",
        "answer_generation_rule": "Make a prediction on what the average glucose value will be for tomorrow and compare it with the average glucose value for today.",
        "answer_instruction": "Make a prediction on what the average glucose value will be for tomorrow and compare it with the average glucose value for today.",
        "answer_type": "string",
        "metric": "accuracy",
    },
    13: {
        "function_name": "predict_insulin",
        "question_id": "pm_13",
        "question_text": "Will I need a correction insulin dose in the next 2 hours?",
        "answer_generation_rule": "Predict if a user has to take an insulin dose in the next 2 hours.",
        "answer_instruction": "Predict if a user has to take an insulin dose in the next 2 hours.",
        "answer_type": "string",
        "metric": "accuracy",
    },
    14: {
        "function_name": "predict_blood_glucose_swings",
        "question_id": "pm_14",
        "question_text": "Will I experience a sudden glucose swing of more than 60 mg/dL in the next 4 hours?",
        "answer_generation_rule": "Predict if the blood glucose might go up by more than 60 mg/dL or go down by more than 60 mg/dL in the next 4 hours from the current value.",
        "answer_instruction": "Predict if the blood glucose might go up by more than 60 mg/dL or go down by more than 60 mg/dL in the next 4 hours from the current value.",
        "answer_type": "string",
        "metric": "accuracy",
    },
    15: {
        "function_name": "predict_blood_glucose_swings",
        "question_id": "pm_15",
        "question_text": "Based on my current blood glucose levels and insulin dose taken, should I eat a snack to ensure my glucose stays in range between 12AM-6AM",
        "answer_generation_rule": "Predict if the patient should eat a snack to ensure their glucose stays in range between 12AM-6AM",
        "answer_instruction": "Predict if the patient should eat a snack to ensure their glucose stays in range between 12AM-6AM",
        "answer_type": "string",
        "metric": "accuracy",
    },
    17: {
        "function_name": "predict_increase",
        "question_id": "pm_17",
        "question_text": "Predict if my blood sugar will go up faster during lunch time (9AM-1PM) or dinner time (6PM-10PM) tomorrow?",
        "answer_generation_rule": "Predict if adjacent time periods with the highest glucose change will be during lunch time (9-1PM) or dinner time (6-10PM). Return a string value of 'lunch' or 'dinner' where 'lunch' means that the blood sugar will go up faster during lunch time (9-1PM) and 'dinner' means that the blood sugar will go up faster during dinner time (6-10PM).",
        "answer_instruction": "Predict if adjacent time periods with the highest glucose change will be during lunch time (9-1PM) or dinner time (6-10PM). Return a string value of 'lunch' or 'dinner' where 'lunch' means that the blood sugar will go up faster during lunch time (9-1PM) and 'dinner' means that the blood sugar will go up faster during dinner time (6-10PM).",
        "answer_type": "string",
        "metric": "accuracy",
    },
    18: {
        "function_name": "predict_insulin_consumption",
        "question_id": "pm_18",
        "question_text": "Will i need more or less insulin than the average insulin consumption in this coming week.",
        "answer_generation_rule": "Calculate the average insulin consumption for the first 3 weeks and predict the future average insulin consumption for the next week and compare. Return a string value of 'more' or 'less' where 'more' means that the user will need more insulin than average in this coming week and 'less' means that the user will need less insulin than average in this coming week.",
        "answer_instruction": "Calculate the average insulin consumption for the first 3 weeks and predict the future average insulin consumption for the next week and compare. Return a string value of 'more' or 'less' where 'more' means that the user will need more insulin than average in this coming week and 'less' means that the user will need less insulin than average in this coming week.",
        "answer_type": "string",
        "metric": "accuracy",
    },
    19: {
        "function_name": "predict_glucose_drop",
        "question_id": "pm_19",
        "question_text": "if i take insulin now, how much will my blood glucose come down in the next 2 hours?",
        "answer_generation_rule": "Assume that insulin will start working immediately and predict the blood glucose level in the next 2 hours.",
        "answer_instruction": "Assume that insulin will start working immediately and predict the blood glucose level in the next 2 hours.",
        "answer_type": "float",
        "metric": "MAE",
    },
    20: {
        "function_name": "predict_glucose_drop",
        "question_id": "pm_20",
        "question_text": "How many hours of stable glucose levels can I expect overnight? (12AM-6AM)",
        "answer_generation_rule": "Based on the data, predict the number of hours during the 12AM–6AM window when blood glucose will stay within a stable range (70-180 mg/dL). Return an integer value of number of hours where the blood glucose level is stable.",
        "answer_instruction": "Based on the data, predict the number of hours during the 12AM–6AM window when blood glucose will stay within a stable range (70-180 mg/dL). Return an integer value of number of hours where the blood glucose level is stable.",
        "answer_type": "int",
        "metric": "MAE",
    },
}

REQUIRED_METADATA_FIELDS = (
    "function_name",
    "question_id",
    "question_text",
    "answer_generation_rule",
    "answer_instruction",
    "answer_type",
    "metric",
)
ALLOWED_METRICS = {"accuracy", "MAE"}
ALLOWED_ANSWER_TYPES = {"string", "float", "int"}


def convert_np(value):
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def patient_id(patient_index: int) -> str:
    return f"Patient_{patient_index}"


def stable_rng(seed: int, question_index: int, patient_index: int) -> random.Random:
    token = f"{seed}:{question_index}:{patient_index}".encode()
    digest = hashlib.sha256(token).hexdigest()
    return random.Random(int(digest[:16], 16))


def stable_choice(seed: int, question_index: int, patient_count: int) -> int:
    return stable_rng(seed, question_index, patient_count).randrange(patient_count)


def alternate_category(answer, options):
    for option in options:
        if option != answer:
            return option
    raise ValueError(f"No alternate category available for {answer!r}")


def scaled_example(rng: random.Random, answer, min_pct: int = 1, max_pct: int = 50):
    example = answer - ((rng.randint(min_pct, max_pct) / 100) * answer)
    if example == answer:
        return answer + 1
    return example


def mean(values):
    return sum(values) / len(values)


def array_split(values, sections: int):
    size, remainder = divmod(len(values), sections)
    start = 0
    output = []
    for section in range(sections):
        width = size + (1 if section < remainder else 0)
        output.append(values[start : start + width])
        start += width
    return output


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def build_qa(patient_index: int, question_index: int, answer, example_answer):
    metadata = QUESTION_METADATA[question_index]
    return {
        "patient_id": patient_id(patient_index),
        "function_name": metadata["function_name"],
        "question_id": metadata["question_id"],
        "question_text": metadata["question_text"],
        "answer_generation_rule": metadata["answer_generation_rule"],
        "answer_instruction": metadata["answer_instruction"],
        "answer_type": metadata["answer_type"],
        "metric": metadata["metric"],
        "example_answer": example_answer,
        "answer": answer,
    }


def split_exercise_events(exercise_events, cutoff_time: float, inclusive: bool):
    running_events = []
    cycling_events = []
    comparator = (lambda t: t <= cutoff_time) if inclusive else (lambda t: t < cutoff_time)

    for event in exercise_events:
        if not comparator(event["time"]):
            continue
        copied = dict(event)
        if copied["exercise_type"] == "running":
            running_events.append(copied)
        elif copied["exercise_type"] == "cycling":
            cycling_events.append(copied)

    return running_events, cycling_events


def filter_carb_events(carb_events, cutoff_time: float, inclusive: bool, normalize_snacks: bool = False):
    comparator = (lambda t: t <= cutoff_time) if inclusive else (lambda t: t < cutoff_time)
    filtered = []
    for event in carb_events:
        if not comparator(event["time"]):
            continue
        copied = dict(event)
        if normalize_snacks and copied.get("meal_type") in {"morning_snack", "afternoon_snack"}:
            copied["meal_type"] = "snack"
        filtered.append(copied)
    return filtered


def build_input_context(
    data: dict,
    insulin_values: list[float],
    cut_idx: int,
    *,
    include_current: bool,
    exercise_inclusive: bool,
    carb_inclusive: bool,
    normalize_snacks: bool = False,
):
    cut_time = cut_idx * 5
    end_idx = cut_idx + 1 if include_current else cut_idx
    running_events, cycling_events = split_exercise_events(
        data.get("exercise_events", []),
        cut_time,
        inclusive=exercise_inclusive,
    )
    carb_events = filter_carb_events(
        data.get("carb_events", []),
        cut_time,
        inclusive=carb_inclusive,
        normalize_snacks=normalize_snacks,
    )

    return {
        "running_events": running_events,
        "cycling_events": cycling_events,
        "carb_events": carb_events,
        "insulin_events": insulin_values[:end_idx],
        "bg_mgdl": data["bg_mgdl"][:end_idx],
    }


def build_record(patient_index: int, input_context: dict, qa: dict):
    return {
        "patient_id": patient_id(patient_index),
        "input_context": input_context,
        "qa_pairs": [qa],
    }


def find_recent_exercise_index(exercise_events, exercise_type: str, window_anchor_idx: int, add_one: bool = False) -> int:
    anchor_time = window_anchor_idx * 5
    matched_index = None
    for event in exercise_events:
        if event["exercise_type"] != exercise_type:
            continue
        if anchor_time - 120 <= event["time"] <= anchor_time:
            matched_index = int(event["time"] // 5) + (1 if add_one else 0)
    if matched_index is None:
        raise ValueError(f"No {exercise_type} event found near index {window_anchor_idx}")
    return matched_index


def find_carb_index_between(carb_events, start_time: float, end_time: float) -> int:
    matched_index = None
    for event in carb_events:
        if start_time <= event["time"] <= end_time:
            matched_index = int(event["time"] // 5)
            if matched_index * 5 > event["time"]:
                matched_index -= 1
    if matched_index is None:
        raise ValueError(f"No carb event found between {start_time} and {end_time}")
    return matched_index


def find_recent_snack_index(carb_events, cutoff_time: float) -> int:
    snack_index = None
    for event in carb_events:
        if cutoff_time - 120 <= event["time"] < cutoff_time:
            snack_index = int(event["time"] // 5) + 1
    if snack_index is None:
        raise ValueError(f"No snack found before cutoff time {cutoff_time}")
    return snack_index


def max_glucose_slope(bg_values, start_idx: int, end_idx: int):
    max_slope = float("-inf")
    for idx in range(start_idx + 1, end_idx + 1):
        max_slope = max(max_slope, bg_values[idx] - bg_values[idx - 1])
    return max_slope


def choose_better_exercise(bg_diff_cycling: float, bg_diff_running: float) -> str:
    if bg_diff_cycling > 0 and bg_diff_running > 0:
        return "cycling" if bg_diff_cycling < bg_diff_running else "running"
    if bg_diff_cycling > 0 and bg_diff_running < 0:
        return "running"
    if bg_diff_cycling < 0 and bg_diff_running > 0:
        return "cycling"
    return "cycling" if bg_diff_cycling < bg_diff_running else "running"


@dataclass
class GenerationContext:
    root: Path
    insulin_cache: dict[str, list[list[float]]] = field(default_factory=dict)
    simulation_cache: dict[tuple[str, int], dict] = field(default_factory=dict)

    def insulin_values(self, csv_name: str, patient_index: int) -> list[float]:
        if csv_name not in self.insulin_cache:
            with (self.root / csv_name).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                self.insulin_cache[csv_name] = [
                    [float(value) for value in row]
                    for row in reader
                    if row
                ]
        return self.insulin_cache[csv_name][patient_index]

    def simulation_data(self, scenario: str, patient_index: int) -> dict:
        cache_key = (scenario, patient_index)
        if cache_key not in self.simulation_cache:
            path = self.root / "SimulationData" / scenario / f"{scenario}_{patient_index}_simulation_data.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                self.simulation_cache[cache_key] = json.load(handle)
        return self.simulation_cache[cache_key]


def question_0(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    bg_values = data["bg_mgdl"]
    counts = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}

    for day_values in array_split(bg_values, 30):
        max_time_of_day = argmax(day_values) * 5
        if 360 <= max_time_of_day < 720:
            counts["morning"] += 1
        elif 720 <= max_time_of_day < 1080:
            counts["afternoon"] += 1
        elif 1080 <= max_time_of_day < 1440:
            counts["evening"] += 1
        else:
            counts["night"] += 1

    answer = max(counts, key=counts.get)
    qa = build_qa(patient_index, 0, answer, alternate_category(answer, list(counts.keys())))
    input_context = {
        "running_events": [dict(event) for event in data.get("exercise_events", []) if event["exercise_type"] == "running"],
        "cycling_events": [dict(event) for event in data.get("exercise_events", []) if event["exercise_type"] == "cycling"],
        "insulin_events": insulin_values,
        "carb_events": [dict(event) for event in data.get("carb_events", [])],
        "bg_mgdl": bg_values,
    }
    return build_record(patient_index, input_context, qa)


def question_1(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    bg_values = data["bg_mgdl"]

    if len(bg_values) <= 6:
        raise ValueError("Not enough blood glucose values to sample 30 minutes ahead")

    cut_idx = rng.randint(15, len(bg_values) - 7)
    future_idx = cut_idx + 6
    answer = bg_values[future_idx]
    qa = build_qa(patient_index, 1, answer, scaled_example(rng, answer, 15, 45))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=True,
        exercise_inclusive=True,
        carb_inclusive=True,
    )
    return build_record(patient_index, input_context, qa)


def question_2(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    insulin_split = array_split(insulin_values, 30)
    average_3_weeks = (sum(insulin_split[0]) + sum(insulin_split[7]) + sum(insulin_split[14])) / 3
    cut_idx = 15 * SAMPLES_PER_DAY

    qa = build_qa(patient_index, 2, average_3_weeks, scaled_example(rng, average_3_weeks, 15, 45))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_3(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("late_night_snack", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_late_night_snack.csv", patient_index)
    cutoff_time = 21 * SAMPLES_PER_DAY * 5
    snack_idx = find_recent_snack_index(data.get("carb_events", []), cutoff_time)
    end_idx = snack_idx + 60
    bg_window = data["bg_mgdl"][snack_idx : end_idx + 1]
    answer = "Yes" if max(bg_window) > 180 else "No"

    qa = build_qa(patient_index, 3, answer, alternate_category(answer, ["Yes", "No"]))
    input_context = build_input_context(
        data,
        insulin_values,
        snack_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
        normalize_snacks=True,
    )
    return build_record(patient_index, input_context, qa)


def question_4(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    cut_idx = 14 * SAMPLES_PER_DAY
    next_day = data["bg_mgdl"][cut_idx : cut_idx + SAMPLES_PER_DAY]
    count = sum(70 <= value <= 180 for value in next_day)
    answer = (count / len(next_day)) * 100

    qa = build_qa(patient_index, 4, answer, scaled_example(rng, answer, 1, 50))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_5(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    running_idx = find_recent_exercise_index(data.get("exercise_events", []), "running", 10 * SAMPLES_PER_DAY + 84)
    end_idx = running_idx + 18
    bg_window = data["bg_mgdl"][running_idx : end_idx + 1]
    answer = "Yes" if min(bg_window) < 70 else "No"

    qa = build_qa(patient_index, 5, answer, alternate_category(answer, ["Yes", "No"]))
    input_context = build_input_context(
        data,
        insulin_values,
        running_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_6(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    start_idx = 10 * SAMPLES_PER_DAY + 72
    end_search_idx = 10 * SAMPLES_PER_DAY + 96
    carb_idx = find_carb_index_between(data.get("carb_events", []), start_idx * 5, end_search_idx * 5)
    end_idx = carb_idx + 12
    answer = data["bg_mgdl"][end_idx]

    qa = build_qa(patient_index, 6, answer, scaled_example(rng, answer, 1, 50))
    input_context = build_input_context(
        data,
        insulin_values,
        carb_idx,
        include_current=True,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_7(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    running_idx = find_recent_exercise_index(data.get("exercise_events", []), "running", 15 * SAMPLES_PER_DAY + 84)
    end_idx = running_idx + 9
    answer = data["bg_mgdl"][end_idx] - data["bg_mgdl"][running_idx]

    qa = build_qa(patient_index, 7, answer, scaled_example(rng, answer, 1, 50))
    input_context = build_input_context(
        data,
        insulin_values,
        running_idx,
        include_current=True,
        exercise_inclusive=True,
        carb_inclusive=True,
    )
    return build_record(patient_index, input_context, qa)


def question_8(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    cycling_idx = find_recent_exercise_index(data.get("exercise_events", []), "cycling", 15 * SAMPLES_PER_DAY + 216)
    end_idx = cycling_idx + 16
    answer = data["bg_mgdl"][end_idx] - data["bg_mgdl"][cycling_idx]

    qa = build_qa(patient_index, 8, answer, scaled_example(rng, answer, 1, 50))
    input_context = build_input_context(
        data,
        insulin_values,
        cycling_idx,
        include_current=True,
        exercise_inclusive=True,
        carb_inclusive=True,
    )
    return build_record(patient_index, input_context, qa)


def question_9(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    main_idx = 15 * SAMPLES_PER_DAY
    cycling_idx = find_recent_exercise_index(data.get("exercise_events", []), "cycling", 15 * SAMPLES_PER_DAY + 216, add_one=True)
    running_idx = find_recent_exercise_index(data.get("exercise_events", []), "running", 15 * SAMPLES_PER_DAY + 84, add_one=True)
    bg_diff_cycling = data["bg_mgdl"][cycling_idx + 4] - data["bg_mgdl"][cycling_idx]
    bg_diff_running = data["bg_mgdl"][running_idx + 6] - data["bg_mgdl"][running_idx]
    answer = choose_better_exercise(bg_diff_cycling, bg_diff_running)

    qa = build_qa(patient_index, 9, answer, alternate_category(answer, ["cycling", "running"]))
    input_context = build_input_context(
        data,
        insulin_values,
        main_idx,
        include_current=False,
        exercise_inclusive=True,
        carb_inclusive=True,
    )
    return build_record(patient_index, input_context, qa)


def question_10(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    cut_idx = 24 * SAMPLES_PER_DAY + 168
    future_values = data["bg_mgdl"][cut_idx : 25 * SAMPLES_PER_DAY]
    answer = "Yes"
    for value in future_values:
        if value < 70 or value > 180:
            answer = "No"
            break

    qa = build_qa(patient_index, 10, answer, alternate_category(answer, ["Yes", "No"]))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=True,
        carb_inclusive=True,
    )
    return build_record(patient_index, input_context, qa)


def question_11(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("overeating_lunch", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_overeating_lunch.csv", patient_index)
    cut_idx = 24 * SAMPLES_PER_DAY
    day_values = data["bg_mgdl"][cut_idx : 25 * SAMPLES_PER_DAY]
    max_val = 0
    max_ind = 1
    for idx in range(1, len(day_values)):
        change = day_values[idx] - day_values[idx - 1]
        if change > max_val:
            max_val = change
            max_ind = idx
    answer = str(max_ind)
    example = str(int(max_ind - ((rng.randint(1, 50) / 100) * max_ind)))
    if example == answer:
        example = "0" if answer != "0" else "1"

    qa = build_qa(patient_index, 11, answer, example)
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=True,
        carb_inclusive=True,
    )
    return build_record(patient_index, input_context, qa)


def question_12(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    start_idx = 14 * SAMPLES_PER_DAY
    cut_idx = 15 * SAMPLES_PER_DAY
    end_idx = 16 * SAMPLES_PER_DAY
    current_average = mean(data["bg_mgdl"][start_idx:cut_idx])
    next_average = mean(data["bg_mgdl"][cut_idx:end_idx])
    answer = "Yes" if current_average > next_average else "No"

    qa = build_qa(patient_index, 12, answer, alternate_category(answer, ["Yes", "No"]))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=True,
        carb_inclusive=True,
    )
    return build_record(patient_index, input_context, qa)


def question_13(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    if patient_index == 0:
        cut_idx = 7 * SAMPLES_PER_DAY + 80
    elif patient_index == 15:
        cut_idx = 12 * SAMPLES_PER_DAY + 67
    else:
        cut_idx = 10 * SAMPLES_PER_DAY + 5
    end_idx = cut_idx + 24
    answer = "Yes" if any(value > 10 for value in insulin_values[cut_idx : end_idx + 1]) else "No"

    qa = build_qa(patient_index, 13, answer, alternate_category(answer, ["Yes", "No"]))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_14(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    cut_idx = 17 * SAMPLES_PER_DAY + 144
    end_idx = cut_idx + 48
    start_value = data["bg_mgdl"][cut_idx]
    answer = "No"
    for value in data["bg_mgdl"][cut_idx + 1 : end_idx + 1]:
        if value > start_value + 60 or value < start_value - 60:
            answer = "Yes"
            break

    qa = build_qa(patient_index, 14, answer, alternate_category(answer, ["Yes", "No"]))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=True,
        exercise_inclusive=True,
        carb_inclusive=True,
    )
    return build_record(patient_index, input_context, qa)


def question_15(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    cut_idx = 23 * SAMPLES_PER_DAY
    end_idx = cut_idx + 72
    bg_window = data["bg_mgdl"][cut_idx : end_idx + 1]
    answer = "Yes" if any(value < 70 for value in bg_window) else "No"

    qa = build_qa(patient_index, 15, answer, alternate_category(answer, ["Yes", "No"]))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_17(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    cut_idx = 26 * SAMPLES_PER_DAY
    lunch_slope = max_glucose_slope(data["bg_mgdl"], 26 * SAMPLES_PER_DAY + 108, 26 * SAMPLES_PER_DAY + 156)
    dinner_slope = max_glucose_slope(data["bg_mgdl"], 26 * SAMPLES_PER_DAY + 216, 26 * SAMPLES_PER_DAY + 264)
    answer = "Lunch" if lunch_slope > dinner_slope else "Dinner"

    qa = build_qa(patient_index, 17, answer, alternate_category(answer, ["Lunch", "Dinner"]))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_18(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    cut_idx = 21 * SAMPLES_PER_DAY
    insulin_split = array_split(insulin_values, 30)

    week1 = sum(sum(insulin_split[idx]) for idx in range(0, 7)) / 7
    week2 = sum(sum(insulin_split[idx]) for idx in range(7, 14)) / 7
    week3 = sum(sum(insulin_split[idx]) for idx in range(14, 21)) / 7
    baseline_average = (week1 + week2 + week3) / 3
    future_average = sum(sum(insulin_split[idx]) for idx in range(21, 28)) / 7
    answer = "Less" if baseline_average > future_average else "More"

    qa = build_qa(patient_index, 18, answer, alternate_category(answer, ["More", "Less"]))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_19(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    search_start = 13 * SAMPLES_PER_DAY + 72
    first_insulin_offset = None
    for idx in range(1, len(insulin_values[search_start : search_start + SAMPLES_PER_DAY])):
        insulin_window = insulin_values[search_start : search_start + SAMPLES_PER_DAY]
        if insulin_window[idx] - insulin_window[idx - 1] > 80:
            first_insulin_offset = idx
            break
    if first_insulin_offset is None:
        raise ValueError(f"No insulin jump found for patient {patient_index}")

    cut_idx = search_start + first_insulin_offset
    end_idx = cut_idx + 24
    answer = data["bg_mgdl"][cut_idx] - data["bg_mgdl"][end_idx]

    qa = build_qa(patient_index, 19, answer, scaled_example(rng, answer, 13, 65))
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=True,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


def question_20(ctx: GenerationContext, patient_index: int, rng: random.Random):
    data = ctx.simulation_data("normal_day", patient_index)
    insulin_values = ctx.insulin_values("insulin_input_normal.csv", patient_index)
    cut_idx = 13 * SAMPLES_PER_DAY
    end_idx = int(cut_idx + (SAMPLES_PER_DAY / 24) * 6)
    bg_chunks = array_split(data["bg_mgdl"][cut_idx : end_idx + 1], 6)
    unstable_hours = 0
    for chunk in bg_chunks:
        if any(value < 70 or value > 180 for value in chunk):
            unstable_hours += 1
    answer = 6 - unstable_hours
    example = next(value for value in range(0, 7) if value != answer)

    qa = build_qa(patient_index, 20, answer, example)
    input_context = build_input_context(
        data,
        insulin_values,
        cut_idx,
        include_current=False,
        exercise_inclusive=False,
        carb_inclusive=False,
    )
    return build_record(patient_index, input_context, qa)


QUESTION_BUILDERS: dict[int, Callable[[GenerationContext, int, random.Random], dict]] = {
    0: question_0,
    1: question_1,
    2: question_2,
    3: question_3,
    4: question_4,
    5: question_5,
    6: question_6,
    7: question_7,
    8: question_8,
    9: question_9,
    10: question_10,
    11: question_11,
    12: question_12,
    13: question_13,
    14: question_14,
    15: question_15,
    17: question_17,
    18: question_18,
    19: question_19,
    20: question_20,
}


def validate_question_metadata():
    errors = []
    builder_indices = set(QUESTION_BUILDERS)
    metadata_indices = set(QUESTION_METADATA)

    missing_metadata = sorted(builder_indices - metadata_indices)
    if missing_metadata:
        errors.append(f"Missing metadata for question indices: {missing_metadata}")

    unused_metadata = sorted(metadata_indices - builder_indices)
    if unused_metadata:
        errors.append(f"Unused metadata entries for question indices: {unused_metadata}")

    seen_question_ids = {}
    for question_index in sorted(metadata_indices):
        metadata = QUESTION_METADATA[question_index]
        missing_fields = [field for field in REQUIRED_METADATA_FIELDS if not metadata.get(field)]
        if missing_fields:
            errors.append(f"Question {question_index} is missing required metadata fields: {missing_fields}")
            continue

        question_id = metadata["question_id"]
        if question_id in seen_question_ids:
            errors.append(
                f"Duplicate question_id {question_id!r} for questions {seen_question_ids[question_id]} and {question_index}"
            )
        else:
            seen_question_ids[question_id] = question_index

        if metadata["metric"] not in ALLOWED_METRICS:
            errors.append(
                f"Question {question_index} has invalid metric {metadata['metric']!r}; allowed: {sorted(ALLOWED_METRICS)}"
            )

        if metadata["answer_type"] not in ALLOWED_ANSWER_TYPES:
            errors.append(
                f"Question {question_index} has invalid answer_type {metadata['answer_type']!r}; allowed: {sorted(ALLOWED_ANSWER_TYPES)}"
            )

    if errors:
        raise ValueError("Invalid QUESTION_METADATA:\n- " + "\n- ".join(errors))


validate_question_metadata()


def build_prompt_text(repo_root: Path, record: dict) -> str:
    sys.path.insert(0, str(repo_root.parent))
    from loopqa.eval import LoopQAEvaluator

    qa_item = record["qa_pairs"][0]
    prompt_context = LoopQAEvaluator.format_patient_context(record["input_context"])
    return LoopQAEvaluator.create_prompt(
        prompt_context,
        qa_item["question_text"],
        qa_item["answer_instruction"],
        qa_item["answer_type"],
        example_answer=qa_item["example_answer"],
    )


def write_prompt(repo_root: Path, prompts_dir: Path, question_index: int, record: dict):
    prompt_text = build_prompt_text(repo_root, record)
    prompt_index = PROMPT_FILE_INDEX[question_index]
    prompts_dir.mkdir(parents=True, exist_ok=True)
    with (prompts_dir / f"{prompt_index}.txt").open("w", encoding="utf-8") as handle:
        handle.write(prompt_text)


def parse_question_indices(raw_value: str) -> list[int]:
    if raw_value.lower() == "all":
        return QUESTION_ORDER

    selected = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        question_index = int(stripped)
        if question_index not in QUESTION_BUILDERS:
            raise ValueError(f"Question {question_index} is not implemented. Available: {QUESTION_ORDER}")
        selected.append(question_index)
    if not selected:
        raise ValueError("No question indices provided")
    return selected


def combine_outputs(output_dir: Path, combined_output: Path, question_indices: list[int]):
    with combined_output.open("w", encoding="utf-8") as out_handle:
        for question_index in question_indices:
            file_path = output_dir / f"questions_and_answers_{question_index}.jsonl"
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)


def generate_question_files(
    repo_root: Path,
    output_dir: Path,
    patient_count: int,
    seed: int,
    question_indices: list[int],
    write_prompts_flag: bool,
    prompts_dir: Path,
):
    ctx = GenerationContext(repo_root)

    for question_index in question_indices:
        output_path = output_dir / f"questions_and_answers_{question_index}.jsonl"
        sample_patient = stable_choice(seed, question_index, patient_count)
        with output_path.open("w", encoding="utf-8") as handle:
            for patient_index in range(patient_count):
                rng = stable_rng(seed, question_index, patient_index)
                record = QUESTION_BUILDERS[question_index](ctx, patient_index, rng)
                handle.write(json.dumps(record, default=convert_np) + "\n")

                if write_prompts_flag and patient_index == sample_patient:
                    write_prompt(repo_root, prompts_dir, question_index, record)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate prediction QA JSONL files from the notebook logic without using Jupyter.",
    )
    parser.add_argument(
        "--output-dir",
        default="questions",
        help="Directory for generated questions_and_answers_*.jsonl files.",
    )
    parser.add_argument(
        "--combined-output",
        default="questions/combined_prediction.jsonl",
        help="Path for the combined prediction JSONL file.",
    )
    parser.add_argument(
        "--questions",
        default="all",
        help="Comma-separated question indices to generate, or 'all'.",
    )
    parser.add_argument(
        "--patient-count",
        type=int,
        default=DEFAULT_PATIENT_COUNT,
        help="Number of patients to generate per question file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used for deterministic sampling in questions with randomness.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Do not regenerate the combined prediction JSONL file.",
    )
    parser.add_argument(
        "--write-prompts",
        action="store_true",
        help="Also write sample prompts to prompts/prediction if loopqa is available.",
    )
    parser.add_argument(
        "--prompts-dir",
        default="prompts/prediction",
        help="Directory for optional prompt outputs.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir).resolve()
    combined_output = (repo_root / args.combined_output).resolve()
    prompts_dir = (repo_root / args.prompts_dir).resolve()
    question_indices = parse_question_indices(args.questions)

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_question_files(
        repo_root=repo_root,
        output_dir=output_dir,
        patient_count=args.patient_count,
        seed=args.seed,
        question_indices=question_indices,
        write_prompts_flag=args.write_prompts,
        prompts_dir=prompts_dir,
    )

    if not args.skip_combined:
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        combine_outputs(output_dir, combined_output, question_indices)


if __name__ == "__main__":
    main()
