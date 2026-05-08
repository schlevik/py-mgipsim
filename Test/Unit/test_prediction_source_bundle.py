import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from QAdataGeneration.prediction.generate_prediction_qa_pairs import (
    SAMPLES_PER_DAY,
    find_late_night_snack_index,
)
from QAdataGeneration.prediction.source_bundle import (
    apply_prediction_day_specific_overrides,
    locate_day_specific_event_index,
    normalize_day_specific_overrides,
    split_prediction_scenario_overrides,
)
from pymgipsim.Utilities.Scenario import input_generation


PATIENT_COUNT = 2
DAY_COUNT = 30


def daily_events(times_by_type):
    events = []
    for day_index in range(DAY_COUNT):
        day_start = day_index * 1440
        events.extend(day_start + time for time in times_by_type)
    return events


def repeated_daily_values(values_by_type):
    values = []
    for _ in range(DAY_COUNT):
        values.extend(values_by_type)
    return values


def repeated_rows(row):
    return [deepcopy(row) for _ in range(PATIENT_COUNT)]


def make_completed_prediction_scenario():
    meal_times = daily_events([420, 780, 1140])
    snack_times = daily_events([600, 960])
    meal_magnitudes = repeated_daily_values([40, 60, 80])
    snack_magnitudes = repeated_daily_values([10, 20])
    meal_durations = [30] * len(meal_times)
    snack_durations = [10] * len(snack_times)
    rng = np.random.default_rng(123)

    return SimpleNamespace(
        settings=SimpleNamespace(
            random_seed=123,
            random_state=deepcopy(rng.bit_generator.state),
        ),
        input_generation=input_generation(),
        inputs=SimpleNamespace(
            meal_carb=SimpleNamespace(
                start_time=repeated_rows(meal_times),
                magnitude=repeated_rows(meal_magnitudes),
                duration=repeated_rows(meal_durations),
            ),
            snack_carb=SimpleNamespace(
                start_time=repeated_rows(snack_times),
                magnitude=repeated_rows(snack_magnitudes),
                duration=repeated_rows(snack_durations),
            ),
        ),
    )


class TestPredictionDaySpecificOverrides(unittest.TestCase):
    def test_overeating_lunch_changes_only_day_30_lunch_magnitude(self):
        scenario = make_completed_prediction_scenario()
        original = deepcopy(scenario.inputs)
        override = normalize_day_specific_overrides(
            [
                {
                    "target": "meal_carb",
                    "event_type": "lunch",
                    "day": 30,
                    "field": "magnitude",
                    "value_range": [200],
                }
            ]
        )

        mutated_fields = apply_prediction_day_specific_overrides(scenario, override, PATIENT_COUNT)

        self.assertEqual(mutated_fields, {("meal_carb", "magnitude")})
        for patient_index in range(PATIENT_COUNT):
            target_index = locate_day_specific_event_index(
                scenario, "meal_carb", 30, "lunch", patient_index
            )
            self.assertEqual(scenario.inputs.meal_carb.magnitude[patient_index][target_index], 200.0)

            expected_magnitudes = deepcopy(original.meal_carb.magnitude[patient_index])
            expected_magnitudes[target_index] = 200.0
            self.assertEqual(scenario.inputs.meal_carb.magnitude[patient_index], expected_magnitudes)
            self.assertEqual(scenario.inputs.meal_carb.start_time, original.meal_carb.start_time)
            self.assertEqual(scenario.inputs.meal_carb.duration, original.meal_carb.duration)
            self.assertEqual(scenario.inputs.snack_carb.start_time, original.snack_carb.start_time)
            self.assertEqual(scenario.inputs.snack_carb.magnitude, original.snack_carb.magnitude)
            self.assertEqual(scenario.inputs.snack_carb.duration, original.snack_carb.duration)

    def test_late_night_snack_changes_day_29_pm_snack_and_window_reaches_day_30(self):
        scenario = make_completed_prediction_scenario()
        original = deepcopy(scenario.inputs)
        override = normalize_day_specific_overrides(
            [
                {
                    "target": "snack_carb",
                    "event_type": "pm_snack",
                    "day": 29,
                    "field": "start_time",
                    "value_range": [1380],
                }
            ]
        )

        mutated_fields = apply_prediction_day_specific_overrides(scenario, override, PATIENT_COUNT)

        self.assertEqual(mutated_fields, {("snack_carb", "start_time")})
        expected_time = (29 - 1) * 1440 + 1380
        for patient_index in range(PATIENT_COUNT):
            target_index = (29 - 1) * 2 + 1
            self.assertEqual(scenario.inputs.snack_carb.start_time[patient_index][target_index], expected_time)

            expected_start_times = deepcopy(original.snack_carb.start_time[patient_index])
            expected_start_times[target_index] = float(expected_time)
            self.assertEqual(scenario.inputs.snack_carb.start_time[patient_index], expected_start_times)
            self.assertEqual(scenario.inputs.snack_carb.magnitude, original.snack_carb.magnitude)
            self.assertEqual(scenario.inputs.snack_carb.duration, original.snack_carb.duration)
            self.assertEqual(scenario.inputs.meal_carb.start_time, original.meal_carb.start_time)
            self.assertEqual(scenario.inputs.meal_carb.magnitude, original.meal_carb.magnitude)
            self.assertEqual(scenario.inputs.meal_carb.duration, original.meal_carb.duration)

        carb_events = [{"meal_type": "afternoon_snack", "time": expected_time}]
        snack_idx = find_late_night_snack_index(carb_events, intervention_day=29)
        self.assertGreater(snack_idx + 60, 29 * SAMPLES_PER_DAY)
        self.assertLess(snack_idx + 60, DAY_COUNT * SAMPLES_PER_DAY)

    def test_invalid_day_specific_override_configs_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "1-based day"):
            normalize_day_specific_overrides(
                [
                    {
                        "target": "meal_carb",
                        "event_type": "lunch",
                        "day": 31,
                        "field": "magnitude",
                        "value_range": [160, 240],
                    }
                ]
            )

        with self.assertRaisesRegex(ValueError, "event_type"):
            normalize_day_specific_overrides(
                [
                    {
                        "target": "snack_carb",
                        "event_type": "lunch",
                        "day": 29,
                        "field": "start_time",
                        "value_range": [1320, 1440],
                    }
                ]
            )

        with self.assertRaisesRegex(ValueError, "lower bound"):
            normalize_day_specific_overrides(
                [
                    {
                        "target": "meal_carb",
                        "event_type": "lunch",
                        "day": 30,
                        "field": "magnitude",
                        "value_range": [240, 160],
                    }
                ]
            )

    def test_split_rejects_unsupported_real_input_fields(self):
        scenario = make_completed_prediction_scenario()

        with self.assertRaisesRegex(ValueError, "Unsupported prediction override field"):
            split_prediction_scenario_overrides(scenario, {"not_a_real_input_field": [1, 2]})


if __name__ == "__main__":
    unittest.main()
