#!/usr/bin/env python3
"""
Script to evaluate LLM responses against ground truth answers using different metrics.
Supports Accuracy, MAE, SMAPE, and Affinity F-score metrics.
"""

import json
import argparse
import os
import re
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging
import ast
import math
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import affiliation metrics
from util.affiliation.metrics import pr_from_events


class ResultsEvaluator:
    def __init__(self):
        self.metric_functions = {
            'accuracy': self.compute_accuracy,
            'mae': self.compute_mae,
            'smape': self.compute_smape,
            'f1': self.compute_f1,
            'affinity f-score': self.compute_affinity_f1,
            'affinity-f1 score': self.compute_affinity_f1,
            'affinity f1 score': self.compute_affinity_f1
        }

    def worst_affinity_f1_score(self) -> Dict[str, float]:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def load_results(self, results_file: str, is_anomaly: bool = False) -> List[Dict[str, Any]]:
        """Load evaluation results from JSONL file."""
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    dic = json.loads(line.strip())
                    if is_anomaly:
                        if dic['question_id'] == 'ad_18' or dic['question_id'] == 'ad_19':
                            continue
                    results.append(dic)
        return results

    def resolve_prediction(self, result: Dict[str, Any]) -> Any:
        """Return the answer value emitted by current inference producers."""
        parsed = result.get('llm_response_parsed')
        if isinstance(parsed, dict) and 'answer' in parsed:
            return parsed['answer']
        return result.get('predicted_answer')

    def resolve_expected(self, result: Dict[str, Any]) -> Any:
        """Return the ground-truth answer value across old and current files."""
        expected = result.get('expected_answer')
        if expected is not None:
            return expected
        return result.get('original_answer', '')

    def _parse_loose_answer_dict(self, text: str) -> Optional[Any]:
        """Parse model output shaped like `{ answer: ... }`."""
        match = re.fullmatch(r"\{\s*['\"]?answer['\"]?\s*:\s*(.*?)\s*,?\s*\}", text, flags=re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    def extract_answer_field(self, value: Any) -> Any:
        """Unwrap prediction objects that contain an answer field."""
        coerced = self.coerce_answer_value(value)
        if isinstance(coerced, dict) and 'answer' in coerced:
            return self.coerce_answer_value(coerced['answer'])
        return coerced

    def coerce_answer_value(self, value: Any) -> Any:
        """Parse serialized answer values while preserving ordinary text labels."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return ""

            if stripped.lower() in {"nan", "+nan", "-nan"}:
                return math.nan

            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    continue
                if parsed is Ellipsis:
                    continue
                return self.coerce_answer_value(parsed)

            loose_answer = self._parse_loose_answer_dict(stripped)
            if loose_answer is not None:
                return {"answer": self.coerce_answer_value(loose_answer)}

            return stripped

        if isinstance(value, list):
            return [self.coerce_answer_value(item) for item in value]

        if isinstance(value, tuple):
            return [self.coerce_answer_value(item) for item in value]

        if isinstance(value, dict):
            return {key: self.coerce_answer_value(item) for key, item in value.items()}

        return value

    def _is_nan(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False

    def _normalize_numeric_text(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        text = value.strip().translate(str.maketrans({
            "\u060c": ",",  # Arabic comma; models sometimes emit it before a number.
            "\u066b": ".",  # Arabic decimal separator.
            "\u066c": ",",  # Arabic thousands separator.
        }))
        text = text.strip(",")

        number_pattern = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?(?:[eE][+-]?\d+)?"
        colon_prefixed_number = re.fullmatch(rf":\s*({number_pattern})", text)
        if colon_prefixed_number:
            text = colon_prefixed_number.group(1)

        if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?([eE][+-]?\d+)?", text):
            text = text.replace(",", "")

        return text

    def _to_float(self, value: Any, value_name: str, allow_nan: bool = False) -> float:
        value = self._normalize_numeric_text(value)
        if value is None or value == "" or isinstance(value, bool):
            raise ValueError(f"Could not convert {value_name} to float: {value}")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Could not convert {value_name} to float: {value}")

        if not math.isfinite(numeric_value):
            if allow_nan and math.isnan(numeric_value):
                return numeric_value
            raise ValueError(f"Could not convert {value_name} to finite float: {value}")

        return numeric_value

    def _to_numeric_list(self, value: Any, value_name: str, allow_nan: bool = False) -> List[float]:
        value = self.coerce_answer_value(value)
        if isinstance(value, list):
            if not value:
                raise ValueError(f"{value_name} numeric list is empty")
            return [self._to_float(item, value_name, allow_nan=allow_nan) for item in value]
        return [self._to_float(value, value_name, allow_nan=allow_nan)]

    def _is_missing_prediction(self, value: Any) -> bool:
        return value is None or value == ""

    def compute_missing_mae_penalty(self, expected: Any) -> float:
        exp_vals = self._to_numeric_list(expected, "expected", allow_nan=False)
        return sum(abs(exp_val) for exp_val in exp_vals) / len(exp_vals)
    
    def normalize_answer(self, answer: Any) -> Any:
        """Normalize answers for comparison."""
        answer = self.coerce_answer_value(answer)
        if isinstance(answer, str):
            # Handle string answers like "yes"/"no", "Unavailable"
            return answer.lower().strip()
        elif isinstance(answer, (int, float)):
            return answer
        elif isinstance(answer, list):
            # Handle list of intervals or integers
            if all(isinstance(item, dict) and 'start' in item and 'end' in item for item in answer):
                # List of intervals, sort by start time
                return sorted(answer, key=lambda x: x['start'])
            else:
                # List of integers or other values
                return sorted(answer) if all(isinstance(x, (int, float)) for x in answer) else answer
        else:
            return answer
    
    def convert_intervals_to_tuples(self, intervals: List[Dict[str, int]]) -> List[tuple]:
        """Convert list of {start, end} dicts to list of (start, end) tuples."""
        if not intervals:
            return []

        converted = []
        for interval in intervals:
            if isinstance(interval, dict) and 'start' in interval and 'end' in interval:
                start = interval['start']
                end = interval['end']
            elif isinstance(interval, (list, tuple)) and len(interval) == 2:
                start, end = interval
            else:
                raise ValueError(f"Invalid interval value: {interval}")

            converted.append((self._to_float(start, "interval start"), self._to_float(end, "interval end")))

        return converted

    def make_hashable(self, item: Any) -> Any:
        """Convert nested structures to hashable representations for set ops."""
        if isinstance(item, dict):
            return tuple(sorted((k, self.make_hashable(v)) for k, v in item.items()))
        if isinstance(item, list):
            return tuple(self.make_hashable(v) for v in item)
        if isinstance(item, set):
            return tuple(sorted(self.make_hashable(v) for v in item))
        return item

    def _normalize_label_value(self, value: Any) -> Optional[Any]:
        """Normalize label candidates to a hashable form for comparison."""
        if value is None:
            return None
        normalized = self.normalize_answer(value)
        try:
            return self.make_hashable(normalized)
        except TypeError:
            logger.warning(f"Could not normalize label value: {value}")
            return None

    def _extract_options_from_text(self, text: str) -> List[str]:
        """Heuristic extraction of label options embedded in free-form text."""
        if not isinstance(text, str):
            return []

        candidates: List[str] = []

        # Capture quoted tokens (single or double quotes)
        candidates.extend(re.findall(r"'([^']+)'", text))
        candidates.extend(re.findall(r'"([^"]+)"', text))

        # Capture bracketed lists like [option1, option2]
        for bracketed in re.findall(r"\[([^\]]+)\]", text):
            parts = re.split(r",|\\/|\bor\b|\||;", bracketed)
            for part in parts:
                cleaned = part.strip().strip("'\"` ")
                if cleaned:
                    candidates.append(cleaned)

        return candidates

    def extract_label_options(self, result: Dict[str, Any]) -> List[Any]:
        """Collect possible label options from structured fields and text."""
        option_fields = ['options', 'answer_options', 'label_options', 'choices']
        options: List[Any] = []

        for field in option_fields:
            value = result.get(field)
            if isinstance(value, list):
                options.extend(value)

        text_fields = [result.get('answer_instruction'), result.get('question_text')]
        for text in text_fields:
            options.extend(self._extract_options_from_text(text))

        example_answer = result.get('example_answer')
        if isinstance(example_answer, list):
            options.extend(example_answer)
        elif example_answer is not None:
            options.append(example_answer)

        cleaned_options: List[Any] = []
        seen = set()
        for opt in options:
            normalized = self._normalize_label_value(opt)
            if normalized is None:
                continue
            if normalized not in seen:
                seen.add(normalized)
                cleaned_options.append(opt)

        return cleaned_options

    def compute_random_guess_baseline(self, result: Dict[str, Any], expected: Any) -> Optional[float]:
        """Probability of a correct answer under uniform random guessing."""
        if expected is None:
            return None

        if isinstance(expected, list):
            accepted_values = [val for val in expected if val is not None]
        else:
            accepted_values = [expected]

        if not accepted_values:
            return None
        option_candidates = self.extract_label_options(result)
        normalized_options = {self._normalize_label_value(opt) for opt in option_candidates}
        normalized_options.discard(None)

        normalized_accepted = {self._normalize_label_value(val) for val in accepted_values}
        normalized_accepted.discard(None)

        if not normalized_accepted:
            return None

        # Ensure the option set contains all accepted answers
        normalized_options.update(normalized_accepted)

        # Require at least as many total options as accepted answers to avoid degenerate baselines
        if len(normalized_options) < len(normalized_accepted):
            return None

        return len(normalized_accepted) / len(normalized_options)
    
    def compute_accuracy(self, predicted: Any, expected: Any) -> float:
        """Compute accuracy score (exact match for single values, Jaccard for sets)."""
        pred_norm = self.normalize_answer(predicted)
        exp_norm = self.normalize_answer(expected)
        
        if isinstance(pred_norm, list) and isinstance(exp_norm, list):
            # For lists, compute Jaccard similarity (set-based accuracy)
            if not pred_norm and not exp_norm:
                return 1.0  # Both empty
            if not pred_norm or not exp_norm:
                return 0.0  # One empty, one not
            
            # Convert to sets for comparison
            pred_set = {self.make_hashable(item) for item in pred_norm}
            exp_set = {self.make_hashable(item) for item in exp_norm}
            
            intersection = len(pred_set & exp_set)
            union = len(pred_set | exp_set)
            return intersection / union if union > 0 else 0.0
        else:
            # For single values, exact match
            if self._is_nan(pred_norm) and self._is_nan(exp_norm):
                return 1.0
            return 1.0 if pred_norm == exp_norm else 0.0
    
    def compute_mae(self, predicted: Any, expected: Any) -> Optional[float]:
        """Compute Mean Absolute Error for numeric values."""
        pred_vals = self._to_numeric_list(predicted, "predicted", allow_nan=True)
        exp_vals = self._to_numeric_list(expected, "expected", allow_nan=True)

        if len(pred_vals) != len(exp_vals):
            raise ValueError("Predicted and expected lists have different lengths")

        absolute_errors = []
        for pred_val, exp_val in zip(pred_vals, exp_vals):
            pred_nan = math.isnan(pred_val)
            exp_nan = math.isnan(exp_val)
            if pred_nan and exp_nan:
                absolute_errors.append(0.0)
            elif pred_nan or exp_nan:
                raise ValueError(f"Cannot compute MAE with NaN mismatch: predicted={predicted}, expected={expected}")
            else:
                absolute_errors.append(abs(pred_val - exp_val))

        return sum(absolute_errors) / len(absolute_errors)

    def compute_smape(self, predicted: Any, expected: Any) -> Optional[float]:
        """Compute Symmetric Mean Absolute Percentage Error for numeric values."""
        pred_vals = self._to_numeric_list(predicted, "predicted", allow_nan=True)
        exp_vals = self._to_numeric_list(expected, "expected", allow_nan=True)

        if len(pred_vals) != len(exp_vals):
            # Penalize mismatched lengths similarly to MAE
            raise ValueError("Predicted and expected lists have different lengths")

        smape_values = []
        for pred_val, exp_val in zip(pred_vals, exp_vals):
            pred_nan = math.isnan(pred_val)
            exp_nan = math.isnan(exp_val)
            if pred_nan and exp_nan:
                smape_values.append(0.0)
                continue
            if pred_nan or exp_nan:
                smape_values.append(2.0)
                continue

            denom = abs(pred_val) + abs(exp_val)
            if denom == 0:
                smape_values.append(0.0)
            else:
                smape_values.append(2 * abs(pred_val - exp_val) / denom)

        if not smape_values:
            return None

        return float(np.mean(smape_values))

    def compute_f1(self, predicted: Any, expected: Any) -> Optional[float]:
        """Compute F1 score treating predictions/answers as sets."""
        pred_norm = self.normalize_answer(predicted)
        exp_norm = self.normalize_answer(expected)

        def to_set(value: Any) -> set:
            if value is None:
                return set()
            if isinstance(value, list):
                return {self.make_hashable(v) for v in value}
            return {self.make_hashable(value)}

        pred_set = to_set(pred_norm)
        exp_set = to_set(exp_norm)

        if not pred_set and not exp_set:
            return 1.0
        if not pred_set or not exp_set:
            return 0.0

        true_positive = len(pred_set & exp_set)
        precision = true_positive / len(pred_set) if pred_set else 0.0
        recall = true_positive / len(exp_set) if exp_set else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def expand_point_anomalies(self, intervals: List[tuple], min_width: int = 1) -> List[tuple]:
        """Expand point anomalies (start == end) to have minimum width."""
        expanded = []
        for start, end in intervals:
            if start == end:
                # Expand point anomaly to have minimum width
                expanded.append((start, start + min_width))
            else:
                expanded.append((start, end))
        return expanded
    
    def sanitize_intervals(self, intervals: List[tuple]) -> List[tuple]:
        """Ensure each (start, end) pair is ordered and drop non-finite spans."""
        cleaned = []
        for start, end in intervals:
            if start is None or end is None:
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                logger.warning("Dropping non-finite interval (%s, %s)", start, end)
                continue
            if start > end:
                logger.warning("Swapping inverted interval (%s, %s)", start, end)
                start, end = end, start
            cleaned.append((start, end))
        cleaned.sort(key=lambda span: span[0])
        return cleaned

    def merge_overlapping_intervals(self, intervals: List[tuple]) -> List[tuple]:
        if not intervals:
            return []
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
    
    def compute_affinity_f1(self, predicted: Any, expected: Any, trange: Optional[tuple] = None) -> Optional[Dict[str, float]]:
        """Compute Affinity F1 score for interval-based predictions."""
        # Convert predicted and expected to interval tuples
        predicted = self.coerce_answer_value(predicted)
        expected = self.coerce_answer_value(expected)
        
        if not isinstance(expected, list):
            raise ValueError(f"Expected intervals must be a list, got {type(expected).__name__}")

        exp_intervals = self.convert_intervals_to_tuples(expected) if expected else []
        if not isinstance(predicted, list):
            pred_intervals = []
        else:
            try:
                pred_intervals = self.convert_intervals_to_tuples(predicted) if predicted else []
            except ValueError as error:
                logger.warning("Scoring malformed predicted intervals as zero affinity F1: %s", error)
                return self.worst_affinity_f1_score()
        
        # If both are empty, perfect match
        if not pred_intervals and not exp_intervals:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        # If one is empty and the other isn't, no overlap
        if not pred_intervals and exp_intervals:
            return self.worst_affinity_f1_score()
        if pred_intervals and not exp_intervals:
            return self.worst_affinity_f1_score()
        
        pred_intervals = self.sanitize_intervals(pred_intervals)
        exp_intervals = self.sanitize_intervals(exp_intervals)

        if not pred_intervals and not exp_intervals:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not pred_intervals and exp_intervals:
            return self.worst_affinity_f1_score()
        if pred_intervals and not exp_intervals:
            return self.worst_affinity_f1_score()
        
        # Expand point anomalies to avoid the "Cannot manage point anomalies" error
        pred_intervals = self.expand_point_anomalies(pred_intervals)
        exp_intervals = self.expand_point_anomalies(exp_intervals)

        # Merge overlapping intervals
        pred_intervals = self.merge_overlapping_intervals(pred_intervals)
        exp_intervals = self.merge_overlapping_intervals(exp_intervals)
        
        # Determine time range if not provided
        if trange is None:
            all_times = []
            for start, end in pred_intervals + exp_intervals:
                all_times.extend([start, end])
            trange = (min(all_times), max(all_times))
        
        # Compute affiliation metrics
        pr_results = pr_from_events(pred_intervals, exp_intervals, trange)
        
        precision = pr_results['precision']
        recall = pr_results['recall']
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
            
    
    def parse_dict_answer(self, answer_str: str) -> Dict[str, Any]:
        """Parse dict answer string to proper dict."""
        if isinstance(answer_str, dict):
            return answer_str
            
        # Parse string like "'Baseline': '95.8', 'Peak': '165.2', 'Peak_time': '(19, '09:00')'"
        try:
            # Handle special case with tuple in Peak_time
            import re
            
            # First extract Peak_time if it has a tuple format
            peak_time_match = re.search(r"'Peak_time':\s*'\((\d+),\s*'([^']+)'\)'", answer_str)
            if peak_time_match:
                # Replace the tuple format with just the time
                answer_str = re.sub(r"'Peak_time':\s*'\(\d+,\s*'([^']+)'\)'", r"'Peak_time': '\1'", answer_str)
            
            # Now parse the rest
            pairs = re.findall(r"'([^']+)':\s*'([^']+)'", answer_str)
            
            result = {}
            for key, value in pairs:
                # Try to convert numeric values
                try:
                    if '.' in value:
                        result[key] = float(value)
                    elif value.isdigit():
                        result[key] = int(value)
                    else:
                        result[key] = value
                except:
                    logger.warning(f"Could not convert numeric value: {value} for key: {key}")
                    result[key] = value
                    
            return result
        except Exception as e:
            logger.warning(f"Could not parse dict answer: {answer_str}, error: {str(e)}")
            raise ValueError(f"Could not parse dict answer: {answer_str}, error: {str(e)}")
    
    def parse_dict_metric(self, metric_str: str) -> Dict[str, str]:
        """Parse dict metric string to proper dict."""
        if isinstance(metric_str, dict):
            return metric_str
            
        # Parse string like "{'Baseline': 'MAE', 'Peak': 'MAE', 'Peak_time': 'Accuracy'}"
        try:
            # Replace single quotes with double quotes for JSON parsing
            metric_str = metric_str.replace("baseline", "Baseline")
            metric_str = metric_str.replace("peak", "Peak")
            metric_str = metric_str.replace("peak_time", "Peak_time")
            json_str = metric_str.replace("'", '"')
            
            return json.loads(json_str)
        except:
            # Fallback to regex parsing
            import re
            pairs = re.findall(r"'([^']+)':\s*'([^']+)'", metric_str)
            return dict(pairs)
    
    def evaluate_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single result entry."""
    
        predicted = self.extract_answer_field(self.resolve_prediction(result))
        expected = self.coerce_answer_value(self.resolve_expected(result))
        metric_type = str(result.get('metric', '')).lower().strip()

        
        evaluation = {
            "question_id": result['question_id'],
            'patient_id': result.get('patient_id', ''),
            "metric_type": metric_type,
            "predicted": predicted,
            "expected": expected,
            "error": None
        }

        numerical_metrics = ['mae', 'smape']
        
        if (predicted is None) and expected is None:
            evaluation["score"] = 1
            return evaluation

        if expected == "":
            raise ValueError(f"No expected answer found for question {result['question_id']}")

        if metric_type == "":
            raise ValueError(f"No metric type found for question {result['question_id']}")
        
        if predicted is not None and expected is None:
            evaluation["score"] = 0
            return evaluation

        if metric_type == "mae" and self._is_missing_prediction(predicted):
            score = self.compute_missing_mae_penalty(expected)
            evaluation["score"] = score
            evaluation["mae"] = score
            evaluation["error"] = None
            return evaluation
      
        
        if metric_type in self.metric_functions:
            try:
                score = self.metric_functions[metric_type](predicted, expected)
                evaluation["score"] = score
                evaluation["error"] = None

                if metric_type in numerical_metrics:
                    evaluation["smape"] = score if metric_type == "smape" else self.compute_smape(predicted, expected)
                    if metric_type == "mae":
                        evaluation["mae"] = score
                    else:
                        try:
                            evaluation["mae"] = self.compute_mae(predicted, expected)
                        except ValueError as secondary_error:
                            logger.warning(
                                "Could not compute secondary MAE for question %s: %s",
                                result['question_id'],
                                secondary_error,
                            )
                            evaluation["mae"] = None

                # if metric_type == "accuracy":
                #     random_guess_baseline_score = self.compute_random_guess_baseline(result, expected)
                #     evaluation["random_guess_baseline"] = random_guess_baseline_score
            except Exception as e:
                logger.warning(f"Error evaluating question {result['question_id']}: {str(e)}")
                evaluation["score"] = None
                evaluation["error"] = str(e)
                return evaluation

        else:
            print("---" * 100)
            print("in branch 2" , expected, predicted, metric_type)
            print("---" * 100)
            if isinstance(expected, dict) and isinstance(predicted, dict):
                score = 0
                count = 0
                metric_type = ast.literal_eval(metric_type)
                if metric_type.get('baseline') == 'mae':
                    score += self.compute_mae(predicted.get('Baseline'), expected.get('Baseline'))
                    count += 1
                if metric_type.get('peak') == 'mae':
                    score += self.compute_mae(predicted.get('Peak'), expected.get('Peak'))
                    count += 1
                if metric_type.get('peak_time') == 'mae':
                    score += self.compute_mae(predicted.get('Peak_time'), expected.get('Peak_time'))
                    count += 1

                evaluation["score"] = score / count
                evaluation['metric_type'] = 'mae'
                return evaluation
            else:
                raise ValueError(f"Invalid metric type: {metric_type}")
         

        return evaluation
    
    def evaluate_all_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all results and compute summary statistics."""
        evaluations = []
        metric_scores = defaultdict(list)
        
        for result in results:
            evaluation = self.evaluate_single_result(result)
            if evaluation['error'] is not None:
                print("*"*100)
                print("score: ", evaluation.get("score", None))
                print("error: ", evaluation['error'])
                print("result: ", result['question_id']) 
                print("original_answer: ", result.get('original_answer'), type(result.get('original_answer')))
                print("resolved_prediction: ", self.resolve_prediction(result))
                print("metric: ", result.get('metric', ''))
                print("*"*100)
                continue
                raise ValueError(f"Error evaluating question {result['question_id']}: {evaluation['error']}")

            # if 'random_guess_baseline' in evaluation and evaluation['random_guess_baseline'] is not None:
            #     metric_scores['accuracy_random_guess_baseline'].append(evaluation['random_guess_baseline'])

            #print(evaluation)
            evaluations.append(evaluation)
            # Collect scores by metric type
            if evaluation.get("score") is not None:
                metric_type = evaluation["metric_type"]
                score = evaluation["score"]

                # Handle different score types
                if isinstance(score, dict):
                    if "f1" in score:
                        # Affinity F1 score
                        metric_scores[f"{metric_type}_precision"].append(score["precision"])
                        metric_scores[f"{metric_type}_recall"].append(score["recall"])
                        metric_scores[f"{metric_type}_f1"].append(score["f1"])
                else:
                    # Single numeric score
                    metric_scores[metric_type].append(score)


        # Compute summary statistics
        summary = {}
        for metric_name, scores in metric_scores.items():
            if scores:
                summary[metric_name] = {
                    "count": len(scores),
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "median": np.median(scores)
                }
        
        
        return {
            "individual_evaluations": evaluations,
            "summary_statistics": summary,
            "total_questions": len(results),
            "successful_evaluations": len([e for e in evaluations if e.get("score") is not None])
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM results against ground truth')
    parser.add_argument('--input', required=True, help='Input JSONL file with LLM results')
    parser.add_argument('--output', help='Output JSON file with evaluation results')
    parser.add_argument('--detailed', action='store_true', help='Include individual evaluations in output')
    parser.add_argument('--is_anomaly', action='store_true', default=False, required=False, help='Whether to evaluate anomaly predictions or not')
    args = parser.parse_args()

    print("is anomaly evaluation is enabled: ", args.is_anomaly)
    
    # Generate output path if not provided
    if not args.output:
        # Convert input path: data/subfolder/file.jsonl -> output/subfolder/file.jsonl
        input_path = args.input
        if input_path.startswith("output/"):
            output_path = input_path.replace("output/", "results/", 1)
        else:
            # If not in data folder, just prepend output/
            output_path = os.path.join("results", os.path.basename(input_path))
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        args.output = output_path
        print(f"Auto-generated output path: {args.output}")

    
    # Initialize evaluator
    evaluator = ResultsEvaluator()
    
    # Load results
    print(f"Loading results from {args.input}")
    results = evaluator.load_results(args.input, args.is_anomaly)
    print(f"Loaded {len(results)} results")
    
    # Evaluate all results
    print("Computing evaluation metrics...")
    evaluation_results = evaluator.evaluate_all_results(results)
    
    # Prepare output
    output_data = {
        "summary_statistics": evaluation_results["summary_statistics"],
        "total_questions": evaluation_results["total_questions"],
        "successful_evaluations": evaluation_results["successful_evaluations"]
    }
    
    if args.detailed:
        output_data["individual_evaluations"] = evaluation_results["individual_evaluations"]
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Evaluation complete! Results saved to {args.output}")
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total questions: {evaluation_results['total_questions']}")
    print(f"Successful evaluations: {evaluation_results['successful_evaluations']}")
    
    for metric_name, stats in evaluation_results["summary_statistics"].items():
        print(f"\n{metric_name}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")


if __name__ == "__main__":
    main()
