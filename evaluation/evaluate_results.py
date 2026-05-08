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
from sklearn.metrics import mean_absolute_error
import logging
import ast
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import affiliation metrics
from util.affiliation.metrics import pr_from_events
from dotenv import load_dotenv
import math
# Load environment variables from .env file
load_dotenv()
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
    
    def normalize_answer(self, answer: Any) -> Any:
        """Normalize answers for comparison."""
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
        return [(interval['start'], interval['end']) for interval in intervals]

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
            pred_set = set(tuple(item.items()) if isinstance(item, dict) else item for item in pred_norm)
            exp_set = set(tuple(item.items()) if isinstance(item, dict) else item for item in exp_norm)
            
            intersection = len(pred_set & exp_set)
            union = len(pred_set | exp_set)
            return intersection / union if union > 0 else 0.0
        else:
            # For single values, exact match
            return 1.0 if pred_norm == exp_norm else 0.0
    
    def compute_mae(self, predicted: Any, expected: Any) -> Optional[float]:
        """Compute Mean Absolute Error for numeric values."""
        # Handle single numeric values
        try:
            expected = float(expected)
        except:
            raise ValueError(f"Could not convert expected to float: {expected}")

        if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
            return abs(predicted - expected)
        
        # Handle lists of numeric values
        if isinstance(predicted, list) and isinstance(expected, list):
            if len(predicted) != len(expected):
                raise ValueError("Predicted and expected lists have different lengths")
            
            # Check if all elements are numeric
            try:
                pred_nums = [float(x) for x in predicted]
                exp_nums = [float(x) for x in expected]
                return mean_absolute_error(exp_nums, pred_nums)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert predicted or expected to float: {predicted} or {expected}")
                return None
        
        # For other types, return None (not applicable)
        return None

    def compute_smape(self, predicted: Any, expected: Any) -> Optional[float]:
        """Compute Symmetric Mean Absolute Percentage Error for numeric values."""
        
        def to_numeric_list(value: Any) -> Optional[List[float]]:
            if isinstance(value, list):
                try:
                    return [float(x) for x in value]
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert predicted or expected to float: {predicted} or {expected}")
                    return None
            if isinstance(value, (int, float)):
                return [float(value)]
            try:
                return [float(value)]
            except (ValueError, TypeError):
                logger.warning(f"Could not convert predicted or expected to float: {predicted} or {expected}")
                return None

        pred_vals = to_numeric_list(predicted)
        exp_vals = to_numeric_list(expected)

        if pred_vals is None or exp_vals is None or pred_vals == [] or exp_vals == []:
            return None

        if len(pred_vals) != len(exp_vals):
            # Penalize mismatched lengths similarly to MAE
            raise ValueError("Predicted and expected lists have different lengths")

        smape_values = []
        for pred_val, exp_val in zip(pred_vals, exp_vals):
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
        
        if isinstance(predicted, list) and isinstance(expected, list):
            pred_intervals = self.convert_intervals_to_tuples(predicted) if predicted else []
            pred_intervals.sort(key=lambda x: x[0])
            exp_intervals = self.convert_intervals_to_tuples(expected) if expected else []
            exp_intervals.sort(key=lambda x: x[0])
        else:
            return None
        
        # If both are empty, perfect match
        if not pred_intervals and not exp_intervals:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        # If one is empty and the other isn't, no overlap
        if not pred_intervals and exp_intervals:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if pred_intervals and not exp_intervals:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # pred_intervals = self.sanitize_intervals(pred_intervals)
        # exp_intervals = self.sanitize_intervals(exp_intervals)
        
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
    
        predicted = result.get('llm_response_parsed')
        if isinstance(predicted, dict) and 'answer' in predicted:
            predicted = predicted['answer']
        expected = result.get('expected_answer', '')
        metric_type = result.get('metric', '').lower()

        
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

        
        if metric_type == 'smape':
            if math.isnan(expected):
                if predicted == "" or not math.isnan(predicted):
                    print(f"expected is nan for question {result['question_id']} but predicted is not nan, setting score for smape to 2")
                    evaluation["score"] = 2.0 # set max score for smape
                    print("*"*100)
                    return evaluation
                else:
                    print(f"expected is nan for question {result['question_id']} and predicted is also nan, setting score for smape to 0")
                    evaluation["score"] = 0.0 # set min score for smape
                    print("*"*100)
                    return evaluation

        if expected == "":
            raise ValueError(f"No expected answer found for question {result['question_id']}")

        if metric_type == "":
            raise ValueError(f"No metric type found for question {result['question_id']}")
        
        if predicted == "" and metric_type not in numerical_metrics:
            return evaluation

        elif (predicted == "" or predicted is None) and metric_type in numerical_metrics and isinstance(expected, (int, float)):
            print(f"Predicted value is {predicted if predicted is None else 'empty'} for question {result['question_id']} for patient {result.get('patient_id', 'unknown')} where metric type is {metric_type}. Setting predicted value to 0. Expected value is {expected}")
            print("*"*100)
            if expected == 0 or expected == 0.0:
                predicted = 2.0 # set max score for mae and smape
            else:
                predicted = 0.0 # set min score for mae and smape

       
        
        if predicted is not None and expected is None:
            evaluation["score"] = 0
            return evaluation
      
        
        if metric_type in self.metric_functions:
            try:
                score = self.metric_functions[metric_type](predicted, expected)
                evaluation["score"] = score
                evaluation["error"] = None

                if metric_type in numerical_metrics:
                    smape_score = self.compute_smape(predicted, expected)
                    mae_score = self.compute_mae(predicted, expected)
                    evaluation["smape"] = smape_score
                    evaluation["mae"] = mae_score

                if metric_type == "accuracy":
                    random_guess_baseline_score = self.compute_random_guess_baseline(result, expected)
                    evaluation["random_guess_baseline"] = random_guess_baseline_score
            except Exception as e:
                logger.warning(f"Error evaluating question {result['question_id']}: {str(e)}")
                evaluation["score"] = None
                evaluation["error"] = str(e)
                return evaluation

        else:
            
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
                print("expected: ", result['expected_answer'], type(result['expected_answer'])) 
                print("llm_response_parsed: ", result['llm_response_parsed'])
                print("metric: ", result.get('metric', ''))
                print("*"*100)
                raise ValueError(f"Error evaluating question {result['question_id']}: {evaluation['error']}")
            
            if 'smape' in evaluation and evaluation['smape'] is not None:
                    metric_scores['smape'].append(evaluation['smape'])
            
            if 'random_guess_baseline' in evaluation and evaluation['random_guess_baseline'] is not None:
                metric_scores['accuracy_random_guess_baseline'].append(evaluation['random_guess_baseline'])
            
            # score returned from metric is None, so manually set score to 0.0.
            if evaluation.get("score") is None and (evaluation.get("metric_type") != "mae" or evaluation.get("metric_type") != "smape"):
                if evaluation.get("metric_type") == "affinity f-score":
                    evaluation['score'] =  {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                    print("*"*100)
                    print(f"No score available for question {evaluation['question_id']}, therefore setting score to 0.0")
                    print(evaluation)
                    print("*"*100)
                else:
                    evaluation['score'] = 0.0
                    print("*"*100)
                    print(f"No score available for question {evaluation['question_id']}, therefore setting score to 0.0")
                    print(evaluation)
                    print("*"*100)
            
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
