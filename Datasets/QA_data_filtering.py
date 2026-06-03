"""
================================================================================
QA BENCHMARK SAMPLER
================================================================================
Purpose:
  Deterministically samples and filters simulated patient QA data (JSON/JSONL)
  based on cognitive levels (Descriptive, Pattern, Memory) to build balanced
  benchmark datasets for three tasks: Anomaly Detection (AD), Pattern Matching
  (PM), and Prediction (PD). Uses SEED=42 for reproducibility.

Inputs:
  - Raw JSON (QA metadata) and JSONL (QA context records) from simulation folders.
  - White Patient lists: `faulty_patient_list`, `normal_patient_list`.

Sampling Proportion (multiplier_dict):
  - AD: 10 Descriptive, 5 Pattern, 5 Memory (using Faulty dataset).
  - PM: 8 Descriptive, 0 Pattern, 9 Memory (using Normal dataset).
  - PD: 0 Descriptive, 16 Pattern, 0 Memory (using Normal dataset).

Outputs:
  - Generates 6 files in 'final_sampled_benchmark/': a pruned subset JSON (metadata)
    and JSONL (context) file for each of the three tasks.
================================================================================
"""

import json
import os
import random
from collections import defaultdict
import glob

SEED = 42
random.seed(SEED)


def filter_nested_qa_pairs(input_file, output_file, valid_keys):
    """
    Filters the 'qa_pairs' list within each JSONL line.
    valid_keys should be a set of (patient_id, question_id) tuples.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        qa_count = 0
        for line in infile:
            record = json.loads(line)
            p_id = record.get('patient_id')

            # Get the original list of questions
            original_qa_list = record.get('qa_pairs', [])

            # Filter the list: Keep only if (patient_id, question_id) is in valid_keys
            filtered_qa_list = [
                qa for qa in original_qa_list
                if (p_id, qa.get('question_id')) in valid_keys
            ]

            # Update the record with the smaller list
            record['qa_pairs'] = filtered_qa_list

            # Only save the line if there are questions left
            if filtered_qa_list:
                outfile.write(json.dumps(record) + '\n')
                qa_count += len(filtered_qa_list)

        print(f"Filtered {qa_count} questions in jsonl.")


def filter_and_collect_qa_sampled(input_json, input_jsonl, output_dir,
                                  target_patients, multiplier_dict,
                                  task_name, reset_output=False):
    """
    Samples n random patients per cognitive level based on multiplier_dict
    and appends them to a master file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_json_path = os.path.join(output_dir, f"simulated_{task_name}_QA.json")
    out_jsonl_path = os.path.join(output_dir, f"simulated_{task_name}_with_context.jsonl")

    if reset_output:
        if os.path.exists(out_json_path): os.remove(out_json_path)
        if os.path.exists(out_jsonl_path): os.remove(out_jsonl_path)

    # 1. Load the metadata
    with open(input_json, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # 2. Organize patient IDs by cognitive level for this specific task
    # Structure: { 'Descriptive': [p1, p2, p5...], 'Memory': [p3, p4...] }
    level_to_patients = defaultdict(set)
    for item in source_data:
        p_id = item.get('patient_id')
        c_level = item.get('cognitive_level')
        # Only consider patients that are in our allowed target_patients list
        if p_id in target_patients:
            level_to_patients[c_level].add(p_id)

    # 3. Select Random Patients based on multiplier_dict
    # Structure: { 'Descriptive': {'Patient_1', 'Patient_5'...}, 'Memory': {...} }
    selected_samples = {}
    task_rules = multiplier_dict.get(task_name, {})

    for level, count in task_rules.items():
        available_patients = list(level_to_patients.get(level, []))

        # Determine how many to sample (don't exceed what's available)
        sample_size = min(len(available_patients), count)

        if sample_size > 0:
            selected_samples[level] = set(random.sample(available_patients, sample_size))
            print(f"[{task_name}] Sampled {sample_size} patients for {level} level.")
        else:
            selected_samples[level] = set()

    # 4. Identify the specific items to keep
    new_filtered_items = []
    valid_keys = set()

    for item in source_data:
        p_id = item.get('patient_id')
        c_level = item.get('cognitive_level')

        # Check if this patient was one of the ones randomly selected for this level
        if p_id in selected_samples.get(c_level, set()):
            new_filtered_items.append(item)
            valid_keys.add((p_id, item['question_id']))

    # 5. Save JSON (Load, Append, Save)
    master_json_list = []
    if os.path.exists(out_json_path):
        with open(out_json_path, 'r', encoding='utf-8') as f:
            master_json_list = json.load(f)

    master_json_list.extend(new_filtered_items)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(master_json_list, f, indent=2)

    # 6. Save JSONL
    filter_nested_qa_pairs(input_jsonl, out_jsonl_path, valid_keys)

    return len(new_filtered_items)


def aggregate_and_count_real_pm_questions(input_dir, output_file):
    """
    Aggregates nested JSONL files quickly as raw text,
    while keeping an accurate count of total individual questions.
    """
    # 1. Find all target files under the root folder
    search_pattern = os.path.join(input_dir, "HUM*", "**", "QA_pattern_with_context.jsonl")
    files_to_merge = glob.glob(search_pattern, recursive=True)
    files_to_merge.sort()
    if not files_to_merge:
        print(f"No matching files found in {input_dir}")
        return
    print(f"Found {len(files_to_merge)} files to aggregate.")
    count_original_files = 0
    count_total_lines = 0
    count_total_questions = 0
    # 2. Stream lines and calculate total questions on the fly
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in files_to_merge:
            count_original_files += 1
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    clean_line = line.strip()
                    if clean_line:
                        # Write the line as-is
                        outfile.write(clean_line + '\n')
                        count_total_lines += 1
                        # Count how many times "question_id" appears in this string block.
                        # This tells us exactly how many questions are in this line's qa_pairs list.
                        count_total_questions += clean_line.count('"question_id"')
    print(f"\nAggregation Complete!")
    print(f"Processed files: {count_original_files}")
    print(f"Total patient contexts (lines): {count_total_lines}")
    print(f"Total question count: {count_total_questions}")


def aggregate_and_count_real_pd_questions(source_folder, output_filename):
    # Find all .jsonl files in the specified folder
    file_pattern = os.path.join(source_folder, "*.jsonl")
    files_to_merge = glob.glob(file_pattern)
    # Sort files to ensure a consistent merge order
    files_to_merge.sort()
    print(f"Found {len(files_to_merge)} files to merge.")
    count_total_lines = 0
    count_total_questions = 0
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for file_path in files_to_merge:
            # Skip the output file if it's in the same folder to avoid infinite loops
            if os.path.abspath(file_path) == os.path.abspath(output_filename):
                continue
            print(f"Processing: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # Strip whitespace and only write non-empty lines
                    clean_line = line.strip()
                    if clean_line:
                        outfile.write(clean_line + '\n')
                        count_total_lines += 1
                        # Count instances of "question_id" to tally the nested questions efficiently
                        count_total_questions += clean_line.count('"question_id"')
    print(f"\nSuccess! Merged file saved as: {output_filename}")
    print(f"Total lines (patient contexts) written: {count_total_lines}")
    print(f"Total individual questions counted: {count_total_questions}")

if __name__ == '__main__':
    multiplier_dict = {
        'AD': {'Descriptive': 10, 'Pattern': 5, 'Memory': 5},
        'PM': {'Descriptive': 8, 'Pattern': 0, 'Memory': 9},
        'PD': {'Descriptive': 0, 'Pattern': 16, 'Memory': 0}
    }

    # For test
    # multiplier_dict = {
    #     'AD': {'Descriptive': 1, 'Pattern': 1, 'Memory': 1},
    #     'PM': {'Descriptive': 1, 'Pattern': 0, 'Memory': 1},
    #     'PD': {'Descriptive': 0, 'Pattern': 1, 'Memory': 0}
    # }

    input_json_ad = 'simulation_30days_faulty/QAData/QA_ad.json'
    input_jsonl_ad = 'simulation_30days_faulty/QAData/QA_ad_with_context.jsonl'
    input_json_pm = 'simulation_30days_normal_running/QAData/QA_pattern.json'
    input_jsonl_pm = 'simulation_30days_normal_running/QAData/QA_pattern_with_context.jsonl'
    input_json_pd = 'simulation_30days_normal/QAData/QA_prediction.json'
    input_jsonl_pd = 'simulation_30days_normal/QAData/QA_prediction_with_context.jsonl'
    output_dir = 'final_sampled_benchmark'

    normal_patient_list = ["Patient_0", 'Patient_1', "Patient_5", "Patient_6", "Patient_7",
                           "Patient_8", "Patient_9", "Patient_10", "Patient_11", "Patient_12", "Patient_13",
                           "Patient_14", "Patient_15", "Patient_16", "Patient_17", "Patient_18"]
    faulty_patient_list = ["Patient_1", "Patient_5", "Patient_6",
                           "Patient_8","Patient_10", "Patient_11", "Patient_12", "Patient_13",
                           "Patient_14", "Patient_16", "Patient_17", "Patient_18"]

    ad_num = filter_and_collect_qa_sampled(
        input_json_ad, input_jsonl_ad, output_dir,
        faulty_patient_list, multiplier_dict, 'AD', reset_output=True
    )
    print(f'AD questions: {ad_num}')

    # Call for PM Task (reset=False to add to the AD data)
    pm_num = filter_and_collect_qa_sampled(
        input_json_pm, input_jsonl_pm, output_dir,
        normal_patient_list, multiplier_dict, 'PM', reset_output=True
    )
    print(f'PM questions: {pm_num}')

    # Call for PD Task
    pd_num = filter_and_collect_qa_sampled(
        input_json_pd, input_jsonl_pd, output_dir,
        normal_patient_list, multiplier_dict, 'PD', reset_output=True
    )
    print(f'PD questions: {pd_num}')