"""
================================================================================================
QA BENCHMARK & CLINICAL GLYCEMIC STATISTICS ANALYZER
================================================================================================

This script serves a dual purpose in evaluating a Type 1 Diabetes (T1D) dataset benchmark:
  A. QA Benchmark Analytics: Parses, aggregates, and evaluates the distribution, complexity,
     and cognitive atomic skills of generated Question-Answer (QA) pairs across different
     tasks (Anomaly Detection, Pattern Matching, Prediction).
  B. Clinical Data Analytics: Processes real-world and simulated Continuous Glucose Monitor
     (CGM) data to compute standard glycemic clinical metrics (GMI, CV, TIR) and generates
     24-hour profile visualizations.

CORE INPUTS
--------------
  - QA Metadata (CSV/JSONL): Contains nested lists of patient QA pairs, cognitive levels,
    and atomic skill structures (e.g., from 'final_sampled_benchmark' folders).
  - Real-World CGM Data (Parquet): Time-series glucose data for clinical baseline analysis.
  - Simulated CGM Data (Excel): Model state results across 'normal' and 'fault-injected'
    simulation runs. Requires unit conversion from IG (mmol/L) to CGM (mg/dL).

CORE OUTPUTS (Saved to 'Datasets/analysis_results/')
-------------------------------------------------------
  - analysis_summary_log.txt
  - 24-Hour Profile Plots (.png)
  - prototype_wordcloud.png

================================================================================================
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import numpy as np
from wordcloud import WordCloud
import os
import glob
import pandas as pd
import json


def generate_prototype_wordcloud(csv_paths, output_image_path='Datasets/prototype_wordcloud.png'):
    """
    Loads multiple CSV metadata files, aggregates the 'question_prototype' column,
    and generates a word cloud based on phrase frequencies.
    """
    all_prototypes = []

    # 1. Load each CSV and extract the target column
    for path in csv_paths:
        try:
            df = pd.read_csv(path)

            # Optional: Inspect columns if you need to double-check names
            # print(f"Columns in {path}: {list(df.columns)}")

            if 'question_prototype' in df.columns:
                # Drop missing values and convert to string
                prototypes = df['question_prototype'].dropna().astype(str)
                all_prototypes.extend(prototypes.tolist())
            else:
                print(f"Warning: 'question_prototype' column not found in {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if not all_prototypes:
        print("No question prototypes found to plot.")
        return

    # 2. Convert to a pandas Series to easily compute exact phrase frequencies
    prototype_series = pd.Series(all_prototypes)
    prototype_series = (
        prototype_series.astype(str)
        .str.replace(r"\n", " ", regex=True)
        .str.replace(r"\r", " ", regex=True)
    )
    frequencies = prototype_series.value_counts().to_dict()

    print(f"Total unique prototypes found: {len(frequencies)}")
    print("Top 5 prototypes by frequency:", list(frequencies.items())[:5])

    # 3. Configure and generate the Word Cloud
    # Using a clean background and distinct colormap for readability
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        max_font_size=70,
        min_font_size=10,
        relative_scaling=0.3,
        colormap='viridis',  # Options: viridis 'plasma', 'inferno', 'coolwarm', etc.
        max_words=100,  # Limit to top 100 to avoid clutter
        prefer_horizontal=0.85  # Keep most text horizontal and readable
    ).generate_from_frequencies(frequencies)

    # 4. Plot using Matplotlib (using subplots to prevent truncation)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # Turn off axis numbers and ticks

    # Clean padding alignment
    plt.tight_layout(pad=0)

    # Save the plot
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Word cloud successfully saved to {output_image_path}")


def question_statistics_analysis_from_nested_jsonl(source_folder):
    """
    Directly aggregates and parses individual question metrics from raw,
    unflattened JSONL files containing a list of objects under 'qa_pairs'.
    """
    # 1. Find all .jsonl files in the target directory
    file_pattern = os.path.join(source_folder, "*.jsonl")
    files_to_process = glob.glob(file_pattern)
    files_to_process.sort()

    if not files_to_process:
        print(f"Error: No .jsonl files found in {source_folder}")
        return None

    print(f"Found {len(files_to_process)} nested JSONL files to analyze.")

    extracted_questions = []
    count_total_lines = 0

    # 2. Extract nested qa_pairs lists line by line
    for file_path in files_to_process:
        task_name = os.path.basename(file_path).upper()
        # Fallback tracking name logic helper
        if 'ANOMALY' in task_name or 'AD' in task_name:
            inferred_task = 'AD'
        elif 'PATTERN' in task_name or 'PM' in task_name:
            inferred_task = 'PM'
        else:
            inferred_task = 'PD'

        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                clean_line = line.strip()
                if not clean_line:
                    continue

                count_total_lines += 1
                try:
                    record = json.loads(clean_line)
                    qa_list = record.get('qa_pairs', [])

                    # Pull each independent question profile dictionary from the list
                    for qa in qa_list:
                        qa_copy = qa.copy()
                        # Explicitly inject the parent task metadata tag if missing
                        if 'task_source' not in qa_copy:
                            qa_copy['task_source'] = inferred_task
                        extracted_questions.append(qa_copy)

                except json.JSONDecodeError:
                    continue

    if not extracted_questions:
        print("No valid evaluation data found across the target file pathways.")
        return None

    # 3. Convert collected item blocks directly to an expanded analysis dataframe
    combined_df = pd.DataFrame(extracted_questions)

    # --- Data Structural Cleaning ---
    combined_df = combined_df[combined_df['cognitive_atomic'].notna()].copy()

    def clean_atomic_skills(x):
        if isinstance(x, list):
            return sorted(list(set([str(item).strip() for item in x])))
        return sorted(list(set([item.strip() for item in str(x).split(',')])))

    combined_df['cognitive_atomic'] = combined_df['cognitive_atomic'].apply(clean_atomic_skills)
    combined_df['complexity_count'] = combined_df['cognitive_atomic'].str.len()

    distinct_q_ids = combined_df['question_id'].nunique()
    print(f"Total Distinct Question IDs:   {distinct_q_ids}\n")

    # --- Metrics Computation Helper ---
    def get_stats(df, group_col):
        stats = df.groupby(group_col).size().reset_index(name='Total_Questions')
        stats['Percentage (%)'] = (stats['Total_Questions'] / stats['Total_Questions'].sum() * 100).round(2)
        return stats

    # --- 1. Global Metrics View ---
    total_q = len(combined_df)
    print("\n## Evaluation Suite Statistics Summary")
    print(f"Total Patient Records Scanned: {count_total_lines}")
    print(f"Total Nested Questions Evaluated: {total_q}\n")

    def print_metric(df, column, title):
        print(f"### {title}")
        stats = get_stats(df, column)
        print(stats.sort_values(by='Total_Questions', ascending=False).to_string(index=False))
        print("\n")

    # --- 2. Descriptive Breakdown Metrics ---
    print_metric(combined_df, 'task_source', "Distribution by Task Source")
    print_metric(combined_df, 'cognitive_level', "Distribution by Cognitive Level")

    print("### Distribution by Task and Cognitive Level")
    task_level = pd.crosstab(combined_df['task_source'], combined_df['cognitive_level'])
    print(task_level)
    print("\n")

    # --- 3. Cognitive Skill Array Metric Extraction ---
    print("### Cognitive Atomic Skills Distribution")
    df_exploded = combined_df.explode('cognitive_atomic')
    atomic_stats = get_stats(df_exploded, 'cognitive_atomic').sort_values(by='Total_Questions', ascending=False)
    print(atomic_stats.to_string(index=False))
    print("\n")

    # --- 4. Complexity Structural Weights ---
    print("### Complexity Distribution (Atomic Count per Question)")
    complexity_stats = get_stats(combined_df, 'complexity_count')
    print(complexity_stats.to_string(index=False))
    print("\n")

    # --- 5. Inter-Task Correlation Audit Matrix ---
    print("### Skill Overlap by Task (Atomic vs Task Source)")
    cross_tab = pd.crosstab(
        df_exploded['cognitive_atomic'],
        df_exploded['task_source']
    ).fillna(0).astype(int)
    print(cross_tab)

    return combined_df

def question_statistics_analysis(multiplier_dict):
    file_paths = [
        'QAdataGeneration/AnomalyDetection/QA_meta.csv',
        'QAdataGeneration/pattern/QA_meta.csv',
        'QAdataGeneration/prediction/QA_meta.csv'
    ]
    task_names = ['AD', 'PM', 'PD']

    all_dfs = []
    for path, name in zip(file_paths, task_names):
        try:
            df = pd.read_csv(path)
            df['task_source'] = name
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping.")

    if not all_dfs:
        return "No data loaded."

    combined_df = pd.concat(all_dfs, join='inner', ignore_index=True)

    # --- Data Cleaning ---
    combined_df = combined_df[combined_df['cognitive_atomic'].notna()].copy()
    combined_df['cognitive_atomic'] = combined_df['cognitive_atomic'].apply(
        lambda x: sorted(list(set([item.strip() for item in str(x).split(',')])))
    )
    combined_df['complexity_count'] = combined_df['cognitive_atomic'].str.len()


    combined_df['Multiplier'] = combined_df.apply(
        lambda row: multiplier_dict.get(row['task_source'], {}).get(row['cognitive_level'], 0), axis=1
    )

    # --- Analysis Helper ---
    def get_stats(df, group_col):
        stats = df.groupby(group_col)['Multiplier'].sum().reset_index(name='Total_Questions')
        stats['Percentage (%)'] = (stats['Total_Questions'] / stats['Total_Questions'].sum() * 100).round(2)
        return stats

    # --- 1. Global Totals ---
    total_q = combined_df['Multiplier'].sum()
    print("## Benchmark Statistics Summary")
    print(f"Total Generated Questions: {total_q}\n")

    # --- 2. Task & Level Breakdown ---
    # --- Analysis Helper Function ---
    def print_metric(df, column, title):
        print(f"### {title}")
        stats = df.groupby(column)['Multiplier'].sum().reset_index(name='Total_Questions')
        stats['Percentage (%)'] = (stats['Total_Questions'] / stats['Total_Questions'].sum() * 100).round(2)
        print(stats.sort_values(by='Total_Questions', ascending=False).to_string(index=False))
        print("\n")

    # --- 1. Distribution by Task Source ---
    # Shows which task (AD, PM, PD) generates the most volume
    print_metric(combined_df, 'task_source', "Distribution by Task Source")

    # --- 2. Distribution by Cognitive Level ---
    # Shows the balance between Descriptive, Pattern, and Memory across the whole benchmark
    print_metric(combined_df, 'cognitive_level', "Distribution by Cognitive Level")

    print("### Distribution by Task and Cognitive Level")
    task_level = combined_df.groupby(['task_source', 'cognitive_level'])['Multiplier'].sum().unstack(fill_value=0)
    print(task_level)
    print("\n")

    # --- 3. Cognitive Atomic Analysis (Exploded) ---
    print("### Cognitive Atomic Skills Distribution")
    df_exploded = combined_df.explode('cognitive_atomic')
    atomic_stats = get_stats(df_exploded, 'cognitive_atomic').sort_values(by='Total_Questions', ascending=False)
    print(atomic_stats.to_string(index=False))
    print("\n")

    # --- 4. Complexity Analysis ---
    print("### Complexity Distribution (Atomic Count per Question)")
    complexity_stats = get_stats(combined_df, 'complexity_count')
    print(complexity_stats.to_string(index=False))
    print("\n")

    # --- 5. Cross-Task Consistency Check ---
    # This helps see if certain tasks are over-relying on specific atomic skills
    print("### Skill Overlap by Task (Atomic vs Task Source)")
    cross_tab = pd.crosstab(
        df_exploded['cognitive_atomic'],
        df_exploded['task_source'],
        values=df_exploded['Multiplier'],
        aggfunc='sum'
    ).fillna(0).astype(int)
    print(cross_tab)

    return combined_df  # Return for further notebook analysis if needed


def analyze_glycemic_statistics(df):
    """
    Calculates glycemic metrics per patient id and returns a summary table
    consistent with Table 2 of the benchmark image.
    """

    def get_patient_stats(group):
        group = group.dropna(subset=['CGM'])
        # Basic Statistics
        num_points = len(group)
        avg_glucose = group['CGM'].mean()
        std_glucose = group['CGM'].std()
        min_glucose = group['CGM'].min()
        max_glucose = group['CGM'].max()
        cv = (std_glucose / avg_glucose) if avg_glucose != 0 else 0

        # GMI (Glucose Management Indicator) formula: 3.31 + 0.02392 * mean_glucose
        gmi = 3.31 + (0.02392 * avg_glucose)

        # Percent Time in Ranges (TIR)
        # Assuming CGM values are recorded at regular intervals
        tir_70_180 = (group['CGM'].between(70, 180)).mean()
        tar_over_180 = (group['CGM'] > 180).mean()
        tar_over_250 = (group['CGM'] > 250).mean()
        tbr_under_70 = (group['CGM'] < 70).mean()
        tbr_under_54 = (group['CGM'] < 54).mean()

        # Percent time sensor active calculation
        duration_delta = group['date'].max() - group['date'].min()
        duration_minutes = duration_delta.total_seconds() / 60
        # Expected records: duration / 5 minutes
        # Note: We add 1 because the range is inclusive of start and end
        expected_records = (duration_minutes / 5) + 1
        actual_records = len(group)

        sensor_active_pct = (actual_records / expected_records) if expected_records > 0 else 0
        # Cap at 1.0 (100%) in case of overlapping timestamps or high-frequency records
        sensor_active_pct = min(sensor_active_pct, 1.0)

        return pd.Series({
            'Number of data points': num_points,
            'Average glucose (mg/dL)': avg_glucose,
            'Glucose management indicator': gmi,
            'Coefficient of variation': cv,
            'Minimum glucose (mg/dL)': min_glucose,
            'Maximum glucose (mg/dL)': max_glucose,
            'Percent time sensor active': sensor_active_pct,
            'Percent time in range (70-180mg/dL)': tir_70_180,
            'Percent time above range 1 (>180mg/dL)': tar_over_180,
            'Percent time above range 2 (>250mg/dL)': tar_over_250,
            'Percent time below range 1 (<70mg/dL)': tbr_under_70,
            'Percent time below range 2 (<54mg/dL)': tbr_under_54
        })

    # 1. Group by patient ID to get individual stats
    patient_summaries = df.groupby('id').apply(get_patient_stats)

    # 2. Aggregate across all patients to match the Table format (Mean, Min, Max)
    summary_final = pd.DataFrame({
        'Mean (SD)': patient_summaries.apply(lambda x: f"{x.mean():.3f} ({x.std():.3f})"),
        'Min': patient_summaries.min(),
        'Max': patient_summaries.max()
    })

    return summary_final


def real_data_statistics_analysis(base_path):
    file_pattern = os.path.join(base_path, '*.parquet')
    files = glob.glob(file_pattern)

    # Read and combine all files into one DataFrame
    all_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(files)} files with a total of {len(all_df)} rows.")

    # Generate the 24h Plot
    plot_title = "24-Hour Profile of Real Data"
    output_dir = "Datasets/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"{plot_title.replace(' ', '_')}_profile.png")
    plot_24h_profile(all_df, plot_title, plot_filename)

    patient_summary = analyze_glycemic_statistics(all_df)

    return patient_summary


def simulation_data_statistics_analysis(folder_paths, target_patients, plot_title=None):
    """
    Loads specific patient sheets from Excel files across multiple folders
    and runs the glycemic analysis.
    """
    all_simulation_dfs = []
    # Generates a continuous 'date' column starting from 2025-01-01.
    patient_clock = {p_id: datetime(2025, 1, 1, 0, 0, 0) for p_id in target_patients}

    for folder in folder_paths:
        xlsx_path = os.path.join(folder, "model_state_results.xlsx")

        if not os.path.exists(xlsx_path):
            print(f"Warning: {xlsx_path} not found. Skipping.")
            continue

        # Load the Excel file to access sheets
        xl = pd.ExcelFile(xlsx_path)

        for p_id in target_patients:
            if p_id in xl.sheet_names:
                # Read the specific patient sheet
                df = pd.read_excel(xl, sheet_name=p_id, index_col=0)
                num_rows = len(df)

                df["CGM"] = (df["IG (mmol/L)"] * 18).round(3)

                start_time = patient_clock[p_id]
                date_range = pd.date_range(
                    start=start_time,
                    periods=num_rows,
                    freq='5min'
                )

                df['date'] = date_range
                df['id'] = p_id

                all_simulation_dfs.append(df)

                patient_clock[p_id] = date_range[-1] + timedelta(minutes=5)
            else:
                print(f"Note: {p_id} not found in {folder}")

    if not all_simulation_dfs:
        return "No simulation data found for the specified patients."

    # Combine all collected data
    combined_sim_df = pd.concat(all_simulation_dfs, ignore_index=True)
    if plot_title is not None:
        output_dir = "Datasets/analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"{plot_title.replace(' ', '_')}_profile.png")
        plot_24h_profile(combined_sim_df, plot_title, plot_filename)

    # Reuse your existing analysis function
    return analyze_glycemic_statistics(combined_sim_df)


def plot_24h_profile(df, title, save_path):
    """
    Plots the 24-hour glucose profile (Mean + SD) and saves to file.
    """
    # 1. Create a 'minutes since midnight' column for consistent grouping
    df['minutes'] = df['date'].dt.hour * 60 + df['date'].dt.minute

    # 2. Group by time and calculate mean/std
    stats_24h = df.groupby('minutes')['CGM'].agg(['mean', 'std']).reset_index()

    # Smooth the lines if data is noisy
    stats_24h = stats_24h.sort_values('minutes')

    plt.figure(figsize=(10, 6))

    # --- Background Color Ranges (matching your image) ---
    # Hyperglycemic (>180) - Light Pink
    plt.axhspan(180, 250, color='pink', alpha=0.3, label='Hyperglycemia')
    # Target Range (70-180) - White (default)
    # Hypoglycemic (<70) - Light Red
    plt.axhspan(50, 70, color='red', alpha=0.1, label='Hypoglycemia')

    # --- Plot Data ---
    mean = stats_24h['mean']
    std = stats_24h['std']
    minutes = stats_24h['minutes']

    plt.plot(minutes, mean, color='navy', linewidth=2, label='Mean Glucose')
    plt.fill_between(minutes, mean - std, mean + std, color='royalblue', alpha=0.3, label='±1 SD')

    # --- Formatting ---
    plt.title(title, fontsize=14)
    plt.ylabel('Glucose (mg/dL)', fontsize=12)
    plt.xlabel('Time of Day', fontsize=12)
    plt.ylim(50, 250)
    plt.xlim(0, 1435)  # 1440 minutes in a day

    # Set X-axis ticks to show time (00:00, 06:00, etc.)
    tick_minutes = np.arange(0, 1441, 180)  # Every 3 hours
    tick_labels = [f"{int(m / 60):02d}:00" for m in tick_minutes]
    plt.xticks(tick_minutes, tick_labels)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        pass


if __name__ == '__main__':
    # Setup the output folder and log file
    output_folder = "Datasets/analysis_results"
    os.makedirs(output_folder, exist_ok=True)
    log_file_path = os.path.join(output_folder, "analysis_summary_log.txt")

    # csv_files = [
    #     'QAdataGeneration/AnomalyDetection/QA_meta.csv',
    #     'QAdataGeneration/pattern/QA_meta.csv',
    #     'QAdataGeneration/prediction/QA_meta.csv'
    # ]
    # generate_prototype_wordcloud(csv_files, f'{output_folder}/question_diversity_wordcloud.png')

    source_directory = 'Datasets/final_sampled_benchmark/simulation_data'
    print("Simulated QA data statistics analysis")
    analyzed_questions_df = question_statistics_analysis_from_nested_jsonl(source_directory)

    real_source_directory = "Datasets/final_sampled_benchmark/real_world_data"
    print("Real-world QA data statistics analysis")
    analyzed_questions_df_real = question_statistics_analysis_from_nested_jsonl(real_source_directory)

    # Redirect stdout to our Logger
    sys.stdout = Logger(log_file_path)


    multiplier_dict = {
        'AD': {'Descriptive': 10, 'Pattern': 5, 'Memory': 5},
        'PM': {'Descriptive': 8, 'Pattern': 0, 'Memory': 9},
        'PD': {'Descriptive': 0, 'Pattern': 16, 'Memory': 0}
    }
    df = question_statistics_analysis(multiplier_dict)

    real_data_path = "Datasets/real_world_data"
    real_patient_summary = real_data_statistics_analysis(real_data_path)
    print(real_patient_summary.to_string(index=True))

    normal_data_path = ["Datasets/simulation_30days_normal/PredictionSource/SimulationResults/normal_day",
                        "Datasets/simulation_30days_normal_running"]
    faulty_data_path = ['Datasets/simulation_30days_faulty']
    normal_patient_list = ["Patient_0", 'Patient_1', "Patient_5", "Patient_6", "Patient_7",
                           "Patient_8", "Patient_9", "Patient_10", "Patient_11", "Patient_12", "Patient_13",
                           "Patient_14", "Patient_15", "Patient_16", "Patient_17", "Patient_18"]
    faulty_patient_list = ["Patient_1", "Patient_5", "Patient_6",
                           "Patient_8","Patient_10", "Patient_11", "Patient_12", "Patient_13",
                           "Patient_14", "Patient_16", "Patient_17", "Patient_18"]  # "Patient_15", "Patient_9", "Patient_0"
    print(f"Used simulation T1D patient list: {normal_patient_list}")
    simulation_patient_summary_normal = simulation_data_statistics_analysis(normal_data_path, normal_patient_list,
                                                                            plot_title="24-Hour Profile of Simulated Data (Regular)")
    print(simulation_patient_summary_normal.to_string(index=True))

    simulation_patient_summary_faulty = simulation_data_statistics_analysis(faulty_data_path, faulty_patient_list,
                                                                            plot_title="24-Hour Profile of Simulated Data (Fault-injected)")
    print(simulation_patient_summary_faulty.to_string(index=True))

