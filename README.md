# py-mgipsim QA Pipeline

This repository extends the original py-mgipsim closed-loop simulation testbed with fault injection, randomized scenarios, QA dataset generation, LLM inference, and evaluation. The current workflow is:

1. Generate simulated patient traces, or load an existing artifact folder, with `data_generation_main.py`.
2. Generate QA datasets for anomaly detection (`ad`), pattern recognition (`pm`), and prediction (`pd`).
3. Run model inference through scripts under `inference/`.
4. Score model outputs with `evaluation/evaluate_results.py`.

The simulator uses the Extended Cambridge patient model in a single-scale setting. OpenAPS support is available when the external OpenAPS server is deployed separately.

## Repository Layout

- `data_generation_main.py` - main entrypoint for simulation and QA generation.
- `prepare_hum_artifacts.py` - optional converter from one HUM parquet file into QA-compatible artifacts.
- `QAdataGeneration/` - anomaly detection, pattern recognition, and prediction QA builders.
- `inference/openrouter_inference.py` - direct prompt-based inference through OpenRouter for contextual JSONL QA files.
- `inference/agent/convert.py` - converts contextual QA records into per-question CSV files for agent/code-execution workflows.
- `inference/agent/run_agent.py` - OpenAI Agents SDK workflow using code interpreter and a final answer agent.
- `inference/agent/run_anthropic_agent.py` - Anthropic code-execution workflow.
- `evaluation/evaluate_results.py` - metric runner for model output JSONL files.

## Setup

Install the Python requirements for the simulator and local evaluation:

```bash
pip install -r requirements.txt
```

The provider inference scripts also require their provider SDKs in the active environment, such as OpenAI Agents/OpenAI packages for `run_agent.py`, Anthropic for `run_anthropic_agent.py`, and OpenAI-compatible client support for OpenRouter.

For OpenAPS controller runs, deploy the OpenAPS server from:

```text
https://github.com/ImperialGlobalSingapore/oref0/tree/kexin
```

Then check out the `kexin` branch and run the server before launching OpenAPS simulations.

For LLM inference, create the relevant API-key environment file before running those scripts:

- `inference/.env` with `OPENROUTER_API_KEY=...` for `openrouter_inference.py`.
- `inference/.env` with `OPENAI_API_KEY=...` for `inference/agent/run_agent.py` when run from `inference/agent`.
- `inference/.env` or the active shell environment with `ANTHROPIC_API_KEY=...` for `inference/agent/run_anthropic_agent.py`.

## Data Generation

Use `data_generation_main.py` from the repository root. For the full argument list:

```bash
python data_generation_main.py -h
```

When `--data_path` is omitted, or points to a path that does not exist, the script creates a new folder under `SimulationResults/`, runs initialization, generates settings, virtual subjects, inputs, results, and plots, then optionally generates QA datasets.

When `--data_path` points to an existing artifact folder, the script skips simulation and generates only the requested QA datasets from that folder.

### Basic Simulation

Run a 3-day simulation for one patient with the OpenLoop controller:

```bash
python data_generation_main.py -d 3 -ns 1 -ctrl OpenLoop -rs 402
```

Run a 30-day OpenAPS simulation with 20 patients, random faults, and QA generation:

```bash
python data_generation_main.py \
  -d 30 \
  -ns 20 \
  -ctrl OpenAPS \
  -rfi 0.01 \
  -ft max_basal positive_spike \
  --qa ad pm pd
```

Generate QA from an existing simulation output without re-running simulation:

```bash
python data_generation_main.py \
  --data_path "SimulationResults/Simulation 05_11_2026_02_41_01" \
  --qa ad pm pd
```

Generate only selected anomaly-detection questions:

```bash
python data_generation_main.py \
  --data_path "SimulationResults/Simulation 05_11_2026_02_41_01" \
  --qa ad \
  --ad_ids 1 5 9 46
```

### QA Types

The `--qa` argument accepts one or more values:

- `ad` - anomaly detection questions from `QAdataGeneration/AnomalyDetection/`.
- `pm` - pattern-recognition questions from `QAdataGeneration/pattern/`.
- `pd` - prediction questions from `QAdataGeneration/prediction/`.

Prediction QA first builds a `PredictionSource/` bundle from the active results folder, then generates prediction questions from that bundle.

### Simulation and QA Outputs

Simulation artifacts are saved under:

```text
SimulationResults/<run folder>/
```

Core simulation files include:

- `model_state_results.xlsx`
- `insulin_input.csv`
- `iob.csv`
- `model.pkl`
- `simulation_settings.json`

QA files are written under:

```text
SimulationResults/<run folder>/QAData/
```

Typical QA outputs are:

- `QA_ad.json`
- `QA_ad_with_context.jsonl`
- `QA_pattern.json`
- `QA_pattern_with_context.jsonl`
- `QA_prediction.json`
- `QA_prediction_with_context.jsonl`

The flat `.json` files are useful for agent workflows that expect one question per record. The `*_with_context.jsonl` files preserve patient context plus nested `qa_pairs`; they are useful for direct contextual prompting and for generating per-question CSV files.

### HUM Artifact Preparation

`prepare_hum_artifacts.py` adapts one HUM parquet file into the artifact shape consumed by `data_generation_main.py`. It does not generate QA directly.

```bash
python prepare_hum_artifacts.py \
  hum_data/example.parquet \
  --output-dir SimulationResults/HUM_example \
  --overwrite
```

Then generate QA from the prepared folder:

```bash
python data_generation_main.py \
  --data_path SimulationResults/HUM_example \
  --qa ad pm
```

## Fault Injection

Faults can be injected randomly or from a CSV file.

### Random Faults

Use `-rfi` / `--random_fault_intensity` with a value between `0` and `1`. Restrict the fault set with `-ft` / `--fault_type`:

```bash
python data_generation_main.py \
  -d 30 \
  -ns 20 \
  -ctrl OpenAPS \
  -rfi 0.01 \
  -ft max_basal positive_spike
```

Supported fault types include:

```text
max_basal, min_basal, positive_basal, negative_basal,
unknown_stop, unknown_under, missing_signal, positive_spike,
negative_spike, negative_bias, positive_bias, min_reading,
max_reading, repeated_reading, false_meal, false_bolus,
repeated_episode
```

Random fault injection guarantees every selected fault type appears at least once, prevents overlapping fault periods, limits single-point faults to one sample, and requires `repeated_episode` to last at least two hours.

### Faults From File

Use `--faults_file` / `-ff` to pass a CSV with:

```text
Start_Time, Period, Data Label, Description
```

Example:

```bash
python data_generation_main.py \
  -d 30 \
  -ns 20 \
  -ctrl HCL0 \
  -ff pymgipsim/faultsGeneration/faults_specification.csv
```

`Start_Time` and `Period` are in minutes. `Data Label` is the fault type, such as `negative_spike`.

## Random Scenarios

Scenario randomness is controlled with:

- `-rc` / `--random_scenario`
- `-rsm` / `--random_scenario_methods`
- `-rsi` / `--random_scenario_intensity`

Targets include meal, snack, cycling, and running magnitude, start time, and duration fields. Methods include `heavy`, `light`, `early`, `delayed`, and `skipped`.

Example:

```bash
python data_generation_main.py \
  -d 30 \
  -ns 1 \
  -ctrl HCL0 \
  -rc meal_start_time \
  -rsm early \
  -rsi 0.1
```

## Inference

Inference outputs are JSONL files where each line is one model answer with the original question metadata, `predicted_answer`, token usage, and cost fields where available.

### OpenRouter Direct Inference

`inference/openrouter_inference.py` reads contextual QA JSONL records with `input_context` and nested `qa_pairs`. Run it from the `inference/` directory:

```bash
cd inference

python openrouter_inference.py \
  --qa_file ../SimulationResults/<run folder>/QAData/QA_pattern_with_context.jsonl \
  --output_file ../agent_outputs/<model>_pattern.jsonl \
  --model gemini-3.1-flash-lite-preview
```

The script formats patient context directly into the prompt and writes flattened evaluation-ready records with `expected_answer`, `predicted_answer`, `input_tokens`, `output_tokens`, and `estimated_cost_usd`.

### Agent CSV Conversion

The OpenAI and Anthropic agent workflows use per-question CSV context files. Convert a contextual QA JSONL file before running those agents:

```bash
python inference/agent/convert.py \
  --input-file "SimulationResults/<run folder>/QAData/QA_pattern_with_context.jsonl" \
  --output-dir "SimulationResults/<run folder>/QAData/agent_csv"
```

For records with nested `qa_pairs`, the converter writes files named:

```text
<patient_id>_question_<question_id>_<qa_index>.csv
```

The agent runners pair those CSV files with the flat QA JSON, such as `QA_pattern.json`, `QA_prediction.json`, or `QA_ad.json`.
Use the flat QA JSON and contextual JSONL generated from the same run so the per-question index in the CSV filename matches the flat QA record order.

### OpenAI Agent Workflow

Run `run_agent.py` from `inference/agent` so it can load `prompt.txt` and `../.env` correctly:

```bash
cd inference/agent

python run_agent.py \
  --qa_file ../../SimulationResults/<run folder>/QAData/QA_pattern.json \
  --output_file ../../agent_outputs/<workflow>_<model>.jsonl \
  --data_type PM \
  --file_path ../../SimulationResults/<run folder>/QAData/agent_csv \
  --model gpt-5.4 \
  --workflow_name <workflow>
```

The OpenAI workflow uploads the matching CSV to a code-interpreter container, runs a coding agent, passes its trace to a final answer agent, and writes `predicted_answer`, `message_traces`, token totals, and workflow-level estimated cost.

If SDK tracing upload warnings are noisy but rows are still written, set:

```bash
export OPENAI_AGENTS_DISABLE_TRACING=true
```

### Anthropic Agent Workflow

Run the Anthropic workflow from `inference/agent`:

```bash
cd inference/agent

python run_anthropic_agent.py \
  --qa_file ../../SimulationResults/<run folder>/QAData/QA_pattern.json \
  --output_file ../../agent_outputs/claude_pattern.jsonl \
  --data_type PM \
  --file_path ../../SimulationResults/<run folder>/QAData/agent_csv \
  --model claude-sonnet-4-6
```

This workflow uploads the matching CSV through Anthropic files/code execution, parses a structured final answer, and writes `predicted_answer`, `message_traces`, token usage, and estimated cost.

## Evaluation

Run evaluation from the `evaluation/` directory:

```bash
cd evaluation

python evaluate_results.py \
  --input ../agent_outputs/<model>_pattern.jsonl \
  --output ../agent_eval_results/<model>_pattern_results.json \
  --detailed
```

For anomaly detection outputs, pass `--is_anomaly` when you want to use the anomaly-specific loader behavior:

```bash
python evaluate_results.py \
  --input ../agent_outputs/<model>_ad.jsonl \
  --output ../agent_eval_results/<model>_ad_results.json \
  --detailed \
  --is_anomaly
```

The evaluator accepts current and older producer schemas:

- Prediction field: `llm_response_parsed.answer` or `predicted_answer`.
- Expected field: `expected_answer` or `original_answer`.

Supported metrics include:

- `accuracy`
- `mae`
- `smape`
- `f1`
- `affinity f-score`

The output JSON includes `summary_statistics`, `total_questions`, `successful_evaluations`, and, when `--detailed` is set, `individual_evaluations`.

## End-to-End Example

```bash
# 1. Generate simulation and QA.
python data_generation_main.py \
  -d 7 \
  -ns 20 \
  -ctrl OpenAPS \
  --qa pm

# 2. Convert contextual QA to CSV files for agent inference.
python inference/agent/convert.py \
  --input-file "SimulationResults/<run folder>/QAData/QA_pattern_with_context.jsonl" \
  --output-dir "SimulationResults/<run folder>/QAData/agent_csv"

# 3. Run OpenAI agent inference.
cd inference/agent
python run_agent.py \
  --qa_file ../../SimulationResults/<run folder>/QAData/QA_pattern.json \
  --output_file ../../agent_outputs/openai_pm.jsonl \
  --data_type PM \
  --file_path ../../SimulationResults/<run folder>/QAData/agent_csv \
  --model gpt-5.4 \
  --workflow_name pm_eval

# 4. Evaluate outputs.
cd ../../evaluation
python evaluate_results.py \
  --input ../agent_outputs/openai_pm.jsonl \
  --output ../agent_eval_results/openai_pm_results.json \
  --detailed
```

Replace `<run folder>` with the generated folder under `SimulationResults/`.

## Fault and Hazard Labels

The simulator labels 17 fault patterns plus hypo/hyperglycemia hazards. Fault labels are stored in simulation artifacts such as `model_state_results.xlsx` and `model.pkl`.

| Failure pattern | Description | Fault label |
| --- | --- | --- |
| Missing signal | Consecutive CGM values replaced with missing values | `missing_signal` |
| Positive spike readings | CGM value increased abruptly | `positive_spike` |
| Negative spike readings | CGM value decreased abruptly | `negative_spike` |
| Repeated readings | Consecutive CGM values replay the same value | `repeated_reading` |
| Negative biased readings | CGM values shifted downward for a period | `negative_bias` |
| Positive biased readings | CGM values shifted upward for a period | `positive_bias` |
| Minimize readings | CGM values clamped to a low reading | `min_reading` |
| Maximize readings | CGM values clamped to a high reading | `max_reading` |
| Repeated episode | A previous episode is replayed to the controller | `repeated_episode` |
| Zero readings | CGM values set to zero | `zero_reading` |
| False meal | Meal registered in the controller but not the patient | `false_meal` |
| False bolus | Previous bolus request repeated at an inappropriate time | `false_bolus` |
| Biased basal | Basal insulin rate shifted up or down | `positive_basal`, `negative_basal` |
| Maximize basal | Basal action forced to maximum | `max_basal` |
| Minimize basal | Basal action forced to minimum | `min_basal` |
| Unknown stopped delivery | Displayed insulin remains normal, delivered basal is zero | `unknown_stop` |
| Unknown under delivery | Displayed insulin remains normal, delivered basal is reduced | `unknown_under` |

Hazard labels:

- `hypoglycemia` - blood glucose below 70 mg/dL.
- `hyperglycemia` - blood glucose above 180 mg/dL.

## Fault Injection Logic

Fault injection runs inside the simulation loop.

For CGM faults such as missing signal, spikes, bias, min/max readings, repeated readings, zero readings, and repeated episodes, the controller receives the faulty BG reading while the patient model state remains based on the true physiology and the impacted insulin dosage.

For insulin-delivery faults such as basal manipulation, unknown stop/under delivery, false bolus, and false meal, the insulin dosage is changed after the controller computes the current insulin action and before delivery into the patient model.

For unknown pump malfunctions, the displayed insulin delivery remains normal while the delivered insulin is changed.
