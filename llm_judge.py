import re
import json
import os
import asyncio
import logging
import argparse
from collections import Counter
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CONCURRENCY_LIMIT = 10  # (QPS limit)
MAX_RETRIES = 3
OUTPUT_FILE = "./results/llm_judge_results.jsonl"
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="empty", timeout=600) if AsyncOpenAI else None

PROMPT_TEMPLATE = """
You are going to evaluate the reasoning processes for long-horizon health time-series question answering.

You will be given:
1. The question
2. The expected reasoning instruction for this question
3. The model's reasoning trace
4. The expected final answer

Your task is to evaluate whether the model's reasoning process:
- follows the intended reasoning instruction,
- is logically reliable and well-supported,
- avoids lazy or unsafe guessing,
- and complies with the required output format.

Evaluate the reasoning trace using the rubric below.

========================
GENERAL INSTRUCTIONS
========================
- Evaluate the reasoning process primarily by comparing it against the provided expected reasoning instruction.
- Do NOT assume access to the original monitoring data.
- Do NOT penalize the model merely because the final answer may be wrong, unless the reasoning itself is unreliable, unsupported, contradictory, or clearly deviates from the intended reasoning path.
- Focus on whether the reasoning path is trustworthy, instruction-consistent, and safe.
- Be strict about unsupported shortcuts, vague reasoning, and lazy guessing.
- Assign failure types only when their definitions are clearly satisfied.

========================
EVALUATION CRITERIA
========================

1. Reasoning Instruction Following

Definition:
Does the reasoning trace follow the expected reasoning instruction provided for the question?

This criterion evaluates whether the model follows the expected reasoning path, rather than whether the answer alone is factually correct.

Key aspects:
- Follows the requested reasoning strategy or sequence
- Covers the key reasoning steps implied by the expected reasoning instruction
- Does not skip essential reasoning stages
- Does not replace the intended reasoning process with an unrelated shortcut
- Stays aligned with the scope of the requested task

Scoring:
5 = Fully follows the expected reasoning instruction, covering the key expected steps in a consistent order
4 = Mostly follows the expected reasoning instruction, with only minor omissions or small deviations
3 = Partially follows the expected reasoning instruction, but misses or alters some important steps
2 = Substantially deviates from the expected reasoning instruction or skips essential steps
1 = Does not follow the expected reasoning instruction at all

----------------------------------------

2. Reasoning Quality

Definition:
Is the reasoning process logically coherent, reliable, and appropriately supported, without relying on unsupported assumptions, contradictions, or lazy guessing?

This criterion focuses on the trustworthiness of the reasoning path itself.

Key aspects:
- Logical coherence across steps
- Clear connection between intermediate reasoning and conclusion
- No internal contradiction
- No unjustified leap from premise to conclusion
- No reliance on vague, generic, or weakly supported claims when more structured reasoning is expected
- Expresses uncertainty appropriately rather than guessing when support is insufficient

Scoring:
5 = Fully coherent, well-supported, and reliable reasoning with no meaningful contradictions or unjustified leaps
4 = Mostly coherent and reliable, with only minor weaknesses in support or clarity
3 = Partially reliable reasoning with noticeable unsupported steps, weak transitions, or mild contradictions
2 = Reasoning is unreliable due to major unsupported assumptions, contradictions, or shortcut-based guessing
1 = Reasoning is fundamentally unreliable, incoherent, or dominated by unsupported guessing

Important rules:
- If the reasoning trace replaces structured reasoning with vague heuristics or plausible-sounding guesses, the score should be no higher than 2.
- If the model acknowledges uncertainty but still commits to a definite conclusion without support, the score should be no higher than 2.
- If the reasoning contains substantial contradictions, the score should be no higher than 2.

----------------------------------------

3. Format Compliance

Definition:
Does the output follow the required response format, structure, and granularity constraints?

This criterion evaluates surface-level compliance only, not reasoning quality.

Key aspects:
- Uses the required output structure
- Preserves required fields or schema
- Follows requested granularity
- Does not merge distinct items when they should remain separate
- Does not return plain text when a structured format is required

Scoring:
5 = Fully compliant with all formatting, structural, and granularity requirements
4 = Minor formatting or structural issues that do not affect usability
3 = Noticeable format deviations, but the output is still partly usable
2 = Major format or structure violations
1 = Completely ignores the required format

========================
FAILURE TYPES (MULTI-LABEL)
========================

Assign failure types ONLY if the following definitions are clearly satisfied.

1. Reluctance to Calculate

Definition:
The model avoids carrying out the reasoning work implied by the expected reasoning instruction, especially when it requires explicit calculation, structured enumeration, or full-step analysis.

Apply this label when the model:
- avoids explicit calculation when the expected reasoning instruction clearly requires it
- substitutes approximate or vague statements for required reasoning steps
- gives partial reasoning instead of completing the intended process
- abandons a full reasoning procedure in favor of a shortcut

Note:
This label is about avoiding reasoning effort, not about numerical correctness against raw data.

----------------------------------------

2. Temporal Misalignment

Definition:
The model fails to follow the temporal logic or temporal scope specified in the expected reasoning instruction or question.

Apply this label when the model:
- reasons about the wrong time period relative to the question or expected reasoning instruction
- confuses the relevant temporal scope
- uses an inconsistent temporal frame in the reasoning trace

Note:
Do not require access to raw data. Judge based on alignment with the question and expected reasoning instruction.

----------------------------------------

3. Unsupported Assumption

Definition:
The model introduces assumptions, priors, or conclusions that are not justified by the question, the expected reasoning instruction, or the reasoning trace itself.

Apply this label when the model:
- inserts plausible but unsupported claims
- uses generic priors in place of grounded reasoning
- reaches conclusions that are not supported by its own intermediate reasoning

----------------------------------------

4. Guessing over Uncertainty

Definition:
The model shows awareness of uncertainty, ambiguity, or missing support, but still gives a definite conclusion instead of maintaining uncertainty.

Apply this label when the model:
- signals doubt, uncertainty, or lack of evidence
- but still commits to a specific answer or conclusion without adequate support

----------------------------------------

5. Formatting Misalignment

Definition:
The model fails to follow the required output format, structure, or granularity.

Apply this label when the model:
- outputs unstructured text instead of the required format
- violates schema requirements
- merges items that should remain separate
- ignores formatting constraints

6. Others

Definition:
A clear reasoning failure exists, but it does not fit any of the five predefined failure types above.

Rules:
- Use ONLY if none of the predefined types apply
- Must include explanation of the failure and why it does not fit existing types
  
 ----------------------------------------

Important rules:
- If any predefined type applies, return those predefined types.
- Use "Others" only for new failure types outside the five predefined categories.
- If no failure is present, return [].

========================
OUTPUT FORMAT (STRICT)
========================

Return your evaluation in the following JSON format:

{
  "scores": {
    "reasoning_instruction_following": {
      "score": <1-5>,
      "justification": "<brief explanation>"
    },
    "reasoning_quality": {
      "score": <1-5>,
      "justification": "<brief explanation>"
    },
    "format_compliance": {
      "score": <1-5>,
      "justification": "<brief explanation>"
    }
  },
  "failure_types": [
    {
      "type": "<Reluctance to Calculate | Temporal Misalignment | Unsupported Assumption | Guessing over Uncertainty | Formatting Misalignment | Others>",
      "explanation": "<brief explanation>"
    }
  ],
  "overall_summary": "<concise overall assessment of whether the reasoning path is reliable and instruction-consistent>"
}

Rules:
- "failure_types" is a multi-label list.
- If multiple failures apply, include multiple objects.
- If "Others" is used, explanation must justify why it is new.
- If no failure exists, return: "failure_types": []
- Do not include any text outside the JSON.
- Base your judgment on the reasoning trace relative to the expected reasoning instruction, not on hidden raw data.

"""

def build_judge_prompt(question: str, cot: str, reasoning_trace: str, expected_answer: str) -> List[Dict[str, str]]:
    """Build the prompt for evaluating whether a reasoning trace follows the intended CoT."""
    sample_block = (
        "\n========================\n"
        "INPUT TO EVALUATE\n"
        "========================\n\n"
        f"### Question\n{question}\n\n"
        f"### Expected reasoning instruction\n{cot}\n\n"
        f"### Model reasoning trace\n{reasoning_trace}\n\n"
        f"### Expected final answer\n{expected_answer}\n"
    )
    return [
        {
            "role": "system",
            "content": "You are a strict evaluator for reasoning-process quality and instruction following.",
        },
        {"role": "user", "content": PROMPT_TEMPLATE + sample_block},
    ]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from judge output, allowing markdown fences or extra text."""
    if not text:
        return None

    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(cleaned[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def get_score(judgment: Dict[str, Any], score_name: str) -> Optional[float]:
    try:
        score = judgment["scores"][score_name]["score"]
        return float(score)
    except (KeyError, TypeError, ValueError):
        return None


def get_record_field(record: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def require_record_field(record: Dict[str, Any], key: str) -> Any:
    if key not in record or record[key] is None:
        record_id = record.get("question_id", "<unknown>")
        raise KeyError(f"Record {record_id} is missing required field: {key}")
    return record[key]


def normalize_answer(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


async def call_vllm_server(
    messages: List[Dict[str, str]],
    client: Optional[AsyncOpenAI] = None,
    model: str = MODEL_NAME,
    temperature: float = 0.0,
    max_tokens: int = 24576,
) -> str:
    """Call the LLM client and return a best-effort string output.

    This centralizes extraction logic so callers don't need to rely on specific
    response shapes. It performs a single request; caller can implement retries.
    """
    active_client = client or globals().get("client")
    if active_client is None:
        raise ImportError("The openai package is required to call the judge server. Install it with `pip install openai`.")
    response = await active_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Try several common response shapes
    llm_output = None
    try:
        llm_output = response.choices[0].message.content
    except Exception:
        try:
            llm_output = response["output"]["choices"][0]["message"]["content"]
        except Exception:
            try:
                llm_output = response["choices"][0]["message"]["content"]
            except Exception:
                try:
                    llm_output = response["choices"][0]["text"]
                except Exception:
                    llm_output = str(response)

    return llm_output


async def process_single_judge(
        sem: asyncio.Semaphore,
        record: Dict[str, Any],
        question: str,
        cot: str,
        reasoning_trace: str,
        expected_answer: str,
        output_path: str,
        write_lock: asyncio.Lock,
        ) -> Dict[str, Any]:

    async with sem:  # Limit concurrency
        messages = build_judge_prompt(question, cot, reasoning_trace, expected_answer)
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:

                llm_output = await call_vllm_server(messages, client=client)
                judgment = extract_json_object(llm_output)
                if judgment is None:
                    raise ValueError("Failed to parse judge JSON output")

                result_data = {
                    "question_id": record.get("question_id"),
                    "question": question,
                    "cot": cot,
                    "reasoning_trace": reasoning_trace,
                    "expected_answer": expected_answer,
                    "judge_output_raw": llm_output,
                    "judge_output_parsed": judgment,
                    "status": "success",
                }             

                async with write_lock:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

                return result_data

            except Exception as e:
                last_error = str(e)
                # Simple exponential backoff
                await asyncio.sleep(2 ** attempt)

        # If all retries fail
        failed_data = {
            "question_id": record.get("question_id"),
            "question": question,
            "cot": cot,
            "reasoning_trace": reasoning_trace,
            "expected_answer": expected_answer,
            "judge_output_raw": f"Error after {MAX_RETRIES} retries: {last_error}",
            "judge_output_parsed": None,
            "status": "failed",
        }

        async with write_lock:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failed_data, ensure_ascii=False) + "\n")

        return failed_data


async def llm_judge(
    records: List[Dict[str, Any]],
    output_file: str = OUTPUT_FILE,
) -> List[Dict[str, Any]]:
    """
    Main async orchestration function.
    """
    if os.path.exists(output_file):
        print(f"Warning: Appending to existing file {output_file}")

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    write_lock = asyncio.Lock()

    # Ensure output directory exists
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Starting evaluation of {len(records)} samples with {CONCURRENCY_LIMIT} workers...")

    tasks = []
    for record in records:
        question = get_record_field(record, "question", "question_text")
        cot = get_record_field(record, "cot", "answer_instruction")
        reasoning_trace = get_record_field(record, "reasoning_trace", "llm_reasoning")
        expected_answer = normalize_answer(require_record_field(record, "expected_answer"))
        task = asyncio.create_task(
            process_single_judge(
                sem,
                record,
                question,
                cot,
                reasoning_trace,
                expected_answer,
                output_file,
                write_lock,
            )
        )
        tasks.append(task)

    results = await tqdm_asyncio.gather(*tasks)

    return results


def compute_metrics(results: List[Dict[str, Any]]):
    total = len(results)
    success_results = [r for r in results if r.get("status") == "success" and r.get("judge_output_parsed")]
    valid_count = len(success_results)
    score_names = [
        "reasoning_instruction_following",
        "reasoning_quality",
        "format_compliance",
    ]
    score_sums = {name: 0.0 for name in score_names}
    score_counts = {name: 0 for name in score_names}
    failure_counter = Counter()

    for result in success_results:
        judgment = result["judge_output_parsed"]
        for name in score_names:
            score = get_score(judgment, name)
            if score is not None:
                score_sums[name] += score
                score_counts[name] += 1
        for failure in judgment.get("failure_types", []):
            if isinstance(failure, dict) and failure.get("type"):
                failure_counter[failure["type"]] += 1

    print("\n" + "=" * 30)
    print("FINAL JUDGE METRICS")
    print("=" * 30)
    print(f"Total processed: {total}")
    print(f"Valid judgments: {valid_count}")
    print(f"Failed/Parse Error: {total - valid_count}")
    for name in score_names:
        avg = score_sums[name] / score_counts[name] if score_counts[name] else 0.0
        print(f"Average {name}: {avg:.3f}")
    if failure_counter:
        print("Failure type counts:")
        for failure_type, count in failure_counter.most_common():
            print(f"  {failure_type}: {count}")
    else:
        print("Failure type counts: none")

    return {
        "total": total,
        "valid_judgments": valid_count,
        "failed_or_parse_error": total - valid_count,
        "average_scores": {
            name: (score_sums[name] / score_counts[name] if score_counts[name] else None)
            for name in score_names
        },
        "failure_type_counts": dict(failure_counter),
    }


def load_jsonl(input_file: str) -> List[Dict[str, Any]]:
    records = []
    with open(input_file, "r", encoding="utf-8") as reader:
        for line_num, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc
    return records



# Example usage in your main script:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge whether model reasoning traces follow the intended CoT.")
    parser.add_argument("--input_file", type=str, default="llm_result_samples.jsonl")
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="empty")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_LIMIT)
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    CONCURRENCY_LIMIT = args.concurrency
    if AsyncOpenAI is None:
        raise SystemExit("The openai package is required to call the judge server. Install it with `pip install openai`.")
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)

    try:
        records = load_jsonl(args.input_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise SystemExit(1)

    results = asyncio.run(llm_judge(records, output_file=args.output_file))
    summary = compute_metrics(results)

    summary_file = args.output_file + ".summary.json"
    with open(summary_file, "w", encoding="utf-8") as writer:
        json.dump(summary, writer, ensure_ascii=False, indent=2)
    print(f"Results saved to: {args.output_file}")
    print(f"Summary saved to: {summary_file}")
