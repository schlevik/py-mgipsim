import argparse
import json
import anthropic
from pydantic import BaseModel
class FinalAnswerAgentSchema(BaseModel):
  answer: str

TOKENS_PER_MILLION = 1_000_000
ANTHROPIC_MODEL_PRICING_PER_1M = {
    "claude-opus-4-7": {
        "input": 5.00,
        "output": 25.00,
    },
    "claude-opus-4-6": {
        "input": 5.00,
        "output": 25.00,
    },
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--qa_file", type=str, required=True)
  parser.add_argument("--output_file", type=str, required=True)
  parser.add_argument("--data_type", type=str, required=True)
  parser.add_argument("--model", type=str, required=True)
  parser.add_argument("--file_path", type=str, required=True)
  return parser.parse_args()


def get_model_pricing(model):
    pricing = ANTHROPIC_MODEL_PRICING_PER_1M.get(model)
    if pricing is None:
        supported_models = ", ".join(sorted(ANTHROPIC_MODEL_PRICING_PER_1M))
        raise ValueError(
            f"Cost calculation is not configured for model {model!r}. "
            f"Supported models: {supported_models}"
        )
    return pricing


def calculate_cost(args, input_tokens, output_tokens):
    pricing = get_model_pricing(args.model)
    cost = (
        input_tokens * pricing["input"]
        + output_tokens * pricing["output"]
    ) / TOKENS_PER_MILLION
    return round(cost, 8)

def run(args):
    get_model_pricing(args.model)
    client = anthropic.Anthropic()
    data = []
    with open(args.qa_file, "r") as f:
        for line in f:
            data.append(json.loads(line))


    client = anthropic.Anthropic()
    total_cost = 0
    with open(args.output_file, "w") as f:
        for h in range(len(data)):
            cur_data = data[h]
            patient_id = cur_data['patient_id']
            question_id = cur_data['question_id']
            question = cur_data['question']
           

            print("patient id: ", patient_id)

            file_path = f"{args.file_path}/{patient_id}_question_{question_id}.csv"


            # Upload a file
            file_object = client.beta.files.upload(
                file=open(file_path, "rb"),
            )
            
            question = cur_data['question_text']
            original_answer = cur_data['answer']
            question_id = cur_data['question_id']                
            answer_instructions = cur_data['answer_instruction']
            answer_type = cur_data['answer_type']
            example_answer = cur_data['example_answer']
            input_txt = f"The Question is: {question} \n The answer instructions are: {answer_instructions} \n The answer type is: {answer_type} \n An example answer is: {example_answer}"
           
            # Use the file_id with code execution
            response = client.beta.messages.parse(
                model=args.model,
                betas=["files-api-2025-04-14"],
                max_tokens=8192,
                thinking={"type": "adaptive", "display" : "summarized"},
                output_config={ "effort": "low"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_txt},
                            {"type": "container_upload", "file_id": file_object.id},
                        ],
                    }
                ],
                tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
                output_format=FinalAnswerAgentSchema,
            )


            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            estimated_cost_usd = calculate_cost(args, input_tokens, output_tokens)
            last_block = ""
            try:
                final_answer = response.content[-1].parsed_output.answer
            except Exception as e:
                print("error parsing output: ", e)
                final_answer = "error"
                last_block = response.content[-1].text

            dic = {}
            dic['question_id'] = question_id
            dic['question'] = question
            dic['patient_id'] = patient_id
            dic['file_used'] = file_path
            dic['answer_instructions'] = answer_instructions
            dic['answer_type'] = answer_type
            dic['example_answer'] = example_answer
            dic['original_answer'] = original_answer
            dic['last_block'] = last_block
            dic['predicted_answer'] = final_answer
            dic['input_tokens'] = input_tokens
            dic['output_tokens'] = output_tokens
            dic['estimated_cost_usd'] = estimated_cost_usd
            dic["message_traces"] = response.content
            total_cost += estimated_cost_usd
            json.dump(dic, f)
            f.write("\n")

    print("total cost: ", total_cost)

if __name__ == "__main__":
    args = parse_args()
    run(args)