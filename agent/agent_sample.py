from dotenv import load_dotenv
from agents import CodeInterpreterTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from pydantic import BaseModel
from openai.types.shared.reasoning import Reasoning
import asyncio

load_dotenv(override=True)

class WorkflowInput(BaseModel):
  input_as_text: str

class FinalAnswerAgentSchema(BaseModel):
  answer: str

def get_coding_agent(file_id: str):
  print("file_id: ", file_id)
  # Tool definitions
  code_interpreter = CodeInterpreterTool(tool_config={
    "type": "code_interpreter",
    "container": ""
    {
      "type": "auto",
      "file_ids": [
      file_id
      ]
    }
  })

  coding_agent = Agent(
    name="Coding_agent",
    instructions="You will be given a user question that you have to answer. You will also be given instruction to get the answer, answer type and answer generation rule. Use python code to get to the answer.  You are given the patients data in the form of a csv file. ",
    model="gpt-5.1",
    tools=[
      code_interpreter
    ],
    model_settings=ModelSettings(
      store=True,
      reasoning=Reasoning(
        effort="low",
        summary="auto"
      )
    )
  )


  final_answer_agent = Agent(
    name="Final_answer_agent",
    instructions="""Return in the answer in the following format: 
  {
  answer: final_answer
  }""",
    model="gpt-5.1",
    output_type=FinalAnswerAgentSchema,
    model_settings=ModelSettings(
      store=True,
      reasoning=Reasoning(
        effort="low",
        summary="auto"
      )
    )
  )

  return coding_agent, final_answer_agent

# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput, file_id: str, question_id: str, patient_id: str):
  with trace(workflow_name=f"Agent_workflow_anomaly_{question_id}_patient_{patient_id}", trace_id=f"trace_yannan_insulin_{question_id}_{patient_id}"):
    state = {
      "file_path": None
    }

    coding_agent, final_answer_agent = get_coding_agent(file_id)
    
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    coding_agent_result_temp = await Runner.run(
      coding_agent,
      input=[
        *conversation_history
      ]
    )

    conversation_history.extend([item.to_input_item() for item in coding_agent_result_temp.new_items])

    coding_agent_result = {
      "output_text": coding_agent_result_temp.final_output_as(str)
    }
    final_answer_agent_result_temp = await Runner.run(
      final_answer_agent,
      input=[
        *conversation_history
      ]
    )

    conversation_history.extend([item.to_input_item() for item in final_answer_agent_result_temp.new_items])

    final_answer_agent_result = {
      "output_text": final_answer_agent_result_temp.final_output.json(),
      "output_parsed": final_answer_agent_result_temp.final_output.model_dump()
    }
    return final_answer_agent_result

async def run():
    import json
    data = []
    with open("/home/srini/time_series/loopqa/data/data/anomaly_latest/16_november/QA_ad_with_context.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    file_ids = []


    for h in range(len(file_ids)):
        patient_idx = h
        patient_id = data[patient_idx]['patient_id']
        print("patient id: ", patient_id)
        with open(f"./agent_outputs/anomaly/anomaly_patient_{patient_id}.jsonl", "w") as f:
            for j,i in enumerate(data[patient_idx]['qa_pairs']):
                print(f"Question: {j+1}")
                question = i['question_text']
                original_answer = i['answer']
                question_id = i['question_id']
                answer_instructions = i['answer_instruction']
                answer_type = i['answer_type']
                example_answer = i['example_answer']
                input_txt = f"The Question is: {question} \n The answer instructions are: {answer_instructions} \n The answer type is: {answer_type} \n An example answer is: {example_answer}"
                input_class = WorkflowInput(input_as_text=input_txt)
                result = await run_workflow(input_class, file_ids[h], question_id, patient_id)
                dic = {}
                dic['question_id'] = question_id
                dic['patient_id'] = patient_id
                dic['original_answer'] = original_answer
                dic['predicted_answer'] = result['output_parsed']['answer']
                print("original answer: ", original_answer)
                print("predicted answer: ", result['output_parsed']['answer'])
                print("--------------------------------")
                json.dump(dic, f)
                f.write("\n")

if __name__ == "__main__":
    asyncio.run(run())