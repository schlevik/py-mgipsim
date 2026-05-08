from dotenv import load_dotenv
from agents import CodeInterpreterTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from pydantic import BaseModel
from openai.types.shared.reasoning import Reasoning
import asyncio
import os
import requests
from openai import OpenAI

import argparse 


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--qa_file", type=str, required=True)
  parser.add_argument("--input_file", type=str, required=True)
  parser.add_argument("--output_file", type=str, required=True)
  return parser.parse_args()

load_dotenv(override=True)

class WorkflowInput(BaseModel):
  input_as_text: str

class FinalAnswerAgentSchema(BaseModel):
  answer: str

def get_container_id():
    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/containers"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.get(url, headers=headers)
    return response.json()['data'][0]['id'], response.json()['data'][0]['status']

def get_file_ids(container_id):
    api_key = os.getenv("OPENAI_API_KEY")
    url = f"https://api.openai.com/v1/containers/{container_id}/files"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.get(url, headers=headers)
    return response.json()['data']

def upload_file(file_path, container_id):# The container ID (from your curl command)
    api_key = os.getenv("OPENAI_API_KEY")
    # Endpoint URL
    url = f"https://api.openai.com/v1/containers/{container_id}/files"

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    with open(file_path, "rb") as file_to_upload:
        files = {
            "file": file_to_upload,
        }
        response = requests.post(url, headers=headers, files=files)

    return response.json()['id']

def delete_file_from_container(file_id, container_id):
    """
    Deletes a file from a specific container using the OpenAI API.

    Args:
        file_id (str): The ID of the file to be deleted.

    Returns:
        dict: The response from the API as a dictionary.
    """
    import requests
    import os

    api_key = os.getenv("OPENAI_API_KEY")

    # Endpoint URL for deleting the file
    url = f"https://api.openai.com/v1/containers/{container_id}/files/{file_id}"

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.delete(url, headers=headers)
    return response.json()

def get_coding_agent(container_id: str):
  code_interpreter = CodeInterpreterTool(tool_config={
    "type": "code_interpreter",
    "container": container_id
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
async def run_workflow(workflow_input: WorkflowInput, container_id: str, question_id: str, patient_id: str):
  with trace(workflow_name=f"Agent_trace_for_paper_{question_id}_patient_{patient_id}", trace_id=f"trace_agent_for_paper_{question_id}_{patient_id}"):
    state = {
      "file_path": None
    }

    coding_agent, final_answer_agent = get_coding_agent(container_id)
    
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



async def run(args):
    import json
    data = []

    container_id, status = get_container_id()
    file_ids = get_file_ids(container_id)
    assert status != "expired" , "Container is not available"
    assert file_ids == [] , "Enusre no files are present in the container"

    with open(args.qa_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    for h in range(0,len(data)):
        patient_idx = h
        patient_id = data[patient_idx]['patient_id']

        #upload file to container
        file_path = f"{args.input_file}"
        #f"/home/srini/time_series/loopqa/agent_data/yannan/insulin/patient_{patient_id}.csv"
        file_id = upload_file(file_path, container_id)
        print("question number" , h+1)
        #print("patient id: ", patient_id)

        with open(args.output_file, "w") as f:
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
                result = await run_workflow(input_class, container_id, question_id, patient_id)
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

        #delete file from container
        delete_status = delete_file_from_container(file_id, container_id)
        print("delete status: ", delete_status)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))