from dotenv import load_dotenv
load_dotenv("./../.env")
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
  parser.add_argument("--output_file", type=str, required=True)
  parser.add_argument("--data_type", type=str, required=True)
  parser.add_argument("--model", type=str, required=True)
  parser.add_argument("--file_path", type=str, required=True)
  return parser.parse_args()

class WorkflowInput(BaseModel):
  input_as_text: str

class FinalAnswerAgentSchema(BaseModel):
  answer: str


def create_container():
  api_key = os.getenv("OPENAI_API_KEY")
  client = OpenAI(api_key=api_key)
  container = client.containers.create(
    name="eval_container")
  
  return container.id

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

def get_coding_agent(args, container_id: str, file_id: str):
  code_interpreter = CodeInterpreterTool(tool_config={
    "type": "code_interpreter",
    "container": container_id
  })

  coding_agent = Agent(
    name="Coding_agent",
    instructions="You will be given a user question that you have to answer. You will also be given instruction to get the answer, answer type and answer generation rule. Use python code to get to the answer.  You are given the patients data in the form of a csv file. ",
    model=args.model,
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
    model=args.model,
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
async def run_workflow(args, workflow_input: WorkflowInput, container_id: str, question_id: str, patient_id: str, file_id: str):
  with trace(workflow_name=f"{args.data_type}_patient_{patient_id}_question_{question_id}", trace_id=f"trace_{args.data_type}_patient_{patient_id}_question_{question_id}"):

    coding_agent, final_answer_agent = get_coding_agent(args, container_id, file_id)

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
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_692291af37788190988f392bd8c107830c08f66636716dbc"
      })
    )

    conversation_history.extend([item.to_input_item() for item in coding_agent_result_temp.new_items])

    coding_agent_result = {
      "output_text": coding_agent_result_temp.final_output_as(str)
    }
    final_answer_agent_result_temp = await Runner.run(
      final_answer_agent,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_692291af37788190988f392bd8c107830c08f66636716dbc"
      })
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
    #create container
    container_id = create_container()

    print("container id: ", container_id)

    assert container_id is not None , "Container is not created"

    with open(args.qa_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    with open(args.output_file, "w") as f:
      for h in range(0,len(data)):

          cur_data = data[h]
          patient_id = data[h]['patient_id']
          
          print("patient id: ", patient_id)
          #upload file to container
          file_path = f"{args.file_path}/{patient_id}_{h}.csv"
          #f"/home/srini/time_series/loopqa/agent_data/yannan/insulin/patient_{patient_id}.csv"
          
          try:
              file_id = upload_file(file_path, container_id)
          except Exception as e:
              print("error uploading file: ", e)
              continue

          assert file_id is not None , "File is not uploaded"
          print("question number" , h+1)
          #print("patient id: ", patient_id)

          
        
          question = cur_data['question_text']
          original_answer = cur_data['answer']
          question_id = cur_data['question_id']                
          answer_instructions = cur_data['answer_instruction']
          answer_type = cur_data['answer_type']
          example_answer = cur_data['example_answer']
          input_txt = f"The Question is: {question} \n The answer instructions are: {answer_instructions} \n The answer type is: {answer_type} \n An example answer is: {example_answer}"
          input_class = WorkflowInput(input_as_text=input_txt)
          result = await run_workflow(args, input_class, container_id, question_id, patient_id, file_id)

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

          break

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))