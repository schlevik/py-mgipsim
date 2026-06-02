from dotenv import load_dotenv

load_dotenv("/home/srini/time_series/py-mgipsim/inference/.env")
from openai import OpenAI
import argparse
import json
import os
from pydantic import BaseModel
from typing import Dict, Any

class Answer(BaseModel):
    answer: str
        

def create_prompt(
        patient_context: str, question_text: str, answer_instruction: str, answer_type: str, example_answer: str
    ) -> str:
        """Create the full prompt for the LLM."""
        # Format example answer based on answer type
      
        formatted_example = example_answer
            
        prompt = f"""You are a medical AI assistant analyzing diabetes management data. Based on the patient health data, answer the following question accurately.

PATIENT DATA OVERVIEW:
This data represents a continuous glucose monitoring (CGM) session for a diabetes patient. The data includes:
- Blood glucose readings taken every 5 minutes, beginning at Week 1, Day 1, 00:00 (normal range: 70–180 mg/dL)
- Carbohydrate intake events with timing and amounts
- Insulin delivery events (basal background insulin and bolus meal insulin)
- Physical activity events (running/cycling with duration and intensity)

The data may contain various artifacts, sensor issues, or abnormal patterns that need to be identified and analyzed.

For all insulin dosage-related questions, use the original insulin unit representation provided in the data (mU/min) and do not convert values into insulin units unless explicitly requested.

{patient_context}

Question: {question_text}

Instructions: {answer_instruction}

Expected Answer Type: {answer_type}

Example Answer: {formatted_example}

Please analyze the data carefully and provide your answer as a **json object** in the exact format specified by the answer type. Be precise and base your response only on the data provided.

Conclude your analysis with:
```json
{{"answer": your answer here}}"""
        return prompt

def convert_minutes_to_time(minutes):
    """
    Convert minutes to weeks, days, hours, and minutes format
    Returns both detailed breakdown and HH:MM format
    """
    
    # Constants
    MINUTES_PER_HOUR = 60
    MINUTES_PER_DAY = 60 * 24  # 1440
    MINUTES_PER_WEEK = 60 * 24 * 7  # 10080
    
    # Calculate weeks, days, hours, minutes
    weeks = minutes // MINUTES_PER_WEEK
    remaining_after_weeks = minutes % MINUTES_PER_WEEK
    
    days = remaining_after_weeks // MINUTES_PER_DAY
    remaining_after_days = remaining_after_weeks % MINUTES_PER_DAY
    
   
    return {
        'weeks': int(weeks)+1,
        'days': int(days)+1,
        'total_minutes': int(minutes)
    }
    
def format_patient_context(input_context: Dict[str, Any]) -> str:
    """Format patient data for LLM prompt."""
    context_str = "Patient Health Data:\n\n"

    # Blood glucose data (continuous readings every 5 minutes)
    if "bg_mgdl" in input_context:
        try:
            bg_data = input_context["bg_mgdl"]['magnitude']
        except:
            bg_data = input_context["bg_mgdl"]
        context_str += f"Blood Glucose Readings (mg/dL, every 5 minutes): {len(bg_data)} readings\n"
        context_str += f"  Values: {', '.join(f'{b:.1f}' for b in bg_data)}\n"
        context_str += "\n"
        # context_str += f"  Range: {min(bg_data):.1f} - {max(bg_data):.1f} mg/dL\n\n"
    # print(input_context.keys())
    if "insulin_mUmin" in input_context:
        insulin_data = input_context["insulin_mUmin"]["magnitude"]
        context_str += f"Insulin Readings (mU/minute, every 5 minutes): {len(insulin_data)} readings\n"
        context_str += f"  Values: {', '.join(f'{b:.1E}' for b in insulin_data)}\n"
        context_str += "\n"
    elif 'insulin_events' in input_context and isinstance(input_context['insulin_events'], dict):
        insulin_data = input_context['insulin_events']['magnitude']
        context_str += f"Insulin Readings (mU/minute, every 5 minutes): {len(insulin_data)} readings\n"
        context_str += f"  Values: {', '.join(f'{b:.1E}' for b in insulin_data)}\n"
        context_str += "\n"
    elif "insulin_events" in input_context and isinstance(input_context['insulin_events'], list) and isinstance(input_context['insulin_events'][0], float):
        insulin_data = input_context['insulin_events']
        context_str += f"Insulin Readings (mU/minute, every 5 minutes): {len(insulin_data)} readings\n"
        context_str += f"  Values: {', '.join(f'{b:.1E}' for b in insulin_data)}\n"
        context_str += "\n"
    elif "insulin_events" in input_context:
        print("entering event loop as dict")
        insulin_events = input_context["insulin_events"]
        context_str += f"Insulin Events ({len(insulin_events)} total):\n"
        for event in insulin_events:
            formatted = convert_minutes_to_time(event['time'])
            context_str += f"  - Week {formatted['weeks']} Day {formatted['days']} {event['time_str']}: {event['dosage']:.1f}U ({event['insulin_type']})\n"
        context_str += "\n"
    
    

    # Carbohydrate events
    if "carb_events" in input_context:
        carb_events = input_context["carb_events"]
        context_str += f"Carbohydrate Events ({len(carb_events)} total):\n"
        for event in carb_events:
            # print(event)
            if "week" not in event:
                formatted = convert_minutes_to_time(event['time'])
                event['week'] = formatted['weeks']
                event['day'] = formatted['days']
                
            context_str += f"  - Week {event['week']} Day {event['day']} {event['time_str']}: {event['carbs']:.1f}g ({event['meal_type']})\n"
        context_str += "\n"
    # Insulin events -- skip for now
    

    # Exercise events - handle both old format and new format
    if "exercise_events" in input_context:
        # Convert exercise_events to running_events and cycling_events
        exercise_events = input_context["exercise_events"]
        running_events = []
        cycling_events = []
        walking_events = []
        extra_events = []
        for event in exercise_events:
            if event['duration'] == 0:
                continue
            
            extra_events.append({
                'time': event['time'],
                'time_str': event['time_str'],
                'exercise_type': event['exercise_type'],
                'power': event['magnitude'],
                'duration': event['duration']
            })


        if extra_events:
            input_context['extra_events'] = extra_events
        
    
    # Process running and cycling events (either original or converted)
    for activity in ["extra_events", "running_events", "cycling_events", "walking_events"]:
        if activity in input_context and input_context[activity]:
            events = input_context[activity]
            activity_name = activity.replace("_events", "").title()
            context_str += f"Exercise Events ({len(events)} total):\n"
            for event in events:
                formatted = convert_minutes_to_time(event['time'])
                context_str += f" - Week {formatted['weeks']} Day {formatted['days']} {event['time_str'].split('.', 1)[0]}: {event['exercise_type']} avg power {event.get("power", event.get("magnitude")):.1f} for {event['duration']:.0f} min\n"
            context_str += "\n"
    # print(context_str)
    return context_str




def main(args):

    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    )

    data = []
    if args.qa_file.endswith(".json"):
        print("QA file is a json file")
        with open(args.qa_file, "r") as f:
            data = json.load(f)
    else:
        print("QA file is a jsonl file")
        with open(args.qa_file, "r") as f:
            for line in f:
                data.append(json.loads(line))

    model_name=args.model
    print("model name: ", model_name)
    qa_count = 0
    with open(args.output_file, "w") as f:
        
        for h in range(0, len(data)):
            
            print("row", h + 1)
            input_context = data[h]['input_context']
            patient_id = data[h]['patient_id']

            for qa in data[h]['qa_pairs']:
                question_id = qa['question_id']
                question = qa['question_text']
                answer_instructions = qa['answer_instruction']
                answer_type = qa['answer_type']
                metric = qa['metric']
                example_answer = qa['example_answer']
                original_answer = qa['answer']
                answer_generation_rule = qa['answer_generation_rule']
                

                patient_context = format_patient_context(input_context)
                prompt = create_prompt(patient_context, question, answer_instructions, answer_type, example_answer)
                
                completion = client.chat.completions.parse(
                        model=model_name,
                        messages=[
                            {
                            "role": "user",
                            "content": prompt
                            }
                        ],
                        response_format=Answer,
                        )
            
                predicted_answer = completion.choices[0].message.content
                print("original answer: ", original_answer)
                print("predicted answer: ", predicted_answer)
                print("--------------")

                dic = {}
                
                # add qa dic into this
                dic.update(qa)
                dic['patient_id'] = patient_id
                dic['predicted_answer'] = predicted_answer
                dic['prompt'] = prompt
                dic['expected_answer'] = original_answer
                dic['input_tokens'] = completion.usage.prompt_tokens
                dic['output_tokens'] = completion.usage.completion_tokens
                dic['estimated_cost_usd'] = completion.usage.cost

                qa_count += 1
                

                json.dump(dic, f)
                f.write("\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)