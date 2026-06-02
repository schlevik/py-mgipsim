







#openai
model_name="gpt-5.4"
workflow_name="final_real_world_PM_5_4"
file_path=./final_sampled_benchmark/real_world_pm
output_file="./agent_outputs/final_run/${workflow_name}_${model_name}_run2.jsonl"
qa_file=./final_sampled_benchmark/real_world_PM_all_367.json
data_type="final_real_world_PM_5_4"

python run_agent.py \
    --qa_file ${qa_file} \
    --output_file ${output_file} \
    --data_type ${data_type} \
    --file_path ${file_path} \
    --model ${model_name} \
    --workflow_name ${workflow_name}


# model_name="claude-opus-4-6
