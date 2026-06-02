

model_name="claude-sonnet-4-6"
output_file="./agent_outputs/final_run/${model_name}_real_world_PM_run2.jsonl"

#anthropic
python run_anthropic_agent.py \
    --qa_file ./final_sampled_benchmark/real_world_PM_all_367.json \
    --output_file ${output_file} \
    --data_type real_world_PM \
    --file_path ./final_sampled_benchmark/real_world_pm \
    --model ${model_name}