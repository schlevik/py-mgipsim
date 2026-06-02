


model_name="gemini-3.1-flash-lite-preview"
python openrouter_inference.py \
    --qa_file ./final_sampled_benchmark/real_world_PD_all_105.jsonl \
    --output_file ./agent_outputs/final_run/${model_name}_real_world_PD_all_105_run_v2.jsonl \
    --model ${model_name}