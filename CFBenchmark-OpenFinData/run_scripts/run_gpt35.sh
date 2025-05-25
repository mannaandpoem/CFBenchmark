#!/bin/bash

model_name="openai/gpt-4.1-mini"
model_name="qwen/qwen3-32b"
model_name="google/gemini-2.0-flash-001"
judge_model_name="openai/gpt-4.1"
result_path="../results"
api_key="sk-xxx"
base_url="https://openrouter.ai/api/v1"


cd ../src

# python exec_fineva_main.py \
#     --model_name ${model_name} \
#     --save_path ${result_path} \
#     --api_key ${api_key} \
#     --base_url ${base_url} \
#     --max_workers 50


python get_score.py \
    --model_name ${model_name} \
    --result_path ${result_path} \
    --api_key ${api_key} \
    --base_url ${base_url} \
    --max_workers 50
