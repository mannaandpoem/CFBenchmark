import argparse
import os
import time
import asyncio

import pandas as pd
from tqdm import tqdm

from evaluator.chatgpt_evaluator import ChatGPTEvaluator
from evaluator.chatglm2_evaluator import ChatGLM2Evaluator
from evaluator.chatglm_evaluator import ChatGLMEvaluator
from evaluator.llama2_evaluator import LLaMA2Evaluator
from evaluator.llama_evaluator import LLaMAEvaluator
from utils.dataloader import load_dataset
from utils.file_utils import save_json


def load_evaluator(model_name_: str, model_path: str, api_key: str = None, base_url: str = None):
    """
    Load evaluator based on model name with improved error handling.
    
    Args:
        model_name_: Model name identifier
        model_path: Path to model or model identifier
        api_key: API key for OpenAI-compatible models
        base_url: Base URL for OpenAI-compatible APIs
        
    Returns:
        Evaluator instance
    """
    model_name = model_name_.lower()
    
    try:
        if "chatglm2" in model_name:
            evaluator = ChatGLM2Evaluator(model_path)
        elif "chatglm" in model_name:
            evaluator = ChatGLMEvaluator(model_path)
        elif "llama2" in model_name:
            generation_config = dict(
                temperature=0.01,
                top_k=40,
                top_p=0.7,
                max_new_tokens=1024
            )
            evaluator = LLaMA2Evaluator(generation_config, model_path)
        elif "llama" in model_name:
            max_new_tokens = 1024
            generation_config = dict(
                temperature=0.2,
                top_k=40,
                top_p=0.7,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens
            )
            evaluator = LLaMAEvaluator(max_new_tokens, generation_config, model_path)
        elif model_name in ["gpt35", "gpt-3.5", "gpt-3.5-turbo"] or "gpt" in model_name or "google" in model_name or "qwen" in model_name:
            # Support for OpenAI and OpenAI-compatible APIs
            evaluator = ChatGPTEvaluator(
                model_type=model_path if model_path and not os.path.exists(model_path) else "gpt-3.5-turbo",
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1"
            )
        else:
            print(f'{model_name} 模型暂不支持，请实现对应的 evaluator 类')
            return None
            
        return evaluator
        
    except Exception as e:
        print(f"Error loading evaluator for {model_name}: {str(e)}")
        print(f"Model path: {model_path}")
        
        # Provide helpful suggestions
        if "baichuan" in model_name:
            print("Suggestion: Use a valid HuggingFace model name like 'baichuan-inc/Baichuan2-7B-Chat' or provide a valid local path")
        elif "gpt" in model_name:
            print("Suggestion: Make sure OPENAI_API_KEY environment variable is set or provide api_key parameter")
        
        raise e


async def fineva_main(args):
    model_name = args.model_name
    model_path = args.model_path
    save_path = args.save_path
    api_key = getattr(args, 'api_key', None)
    base_url = getattr(args, 'base_url', None)
    max_workers = getattr(args, 'max_workers', 5)

    print(f"Loading model: {model_name}")
    print(f"Model path: {model_path}")
    if api_key:
        print(f"Using custom API key: {api_key[:10]}...")
    if base_url:
        print(f"Using custom base URL: {base_url}")
    print(f"Using max workers: {max_workers}")

    # Sanitize model name for use as dictionary keys by replacing slashes with underscores
    sanitized_model_name = model_name.replace('/', '_')

    # 导入模型
    try:
        evaluator = load_evaluator(model_name, model_path, api_key, base_url)
        if evaluator is None:
            print(f"Failed to load evaluator for {model_name}")
            return
        print(f'模型 {model_name} 加载成功')
    except Exception as e:
        print(f"Failed to load model {model_name}: {str(e)}")
        return

    # 导入数据
    try:
        # Use correct path relative to src directory
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        dataset = load_dataset(data_path)
        print(f"Loaded {len(dataset)} examples from dataset")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print(f"Attempted to load from: {data_path if 'data_path' in locals() else '../data'}")
        return

    # 大模型推理 - 异步并行处理
    successful_count = 0
    error_count = 0
    
    async def process_single_item(i, data):
        """Process a single data item asynchronously"""
        nonlocal successful_count, error_count
        
        try:
            prompt_query = data['prompt']
            
            # Check if evaluator has async support
            if hasattr(evaluator, 'answer_async'):
                # Use async method if available
                model_answer = await evaluator.answer_async(prompt_query)
            else:
                # Fallback to sync method (run in thread pool for non-blocking)
                loop = asyncio.get_event_loop()
                model_answer = await loop.run_in_executor(None, evaluator.answer, prompt_query)
            
            # # Rate limiting for API-based models
            # if model_name in ['gpt35', 'gpt-3.5', 'gpt-3.5-turbo'] or 'gpt' in model_name:
            #     await asyncio.sleep(0.1)  # Reduced sleep time for better throughput
            
            # Use sanitized model name for the key to avoid issues with slashes
            data[f'{sanitized_model_name}_answer'] = model_answer
            successful_count += 1
            return True
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            # Use sanitized model name for the key to avoid issues with slashes
            data[f'{sanitized_model_name}_answer'] = "Error"
            error_count += 1
            return False
    
    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(i, data):
        async with semaphore:
            return await process_single_item(i, data)
    
    # # Process first 10 examples for testing with progress bar
    # test_dataset = dataset[:10]
    
    with tqdm(total=len(dataset), desc="Processing") as pbar:
        tasks = [process_with_semaphore(i, data) for i, data in enumerate(dataset)]
        
        # Process tasks as they complete
        for coro in asyncio.as_completed(tasks):
            try:
                success = await coro
                pbar.update(1)
                
                # Stop if too many errors
                if error_count > 10:
                    print("Too many errors, stopping execution")
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
                    
            except Exception as e:
                print(f"Error in task: {str(e)}")
                pbar.update(1)

    print(f"Processing completed: {successful_count} successful, {error_count} errors")

    # 确保保存目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存为 json
    try:
        output_file = os.path.join(save_path, f'{sanitized_model_name}_ga.json')
        save_json(dataset, output_file)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error occurred while saving as json: {e}")
        # Try to save as backup
        try:
            backup_file = os.path.join(save_path, f'{sanitized_model_name}_ga_backup.json')
            save_json(dataset, backup_file)
            print(f"Backup saved to: {backup_file}")
        except Exception as e2:
            print(f"Failed to save backup: {e2}")
            print("Dataset preview:")
            print(dataset[:2] if len(dataset) > 2 else dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Evaluation Main Script")
    parser.add_argument('--model_name', required=True, type=str, help='Model name identifier')
    parser.add_argument('--model_path', required=False, type=str, help='Path to model or model identifier')
    parser.add_argument('--save_path', required=True, type=str, help='Path to save results')
    parser.add_argument('--api_key', required=False, type=str, help='API key for OpenAI-compatible models')
    parser.add_argument('--base_url', required=False, type=str, help='Base URL for OpenAI-compatible APIs')
    parser.add_argument('--max_workers', required=False, type=int, default=5, help='Maximum number of concurrent workers')
    
    args = parser.parse_args()
    asyncio.run(fineva_main(args))
