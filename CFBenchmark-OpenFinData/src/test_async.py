#!/usr/bin/env python3
"""
Test script for async functionality in the CFBenchmark system.
"""

import asyncio
import time
import os
from utils.gpt_utils import gpt_api
from evaluator.chatgpt_evaluator import ChatGPTEvaluator


async def test_async_gpt_api():
    """Test the async gpt_api function"""
    print("Testing async gpt_api...")
    
    start_time = time.time()
    
    # Test multiple concurrent requests
    tasks = [
        gpt_api("What is 1+1?", model_type="gpt-3.5-turbo"),
        gpt_api("What is 2+2?", model_type="gpt-3.5-turbo"),
        gpt_api("What is 3+3?", model_type="gpt-3.5-turbo"),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    print(f"Async requests completed in {end_time - start_time:.2f} seconds")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i+1} failed: {result}")
        else:
            print(f"Request {i+1} result: {result[:50]}...")


async def test_async_evaluator():
    """Test the async ChatGPT evaluator"""
    print("\nTesting async ChatGPT evaluator...")
    
    evaluator = ChatGPTEvaluator()
    
    start_time = time.time()
    
    # Test multiple concurrent evaluations
    tasks = [
        evaluator.answer_async("Explain machine learning in one sentence."),
        evaluator.answer_async("What is artificial intelligence?"),
        evaluator.answer_async("Define deep learning briefly."),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    print(f"Async evaluations completed in {end_time - start_time:.2f} seconds")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Evaluation {i+1} failed: {result}")
        else:
            print(f"Evaluation {i+1} result: {result[:50]}...")


async def main():
    """Main test function"""
    print("Starting async functionality tests...")
    
    # Check if API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Some tests may fail. Please set your OpenAI API key.")
        return
    
    try:
        await test_async_gpt_api()
        await test_async_evaluator()
        print("\nAll async tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 