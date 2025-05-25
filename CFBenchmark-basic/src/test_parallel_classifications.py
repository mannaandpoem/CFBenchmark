#!/usr/bin/env python3
"""
Test script to demonstrate parallel classification processing in CFBenchmark.
This script shows how classifications are now processed in parallel.
"""

from CFBenchmark import CFBenchmark
import sys
import os
import time

# Add parent directory to path to import parallel_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parallel_config import get_config

def test_parallel_classifications():
    """Test the parallel classification processing functionality."""
    
    print("=" * 60)
    print("Testing Parallel Classification Processing")
    print("=" * 60)
    
    # EXPERIMENT SETUP
    api_key = "sk-xxx"
    base_url = "https://openrouter.ai/api/v1"
    model_name = "openai/gpt-4o-mini"

    # Change work directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current working directory: {os.getcwd()}")

    # Paths setup
    fewshot_text_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/fewshot'
    test_type = 'zero-shot'
    response_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/cfbenchmark-response'
    scores_path = '../cfbenchmark-scores'
    benchmark_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/data'

    # Use conservative configuration for testing
    config_type = "conservative"
    parallel_config = get_config(config_type)
    
    print(f"\nUsing {config_type} parallel configuration:")
    print(f"  - Max workers: {parallel_config['max_workers']}")
    print(f"  - API rate limit: {parallel_config['api_rate_limit']} seconds")

    # Create CFBenchmark instance with parallel processing
    cfb = CFBenchmark(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        fewshot_text_path=fewshot_text_path,
        test_type=test_type,
        response_path=response_path,
        scores_path=scores_path,
        benchmark_path=benchmark_path,
        **parallel_config
    )
    
    print(f"\nClassifications to process: {cfb.classifications}")
    print(f"Number of classifications: {len(cfb.classifications)}")
    
    # Test parallel model results processing
    print("\n" + "=" * 40)
    print("Testing Parallel Model Results Processing")
    print("=" * 40)
    
    start_time = time.time()
    cfb.get_model_results()
    model_results_time = time.time() - start_time
    
    print(f"\nModel results processing completed in {model_results_time:.2f} seconds")
    
    # Test parallel test scores processing
    print("\n" + "=" * 40)
    print("Testing Parallel Test Scores Processing")
    print("=" * 40)
    
    start_time = time.time()
    cfb.get_test_scores()
    test_scores_time = time.time() - start_time
    
    print(f"\nTest scores processing completed in {test_scores_time:.2f} seconds")
    
    # Summary
    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total classifications processed: {len(cfb.classifications)}")
    print(f"Model results time: {model_results_time:.2f} seconds")
    print(f"Test scores time: {test_scores_time:.2f} seconds")
    print(f"Total time: {model_results_time + test_scores_time:.2f} seconds")
    print(f"Configuration used: {config_type}")
    print(f"Max workers: {parallel_config['max_workers']}")
    print("\nParallel processing benefits:")
    print("✓ Classifications processed concurrently")
    print("✓ API calls within each classification parallelized")
    print("✓ Evaluation metrics calculated in parallel")
    print("✓ Thread-safe rate limiting")
    print("=" * 60)

if __name__ == "__main__":
    test_parallel_classifications() 