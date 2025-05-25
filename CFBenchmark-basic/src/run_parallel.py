from CFBenchmark import CFBenchmark
import sys
import os

# Add parent directory to path to import parallel_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parallel_config import get_config

if __name__=='__main__':

    # EXPERIMENT SETUP
    api_key = "sk-xxx"  # Replace with your actual API key
    base_url = "https://openrouter.ai/api/v1"  # Default OpenAI API URL, can be changed for other endpoints
    # model_name = "openai/gpt-4o-mini"  # or any other OpenAI model like "gpt-4"
    model_name = "openai/gpt-4.1"  # or any other OpenAI model like "gpt-4"

    # Change work directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current working directory: {os.getcwd()}")

    # Paths setup
    fewshot_text_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/fewshot'  # Path to few-shot examples
    test_type = 'few-shot'  # 'few-shot' or 'zero-shot'
    response_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/cfbenchmark-response'  # Path to store model responses
    scores_path = '../cfbenchmark-scores'  # Path to store evaluation scores
    benchmark_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/data'  # Path to benchmark data

    # Choose parallel configuration based on your API limits
    # Options: "high_throughput", "moderate", "conservative", "free_tier"
    config_type = "high_throughput"  # Change this based on your needs
    parallel_config = get_config(config_type)
    
    print(f"Using {config_type} parallel configuration:")
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
        **parallel_config  # Unpack the parallel configuration
    )
    
    print("Starting parallel benchmark execution...")
    
    # Run the benchmark with parallel processing
    cfb.get_model_results()  # Get responses from the OpenAI model (parallel API calls)
    cfb.get_test_scores()    # Calculate scores based on model responses (parallel evaluation)
    
    print("Benchmark execution completed!") 