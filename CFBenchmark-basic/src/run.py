from CFBenchmark import CFBenchmark
if __name__=='__main__':

    # EXPERIMENT SETUP
    api_key = "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"  # Replace with your actual API key
    base_url = "https://openrouter.ai/api/v1"  # Default OpenAI API URL, can be changed for other endpoints
    # model_name = "google/gemini-2.0-flash-001"  # or any other OpenAI model like "gpt-4"
    model_name = "openai/gpt-4.1-mini"  # or any other OpenAI model like "gpt-4"

    # Change work directory to the script's directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current working directory: {os.getcwd()}")

    # Paths setup
    fewshot_text_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/fewshot'  # Path to few-shot examples
    test_type = 'few-shot'  # 'few-shot' or 'zero-shot'
    response_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/cfbenchmark-response'  # Path to store model responses
    scores_path = '../cfbenchmark-scores'  # Path to store evaluation scores
    benchmark_path = '/Users/manna/PycharmProjects/CFBenchmark/CFBenchmark-basic/data'  # Path to benchmark data

    # Create CFBenchmark instance
    cfb = CFBenchmark(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        fewshot_text_path=fewshot_text_path,
        test_type=test_type,
        response_path=response_path,
        scores_path=scores_path,
        benchmark_path=benchmark_path
    )
    
    # Run the benchmark
    cfb.get_model_results()  # Get responses from the OpenAI model
    cfb.get_test_scores()    # Calculate scores based on model responses