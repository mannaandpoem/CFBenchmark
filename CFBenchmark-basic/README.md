# CFBenchmark with OpenAI API and Parallel Processing

CFBenchmark is a tool for evaluating language models on financial text processing tasks. This version has been modified to work with OpenAI API instead of local Hugging Face models and includes parallel processing capabilities for improved performance.

## Features

- **Parallel API Calls**: Process multiple requests concurrently using ThreadPoolExecutor
- **Parallel Evaluation**: Calculate metrics in parallel using ProcessPoolExecutor
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Configurable Workers**: Adjust the number of parallel workers based on your needs
- **Thread-Safe**: Safe concurrent access to shared resources

## Setup

1. Install required packages:
```bash
pip install openai pandas numpy torch sentence-transformers datasets
```

2. Clone this repository:
```bash
git clone <repository-url>
cd CFBenchmark-basic
```

3. Prepare your benchmark data and few-shot examples in the appropriate directories.

## Configuration

### Basic Configuration

Edit the `run.py` file to configure your benchmark:

```python
# EXPERIMENT SETUP
api_key = "YOUR-OPENAI-API-KEY"  # Replace with your actual API key
base_url = "https://api.openai.com/v1"  # Default OpenAI API URL
model_name = "gpt-3.5-turbo"  # or any other OpenAI model like "gpt-4"

# Parallel processing configuration
max_workers = 4  # Number of parallel workers
api_rate_limit = 0.5  # Minimum seconds between API calls
```

### Advanced Parallel Configuration

Use the `parallel_config.py` file for predefined configurations:

```python
from parallel_config import get_config

# Choose configuration based on your API limits:
# - "high_throughput": 8 workers, 0.1s rate limit
# - "moderate": 4 workers, 0.5s rate limit  
# - "conservative": 2 workers, 1.0s rate limit
# - "free_tier": 1 worker, 2.0s rate limit

config = get_config("moderate")
cfb = CFBenchmark(..., **config)
```

## Running the Benchmark

### Standard Execution
```bash
cd src
python run.py
```

### Parallel Execution with Configuration
```bash
cd src
python run_parallel.py
```

## Parallel Processing Details

### Classification-Level Parallelization
- **Concurrent Classifications**: Different classification tasks (sector, event, etc.) are processed simultaneously
- **Independent Processing**: Each classification task runs in its own thread
- **Scalable Workers**: Number of parallel classifications limited by available workers

### API Call Parallelization
- Uses `ThreadPoolExecutor` for concurrent API requests
- Built-in rate limiting prevents exceeding API limits
- Thread-safe implementation with proper locking mechanisms

### Evaluation Parallelization
- Uses `ThreadPoolExecutor` for CPU-intensive metric calculations (instead of ProcessPoolExecutor to avoid serialization issues)
- Parallel processing of cosine similarity calculations
- Parallel F1 score computation for classification tasks

### Performance Benefits
- **Classification Processing**: Up to Nx faster where N = number of classifications
- **API Calls**: Up to 4x faster with 4 workers (depending on rate limits)
- **Evaluation**: Significant speedup for large datasets
- **Memory Efficient**: Batch processing prevents memory overflow

## Configuration Guidelines

Choose your configuration based on:

1. **API Rate Limits**: Higher limits allow more workers and lower rate limits
2. **Dataset Size**: Larger datasets benefit more from parallelization
3. **System Resources**: More CPU cores support higher worker counts
4. **Cost Considerations**: More parallel calls = higher API costs

### Recommended Settings

| API Tier | Workers | Rate Limit | Use Case |
|----------|---------|------------|----------|
| Free | 1 | 2.0s | Testing, small datasets |
| Basic | 2 | 1.0s | Development, medium datasets |
| Standard | 4 | 0.5s | Production, large datasets |
| Premium | 8 | 0.1s | High-throughput processing |

## Understanding the Code

The CFBenchmark class now includes parallel processing capabilities:

1. `CFBenchmark.__init__()`: Initializes with parallel configuration
2. `CFBenchmark._rate_limited_api_call()`: Thread-safe API calls with rate limiting
3. `CFBenchmark._process_single_row()`: Worker function for parallel row processing
4. `CFBenchmark._process_single_classification()`: Worker function for parallel classification processing
5. `CFBenchmark._process_single_classification_scores()`: Worker function for parallel score calculation
6. `CFBenchmark.get_model_results()`: Parallel classification and API call processing
7. `CFBenchmark.get_test_scores()`: Parallel classification evaluation

## Evaluation Metrics

The benchmark uses different metrics depending on the task:
- For classification tasks (company, product, sector, event, sentiment): F1 score
- For generation tasks (summary, risk, suggestion): Cosine similarity

## Error Handling

- Automatic retry logic for failed API calls
- Graceful handling of rate limit errors
- Fallback mechanisms for evaluation failures
- Comprehensive error logging

## Monitoring and Debugging

- Progress indicators for parallel processing
- Detailed error messages with row indices
- Performance metrics and timing information
- Configurable logging levels

## Customization

You can modify the following aspects:
- OpenAI model (gpt-3.5-turbo, gpt-4, etc.)
- Test type (few-shot or zero-shot)
- API base URL (for using with OpenAI-compatible APIs)
- Various file paths for data and results

## Requirements

- Python 3.6+
- OpenAI Python client
- PyTorch
- Pandas
- NumPy
- Datasets (HuggingFace)
- Sentence Transformers (for evaluation) 