#!/usr/bin/env python3
"""
Example usage of the updated CFBenchmark with transformers-based embedding
"""

import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from CFBenchmark import CFBenchmark

def main():
    """Example usage of CFBenchmark with the new embedding functionality."""
    
    print("CFBenchmark Example Usage")
    print("=" * 50)
    
    # Example 1: Basic initialization
    print("\n1. Basic initialization:")
    try:
        benchmark = CFBenchmark(
            api_key="your_openai_api_key_here",  # Replace with your actual API key
            model_name="gpt-3.5-turbo",
            embedding_model_path="BAAI/bge-large-zh-v1.5",  # Default embedding model
            test_type="few-shot",
            max_workers=2,  # Reduce for testing
            api_rate_limit=1.0  # 1 second between API calls
        )
        print("✓ CFBenchmark initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing CFBenchmark: {str(e)}")
        return
    
    # Example 2: Test embedding functionality
    print("\n2. Testing embedding functionality:")
    test_data = [
        {
            'output': '建议投资者关注科技股的发展前景',
            'response': '推荐关注科技类股票的投资机会'
        },
        {
            'output': '市场风险较高，建议谨慎投资',
            'response': '当前市场存在较大风险，投资需谨慎'
        },
        {
            'output': '完全不相关的内容',
            'response': '建议投资者关注科技股的发展前景'
        }
    ]
    
    for i, test_row in enumerate(test_data, 1):
        similarity = benchmark.get_cosine_similarities(test_row)
        print(f"   Test {i}: Similarity = {similarity:.4f}")
        print(f"     Output:   {test_row['output']}")
        print(f"     Response: {test_row['response']}")
        print()
    
    # Example 3: Custom embedding model
    print("\n3. Using a different embedding model:")
    try:
        custom_benchmark = CFBenchmark(
            api_key="your_openai_api_key_here",
            embedding_model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Alternative model
            max_workers=1
        )
        print("✓ Custom embedding model initialized")
    except Exception as e:
        print(f"⚠️  Custom model failed (expected if model not available): {str(e)}")
    
    # Example 4: Install dependencies
    print("\n4. Installing optional dependencies:")
    print("   To install required dependencies, run:")
    print("   CFBenchmark.install_optional_dependencies()")
    
    # Example 5: Running the benchmark (commented out to avoid API calls)
    print("\n5. Running the full benchmark:")
    print("   # Uncomment the following lines to run the actual benchmark:")
    print("   # benchmark.get_model_results()  # Generate model responses")
    print("   # benchmark.get_test_scores()    # Calculate evaluation scores")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nNext steps:")
    print("1. Replace 'your_openai_api_key_here' with your actual OpenAI API key")
    print("2. Ensure you have the required dependencies installed")
    print("3. Prepare your benchmark data in the correct format")
    print("4. Run benchmark.get_model_results() and benchmark.get_test_scores()")

if __name__ == "__main__":
    main() 