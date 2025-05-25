#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced cosine similarity functionality
without requiring SentenceTransformer.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from CFBenchmark import CFBenchmark
import pandas as pd

def test_cosine_similarity():
    """Test the enhanced cosine similarity implementation."""
    
    print("Testing Enhanced Cosine Similarity (without SentenceTransformer)")
    print("=" * 60)
    
    # Create a CFBenchmark instance (without API key for testing similarity only)
    benchmark = CFBenchmark(
        api_key="dummy_key",  # Not needed for similarity testing
        model_name="test-model"
    )
    
    # Test cases with different similarity levels
    test_cases = [
        {
            "output": "这是一个很好的建议",
            "response": "这是一个很好的建议",
            "expected": "High similarity (identical)"
        },
        {
            "output": "这是一个很好的建议",
            "response": "这是一个不错的建议",
            "expected": "High similarity (similar meaning)"
        },
        {
            "output": "这是一个很好的建议",
            "response": "这是一个糟糕的想法",
            "expected": "Medium similarity (some overlap)"
        },
        {
            "output": "这是一个很好的建议",
            "response": "完全不同的内容",
            "expected": "Low similarity (different content)"
        },
        {
            "output": "Good suggestion for improvement",
            "response": "Excellent recommendation for enhancement",
            "expected": "Medium similarity (English, similar meaning)"
        },
        {
            "output": "Mixed 中英文 content here",
            "response": "Mixed 中英文 content here",
            "expected": "High similarity (mixed language, identical)"
        },
        {
            "output": "",
            "response": "",
            "expected": "Perfect similarity (both empty)"
        },
        {
            "output": "Some content",
            "response": "",
            "expected": "No similarity (one empty)"
        }
    ]
    
    print("Testing cosine similarity calculations:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        # Create a row-like dictionary
        row = {
            "output": test_case["output"],
            "response": test_case["response"]
        }
        
        # Calculate similarity
        similarity = benchmark.get_cosine_similarities(row)
        
        print(f"Test {i}: {test_case['expected']}")
        print(f"  Output:  '{test_case['output']}'")
        print(f"  Response: '{test_case['response']}'")
        print(f"  Similarity: {similarity:.4f}")
        print()
    
    print("✓ All similarity calculations completed successfully!")
    print("\nNote: The enhanced TF-IDF based cosine similarity provides")
    print("reasonable similarity scores without requiring SentenceTransformer.")

def test_installation_helper():
    """Test the installation helper method."""
    print("\nTesting Installation Helper")
    print("=" * 30)
    
    print("To install optional dependencies for better performance, you can run:")
    print("CFBenchmark.install_optional_dependencies()")
    print("\nOr manually install with:")
    print("pip install sentence-transformers datasets")

if __name__ == "__main__":
    test_cosine_similarity()
    test_installation_helper() 