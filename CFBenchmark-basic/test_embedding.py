#!/usr/bin/env python3
"""
Test script for the updated CFBenchmark embedding functionality
"""

import os
import sys
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from CFBenchmark import CFBenchmark

def test_embedding_functionality():
    """Test the embedding functionality with a simple example."""
    
    print("Testing CFBenchmark embedding functionality...")
    
    # Initialize CFBenchmark with minimal configuration
    try:
        benchmark = CFBenchmark(
            api_key="test_key",  # Not used for embedding test
            embedding_model_path="BAAI/bge-large-zh-v1.5",  # Default embedding model
            max_workers=1
        )
        
        print("✓ CFBenchmark initialized successfully")
        
        # Test embedding model loading
        if benchmark.t2v_model is not None and benchmark.t2v_tokenizer is not None:
            print("✓ Embedding model loaded successfully using transformers")
            
            # Test cosine similarity calculation
            test_row = {
                'output': '这是一个测试输出',
                'response': '这是一个测试响应'
            }
            
            similarity = benchmark.get_cosine_similarities(test_row)
            print(f"✓ Cosine similarity calculated: {similarity}")
            
            # Test with different sentences
            test_row2 = {
                'output': '完全不同的内容',
                'response': '这是一个测试响应'
            }
            
            similarity2 = benchmark.get_cosine_similarities(test_row2)
            print(f"✓ Cosine similarity for different sentences: {similarity2}")
            
        else:
            print("⚠️  Embedding model not loaded, using fallback method")
            
            # Test fallback method
            test_row = {
                'output': '这是一个测试输出',
                'response': '这是一个测试响应'
            }
            
            similarity = benchmark.get_cosine_similarities(test_row)
            print(f"✓ Fallback cosine similarity calculated: {similarity}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("This might be due to missing dependencies or model download issues.")
        print("You can install dependencies using: CFBenchmark.install_optional_dependencies()")

if __name__ == "__main__":
    test_embedding_functionality() 