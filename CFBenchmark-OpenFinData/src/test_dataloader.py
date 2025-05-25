#!/usr/bin/env python3
"""
Test script for the dataloader functionality.
"""

import os
import sys
from utils.dataloader import load_dataset

def test_dataloader():
    """Test the dataloader functionality"""
    print("Testing dataloader...")
    
    # Get the correct path to data directory
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(os.path.dirname(current_dir), "data")
    
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: Data directory not found at {data_path}")
        return False
    
    try:
        dataset = load_dataset(data_path)
        print(f"Successfully loaded {len(dataset)} examples")
        
        if len(dataset) > 0:
            # Show sample data
            sample = dataset[0]
            print("\nSample data structure:")
            for key, value in sample.items():
                if isinstance(value, dict):
                    print(f"  {key}: {type(value)} with {len(value)} items")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
            
            # Count by domain
            domain_counts = {}
            for item in dataset:
                domain = item.get('domain', 'unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            print("\nData distribution by domain:")
            for domain, count in sorted(domain_counts.items()):
                print(f"  {domain}: {count} examples")
                
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataloader()
    if success:
        print("\n✅ Dataloader test passed!")
    else:
        print("\n❌ Dataloader test failed!")
        sys.exit(1) 