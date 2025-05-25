# Parallel Processing Configuration for CFBenchmark
# This file contains different configuration examples for parallel processing

# Configuration for high-throughput processing (if you have high API rate limits)
HIGH_THROUGHPUT_CONFIG = {
    "max_workers": 50,
    "api_rate_limit": 0.0001  # 100ms between calls
}

# Configuration for moderate processing (balanced approach)
MODERATE_CONFIG = {
    "max_workers": 4,
    "api_rate_limit": 0.5  # 500ms between calls
}

# Configuration for conservative processing (to avoid rate limiting)
CONSERVATIVE_CONFIG = {
    "max_workers": 2,
    "api_rate_limit": 1.0  # 1 second between calls
}

# Configuration for free tier or limited API access
FREE_TIER_CONFIG = {
    "max_workers": 1,
    "api_rate_limit": 2.0  # 2 seconds between calls
}

def get_config(config_type="moderate"):
    """
    Get configuration based on your API limits and requirements.
    
    Args:
        config_type (str): One of "high_throughput", "moderate", "conservative", "free_tier"
    
    Returns:
        dict: Configuration dictionary with max_workers and api_rate_limit
    """
    configs = {
        "high_throughput": HIGH_THROUGHPUT_CONFIG,
        "moderate": MODERATE_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
        "free_tier": FREE_TIER_CONFIG
    }
    
    return configs.get(config_type, MODERATE_CONFIG)

# Example usage:
# from parallel_config import get_config
# config = get_config("conservative")
# cfb = CFBenchmark(..., **config) 