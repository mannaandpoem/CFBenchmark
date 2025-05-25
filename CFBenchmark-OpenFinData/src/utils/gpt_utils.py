import os
import time
import asyncio
from openai import AsyncOpenAI
from typing import Optional

async def gpt_api(query: str, 
                 model_type: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 max_retries: int = 3,
                 retry_delay: float = 2.0) -> str:
    """
    Generate text using OpenAI API with async client and error handling.
    
    Args:
        query: The input prompt
        model_type: Model name to use
        api_key: API key (if None, uses environment variable)
        base_url: API base URL for custom endpoints
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        Generated text response
    """
    
    # Set up API key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    # Create async OpenAI client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            # Use chat completions for modern models
            if model_type.startswith("gpt-") or "turbo" in model_type.lower():
                response = await client.chat.completions.create(
                    model=model_type,
                    messages=[
                        {"role": "user", "content": query}
                    ],
                    max_tokens=1024,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            else:
                # Fallback for older completion models
                response = await client.completions.create(
                    model=model_type,
                    prompt=query,
                    max_tokens=1024,
                    temperature=0.7
                )
                return response.choices[0].text.strip()
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error: {str(e)}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return "API Error"
    
    return "API Error"


if __name__ == '__main__':
    async def test():
        result = await gpt_api("你是谁")
        print(result)
    
    asyncio.run(test())
