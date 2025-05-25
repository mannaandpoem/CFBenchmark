from utils.gpt_utils import gpt_api
from typing import Optional
import asyncio


class ChatGPTEvaluator:
    def __init__(self, 
                 model_type: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1"):
        """
        Initialize ChatGPT evaluator with configurable parameters.
        
        Args:
            model_type: OpenAI model name
            api_key: API key (if None, uses environment variable)
            base_url: API base URL for custom endpoints
        """
        self.model_type = model_type
        self.api_key = api_key
        self.base_url = base_url

    def answer(self, query: str, model_type: Optional[str] = None) -> str:
        """
        Generate answer using OpenAI API (synchronous).
        
        Args:
            query: Input prompt
            model_type: Override model type for this call
            
        Returns:
            Generated response
        """
        model_to_use = model_type if model_type is not None else self.model_type
        return asyncio.run(gpt_api(
            query=query, 
            model_type=model_to_use,
            api_key=self.api_key,
            base_url=self.base_url
        ))

    async def answer_async(self, query: str, model_type: Optional[str] = None) -> str:
        """
        Generate answer using OpenAI API (asynchronous).
        
        Args:
            query: Input prompt
            model_type: Override model type for this call
            
        Returns:
            Generated response
        """
        model_to_use = model_type if model_type is not None else self.model_type
        return await gpt_api(
            query=query, 
            model_type=model_to_use,
            api_key=self.api_key,
            base_url=self.base_url
        )


if __name__ == "__main__":
    evaluator = ChatGPTEvaluator()
    print(evaluator.answer("你好"))
    
    # Test async version
    async def test_async():
        result = await evaluator.answer_async("你好")
        print(f"Async result: {result}")
    
    asyncio.run(test_async())
