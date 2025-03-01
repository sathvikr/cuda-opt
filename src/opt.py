import os
from dotenv import load_dotenv
import requests
import logging

logger = logging.getLogger(__name__)

def opt(code: str) -> str:
    logger.info("Starting CUDA kernel optimization")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up the API endpoint and headers
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not found in environment variables")
        raise ValueError("DEEPSEEK_API_KEY not found")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare the request payload
    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": f"Optimize this CUDA kernel.\n\n{code}"}]
    }
    
    logger.debug("Making API request to DeepSeek")
    # Make the API request
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=payload,
            headers=headers
        )
        
        # Handle the response
        if response.status_code == 200:
            optimized_code = response.json()["choices"][0]["message"]["content"]
            logger.info("Successfully received optimized code from API")
            return optimized_code
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during API request: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during optimization: {e}")
        raise
