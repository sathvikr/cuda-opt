import os
from dotenv import load_dotenv
import requests
import logging
from tqdm import tqdm
from .arxiv import ArxivClient
import json
import re

logger = logging.getLogger(__name__)

class CUDAOptimizer:
    def __init__(self, deepseek_api_key: str, arxiv_client: ArxivClient, k: int = 3):
        self.deepseek_api_key = deepseek_api_key
        self.arxiv_client = arxiv_client
        self.k = k
        self.api_endpoint = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-reasoner"

    def optimize(self, code: str) -> list[str]:
        """Optimize CUDA kernel using DeepSeek API and ArXiv papers
        Returns a list of k optimized kernel candidates"""
        logger.info(f"Starting optimization process (targeting {self.k} kernels)...")
        
        # Get relevant papers
        paper_urls = self.arxiv_client.search_papers(code, num_results=self.k)
        optimized_kernels = []
        
        # Generate one kernel per paper with progress bar
        with tqdm(total=len(paper_urls), desc="Generating optimized kernels", unit="kernel") as pbar:
            for i, url in enumerate(paper_urls, 1):
                logger.info(f"\n📝 Processing paper {i}/{len(paper_urls)}")
                text = self.arxiv_client.extract_paper_text(url)
                if not text:
                    pbar.update(1)
                    continue
                
                logger.info("🤖 Implementing draft kernel optimization...")
                # Prepare API request
                headers = {
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json"
                }

                prompt = f"""
                Using this paper as reference:
                <paper>
                {text}
                </paper>
                Optimize this CUDA kernel:
                <kernel>
                {code}
                </kernel>
                Output the optimized CUDA kernel in <cuda></cuda> tags.
                """
                
                payload = {
                    "model": self.model,
                    "messages": [{
                        "role": "user", 
                        "content": prompt
                    }]
                }
                
                # Make API request
                try:
                    logger.info(f"Making request to DeepSeek API...")
                    response = requests.post(
                        self.api_endpoint,
                        json=payload,
                        headers=headers,
                        timeout=60  # Add timeout
                    )
                    
                    # Log raw response first
                    logger.debug(f"Raw response: {response.text}")
                    logger.debug(f"Response status: {response.status_code}")
                    logger.debug(f"Response headers: {response.headers}")
                    
                    # Check response status
                    if response.status_code != 200:
                        logger.error(f"❌ API request failed with status {response.status_code}")
                        logger.error(f"Response: {response.text}")
                        continue
                        
                    if not response.text:
                        logger.error("❌ Empty response from API")
                        continue
                    
                    try:
                        response_json = response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse JSON response: {e}")
                        logger.error(f"Response content: {response.text[:1000]}")  # Show more content
                        continue
                    
                    if "choices" not in response_json or not response_json["choices"]:
                        logger.error("❌ No choices in API response")
                        logger.error(f"Response JSON: {response_json}")
                        continue
                        
                    optimized_code = response_json["choices"][0]["message"]["content"]
                    
                    # Extract code between <cuda> tags
                    cuda_match = re.search(r'<cuda>(.*?)</cuda>', optimized_code, re.DOTALL)
                    if cuda_match:
                        optimized_code = cuda_match.group(1).strip()
                        logger.info("✓ Successfully extracted CUDA code")
                    else:
                        logger.error("❌ No CUDA code found in response")
                        logger.debug(f"Response content: {optimized_code[:500]}...")
                        continue
                        
                    optimized_kernels.append(optimized_code)
                    pbar.set_postfix({"status": "success"})
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"❌ Network error during API request: {str(e)}")
                    logger.error(f"Request payload: {payload}")
                    pbar.set_postfix({"status": "failed - network"})
                except Exception as e:
                    logger.error(f"❌ Unexpected error: {str(e)}")
                    logger.error(f"Request payload: {payload}")
                    pbar.set_postfix({"status": "failed - unknown"})
                
                pbar.update(1)
        
        if not optimized_kernels:
            raise Exception("Failed to generate any optimized kernels")
            
        logger.info(f"Successfully generated {len(optimized_kernels)} optimized kernels")
        return optimized_kernels
