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

                # Truncate paper text to avoid exceeding token limits
                max_paper_length = 4000  # Adjust this value as needed
                truncated_text = text[:max_paper_length] + ("..." if len(text) > max_paper_length else "")

                prompt = (
                    f"Optimize this CUDA kernel using insights from this research paper.\n\n"
                    f"Paper excerpt:\n{truncated_text}\n\n"
                    f"CUDA kernel to optimize:\n{code}\n\n"
                    f"Provide only the optimized CUDA kernel code in your response, wrapped in <cuda></cuda> tags."
                )
                
                payload = {
                    "model": self.model,
                    "messages": [{
                        "role": "user", 
                        "content": prompt
                    }]
                }

                try:
                    # Make API request
                    logger.debug("Sending request to DeepSeek API...")
                    response = requests.post(
                        self.api_endpoint,
                        headers=headers,
                        json=payload,
                        timeout=120  # Increased timeout for large prompts
                    )
                    
                    response.raise_for_status()  # Raise exception for non-200 status codes
                    
                    # Parse response
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON response. Response text: {response.text[:500]}...")
                        pbar.set_postfix({"status": "failed - invalid JSON"})
                        continue

                    if "choices" not in response_data or not response_data["choices"]:
                        logger.error("❌ No choices in API response")
                        logger.error(f"Response JSON: {response_data}")
                        pbar.set_postfix({"status": "failed - no choices"})
                        continue
                        
                    optimized_code = response_data["choices"][0]["message"]["content"]
                    
                    # Extract code between <cuda> tags
                    cuda_match = re.search(r'<cuda>(.*?)</cuda>', optimized_code, re.DOTALL)
                    if cuda_match:
                        optimized_code = cuda_match.group(1).strip()
                        logger.info("✓ Successfully extracted CUDA code")
                        optimized_kernels.append(optimized_code)
                        pbar.set_postfix({"status": "success"})
                    else:
                        logger.error("❌ No CUDA code found in response")
                        logger.debug(f"Response content: {optimized_code[:500]}...")
                        pbar.set_postfix({"status": "failed - no CUDA tags"})
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"❌ Network error during API request: {str(e)}")
                    pbar.set_postfix({"status": "failed - network"})
                except Exception as e:
                    logger.error(f"❌ Unexpected error: {str(e)}")
                    logger.debug(f"Response content if available: {getattr(response, 'text', '')[:500]}...")
                    pbar.set_postfix({"status": "failed - unknown"})
                
                pbar.update(1)
        
        if not optimized_kernels:
            raise Exception("Failed to generate any optimized kernels")
            
        logger.info(f"Successfully generated {len(optimized_kernels)} optimized kernels")
        return optimized_kernels
