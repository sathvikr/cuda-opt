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
                # check this
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

    def tree_of_thought(self, paper: str, kernel: str, nvcc_metrics: dict, max_iterations: int = 3, 
                       branching_factor: int = 3, max_depth: int = 2) -> str:
        """
        Optimize CUDA kernel using a tree-of-thought approach with DeepSeek API.
        
        Args:
            paper: Text content of the research paper to use for optimization
            kernel: Original CUDA kernel code to optimize
            nvcc_metrics: Initial performance metrics from NVCC for the kernel
            max_iterations: Maximum number of optimization iterations per branch
            branching_factor: Number of optimization techniques to try at each step
            max_depth: Maximum depth of the optimization tree
            
        Returns:
            The best optimized kernel found during the search
        """
        logger.info("Starting Tree of Thought optimization process...")
        
        # Track the best kernel and its metrics
        best_kernel = kernel
        best_metrics = nvcc_metrics
        
        def dfs_optimize(current_kernel, current_metrics, depth=0):
            nonlocal best_kernel, best_metrics
            
            if depth >= max_depth:
                logger.info(f"Reached maximum depth {max_depth}, stopping this branch")
                return
            
            # Generate optimization techniques based on paper and current kernel
            logger.info(f"Generating optimization techniques at depth {depth}...")
            techniques = self._generate_optimization_techniques(paper, current_kernel, current_metrics, branching_factor)
            
            for i, technique in enumerate(techniques):
                logger.info(f"Trying optimization technique {i+1}/{len(techniques)} at depth {depth}")
                
                # Apply the technique to generate a new kernel
                new_kernel = self._apply_optimization_technique(current_kernel, technique, paper)
                if not new_kernel:
                    logger.warning(f"Failed to apply technique {i+1}, skipping")
                    continue
                    
                # Evaluate the new kernel (this would call your NVCC evaluation code)
                new_metrics = self._evaluate_kernel(new_kernel)
                
                # Check if this is the best kernel so far
                if self._is_better_kernel(new_metrics, best_metrics):
                    logger.info(f"Found better kernel at depth {depth}, technique {i+1}")
                    best_kernel = new_kernel
                    best_metrics = new_metrics
                
                # Continue DFS with this new kernel
                dfs_optimize(new_kernel, new_metrics, depth + 1)
        
        # Start DFS from the original kernel
        dfs_optimize(kernel, nvcc_metrics)
        
        logger.info("Tree of Thought optimization completed")
        return best_kernel

    def _generate_optimization_techniques(self, paper: str, kernel: str, metrics: dict, num_techniques: int) -> list[str]:
        """Generate different optimization techniques based on the paper and current kernel"""
        logger.info("Generating optimization techniques...")
        
        # Truncate paper text to avoid exceeding token limits
        max_paper_length = 3000
        truncated_paper = paper[:max_paper_length] + ("..." if len(paper) > max_paper_length else "")
        
        # Format metrics as a string
        metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        
        prompt = (
            f"Based on this research paper and the current CUDA kernel metrics, suggest {num_techniques} different "
            f"optimization techniques that could improve performance. For each technique, provide:\n"
            f"1. A name for the technique\n"
            f"2. A brief explanation of how it works\n"
            f"3. Why it might improve performance for this specific kernel\n\n"
            f"Paper excerpt:\n{truncated_paper}\n\n"
            f"Current CUDA kernel:\n{kernel}\n\n"
            f"Current performance metrics:\n{metrics_str}\n\n"
            f"Format each technique as: <technique>name: explanation and reasoning</technique>"
        )
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user", 
                "content": prompt
            }]
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                logger.error("No choices in API response")
                return []
                
            content = response_data["choices"][0]["message"]["content"]
            
            # Extract techniques using regex
            techniques = re.findall(r'<technique>(.*?)</technique>', content, re.DOTALL)
            logger.info(f"Generated {len(techniques)} optimization techniques")
            return techniques
            
        except Exception as e:
            logger.error(f"Error generating optimization techniques: {str(e)}")
            return []

    def _apply_optimization_technique(self, kernel: str, technique: str, paper: str) -> str:
        """Apply a specific optimization technique to the kernel"""
        logger.info(f"Applying optimization technique: {technique[:50]}...")
        
        prompt = (
            f"Apply the following optimization technique to the CUDA kernel:\n\n"
            f"Technique: {technique}\n\n"
            f"Original CUDA kernel:\n{kernel}\n\n"
            f"Paper context:\n{paper[:2000]}...\n\n"
            f"Provide only the optimized CUDA kernel code in your response, wrapped in <cuda></cuda> tags."
        )
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user", 
                "content": prompt
            }]
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                logger.error("No choices in API response")
                return ""
                
            content = response_data["choices"][0]["message"]["content"]
            
            # Extract code between <cuda> tags
            cuda_match = re.search(r'<cuda>(.*?)</cuda>', content, re.DOTALL)
            if cuda_match:
                return cuda_match.group(1).strip()
            else:
                logger.error("No CUDA code found in response")
                return ""
                
        except Exception as e:
            logger.error(f"Error applying optimization technique: {str(e)}")
            return ""

    def _evaluate_kernel(self, kernel: str) -> dict:
        """
        Evaluate a kernel using NVIDIA Nsight Compute (NCU) metrics
        """
        logger.info("Evaluating kernel performance with NVIDIA Nsight Compute...")
        
        try:
            # Create a temporary file with the kernel code
            import tempfile
            import subprocess
            import pandas as pd
            import os
            
            # Create a temporary directory for our files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write the kernel to a temporary file
                kernel_file = os.path.join(temp_dir, "kernel.cu")
                with open(kernel_file, "w") as f:
                    f.write(kernel)
                
                # Compile the kernel
                logger.info("Compiling kernel...")
                compile_cmd = f"nvcc -o {os.path.join(temp_dir, 'kernel')} {kernel_file}"
                subprocess.run(compile_cmd, shell=True, check=True)
                
                # Run NCU profiling
                logger.info("Running NCU profiling...")
                profile_path = os.path.join(temp_dir, "profile")
                ncu_cmd = f"sudo /usr/local/cuda-12.4/bin/ncu -o {profile_path} {os.path.join(temp_dir, 'kernel')}"
                subprocess.run(ncu_cmd, shell=True, check=True)
                
                # Export to CSV
                logger.info("Exporting profile data to CSV...")
                csv_path = os.path.join(temp_dir, "profile.csv")
                export_cmd = f"sudo /usr/local/cuda-12.4/bin/ncu --import {profile_path}.ncu-rep --csv --page raw > {csv_path}"
                subprocess.run(export_cmd, shell=True, check=True)
                
                # Parse the CSV to extract metrics
                logger.info("Parsing profiling results...")
                metrics = {}
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Extract key metrics
                    # Look for throughput metric
                    throughput_row = df[df['Metric Name'].str.contains('sm__throughput.avg.pct_of_peak_sustained_elapsed', na=False)]
                    if not throughput_row.empty:
                        metrics['sm_throughput_pct'] = float(throughput_row['Metric Value'].iloc[0])
                    
                    # Look for execution time
                    exec_time_row = df[df['Metric Name'].str.contains('gpu__time_duration.sum', na=False)]
                    if not exec_time_row.empty:
                        metrics['execution_time_ms'] = float(exec_time_row['Metric Value'].iloc[0])
                    
                    # Look for register usage
                    reg_row = df[df['Metric Name'].str.contains('launch__registers_per_thread', na=False)]
                    if not reg_row.empty:
                        metrics['register_usage'] = int(reg_row['Metric Value'].iloc[0])
                    
                    # Look for occupancy
                    occ_row = df[df['Metric Name'].str.contains('sm__warps_launched.avg.pct_of_peak_sustained_elapsed', na=False)]
                    if not occ_row.empty:
                        metrics['occupancy'] = float(occ_row['Metric Value'].iloc[0]) / 100.0  # Convert percentage to decimal
                    
                    # Look for shared memory usage
                    shared_mem_row = df[df['Metric Name'].str.contains('launch__shared_mem_per_block_allocated', na=False)]
                    if not shared_mem_row.empty:
                        metrics['shared_memory_usage'] = int(shared_mem_row['Metric Value'].iloc[0])
                    
                except Exception as e:
                    logger.error(f"Error parsing CSV: {str(e)}")
                    
                # If we couldn't get all metrics, add defaults
                if 'execution_time_ms' not in metrics:
                    logger.warning("Could not find execution time metric, using placeholder")
                    metrics['execution_time_ms'] = 1.0
                
                if 'register_usage' not in metrics:
                    logger.warning("Could not find register usage metric, using placeholder")
                    metrics['register_usage'] = 32
                    
                if 'shared_memory_usage' not in metrics:
                    logger.warning("Could not find shared memory usage metric, using placeholder")
                    metrics['shared_memory_usage'] = 0
                    
                if 'occupancy' not in metrics:
                    logger.warning("Could not find occupancy metric, using placeholder")
                    metrics['occupancy'] = 0.7
                    
                if 'sm_throughput_pct' not in metrics:
                    logger.warning("Could not find SM throughput metric, using placeholder")
                    metrics['sm_throughput_pct'] = 50.0
                    
                logger.info(f"Collected metrics: {metrics}")
                return metrics
                
        except Exception as e:
            logger.error(f"Error during kernel evaluation: {str(e)}")
            # Return default metrics if evaluation fails
            return {
                "execution_time_ms": 1.0,
                "register_usage": 32,
                "shared_memory_usage": 0,
                "occupancy": 0.7,
                "sm_throughput_pct": 50.0
            }

    def _is_better_kernel(self, new_metrics: dict, best_metrics: dict) -> bool:
        """
        Compare metrics to determine if the new kernel is better
        Prioritizes execution time but also considers other factors including SM throughput
        """
        # Primary metric is execution time (lower is better)
        if new_metrics["execution_time_ms"] < best_metrics["execution_time_ms"] * 0.95:
            # If at least 5% faster, consider it better
            return True
            
        # If execution time is similar (within 5%)
        if (new_metrics["execution_time_ms"] <= best_metrics["execution_time_ms"] * 1.05 and
            new_metrics["execution_time_ms"] >= best_metrics["execution_time_ms"] * 0.95):
            
            # Check SM throughput - higher is better
            if 'sm_throughput_pct' in new_metrics and 'sm_throughput_pct' in best_metrics:
                if new_metrics["sm_throughput_pct"] > best_metrics["sm_throughput_pct"] * 1.1:
                    return True
            
            # Check secondary metrics - better occupancy is good
            if new_metrics["occupancy"] > best_metrics["occupancy"] * 1.1:
                return True
                
            # Lower register usage is good if it doesn't hurt performance
            if new_metrics["register_usage"] < best_metrics["register_usage"] * 0.9:
                return True
        
        return False

        
