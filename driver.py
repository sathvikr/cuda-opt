import logging
from dotenv import load_dotenv
import os
from pathlib import Path
import click
from datetime import datetime
from src.loader import CUDAFileHandler
from src.opt import CUDAOptimizer
from src.arxiv import ArxivClient

def setup_logging(output_dir: Path, verbose: bool):
    """Configure logging with timestamp-based file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f"cuda_opt_{timestamp}.log"
    
    handlers = [logging.FileHandler(log_file)]
    if verbose:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s | %(levelname)8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def setup_directories(output_dir: Path):
    """Create required directory structure"""
    (output_dir / "papers").mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_kernels").mkdir(exist_ok=True)
    return output_dir

@click.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True),
              help='Path to input CUDA kernel file')
@click.option('--output-dir', '-o', required=True, type=click.Path(),
              help='Directory for output files and logs')
@click.option('--kernels', '-k', default=3, type=int,
              help='Number of candidate kernels to generate')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def main(input_file, output_dir, kernels, verbose):
    """CUDA Kernel Optimizer using ArXiv papers and LLMs"""
    # Setup directories and logging
    output_path = Path(output_dir)
    setup_directories(output_path)
    logger = setup_logging(output_path, verbose)
    
    logger.info("🚀 Starting CUDA optimization process")
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not deepseek_api_key or not perplexity_api_key:
        raise click.ClickException("❌ Missing required API keys in environment variables")
    
    try:
        # Initialize components
        file_handler = CUDAFileHandler(
            input_path=input_file,
            output_path=output_path / "candidate_kernels"
        )
        
        arxiv_client = ArxivClient(
            perplexity_api_key=perplexity_api_key,
            papers_dir=output_path / "papers"
        )
        
        optimizer = CUDAOptimizer(
            deepseek_api_key=deepseek_api_key,
            arxiv_client=arxiv_client,
            k=kernels
        )
        
        # Process the optimization
        logger.info("📖 Reading input CUDA kernel...")
        cuda_code = file_handler.read_input()
        optimized_kernels = optimizer.optimize(cuda_code)
        #optimized_kernel = optimizer.tree_of_thought(papers, cuda_code, nvcc_metrics, max_iterations: int = 2, branching_factor: int = 2, max_depth: int = 2)
        
        # Write kernels
        logger.info("\n💾 Saving optimized kernels...")
        for i, kernel in enumerate(optimized_kernels, 1):
            output_path = f"candidate_{i}.cu"
            file_handler.write_output(kernel, output_path)
            logger.info(f"✓ Saved kernel {i} to {output_path}")
        
        logger.info("✨ CUDA optimization process completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in optimization process: {e}")
        raise click.ClickException(str(e))

if __name__ == "__main__":
    main()
