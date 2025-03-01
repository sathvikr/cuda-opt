import logging
from src.opt import opt
from src.loader import read_cu, write_cu

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cuda_optimizer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting CUDA optimization process")
    
    try:
        input_file = "tests/matmul.cu"
        output_file = "tests/matmul_opt.cu"
        
        logger.info(f"Reading input file: {input_file}")
        code = read_cu(input_file)
        
        logger.info("Optimizing CUDA code")
        optimized_code = opt(code)
        
        logger.info(f"Writing optimized code to: {output_file}")
        write_cu(output_file, optimized_code)
        
        logger.info("CUDA optimization process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in optimization process: {e}")
        raise
