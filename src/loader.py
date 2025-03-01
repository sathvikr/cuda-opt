import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CUDAFileHandler:
    def __init__(self, input_path: str, output_path: Path):
        self.input_path = input_path
        self.output_dir = output_path
        
    def read_input(self) -> str:
        """Read CUDA code from input file"""
        logger.info(f"Reading CUDA file from: {self.input_path}")
        try:
            with open(self.input_path, 'r') as f:
                contents = f.read()
                logger.debug(f"Successfully read {len(contents)} bytes from file")
                return contents
        except FileNotFoundError:
            logger.error(f"Error: Could not find file {self.input_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise

    def write_output(self, code: str, filename: str) -> None:
        """Write optimized CUDA code to output file in the output directory"""
        output_path = self.output_dir / filename
        logger.info(f"Writing CUDA file to: {output_path}")
        try:
            with open(output_path, 'w') as f:
                f.write(code)
            logger.debug(f"Successfully wrote {len(code)} bytes to file")
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            raise
