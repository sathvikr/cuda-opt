import logging

logger = logging.getLogger(__name__)

def read_cu(file_path: str) -> str:
    logger.info(f"Reading CUDA file from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            contents = f.read()
            logger.debug(f"Successfully read {len(contents)} bytes from file")
            return contents
    except FileNotFoundError:
        logger.error(f"Error: Could not find file {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise

def write_cu(file_path: str, code: str):
    logger.info(f"Writing CUDA file to: {file_path}")
    try:
        with open(file_path, 'w') as f:
            f.write(code)
        logger.debug(f"Successfully wrote {len(code)} bytes to file")
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        raise
