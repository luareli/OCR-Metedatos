import logging
import os
from datetime import datetime, date
from typing import Dict, Any, Optional
import json


def setup_logging():
    """Setup logging configuration

    Sets up both file and console logging with appropriate format.

    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        handler = logging.FileHandler('app.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def validate_file_type(file_name: str, allowed_extensions: set) -> bool:
    """Validate if the file type is allowed"""
    if not file_name or not isinstance(file_name, str):
        return False
    if not allowed_extensions or not isinstance(allowed_extensions, set):
        return False

    _, ext = os.path.splitext(file_name.lower())
    return ext[1:] in allowed_extensions


def validate_file_size(file_obj, max_size_mb: int) -> bool:
    """Validate if the file size is within the allowed limit"""
    if not file_obj:
        return False
    if not isinstance(max_size_mb, int) or max_size_mb <= 0:
        return False

    file_obj.seek(0, 2)  # Move to end of file
    size = file_obj.tell()  # Get current position (file size)
    file_obj.seek(0)  # Move back to the beginning
    max_size_bytes = max_size_mb * 1024 * 1024
    return size <= max_size_bytes


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and other attacks"""
    import re
    # Remove path separators and other potentially dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove any path components to prevent directory traversal
    sanitized = os.path.basename(sanitized)
    return sanitized


def is_binary_file(file_obj, chunk_size: int = 1024) -> bool:
    """Check if a file is binary by examining its content"""
    # Read the first chunk of the file
    chunk = file_obj.read(chunk_size)
    file_obj.seek(0)  # Reset file pointer

    # Check for null bytes or other binary indicators
    if b'\x00' in chunk:
        return True

    # Count printable characters
    printable_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in {9, 10, 13})
    printable_ratio = printable_chars / len(chunk) if len(chunk) > 0 else 1

    # If less than 70% of characters are printable, consider it binary
    return printable_ratio < 0.7


def calculate_file_hash(file_obj) -> str:
    """Calculate SHA-256 hash of a file to identify if it has been processed before"""
    import hashlib

    # Handle both file-like objects and bytes
    if isinstance(file_obj, bytes):
        # If input is bytes, calculate hash directly
        hasher = hashlib.sha256()
        hasher.update(file_obj)
        return hasher.hexdigest()
    else:
        # Handle file-like objects
        # Store current position to restore later
        original_pos = file_obj.tell()
        # Move to beginning of the file
        file_obj.seek(0)

        # Calculate hash
        hasher = hashlib.sha256()
        # Read file in chunks to avoid memory issues with large files
        while chunk := file_obj.read(8192):
            hasher.update(chunk)

        # Restore original position
        file_obj.seek(original_pos)

        return hasher.hexdigest()


def save_processed_result(file_hash: str, extracted_text: str, metadata: Dict[str, Any]) -> bool:
    """Save processed result to cache"""
    import json
    import os
    from pathlib import Path

    try:
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)

        cache_file = cache_dir / f"{file_hash}.json"

        cache_data = {
            "extracted_text": extracted_text,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error saving to cache: {e}")
        return False


def load_cached_result(file_hash: str) -> Optional[Dict[str, Any]]:
    """Load cached result if it exists"""
    import json
    import os
    from pathlib import Path

    try:
        cache_file = Path("./cache") / f"{file_hash}.json"

        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Check if cache is still valid (not older than 24 hours)
            if 'timestamp' in cache_data:
                from datetime import datetime
                timestamp = datetime.fromisoformat(cache_data['timestamp'])
                if (datetime.now() - timestamp).total_seconds() > 24 * 3600:  # 24 hours
                    return None

            return cache_data
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error loading from cache: {e}")

    return None


def format_metadata_for_download(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Format metadata for JSON download, handling special cases like keywords and date objects"""
    formatted = metadata.copy()
    
    # Handle keywords - if they're a string separated by commas, convert to list
    if 'Palabras Clave' in formatted and isinstance(formatted['Palabras Clave'], str):
        keywords_str = formatted['Palabras Clave']
        if keywords_str.strip():
            formatted['Palabras Clave'] = [
                kw.strip() for kw in keywords_str.split(',') if kw.strip()
            ]
        else:
            formatted['Palabras Clave'] = []
    
    # Handle any date objects in the metadata by converting them to ISO format strings
    for key, value in formatted.items():
        if isinstance(value, (datetime, date)):
            formatted[key] = value.isoformat()
    
    # Add processing timestamp
    formatted['timestamp'] = datetime.now().isoformat()
    
    return formatted


def save_to_json(data: Dict[str, Any], filepath: str) -> bool:
    """Save data to JSON file"""
    try:
        # Convert datetime objects to strings for JSON serialization
        serialized_data = _serialize_for_json(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False


def _serialize_for_json(obj):
    """Recursively convert objects for JSON serialization"""
    if isinstance(obj, dict):
        return {key: _serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return obj


def load_from_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading from JSON: {e}")
        return {}